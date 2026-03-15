from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf

from .ingest import sha256_file, _utc_now_iso


_EXPECTED_STEMS = ("drums", "bass", "vocals", "other")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def stems_dir_for_output_dir(out_dir: Path) -> Path:
    return out_dir / "stems"


def stems_meta_path(out_dir: Path) -> Path:
    return stems_dir_for_output_dir(out_dir) / "stems.json"


def _require_demucs() -> str:
    path = shutil.which("demucs")
    if path:
        return path

    # If running from a venv, the console script might live next to sys.executable
    # (even if the python binary itself is a symlink).
    try:
        cand = Path(sys.executable).parent / "demucs"
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)
    except Exception:
        pass

    # Or under sys.prefix (venv root).
    try:
        cand = Path(sys.prefix) / "bin" / "demucs"
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)
    except Exception:
        pass

    raise RuntimeError(
        "demucs was not found. Install it to enable stem separation.\n"
        "In your venv: python3 -m pip install -U demucs\n"
        "Or: pip install -e '.[stems]'"
    )


def _demucs_version() -> str | None:
    # Avoid importing demucs/torch; read installed distribution metadata instead.
    try:
        from importlib.metadata import version  # py3.10+

        return version("demucs")
    except Exception:
        return None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            dst.unlink()
        dst.hardlink_to(src)
        return
    except Exception:
        pass
    shutil.copyfile(src, dst)


@dataclass(frozen=True)
class StemsResult:
    stems_dir: Path
    stems: dict[str, Path]
    meta_path: Path
    cached: bool


def ensure_demucs_stems(
    audio_path: str | Path,
    *,
    out_dir: Path,
    model: str = "htdemucs",
    device: str = "auto",  # "auto" | "cpu" | "cuda"
    force: bool = False,
) -> StemsResult:
    """
    Ensure Demucs stems exist under outputs/<song>/stems/.

    Writes:
      - stems/{drums,bass,vocals,other}.wav
      - stems/stems.json
    """
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    stems_dir = stems_dir_for_output_dir(out_dir)
    stems_dir.mkdir(parents=True, exist_ok=True)
    meta_path = stems_meta_path(out_dir)

    input_sha256 = sha256_file(p)

    if not force and meta_path.exists():
        meta = _load_json(meta_path)
        if meta:
            try:
                if (
                    str(meta.get("schema_version")) == "1"
                    and meta.get("backend", {}).get("name") == "demucs"
                    and str(meta.get("input", {}).get("sha256")) == input_sha256
                    and str(meta.get("backend", {}).get("model")) == str(model)
                    and str(meta.get("backend", {}).get("device")) == str(device)
                ):
                    stems: dict[str, Path] = {}
                    ok = True
                    for name in _EXPECTED_STEMS:
                        sp = stems_dir / f"{name}.wav"
                        if not sp.exists() or sp.stat().st_size <= 0:
                            ok = False
                            break
                        stems[name] = sp
                    if ok:
                        return StemsResult(stems_dir=stems_dir, stems=stems, meta_path=meta_path, cached=True)
            except Exception:
                pass

    demucs = _require_demucs()

    # Demucs writes stems using torchaudio.save(), which requires TorchCodec on
    # newer torchaudio versions. Fail fast with a clear message.
    try:
        import torchcodec  # type: ignore  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "TorchCodec is required to write Demucs stems (torchaudio backend).\n"
            "Install: python3 -m pip install -U torchcodec\n"
            "Or: pip install -e '.[stems]'"
        ) from e

    tmp_root = stems_dir / "_tmp_demucs"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [demucs, "-n", str(model), "-o", str(tmp_root)]
    if device in ("cpu", "cuda"):
        cmd += ["-d", device]
    elif device != "auto":
        raise ValueError(f"Unsupported device: {device!r} (expected: auto|cpu|cuda)")
    cmd.append(str(p))

    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"demucs exited with code {res.returncode}")

    # Locate the single directory containing the expected stem wavs.
    candidates: list[Path] = []
    for d in tmp_root.rglob("*"):
        if not d.is_dir():
            continue
        if all((d / f"{name}.wav").exists() for name in _EXPECTED_STEMS):
            candidates.append(d)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one demucs output directory, found {len(candidates)} under {tmp_root}")

    src_dir = candidates[0]

    stems: dict[str, Path] = {}
    for name in _EXPECTED_STEMS:
        src = src_dir / f"{name}.wav"
        dst = stems_dir / f"{name}.wav"
        _copy_or_link(src, dst)
        stems[name] = dst

    # Basic audio metadata from one stem header (all stems should match).
    info = sf.info(str(stems["drums"]))

    meta: dict[str, Any] = {
        "schema_version": 1,
        "created_at": _utc_now_iso(),
        "input": {
            "path": str(p),
            "sha256": input_sha256,
            "bytes": int(p.stat().st_size),
        },
        "backend": {
            "name": "demucs",
            "version": _demucs_version(),
            "model": str(model),
            "device": str(device),
            "cmd": cmd,
        },
        "audio": {
            "sample_rate_hz": int(info.samplerate),
            "channels": int(info.channels),
            "duration_s": float(info.duration),
        },
        "stems": [
            {
                "name": name,
                "path": str(stems[name]),
                "bytes": int(stems[name].stat().st_size),
            }
            for name in _EXPECTED_STEMS
        ],
    }

    # Small integrity signal without hashing huge WAVs.
    meta_bytes = json.dumps(meta, sort_keys=True).encode("utf-8")
    meta["meta_sha256"] = _sha256_bytes(meta_bytes)

    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Best-effort cleanup of demucs intermediate outputs.
    try:
        shutil.rmtree(tmp_root)
    except Exception:
        pass

    return StemsResult(stems_dir=stems_dir, stems=stems, meta_path=meta_path, cached=False)


# ---------------------------------------------------------------------------
# DrumSep: optional component separation on a Demucs drum stem
# ---------------------------------------------------------------------------

_DRUMSEP_MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
_DRUMSEP_COMPONENTS = ("kick", "snare", "toms", "hh", "ride", "crash")


def _audio_separator_available() -> bool:
    try:
        from audio_separator.separator import Separator  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _audio_separator_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("audio-separator")
    except Exception:
        return None


def _source_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime": float(stat.st_mtime),
    }


@dataclass(frozen=True)
class DrumSepResult:
    drumsep_dir: Path
    components: dict[str, Path]
    meta_path: Path
    cached: bool


def ensure_drumsep_components(
    drum_stem_path: str | Path,
    *,
    out_dir: Path,
    force: bool = False,
) -> DrumSepResult | None:
    """Run DrumSep on a Demucs drum stem, producing per-component WAVs.

    Returns None if audio-separator is not installed (graceful degradation).
    """
    if not _audio_separator_available():
        return None

    p = Path(drum_stem_path)
    if not p.exists():
        return None

    drumsep_dir = stems_dir_for_output_dir(out_dir) / "drumsep"
    drumsep_dir.mkdir(parents=True, exist_ok=True)
    meta_path = drumsep_dir / "drumsep.json"

    source_fp = _source_fingerprint(p)

    # Check cache
    if not force and meta_path.exists():
        meta = _load_json(meta_path)
        if meta:
            try:
                cached_fp = meta.get("source", {})
                if (
                    str(meta.get("schema_version")) == "1"
                    and str(meta.get("model")) == _DRUMSEP_MODEL
                    and int(cached_fp.get("size_bytes", -1)) == source_fp["size_bytes"]
                    and float(cached_fp.get("mtime", -1)) == source_fp["mtime"]
                ):
                    components: dict[str, Path] = {}
                    ok = True
                    for comp in _DRUMSEP_COMPONENTS:
                        cp = drumsep_dir / f"{comp}.wav"
                        if not cp.exists() or cp.stat().st_size <= 0:
                            ok = False
                            break
                        components[comp] = cp
                    if ok:
                        return DrumSepResult(
                            drumsep_dir=drumsep_dir,
                            components=components,
                            meta_path=meta_path,
                            cached=True,
                        )
            except Exception:
                pass

    from audio_separator.separator import Separator  # type: ignore

    sep = Separator(
        output_dir=str(drumsep_dir),
        output_format="WAV",
    )
    sep.load_model(model_filename=_DRUMSEP_MODEL)
    output_files = sep.separate(str(p))

    # Map audio-separator output filenames → canonical component names.
    # separate() returns bare filenames; actual files are in drumsep_dir.
    components: dict[str, Path] = {}
    for comp in _DRUMSEP_COMPONENTS:
        # audio-separator names outputs like: drums_(kick)_MDX23C-DrumSep-...wav
        matched = [f for f in output_files if f"({comp})" in f]
        if not matched:
            raise RuntimeError(
                f"DrumSep did not produce expected '{comp}' component. "
                f"Output files: {output_files}"
            )
        src = drumsep_dir / Path(matched[0]).name
        dst = drumsep_dir / f"{comp}.wav"
        if src != dst:
            _copy_or_link(src, dst)
            try:
                if src.exists():
                    src.unlink()
            except Exception:
                pass
        components[comp] = dst

    # Clean up any other files left by audio-separator
    for f in output_files:
        fp = drumsep_dir / Path(f).name
        if fp.exists() and fp.name not in {f"{c}.wav" for c in _DRUMSEP_COMPONENTS}:
            try:
                fp.unlink()
            except Exception:
                pass

    meta: dict[str, Any] = {
        "schema_version": 1,
        "created_at": _utc_now_iso(),
        "source": source_fp,
        "model": _DRUMSEP_MODEL,
        "audio_separator_version": _audio_separator_version(),
        "components": [
            {
                "name": comp,
                "path": str(components[comp]),
                "bytes": int(components[comp].stat().st_size),
            }
            for comp in _DRUMSEP_COMPONENTS
        ],
    }
    meta_bytes = json.dumps(meta, sort_keys=True).encode("utf-8")
    meta["meta_sha256"] = _sha256_bytes(meta_bytes)

    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return DrumSepResult(
        drumsep_dir=drumsep_dir,
        components=components,
        meta_path=meta_path,
        cached=False,
    )
