from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_HEX16_RE = re.compile(r"^[0-9a-f]{16}$")
_EXPORT_EXTS = {".mp4", ".mov", ".mkv", ".avi"}


@dataclass(frozen=True)
class TidyResult:
    moved: list[tuple[Path, Path]]
    skipped: list[Path]


def _unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    for i in range(1, 10_000):
        cand = dest.with_name(f"{stem}__{i}{suffix}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"Could not find a free destination name near: {dest}")


def tidy_outputs(
    outputs_dir: str | Path = "outputs",
    *,
    legacy_dirname: str = ".legacy",
    exports_dirname: str = ".exports",
    scratch_dir: str | Path = ".songviz/scratch",
    dry_run: bool = False,
) -> TidyResult:
    out = Path(outputs_dir)
    moved: list[tuple[Path, Path]] = []
    skipped: list[Path] = []

    if not out.exists():
        return TidyResult(moved=moved, skipped=skipped)

    legacy_dir = out / legacy_dirname
    exports_dir = out / exports_dirname
    scratch = Path(scratch_dir)

    for p in sorted(out.iterdir(), key=lambda x: x.name.lower()):
        if p.name in (legacy_dirname, exports_dirname):
            continue
        if p.name.startswith("."):
            skipped.append(p)
            continue

        if p.is_dir():
            is_legacy_song_id_dir = _HEX16_RE.match(p.name) is not None
            is_legacy_flat_analysis = (p / "analysis.json").exists()
            if is_legacy_song_id_dir or is_legacy_flat_analysis:
                legacy_dir.mkdir(parents=True, exist_ok=True)
                dest = _unique_dest(legacy_dir / p.name)
                moved.append((p, dest))
                if not dry_run:
                    p.rename(dest)
            else:
                skipped.append(p)
            continue

        if p.is_file():
            ext = p.suffix.lower()
            if ext in _EXPORT_EXTS:
                exports_dir.mkdir(parents=True, exist_ok=True)
                dest = _unique_dest(exports_dir / p.name)
            else:
                scratch.mkdir(parents=True, exist_ok=True)
                dest = _unique_dest(scratch / p.name)
            moved.append((p, dest))
            if not dry_run:
                p.rename(dest)
            continue

        skipped.append(p)

    return TidyResult(moved=moved, skipped=skipped)

