#!/usr/bin/env python3
"""Build a fresh, offline listening package for the 119--132s vocal reference."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


SOURCE_SHA256 = "657af9333edafbb6cf8cc573446650773d1169b0117c784106d1895cf802cb44"
SOURCE_RELATIVE = "songs/Gorillaz - Feel Good Inc (featuring De La Soul).flac"
START_S, END_S = 119, 132
PACKAGE_KIND = "songviz-vocal-behavior-reference"
TEMPLATE_RELATIVE = "experiments/templates/vocal_behavior_reference.html"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_pcm24_excerpt(source: Path, destination: Path, *, expected_hash: str,
                          start_s: int = START_S, end_s: int = END_S) -> dict[str, Any]:
    """Decode integer frames once, write PCM_24, and prove decoded-frame parity."""
    if sha256(source) != expected_hash:
        raise ValueError("provenance failure for original FLAC hash")
    info = sf.info(source)
    if info.subtype != "PCM_24":
        raise ValueError(f"original audio must be PCM_24 (got {info.subtype})")
    if info.channels < 1 or info.samplerate < 1 or end_s <= start_s:
        raise ValueError("invalid original audio format or excerpt bounds")
    start_frame, frames = start_s * info.samplerate, (end_s - start_s) * info.samplerate
    if info.frames < start_frame + frames:
        raise ValueError("original audio lacks the requested excerpt")
    with sf.SoundFile(source) as audio:
        audio.seek(start_frame)
        decoded = audio.read(frames, dtype="int32", always_2d=True)
    if decoded.shape != (frames, info.channels):
        raise ValueError("could not decode the exact requested source frames")
    sf.write(destination, decoded, info.samplerate, subtype="PCM_24", format="WAV")
    written = sf.info(destination)
    if (written.samplerate, written.channels, written.frames, written.subtype) != (info.samplerate, info.channels, frames, "PCM_24"):
        raise ValueError("written excerpt format mismatch")
    with sf.SoundFile(destination) as audio:
        replay = audio.read(frames, dtype="int32", always_2d=True)
    if not np.array_equal(decoded, replay):
        raise ValueError("written PCM_24 excerpt does not match decoded source frames")
    return {
        "path": destination.name, "sha256": sha256(destination), "bytes": destination.stat().st_size,
        "sample_rate": info.samplerate, "channels": info.channels, "subtype": "PCM_24",
        "frames": frames, "start_song_s": start_s, "end_song_s": end_s,
        "source_start_frame": start_frame, "decode_parity": "int32 decoded PCM frames equal after PCM_24 WAV round trip",
    }


def build(repo: Path, output: Path, *, source: Path | None = None,
          expected_source_hash: str = SOURCE_SHA256, start_s: int = START_S, end_s: int = END_S) -> Path:
    """Create a package. Existing destinations are refused before inspecting inputs."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    source = (source or repo / SOURCE_RELATIVE).resolve()
    template = repo / TEMPLATE_RELATIVE
    if not source.is_file() or not template.is_file():
        raise ValueError("required source audio or page template is missing")
    # Hash all inputs before output creation so no partial package represents bad provenance.
    source_hash, template_hash, builder_hash = sha256(source), sha256(template), sha256(Path(__file__))
    if source_hash != expected_source_hash:
        raise ValueError("provenance failure for original FLAC hash")
    output.mkdir(parents=True)
    try:
        audio = extract_pcm24_excerpt(source, output / "excerpt.wav", expected_hash=expected_source_hash,
                                      start_s=start_s, end_s=end_s)
        identity = {
            "kind": PACKAGE_KIND, "schema_version": 1, "source_sha256": source_hash,
            "excerpt_sha256": audio["sha256"], "start_song_s": start_s, "end_song_s": end_s,
            "builder_sha256": builder_hash, "template_sha256": template_hash,
        }
        identity["package_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        schema = {
            "kind": PACKAGE_KIND, "schema_version": 1,
            "purpose": "Human listening annotations of vocal behavior. It does not establish musical leadership, importance, or an exhaustive absence label.",
            "identity": identity,
            "span_schema": {"start_song_s": "number", "end_song_s": "number", "labels": ["speech_rap", "singing", "laughter", "other_vocal", "uncertain"], "timing_certainty": ["unspecified", "exact", "approximate", "uncertain"], "note": "optional string"},
            "blank_export": {"review_completed": False, "spans": []},
        }
        write_json(output / "schema.json", schema)
        page = template.read_text(encoding="utf-8").replace("{{REFERENCE_CONFIG}}", json.dumps({"identity": identity, "audio": audio}, sort_keys=True))
        (output / "index.html").write_text(page, encoding="utf-8")
        shutil.copy2(Path(__file__), output / "build_vocal_behavior_reference.py")
        shutil.copy2(template, output / "vocal_behavior_reference.html")
        manifest = {
            "kind": f"{PACKAGE_KIND}-package", "schema_version": 1, "identity": identity,
            "source": {"path": str(source.relative_to(repo)) if source.is_relative_to(repo) else str(source), "sha256": source_hash,
                       "bytes": source.stat().st_size, "native_format": {"sample_rate": audio["sample_rate"], "channels": audio["channels"], "subtype": "PCM_24"}},
            "excerpt": audio,
            "snapshots": {"builder": {"path": "build_vocal_behavior_reference.py", "sha256": sha256(output / "build_vocal_behavior_reference.py")}, "template": {"path": "vocal_behavior_reference.html", "sha256": sha256(output / "vocal_behavior_reference.html")}},
            "outputs": {name: sha256(output / name) for name in ("excerpt.wav", "schema.json", "index.html")},
            "input_hashes_before_creation": {"source": source_hash, "template": template_hash, "builder": builder_hash},
        }
        write_json(output / "manifest.json", manifest)
    except Exception:
        # The caller receives the failure; retain no misleading package directory.
        shutil.rmtree(output)
        raise
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = (args.output or repo / "outputs/reviews/vocal-behavior-reference-01").resolve()
    try:
        build(repo, output)
    except (OSError, ValueError, FileExistsError) as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
