"""Build one artistic preview from the exact, user-reviewed rhythm inputs.

This consumes immutable review artifacts rather than re-running extraction or
fitting a beat grid. The reference is the accepted timing diagnostic, not a
competing artistic design. Never overwrites a review or production caches.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

from PIL import Image
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.build_review import fingerprint, run
from songviz.ingest import sha256_file
from songviz.render import RenderConfig, _render_mp4_with_visualizer


def verify_hash(path: Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"Review input changed: {path}")


def reviewed_inputs(review: Path) -> dict:
    """Check the exact parent package before trusting accepted timing or audio."""
    manifest_path = review / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    feedback = json.loads((ROOT / "benchmark/feedback/rhythm-review-01.json").read_text())
    verify_hash(manifest_path, feedback["manifest_sha256"])
    clip = next(c for c in manifest["clips"] if c["id"] == "01-opening")
    if (clip["start_s"], clip["end_s"]) != (0, 22):
        raise ValueError("This study expects the reviewed 0–22s opening")
    paths = {"timing": review / "timing.json", "percussion": review / "percussion-candidates.json"}
    for key, path in paths.items():
        verify_hash(path, manifest[key]["sha256"])
    paths["audio"] = review / clip["id"] / "original.wav"
    paths["reference"] = review / clip["id"] / "regular.mp4"
    for key in ["audio", "reference"]:
        record = next(r for r in clip["outputs"] if Path(r["path"]).name == paths[key].name)
        verify_hash(paths[key], record["sha256"])
    source = Path(manifest["source_audio"]["path"])
    if not source.is_absolute():
        source = ROOT / source
    verify_hash(source, manifest["source_audio"]["sha256"])
    timing = json.loads(paths["timing"].read_text())
    if timing["candidate"]["status"] != "candidate":
        raise ValueError("Reviewed package has no supported timing candidate")
    info = sf.info(paths["audio"])
    if abs(info.duration - 22) > 1 / info.samplerate:
        raise ValueError("Reviewed audio has an unexpected duration")
    return {"manifest": manifest, "clip": clip, "paths": paths, "timing": timing,
            "events": json.loads(paths["percussion"].read_text())}


def build(review: Path, out: Path) -> None:
    from experiments.passage_visualizer import PassageVisualizer

    data = reviewed_inputs(review)
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "inputs"
    inputs.mkdir()
    for key in ["timing", "percussion"]:
        shutil.copy2(data["paths"][key], inputs / data["paths"][key].name)
    shutil.copy2(review / "manifest.json", inputs / "parent-manifest.json")
    for rel in ["experiments/passage_visualizer.py", "experiments/build_visual_passage.py",
                "experiments/templates/passage_review.html", "songviz/render.py",
                "benchmark/feedback/rhythm-review-01.json",
                "benchmark/feedback/rhythm-review-01-clarification.md"]:
        dest = inputs / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    shutil.copy2(data["paths"]["audio"], out / "original.wav")
    shutil.copy2(data["paths"]["reference"], out / "reference.mp4")
    cfg = RenderConfig(width=960, height=540, fps=60, audio_codec="aac", audio_bitrate="192k")
    visualizer = PassageVisualizer(data["timing"]["candidate"]["beat_times_s"],
                                  data["events"], start=0, end=22, width=960, height=540)
    print("Rendering opening visual study (22 seconds, 60 FPS)", flush=True)
    _render_mp4_with_visualizer(audio_path=out / "original.wav", out_path=out / "visual.mp4",
                               cfg=cfg, duration_s=22, visualizer=visualizer)
    snapshots = out / "frames"
    snapshots.mkdir()
    for t in [0, 5.5, 6.35, 6.8, 7.7, 12, 18, 21.5]:
        Image.frombytes("RGB", (960, 540), visualizer.frame_rgb24(t)).save(snapshots / f"{t:05.2f}.png")
    shutil.copy2(snapshots / "06.80.png", out / "poster.png")
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": run("git", "rev-parse", "HEAD"),
        "parent_review": fingerprint(review / "manifest.json"),
        "source_audio": data["manifest"]["source_audio"],
        "passage": data["clip"]["id"], "start_s": 0, "end_s": 22,
        "render_config": asdict(cfg), "maximum_frame_quantization_ms": 1000 / cfg.fps,
        "comparison": "same original audio and accepted pulse/percussion; artistic layout vs timing diagnostic",
        "design": {"snare": "dominant warm rings/arcs", "kick": "grounded cyan impulse",
                   "hh": "small gold peripheral detail", "pulse": "subdued central breathing"},
        "production_pipeline_changed": False, "extraction_rerun": False,
        "user_review": "pending for artistic design; underlying timing supported on reviewed excerpts",
        "delegation": "Terra implemented visualizer, its tests, and review page; lead built package and verified integration",
        "inputs": [fingerprint(p) for p in sorted(inputs.rglob("*")) if p.is_file()],
        "outputs": [fingerprint(p) for p in sorted(out.rglob("*")) if p.is_file() and inputs not in p.parents],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    template = (ROOT / "experiments/templates/passage_review.html").read_text()
    (out / "index.html").write_text(template.replace("{{MANIFEST_SHA}}", sha256_file(out / "manifest.json")))
    print(f"Ready: {out / 'index.html'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=ROOT / "outputs/reviews/rhythm-review-01")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.review.resolve(), args.out.resolve())
