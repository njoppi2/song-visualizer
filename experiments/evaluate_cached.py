"""Evaluate existing benchmark artifacts without allowing extraction or downloads.

Usage: .songviz/venv/bin/python experiments/evaluate_cached.py --out outputs/reviews/restart-02/evaluation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.build_review import fingerprint, run
from songviz.bench import evaluate_all_songs, find_benchmark_songs, format_bench_report
from songviz.paths import output_dir_for_audio, reduced_path_for_output_dir


def evaluate(out: Path) -> None:
    songs_dir = ROOT / "songs"
    songs = find_benchmark_songs(songs_dir)
    if not songs:
        raise RuntimeError("No benchmark inputs found")
    provenance = {"mode": "cached_only", "extraction_revision": "unknown",
                  "git_head": run("git", "rev-parse", "HEAD"),
                  "git_status": run("git", "status", "--short"), "inputs": []}
    paths = []
    for song in songs:
        cached = output_dir_for_audio(song["audio_path"], song["song_id"], outputs_root=ROOT / "outputs")
        if not reduced_path_for_output_dir(cached).is_file():
            raise RuntimeError(f"Missing reduced cache: {cached}; refusing to run extraction")
        paths.append(song["audio_path"])
        paths.extend(p for p in (cached / "analysis").glob("*.json") if p.name in {"analysis.json", "story.json", "reduced.json"})
        paths.extend(p for p in song["ref_dir"].iterdir() if p.is_file())
    out.mkdir(parents=True, exist_ok=False)
    for p in paths:
        provenance["inputs"].append(fingerprint(p))
        if p.suffix in {".json", ".mid"}:
            dest = out / "inputs" / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    for rel in ("songviz/bench.py", "songviz/eval.py", "experiments/evaluate_cached.py"):
        src = ROOT / rel
        dest = out / "inputs" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        provenance["inputs"].append(fingerprint(src))
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    result = evaluate_all_songs(songs_dir)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "report.txt").write_text(format_bench_report(result) + "\n")
    print(json.dumps({"songs": result.get("song_count"), "success": result.get("success_count"),
                      "errors": result.get("errors"), "report": str(out / "report.txt")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    # Existing benchmark helpers use repo-relative paths.
    import os
    os.chdir(ROOT)
    evaluate(args.out.resolve())
