from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze import analyze_file
from .ingest import song_id_for_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="songviz", description="SongViz CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    analyze = sub.add_parser("analyze", help="Analyze an audio file and write analysis.json")
    analyze.add_argument("audio_path", help="Path to audio (flac/mp3/wav)")
    analyze.add_argument("--out", default=None, help="Output analysis.json path (optional)")

    render = sub.add_parser("render", help="Render a music-reactive MP4")
    render.add_argument("audio_path", help="Path to audio (flac/mp3/wav)")
    render.add_argument("--out", default=None, help="Output mp4 path (optional)")
    render.add_argument("--seed", type=int, default=0, help="Deterministic seed")

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "analyze":
        song_id = song_id_for_path(args.audio_path)
        default_out = Path("outputs") / song_id / "analysis.json"
        out_path = Path(args.out) if args.out else default_out
        out_path.parent.mkdir(parents=True, exist_ok=True)

        analysis = analyze_file(args.audio_path)
        out_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(str(out_path))
        return 0
    if args.cmd == "render":
        raise NotImplementedError("Render is not implemented yet (v0 scaffold).")
    raise AssertionError(f"Unhandled command: {args.cmd}")
