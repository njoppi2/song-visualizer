from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyze import analyze_file
from .render import RenderConfig, render_mp4


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
    render.add_argument("--width", type=int, default=960, help="Video width (pixels)")
    render.add_argument("--height", type=int, default=540, help="Video height (pixels)")
    render.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        if args.cmd == "analyze":
            analysis = analyze_file(args.audio_path)
            song_id = analysis["meta"]["song_id"]
            default_out = Path("outputs") / str(song_id) / "analysis.json"
            out_path = Path(args.out) if args.out else default_out
            out_path.parent.mkdir(parents=True, exist_ok=True)

            out_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(str(out_path))
            return 0

        if args.cmd == "render":
            analysis = analyze_file(args.audio_path)
            song_id = analysis["meta"]["song_id"]
            out_dir = Path("outputs") / str(song_id)
            out_dir.mkdir(parents=True, exist_ok=True)

            analysis_path = out_dir / "analysis.json"
            analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            mp4_path = Path(args.out) if args.out else (out_dir / "video.mp4")
            cfg = RenderConfig(
                width=int(args.width),
                height=int(args.height),
                fps=int(args.fps),
                seed=int(args.seed),
            )
            render_mp4(analysis=analysis, audio_path=args.audio_path, out_path=mp4_path, cfg=cfg)
            print(str(mp4_path))
            return 0
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    raise AssertionError(f"Unhandled command: {args.cmd}")
