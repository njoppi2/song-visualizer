from __future__ import annotations

import argparse


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
        raise NotImplementedError("Analyze is not implemented yet (v0 scaffold).")
    if args.cmd == "render":
        raise NotImplementedError("Render is not implemented yet (v0 scaffold).")
    raise AssertionError(f"Unhandled command: {args.cmd}")

