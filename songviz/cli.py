from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .analyze import analyze_file
from .ingest import song_id_for_path
from .paths import (
    analysis_path_for_output_dir,
    output_dir_for_audio,
    video_path_for_output_dir,
)
from .render import RenderConfig, render_mp4
from .stems import ensure_demucs_stems
from .tidy import tidy_outputs
from .ui import UIConfig, run_ui


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Try hardlink first (fast, no extra disk) then fall back to copy.
    try:
        if dst.exists():
            dst.unlink()
        dst.hardlink_to(src)
        return
    except Exception:
        pass
    shutil.copyfile(src, dst)


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
    render.add_argument("--audio-codec", default="mp3", choices=["aac", "mp3"], help="Audio codec for MP4 (aac|mp3)")
    render.add_argument("--audio-bitrate", default="128k", help="Audio bitrate (e.g. 96k, 128k, 160k)")

    stems = sub.add_parser("stems", help="Separate an audio file into stems (Demucs)")
    stems.add_argument("audio_path", help="Path to audio (flac/mp3/wav)")
    stems.add_argument("--model", default="htdemucs", help="Demucs model name (default: htdemucs)")
    stems.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device (auto|cpu|cuda)")
    stems.add_argument("--force", action="store_true", help="Re-run separation even if cached")

    ui = sub.add_parser("ui", help="Interactive terminal UI (pick a song from songs/ and render)")
    ui.add_argument("--songs-dir", default="songs", help="Directory containing local audio files (default: songs/)")
    ui.add_argument("--outputs-dir", default="outputs", help="Directory for outputs (default: outputs/)")
    ui.add_argument("--seed", type=int, default=0, help="Deterministic seed")
    ui.add_argument("--width", type=int, default=960, help="Video width (pixels)")
    ui.add_argument("--height", type=int, default=540, help="Video height (pixels)")
    ui.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    ui.add_argument("--audio-codec", default="mp3", choices=["aac", "mp3"], help="Audio codec for MP4 (aac|mp3)")
    ui.add_argument("--audio-bitrate", default="128k", help="Audio bitrate (e.g. 96k, 128k, 160k)")

    tidy = sub.add_parser("tidy", help="Tidy outputs/ (move legacy dirs and loose files into hidden folders)")
    tidy.add_argument("--outputs-dir", default="outputs", help="Outputs directory (default: outputs/)")
    tidy.add_argument("--dry-run", action="store_true", help="Print what would happen without moving files")

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        if args.cmd == "analyze":
            analysis = analyze_file(args.audio_path)
            song_id = analysis["meta"]["song_id"]
            out_dir = output_dir_for_audio(args.audio_path, str(song_id))
            canonical = analysis_path_for_output_dir(out_dir)
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            if args.out:
                out_path = Path(args.out)
                if out_path.resolve() != canonical.resolve():
                    _copy_or_link(canonical, out_path)
                print(str(out_path))
            else:
                print(str(canonical))
            return 0

        if args.cmd == "render":
            analysis = analyze_file(args.audio_path)
            song_id = analysis["meta"]["song_id"]
            out_dir = output_dir_for_audio(args.audio_path, str(song_id))
            out_dir.mkdir(parents=True, exist_ok=True)

            analysis_path = analysis_path_for_output_dir(out_dir)
            analysis_path.parent.mkdir(parents=True, exist_ok=True)
            analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            canonical_mp4 = video_path_for_output_dir(out_dir)
            cfg = RenderConfig(
                width=int(args.width),
                height=int(args.height),
                fps=int(args.fps),
                seed=int(args.seed),
                audio_codec=str(args.audio_codec),
                audio_bitrate=str(args.audio_bitrate),
            )

            # Always render into the per-song output directory.
            render_mp4(analysis=analysis, audio_path=args.audio_path, out_path=canonical_mp4, cfg=cfg)

            # If --out is given, also place a copy/hardlink there.
            if args.out:
                out_mp4 = Path(args.out)
                if out_mp4.resolve() != canonical_mp4.resolve():
                    _copy_or_link(canonical_mp4, out_mp4)
                print(str(out_mp4))
            else:
                print(str(canonical_mp4))
            return 0

        if args.cmd == "stems":
            song_id = song_id_for_path(args.audio_path)
            out_dir = output_dir_for_audio(args.audio_path, str(song_id))
            out_dir.mkdir(parents=True, exist_ok=True)

            res = ensure_demucs_stems(
                args.audio_path,
                out_dir=out_dir,
                model=str(args.model),
                device=str(args.device),
                force=bool(args.force),
            )

            for name, p in res.stems.items():
                print(f"{name}: {p}")
            print(f"meta: {res.meta_path}", file=sys.stderr)
            return 0

        if args.cmd == "ui":
            cfg = UIConfig(
                songs_dir=Path(args.songs_dir),
                outputs_dir=Path(args.outputs_dir),
                seed=int(args.seed),
                width=int(args.width),
                height=int(args.height),
                fps=int(args.fps),
                audio_codec=str(args.audio_codec),
                audio_bitrate=str(args.audio_bitrate),
            )
            return int(run_ui(cfg) or 0)

        if args.cmd == "tidy":
            res = tidy_outputs(outputs_dir=args.outputs_dir, dry_run=bool(args.dry_run))
            for src, dst in res.moved:
                print(f"move: {src} -> {dst}")
            if not res.moved:
                print("Nothing to move.")
            return 0
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    raise AssertionError(f"Unhandled command: {args.cmd}")
