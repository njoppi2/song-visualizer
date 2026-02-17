from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import librosa
import numpy as np

from .analyze import analyze_file
from .analyze import analyze_audio
from .features import (
    bass_pitch_hz,
    drums_band_energy_3,
    other_chroma_12,
    vocals_note_events_basic_pitch,
    vocals_pitch_hz,
)
from .ingest import song_id_for_path
from .paths import (
    analysis_path_for_output_dir,
    output_dir_for_audio,
    story_path_for_output_dir,
    video_path_for_output_dir,
)
from .render import RenderConfig, render_mp4, render_mp4_stems4
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
    render.add_argument("--layout", default="mix", choices=["mix", "stems4"], help="Layout: mix | stems4")
    render.add_argument("--audio-codec", default="mp3", choices=["aac", "mp3"], help="Audio codec for MP4 (aac|mp3)")
    render.add_argument("--audio-bitrate", default="128k", help="Audio bitrate (e.g. 96k, 128k, 160k)")
    render.add_argument("--stems-model", default="htdemucs", help="Demucs model for stems4 layout (default: htdemucs)")
    render.add_argument("--stems-device", default="auto", choices=["auto", "cpu", "cuda"], help="Device for stems4 (auto|cpu|cuda)")
    render.add_argument("--stems-force", action="store_true", help="Re-run stem separation for stems4 even if cached")
    render.add_argument("--lyrics", action="store_true", help="Load lyrics/alignment.json (if present) and render word overlays")

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
    ui.add_argument("--layout", default="mix", choices=["mix", "stems4"], help="Layout: mix | stems4")
    ui.add_argument("--audio-codec", default="mp3", choices=["aac", "mp3"], help="Audio codec for MP4 (aac|mp3)")
    ui.add_argument("--audio-bitrate", default="128k", help="Audio bitrate (e.g. 96k, 128k, 160k)")
    ui.add_argument("--stems-model", default="htdemucs", help="Demucs model for stems4 layout (default: htdemucs)")
    ui.add_argument("--stems-device", default="auto", choices=["auto", "cpu", "cuda"], help="Device for stems4 (auto|cpu|cuda)")
    ui.add_argument("--stems-force", action="store_true", help="Re-run stem separation for stems4 even if cached")

    tidy = sub.add_parser("tidy", help="Tidy outputs/ (move legacy dirs and loose files into hidden folders)")
    tidy.add_argument("--outputs-dir", default="outputs", help="Outputs directory (default: outputs/)")
    tidy.add_argument("--dry-run", action="store_true", help="Print what would happen without moving files")

    lyrics_p = sub.add_parser("lyrics", help="Run lyrics alignment and write lyrics/alignment.json")
    lyrics_p.add_argument("audio_path", help="Path to audio (flac/mp3/wav)")
    lyrics_p.add_argument("--language", default="en", help="Language code for Whisper (default: en)")
    lyrics_p.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    lyrics_p.add_argument("--force", action="store_true", help="Re-run alignment even if cached")

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
            # Also export story as a separate artifact for easier human inspection.
            story_p = story_path_for_output_dir(out_dir)
            story_p.parent.mkdir(parents=True, exist_ok=True)
            story_p.write_text(json.dumps(analysis.get("story", {}), indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
            story_p = story_path_for_output_dir(out_dir)
            story_p.parent.mkdir(parents=True, exist_ok=True)
            story_p.write_text(json.dumps(analysis.get("story", {}), indent=2, sort_keys=True) + "\n", encoding="utf-8")

            canonical_mp4 = video_path_for_output_dir(out_dir)
            cfg = RenderConfig(
                width=int(args.width),
                height=int(args.height),
                fps=int(args.fps),
                seed=int(args.seed),
                audio_codec=str(args.audio_codec),
                audio_bitrate=str(args.audio_bitrate),
            )

            # Load lyrics alignment when --lyrics flag is set.
            alignment = None
            if getattr(args, "lyrics", False):
                from .lyrics import load_alignment
                alignment = load_alignment(out_dir)
                if alignment is None:
                    print(
                        f"warning: --lyrics set but lyrics/alignment.json not found under {out_dir}. "
                        "Run `songviz lyrics` first.",
                        file=sys.stderr,
                    )

            # Always render into the per-song output directory.
            layout = str(args.layout)
            if layout == "mix":
                render_mp4(analysis=analysis, audio_path=args.audio_path, out_path=canonical_mp4, cfg=cfg, alignment=alignment)
            elif layout == "stems4":
                stems = ensure_demucs_stems(
                    args.audio_path,
                    out_dir=out_dir,
                    model=str(args.stems_model),
                    device=str(args.stems_device),
                    force=bool(args.stems_force),
                )

                stem_analyses: dict[str, dict] = {}
                hop_length = 512
                frame_length = 2048
                for name, stem_path in stems.stems.items():
                    y, sr = librosa.load(stem_path, sr=22050, mono=True)
                    y = np.asarray(y, dtype=np.float32)
                    a = analyze_audio(y, int(sr), hop_length=hop_length, frame_length=frame_length)
                    # Use the mix story/sections for all stems (keeps a single "narrative timeline").
                    a["story"] = analysis.get("story", {})

                    # Optional stem-specific features (in-memory only).
                    feats: dict[str, np.ndarray] = {}
                    if name == "drums":
                        feats["drums_bands_3"] = drums_band_energy_3(
                            y, int(sr), hop_length=hop_length, n_fft=frame_length
                        )
                    elif name == "bass":
                        feats["pitch_hz"] = bass_pitch_hz(
                            y, int(sr), hop_length=hop_length, frame_length=frame_length
                        )
                    elif name == "vocals":
                        feats["pitch_hz"] = vocals_pitch_hz(
                            y, int(sr), hop_length=hop_length, frame_length=frame_length
                        )
                        # Prefer note-events for a readable "what note is sung" visualization.
                        try:
                            feats["note_events"] = vocals_note_events_basic_pitch(str(stem_path))
                        except Exception:
                            pass
                    elif name == "other":
                        feats["chroma_12"] = other_chroma_12(
                            y, int(sr), hop_length=hop_length, n_fft=frame_length
                        )
                    if feats:
                        a["features"] = feats

                    if name != "drums":
                        a["beats"]["beat_times_s"] = []
                        a["beats"]["tempo_bpm"] = 0.0
                    stem_analyses[name] = a

                render_mp4_stems4(
                    stem_analyses=stem_analyses,
                    duration_s=float(analysis["meta"]["duration_s"]),
                    audio_path=args.audio_path,
                    out_path=canonical_mp4,
                    cfg=cfg,
                    alignment=alignment,
                )
            else:
                raise AssertionError(f"Unknown layout: {layout!r}")

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
                layout=str(args.layout),
                audio_codec=str(args.audio_codec),
                audio_bitrate=str(args.audio_bitrate),
                stems_model=str(args.stems_model),
                stems_device=str(args.stems_device),
                stems_force=bool(args.stems_force),
            )
            return int(run_ui(cfg) or 0)

        if args.cmd == "tidy":
            res = tidy_outputs(outputs_dir=args.outputs_dir, dry_run=bool(args.dry_run))
            for src, dst in res.moved:
                print(f"move: {src} -> {dst}")
            if not res.moved:
                print("Nothing to move.")
            return 0

        if args.cmd == "lyrics":
            from .lyrics import align_lyrics

            song_id = song_id_for_path(args.audio_path)
            out_dir = output_dir_for_audio(args.audio_path, str(song_id))
            out_dir.mkdir(parents=True, exist_ok=True)

            # Use isolated vocals stem when available for better alignment quality.
            vocals_stem = out_dir / "stems" / "vocals.wav"
            if not vocals_stem.exists():
                vocals_stem = None  # type: ignore[assignment]

            result = align_lyrics(
                args.audio_path,
                song_id=str(song_id),
                output_dir=out_dir,
                vocals_stem=vocals_stem,
                language=str(args.language),
                model_name=str(args.model),
                force=bool(args.force),
            )
            print(str(result))
            return 0

    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    raise AssertionError(f"Unhandled command: {args.cmd}")
