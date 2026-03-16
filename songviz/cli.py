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
    reduced_path_for_output_dir,
    sonify_path_for_output_dir,
    story_path_for_output_dir,
    video_path_for_output_dir,
)
from .pipeline import run_render_pipeline
from .render import RenderConfig
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
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: small)",
    )
    lyrics_p.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "whisper", "whisperx", "stable_whisper"],
        help="Alignment backend (default: auto; prefers whisperx > stable_whisper > whisper)",
    )
    lyrics_p.add_argument(
        "--no-auto-calibrate",
        action="store_true",
        help="Disable automatic global timing-offset calibration",
    )
    lyrics_p.add_argument("--force", action="store_true", help="Re-run alignment even if cached")
    lyrics_p.add_argument("--artist", default=None, help="Artist name (overrides ID3 tag)")
    lyrics_p.add_argument("--title", default=None, help="Track title (overrides ID3 tag)")

    lyrics_tap = sub.add_parser("lyrics-tap", help="Tap along with a song to correct word timing")
    lyrics_tap.add_argument("audio_path", help="Path to audio (flac/mp3/wav)")
    lyrics_tap.add_argument("--offset", type=float, default=0.0, help="Global offset added to all taps (compensate ffplay latency)")

    lyrics_template = sub.add_parser("lyrics-template", help="Generate blank corrections.yaml for manual editing")
    lyrics_template.add_argument("audio_path", help="Path to audio (flac/mp3/wav)")

    lyrics_correct = sub.add_parser("lyrics-correct", help="Apply corrections.yaml to alignment.json")
    lyrics_correct.add_argument("audio_path", help="Path to audio (flac/mp3/wav)")

    lyrics_preview = sub.add_parser("lyrics-preview", help="Render a lyrics-only preview video")
    lyrics_preview.add_argument("audio_path", help="Path to audio (flac/mp3/wav)")
    lyrics_preview.add_argument("--out", default=None, help="Output mp4 path (optional)")

    sonify_p = sub.add_parser("sonify", help="Sonify reduced.json into a debug WAV")
    sonify_p.add_argument("audio_path", help="Path to audio (flac/mp3/wav)")
    sonify_p.add_argument("--out", default=None, help="Output WAV path (optional)")
    sonify_p.add_argument("--diagnose", action="store_true", help="Print diagnostic stats, warnings, and write per-layer debug WAVs")

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

            from .viz import generate_all
            generate_all(
                analysis,
                out_dir,
                stems_dir=out_dir / "stems",
                has_lyrics=(out_dir / "lyrics" / "alignment.json").exists(),
            )

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

            canonical_mp4 = run_render_pipeline(
                Path(args.audio_path),
                out_dir,
                analysis,
                str(args.layout),
                cfg,
                stems_model=str(args.stems_model),
                stems_device=str(args.stems_device),
                stems_force=bool(args.stems_force),
                alignment=alignment,
            )

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
                align_backend=str(args.backend),
                auto_calibrate=not bool(args.no_auto_calibrate),
                force=bool(args.force),
                artist=args.artist,
                title=args.title,
            )
            print(str(result))

            # Generate alignment diagnostic in a subprocess to avoid OOM
            # (Whisper model may still be in memory from alignment).
            diag_audio = str(vocals_stem) if vocals_stem is not None else str(args.audio_path)
            lyrics_dir = str(out_dir / "lyrics")
            try:
                import subprocess
                cp = subprocess.run(
                    [
                        sys.executable, "-c",
                        "import json, sys; "
                        "from pathlib import Path; "
                        "from songviz.lyrics import load_alignment; "
                        "from songviz.viz import generate_lyrics_diagnostic; "
                        "alignment = load_alignment(Path(sys.argv[1])); "
                        "stats = generate_lyrics_diagnostic(alignment, Path(sys.argv[2]), Path(sys.argv[3])) if alignment else {}; "
                        "print(json.dumps(stats, default=str))",
                        str(out_dir), diag_audio, lyrics_dir,
                    ],
                    capture_output=True, text=True, timeout=120,
                )
                if cp.returncode == 0 and cp.stdout.strip():
                    import json as _json
                    stats = _json.loads(cp.stdout.strip())
                    total = stats.get("total_words", 0)
                    ok = stats.get("well_aligned", 0)
                    silence = stats.get("in_silence", 0)
                    early = stats.get("significantly_early", 0)
                    print(
                        f"diagnostic: {ok}/{total} well-aligned, "
                        f"{silence} in-silence, {early} >0.1s early"
                    )
                    if stats.get("png_path"):
                        print(str(stats["png_path"]))
            except Exception:
                pass  # matplotlib or librosa not installed, or subprocess failed

            return 0

        if args.cmd == "lyrics-tap":
            from .lyrics import apply_corrections, load_alignment, measure_alignment_quality
            from .tap import run_tap_session, taps_to_corrections, write_corrections

            song_id = song_id_for_path(args.audio_path)
            out_dir = output_dir_for_audio(args.audio_path, str(song_id))
            alignment = load_alignment(out_dir)
            if alignment is None:
                print("error: alignment.json not found. Run `songviz lyrics` first.", file=sys.stderr)
                return 2

            result = run_tap_session(
                args.audio_path,
                alignment,
                offset=float(args.offset),
            )
            print(f"Tapped {result.tapped_count}/{result.total_words} words")
            if result.taps:
                corrections = taps_to_corrections(alignment, result.taps)
                write_corrections(out_dir, alignment, corrections, source="tap_along")
                stats = apply_corrections(out_dir)
                print(f"Applied: {stats['applied']}, verified: {stats['verified']}, skipped: {stats['skipped']}, mismatched: {stats['mismatched']}")
                quality = measure_alignment_quality(out_dir)
                if quality.get("total_corrected_words", 0) > 0:
                    print(f"Quality: mean={quality['mean_error_s']:.3f}s, median={quality['median_error_s']:.3f}s, "
                          f"within 100ms={quality['pct_within_100ms']:.0f}%, "
                          f"systematic offset={quality['systematic_offset_s']:+.3f}s")
            return 0

        if args.cmd == "lyrics-template":
            from .lyrics import generate_corrections_template

            song_id = song_id_for_path(args.audio_path)
            out_dir = output_dir_for_audio(args.audio_path, str(song_id))
            result = generate_corrections_template(out_dir)
            print(str(result))
            return 0

        if args.cmd == "lyrics-correct":
            from .lyrics import apply_corrections, measure_alignment_quality

            song_id = song_id_for_path(args.audio_path)
            out_dir = output_dir_for_audio(args.audio_path, str(song_id))
            stats = apply_corrections(out_dir)
            print(f"Applied: {stats['applied']}, verified: {stats['verified']}, skipped: {stats['skipped']}, mismatched: {stats['mismatched']}")
            quality = measure_alignment_quality(out_dir)
            if quality.get("total_corrected_words", 0) > 0:
                print(f"Quality: mean={quality['mean_error_s']:.3f}s, median={quality['median_error_s']:.3f}s, "
                      f"max={quality['max_error_s']:.3f}s, "
                      f"within 100ms={quality['pct_within_100ms']:.0f}%, "
                      f"within 200ms={quality['pct_within_200ms']:.0f}%, "
                      f"systematic offset={quality['systematic_offset_s']:+.3f}s")
            else:
                print("No corrected words found in corrections.yaml.")
            return 0

        if args.cmd == "lyrics-preview":
            from .lyrics import load_alignment
            from .render import render_mp4_lyrics_only

            song_id = song_id_for_path(args.audio_path)
            out_dir = output_dir_for_audio(args.audio_path, str(song_id))
            alignment = load_alignment(out_dir)
            if alignment is None:
                print("error: alignment.json not found. Run `songviz lyrics` first.", file=sys.stderr)
                return 2

            cfg = RenderConfig(width=960, height=270, fps=15)
            out_mp4 = Path(args.out) if args.out else out_dir / "lyrics" / "lyrics_preview.mp4"
            # Estimate duration from last segment end.
            segs = alignment.get("segments", [])
            if segs:
                duration_s = max(float(s.get("end_s", 0.0)) for s in segs) + 2.0
            else:
                duration_s = 30.0

            render_mp4_lyrics_only(
                alignment=alignment,
                audio_path=args.audio_path,
                out_path=out_mp4,
                cfg=cfg,
                duration_s=duration_s,
            )
            print(str(out_mp4))
            return 0

        if args.cmd == "sonify":
            from .sonify import diagnose_reduced, sonify_reduced, sonify_reduced_layers

            song_id = song_id_for_path(args.audio_path)
            out_dir = output_dir_for_audio(args.audio_path, str(song_id))
            reduced_path = reduced_path_for_output_dir(out_dir)
            if not reduced_path.exists():
                print(
                    f"error: {reduced_path} not found — run the reduction pipeline first "
                    "(e.g. `songviz render --layout stems4`).",
                    file=sys.stderr,
                )
                return 2
            reduced = json.loads(reduced_path.read_text(encoding="utf-8"))
            wav_path = Path(args.out) if args.out else sonify_path_for_output_dir(out_dir)
            sonify_reduced(reduced, wav_path)
            print(str(wav_path))

            if args.diagnose:
                diag = diagnose_reduced(reduced)
                analysis_dir = out_dir / "analysis"

                # Print stats table
                print(f"\n{'=== Diagnostic Report ===':=^60}")
                print(f"Song duration: {diag['song_duration_s']:.1f}s\n")

                d = diag["drums"]
                print(f"--- Drums ---")
                print(f"  hits: {d['event_count']}  density: {d['density_hits_per_s']:.1f}/s")
                print(f"  components: {d['component_counts']}")
                print(f"  RMS: {d['rms']:.5f}  energy: {d['energy_pct']:.1f}%")

                for layer in ("vocals", "bass"):
                    s = diag[layer]
                    print(f"\n--- {layer.capitalize()} (source: {s['source']}) ---")
                    print(f"  notes: {s['event_count']}  coverage: {s['coverage_pct']:.1f}%  active: {s['total_active_s']:.1f}s")
                    if s["event_count"] > 0:
                        print(f"  duration: mean={s['mean_duration_s']:.3f}s  median={s['median_duration_s']:.3f}s")
                        print(f"  MIDI: {s['midi_min']:.0f}-{s['midi_max']:.0f}  mean={s['midi_mean']:.1f}  median={s['midi_median']:.1f}")
                    print(f"  RMS: {s['rms']:.5f}  energy: {s['energy_pct']:.1f}%")

                if diag["warnings"]:
                    print(f"\n{'!!! Warnings !!!':!^60}")
                    for w in diag["warnings"]:
                        print(f"  - {w}")
                else:
                    print("\nNo warnings.")

                # Write per-layer debug WAVs
                layer_paths = sonify_reduced_layers(reduced, analysis_dir)
                print(f"\nDebug WAVs:")
                for name, p in sorted(layer_paths.items()):
                    print(f"  {name}: {p}")

            return 0

    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    raise AssertionError(f"Unhandled command: {args.cmd}")
