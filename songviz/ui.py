from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .analyze import analyze_file
from .paths import (
    analysis_path_for_output_dir,
    output_dir_for_audio,
    safe_dirname,
    story_path_for_output_dir,
    video_path_for_output_dir,
)
from .pipeline import run_render_pipeline
from .render import RenderConfig


_AUDIO_EXTS = {
    ".flac",
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".aiff",
    ".aif",
}


@dataclass(frozen=True)
class UIConfig:
    songs_dir: Path = Path("songs")
    outputs_dir: Path = Path("outputs")
    seed: int = 0
    width: int = 960
    height: int = 540
    fps: int = 30
    layout: str = "mix"  # "mix" | "stems4"
    audio_codec: str = "mp3"
    audio_bitrate: str = "128k"
    stems_model: str = "htdemucs"
    stems_device: str = "auto"  # "auto" | "cpu" | "cuda"
    stems_force: bool = False


def _clear_screen() -> None:
    # ANSI clear + home
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def _human_size(n: int) -> str:
    v = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if v < 1024.0:
            return f"{v:.1f}{unit}"
        v /= 1024.0
    return f"{v:.1f}PB"


def _fmt_mtime(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
    except FileNotFoundError:
        return "-"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
    return dt.strftime("%Y-%m-%d %H:%M")


def _iter_song_files(songs_dir: Path) -> Iterable[Path]:
    if not songs_dir.exists():
        return []
    files = [p for p in songs_dir.iterdir() if p.is_file() and p.suffix.lower() in _AUDIO_EXTS]
    files.sort(key=lambda p: p.name.lower())
    return files


def _print_song_list(cfg: UIConfig, songs: list[Path]) -> None:
    print("SongViz UI")
    print()
    print(f"Songs dir: {cfg.songs_dir}")
    print(f"Outputs dir: {cfg.outputs_dir}")
    print(f"Layout: {cfg.layout}")
    if str(cfg.layout) == "stems4":
        print(f"Stems: model={cfg.stems_model} device={cfg.stems_device} force={1 if cfg.stems_force else 0}")
    print()
    if not songs:
        print("No audio files found.")
        print()
        print(f"Put songs under: {cfg.songs_dir}/ (gitignored)")
        print("Supported extensions: " + ", ".join(sorted(_AUDIO_EXTS)))
        print()
        return

    print("Pick a song:")
    for i, p in enumerate(songs, start=1):
        label = safe_dirname(p.stem)
        out_dir = cfg.outputs_dir / label
        vid = out_dir / "video.mp4"
        status = "no video"
        if vid.exists():
            status = f"video {_human_size(vid.stat().st_size)} @ {_fmt_mtime(vid)}"
        print(f"{i:>2}. {p.name}  [{status}]")
    print()
    print("Commands: <number> select, r refresh, q quit")


def run_ui(cfg: UIConfig) -> int:
    cfg.songs_dir.mkdir(parents=True, exist_ok=True)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

    while True:
        songs = list(_iter_song_files(cfg.songs_dir))
        _clear_screen()
        _print_song_list(cfg, songs)

        if not songs:
            cmd = input("> ").strip().lower()
            if cmd in ("q", "quit", "exit"):
                return 0
            continue

        cmd = input("> ").strip().lower()
        if cmd in ("q", "quit", "exit"):
            return 0
        if cmd in ("r", "refresh"):
            continue
        if not cmd.isdigit():
            continue

        idx = int(cmd)
        if idx < 1 or idx > len(songs):
            continue
        audio_path = songs[idx - 1]

        label = safe_dirname(audio_path.stem)
        out_dir = cfg.outputs_dir / label
        canonical_video = video_path_for_output_dir(out_dir)
        canonical_analysis = analysis_path_for_output_dir(out_dir)
        stems_meta = out_dir / "stems" / "stems.json"

        _clear_screen()
        print("SongViz UI")
        print()
        print(f"Selected: {audio_path}")
        print(f"Outputs:  {out_dir}")
        print(f"Layout:   {cfg.layout}")
        print()
        if canonical_video.exists():
            print(f"Existing video: {canonical_video} ({_human_size(canonical_video.stat().st_size)})")
            if str(cfg.layout) == "stems4" and not stems_meta.exists():
                print("Note: stems cache not found; this video was probably rendered as mix.")
            action = input("Regenerate video? [Y/n] ").strip().lower()
            if action in ("n", "no"):
                continue
        else:
            action = input("Generate video now? [Y/n] ").strip().lower()
            if action in ("n", "no"):
                continue

        print()
        print("Rendering...")

        analysis = analyze_file(audio_path)
        song_id = str(analysis["meta"]["song_id"])
        out_dir = output_dir_for_audio(audio_path, song_id, outputs_root=cfg.outputs_dir)

        rcfg = RenderConfig(
            width=int(cfg.width),
            height=int(cfg.height),
            fps=int(cfg.fps),
            seed=int(cfg.seed),
            audio_codec=str(cfg.audio_codec),
            audio_bitrate=str(cfg.audio_bitrate),
        )
        layout = str(cfg.layout)

        # ── Lyrics alignment (auto; cached — only runs once per song) ────────
        # Run before pipeline so vocals.wav is used when stems4 is also selected.
        from .lyrics import align_lyrics, load_alignment
        lyrics_status = "Lyrics: unavailable"
        alignment = None
        try:
            vocals_stem_path = out_dir / "stems" / "vocals.wav"
            align_path = align_lyrics(
                audio_path,
                song_id=song_id,
                output_dir=out_dir,
                vocals_stem=vocals_stem_path if vocals_stem_path.exists() else None,
                force=False,
            )
            alignment = load_alignment(out_dir)
            if alignment is not None:
                meta = alignment.get("metadata", {})
                tool = str(meta.get("alignment_tool", "unknown"))
                backend = str(meta.get("backend_used", "unknown"))
                calibrated = bool(meta.get("calibration_applied", False))
                offset_s = float(meta.get("auto_offset_s", 0.0) or 0.0)
                flags = alignment.get("quality_flags", [])
                flag_text = ""
                if isinstance(flags, list) and flags:
                    flag_text = f" flags={','.join(str(f) for f in flags)}"
                cal_text = f" calibrated={offset_s:+.3f}s" if calibrated else ""
                lyrics_status = f"Lyrics: ok ({tool}, backend={backend}{cal_text}{flag_text}) [{align_path}]"
            else:
                lyrics_status = "Lyrics: missing alignment file after run"
        except ImportError:
            lyrics_status = "Lyrics: skipped (alignment backend not installed)"
            alignment = load_alignment(out_dir)
        except Exception as e:
            lyrics_status = f"Lyrics: failed ({type(e).__name__}: {e})"
            alignment = load_alignment(out_dir)

        # ── Render ───────────────────────────────────────────────────────────
        canonical_video = run_render_pipeline(
            audio_path,
            out_dir,
            analysis,
            layout,
            rcfg,
            stems_model=str(cfg.stems_model),
            stems_device=str(cfg.stems_device),
            stems_force=bool(cfg.stems_force),
            alignment=alignment,
        )

        canonical_analysis = analysis_path_for_output_dir(out_dir)
        story_p = story_path_for_output_dir(out_dir)

        print()
        print("Done.")
        print(f"Wrote: {canonical_video}")
        print(f"Wrote: {canonical_analysis}")
        print(f"Wrote: {story_p}")
        print(lyrics_status)
        input("Press Enter to continue...")


def main() -> int:
    return run_ui(UIConfig())
