from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np

from .analyze import analyze_audio, analyze_file
from .features import other_chroma_12, vocals_pitch_hz
from .paths import (
    analysis_path_for_output_dir,
    output_dir_for_audio,
    safe_dirname,
    video_path_for_output_dir,
)
from .render import RenderConfig, render_mp4, render_mp4_stems4
from .stems import ensure_demucs_stems


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

        _clear_screen()
        print("SongViz UI")
        print()
        print(f"Selected: {audio_path}")
        print(f"Outputs:  {out_dir}")
        print()
        if canonical_video.exists():
            print(f"Existing video: {canonical_video} ({_human_size(canonical_video.stat().st_size)})")
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
        out_dir.mkdir(parents=True, exist_ok=True)

        canonical_analysis = analysis_path_for_output_dir(out_dir)
        canonical_analysis.parent.mkdir(parents=True, exist_ok=True)
        canonical_analysis.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        canonical_video = video_path_for_output_dir(out_dir)
        rcfg = RenderConfig(
            width=int(cfg.width),
            height=int(cfg.height),
            fps=int(cfg.fps),
            seed=int(cfg.seed),
            audio_codec=str(cfg.audio_codec),
            audio_bitrate=str(cfg.audio_bitrate),
        )

        layout = str(cfg.layout)
        if layout == "mix":
            render_mp4(analysis=analysis, audio_path=audio_path, out_path=canonical_video, cfg=rcfg)
        elif layout == "stems4":
            stems = ensure_demucs_stems(
                audio_path,
                out_dir=out_dir,
                model=str(cfg.stems_model),
                device=str(cfg.stems_device),
                force=bool(cfg.stems_force),
            )

            stem_analyses: dict[str, dict] = {}
            hop_length = 512
            frame_length = 2048
            for name, stem_path in stems.stems.items():
                y, sr = librosa.load(stem_path, sr=22050, mono=True)
                y = np.asarray(y, dtype=np.float32)
                a = analyze_audio(y, int(sr), hop_length=hop_length, frame_length=frame_length)

                feats: dict[str, np.ndarray] = {}
                if name == "vocals":
                    feats["pitch_hz"] = vocals_pitch_hz(
                        y, int(sr), hop_length=hop_length, frame_length=frame_length
                    )
                elif name == "other":
                    feats["chroma_12"] = other_chroma_12(y, int(sr), hop_length=hop_length, n_fft=frame_length)
                if feats:
                    a["features"] = feats

                if name != "drums":
                    a["beats"]["beat_times_s"] = []
                    a["beats"]["tempo_bpm"] = 0.0
                stem_analyses[name] = a

            render_mp4_stems4(
                stem_analyses=stem_analyses,
                duration_s=float(analysis["meta"]["duration_s"]),
                audio_path=audio_path,
                out_path=canonical_video,
                cfg=rcfg,
            )
        else:
            raise AssertionError(f"Unknown layout: {layout!r}")

        print()
        print("Done.")
        print(f"Wrote: {canonical_video}")
        print(f"Wrote: {canonical_analysis}")
        input("Press Enter to continue...")


def main() -> int:
    return run_ui(UIConfig())
