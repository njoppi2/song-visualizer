from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .ffmpeg import require_ffmpeg


@dataclass(frozen=True)
class RenderConfig:
    width: int = 960
    height: int = 540
    fps: int = 30
    seed: int = 0
    crf: int = 18
    preset: str = "veryfast"
    # Some Linux browser builds don't ship AAC decoders. MP3-in-MP4 is often more
    # broadly playable, even if it's less "standard" than AAC.
    audio_codec: str = "mp3"  # "aac" | "mp3"
    audio_bitrate: str = "128k"


def _as_np_float(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _pick_palette(rng: np.random.Generator) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    palettes = [
        ((12, 18, 38), (24, 70, 140), (240, 238, 232)),   # deep blue
        ((18, 10, 24), (124, 24, 100), (245, 238, 250)),  # magenta
        ((8, 22, 18), (22, 120, 88), (240, 246, 242)),    # teal
        ((22, 14, 6), (180, 90, 18), (248, 240, 232)),    # amber
    ]
    return palettes[int(rng.integers(0, len(palettes)))]


def _make_gradient(width: int, height: int, top: tuple[int, int, int], bot: tuple[int, int, int]) -> np.ndarray:
    # uint8 [H,W,3]
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    top_v = np.asarray(top, dtype=np.float32)[None, :]
    bot_v = np.asarray(bot, dtype=np.float32)[None, :]
    col = (1.0 - t) * top_v + t * bot_v  # [H,3]
    img = np.repeat(col[:, None, :], width, axis=1)
    return np.clip(img, 0, 255).astype(np.uint8)


class Visualizer:
    def __init__(self, analysis: dict[str, Any], cfg: RenderConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        env = analysis["envelopes"]
        self.env_times = _as_np_float(env["times_s"])
        self.loudness = _as_np_float(env["loudness"])
        self.onset = _as_np_float(env["onset_strength"])

        beats = analysis["beats"]
        self.beat_times = _as_np_float(beats["beat_times_s"])

        self._beat_idx = -1

        c_top, c_bot, c_accent = _pick_palette(self.rng)
        self.c_top = c_top
        self.c_bot = c_bot
        self.c_accent = c_accent
        self.base = _make_gradient(cfg.width, cfg.height, c_top, c_bot)

        # Static film grain; scaled per-frame by onset.
        grain = self.rng.random((cfg.height, cfg.width, 1), dtype=np.float32)
        self.grain = (grain - 0.5)  # [-0.5, 0.5]

    def _interp_env(self, t: float) -> tuple[float, float]:
        if self.env_times.size == 0:
            return 0.0, 0.0
        loud = float(np.interp(t, self.env_times, self.loudness))
        onset = float(np.interp(t, self.env_times, self.onset))
        return loud, onset

    def _beat_flash(self, t: float, *, decay_s: float = 0.12) -> float:
        if self.beat_times.size == 0:
            return 0.0

        # Advance beat cursor (render is sequential in time).
        while self._beat_idx + 1 < self.beat_times.size and float(self.beat_times[self._beat_idx + 1]) <= t:
            self._beat_idx += 1
        if self._beat_idx < 0:
            return 0.0
        dt = t - float(self.beat_times[self._beat_idx])
        if dt < 0:
            return 0.0
        return float(math.exp(-dt / decay_s))

    def frame_rgb24(self, t: float) -> bytes:
        w, h = self.cfg.width, self.cfg.height
        loud, onset = self._interp_env(t)
        flash = self._beat_flash(t)

        # Background (layer 1): gradient + loudness brightness + onset grain.
        bg = self.base.astype(np.float32)
        brightness = 0.55 + 0.65 * (loud ** 0.6)
        bg *= brightness
        bg += (self.grain * (40.0 * onset))  # subtle spark on onsets
        bg = np.clip(bg, 0, 255).astype(np.uint8)

        img = Image.fromarray(bg, mode="RGB")
        draw = ImageDraw.Draw(img, mode="RGBA")

        # Motion center (deterministic, smooth).
        cx = (w * 0.5) + math.sin(t * 0.7) * (w * 0.05) * (0.3 + onset)
        cy = (h * 0.5) + math.cos(t * 0.9) * (h * 0.05) * (0.3 + onset)

        # Pulse orb (layer 2): loudness-driven radius.
        r = (min(w, h) * (0.08 + 0.22 * (loud ** 0.8)))
        orb_alpha = int(90 + 130 * loud)
        orb = (*self.c_accent, max(0, min(255, orb_alpha)))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=orb)

        # Onset rays (layer 3): onset-driven bursts.
        ray_n = int(onset * 18)
        if ray_n:
            base_len = min(w, h) * (0.18 + 0.22 * loud)
            for _ in range(ray_n):
                ang = float(self.rng.random()) * (2.0 * math.pi)
                j = (float(self.rng.random()) - 0.5) * 0.35
                length = base_len * (0.65 + 0.9 * float(self.rng.random()))
                x2 = cx + math.cos(ang + j) * length
                y2 = cy + math.sin(ang + j) * length
                a = int(40 + 120 * onset)
                col = (255, 255, 255, a)
                draw.line((cx, cy, x2, y2), fill=col, width=2)

        # Beat flash overlay (layer 4): quick full-frame strobe.
        if flash > 1e-3:
            a = int(140 * flash)
            draw.rectangle((0, 0, w, h), fill=(255, 255, 255, a))

        return img.tobytes()


def render_mp4(
    *,
    analysis: dict[str, Any],
    audio_path: str | Path,
    out_path: str | Path,
    cfg: RenderConfig,
) -> None:
    ffmpeg = require_ffmpeg()

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    duration_s = float(analysis["meta"]["duration_s"])
    n_frames = int(math.ceil(duration_s * cfg.fps))

    vis = Visualizer(analysis, cfg)

    audio_codec = cfg.audio_codec.strip().lower()
    if audio_codec == "aac":
        audio_args = ["-c:a", "aac", "-b:a", cfg.audio_bitrate]
    elif audio_codec in ("mp3", "libmp3lame"):
        audio_args = ["-c:a", "libmp3lame", "-b:a", cfg.audio_bitrate]
    else:
        raise ValueError(f"Unsupported audio codec: {cfg.audio_codec!r} (expected: aac|mp3)")

    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{cfg.width}x{cfg.height}",
        "-r",
        str(cfg.fps),
        "-i",
        "pipe:0",
        "-i",
        str(audio_path),
        # Be explicit: input 0 is our generated frames, input 1 is the original audio.
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        cfg.preset,
        "-crf",
        str(cfg.crf),
        "-pix_fmt",
        "yuv420p",
        *audio_args,
        "-shortest",
        "-movflags",
        "+faststart",
        str(out_p),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None

    try:
        for i in range(n_frames):
            t = i / cfg.fps
            proc.stdin.write(vis.frame_rgb24(t))
    except BrokenPipeError as e:
        raise RuntimeError("ffmpeg pipeline failed while writing video frames") from e
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with code {rc}")


def write_vscode_preview_webm(
    *,
    src_video_path: str | Path,
    out_path: str | Path,
    audio_bitrate: str,
) -> None:
    """
    Write a VS Code-friendly preview file.

    VS Code's bundled "Media Preview" extension documents that `.mp4` preview on
    Linux does not support AAC audio tracks. WebM (VP8) works reliably.

    We transcode from the already-rendered MP4 to avoid re-rendering frames in
    Python (much faster iterations while tweaking visuals).
    """
    ffmpeg = require_ffmpeg()

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src_video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        # Drop MP4-global metadata that can confuse some WebM consumers.
        "-map_metadata",
        "-1",
        "-c:v",
        "libvpx",
        "-b:v",
        "0",
        "-crf",
        "32",
        "-deadline",
        "good",
        "-cpu-used",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "libvorbis",
        "-b:a",
        str(audio_bitrate),
        "-shortest",
        str(out_p),
    ]

    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {res.returncode} while writing VS Code preview webm")
