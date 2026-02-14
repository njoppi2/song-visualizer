from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

def _stem_palette(stem: str) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    # Keep stem colors stable across songs; seed drives motion/variation, not meaning.
    palettes = {
        "drums": ((12, 18, 38), (24, 70, 140), (240, 238, 232)),   # deep blue
        "bass": ((8, 22, 18), (22, 120, 88), (240, 246, 242)),     # teal
        "vocals": ((22, 14, 6), (180, 90, 18), (248, 240, 232)),   # amber
        "other": ((18, 10, 24), (124, 24, 100), (245, 238, 250)),  # magenta
    }
    return palettes.get(stem, palettes["other"])


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


class StemQuadVisualizer:
    """
    A per-stem visual grammar, so each quadrant "reads" differently.

    Inputs are still envelopes + beats, but mapped to different shapes:
    - drums: onset history bars + impact ring + beat border flash
    - bass: sub ring + oscilloscope wave
    - vocals: pitch trail (if available) + syllable pulses
    - other: chroma spokes wheel (if available) + texture
    """

    def __init__(self, stem: str, analysis: dict[str, Any], cfg: RenderConfig):
        self.stem = str(stem)
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        env = analysis["envelopes"]
        self.env_times = _as_np_float(env["times_s"])
        self.loudness = _as_np_float(env["loudness"])
        self.onset = _as_np_float(env["onset_strength"])

        beats = analysis.get("beats") or {}
        self.beat_times = _as_np_float(beats.get("beat_times_s", []))
        self._beat_idx = -1

        feats = analysis.get("features") or {}
        self.pitch_hz = _as_np_float(feats.get("pitch_hz", []))
        self.chroma_12 = np.asarray(feats.get("chroma_12", []), dtype=np.float32)

        # Pre-process pitch into a stable [0,1] range for drawing.
        self.pitch_norm = None
        self.pitch_voiced = None
        if self.pitch_hz.size:
            p = self.pitch_hz.astype(np.float32)
            voiced = np.isfinite(p) & (p > 0.0)
            p2 = np.where(voiced, p, np.nan).astype(np.float32)
            # MIDI conversion without librosa dependency here.
            midi = 69.0 + 12.0 * np.log2(np.maximum(p2, 1e-6) / 440.0)
            midi_min, midi_max = 48.0, 84.0  # C3..C6
            norm = (np.clip(midi, midi_min, midi_max) - midi_min) / (midi_max - midi_min)
            # Forward-fill unvoiced regions to avoid jittery jumps.
            last = 0.5
            out = np.empty_like(norm, dtype=np.float32)
            for i in range(norm.size):
                if np.isfinite(norm[i]):
                    last = float(norm[i])
                out[i] = last
            self.pitch_norm = out
            self.pitch_voiced = voiced.astype(np.float32)

        c_top, c_bot, c_accent = _stem_palette(self.stem)
        self.c_top = c_top
        self.c_bot = c_bot
        self.c_accent = c_accent
        self.base = _make_gradient(cfg.width, cfg.height, c_top, c_bot)

        # Static grain; scaled per-frame by onset.
        grain = self.rng.random((cfg.height, cfg.width, 1), dtype=np.float32)
        self.grain = (grain - 0.5)

        self._trail: list[tuple[float, float]] = []

    def _env_index(self, t: float) -> int:
        if self.env_times.size == 0:
            return 0
        k = int(np.searchsorted(self.env_times, t, side="right") - 1)
        return max(0, min(int(self.env_times.size - 1), k))

    def _interp_env(self, t: float) -> tuple[float, float]:
        if self.env_times.size == 0:
            return 0.0, 0.0
        loud = float(np.interp(t, self.env_times, self.loudness))
        onset = float(np.interp(t, self.env_times, self.onset))
        return loud, onset

    def _beat_flash(self, t: float, *, decay_s: float = 0.12) -> float:
        if self.beat_times.size == 0:
            return 0.0
        while self._beat_idx + 1 < self.beat_times.size and float(self.beat_times[self._beat_idx + 1]) <= t:
            self._beat_idx += 1
        if self._beat_idx < 0:
            return 0.0
        dt = t - float(self.beat_times[self._beat_idx])
        if dt < 0:
            return 0.0
        return float(math.exp(-dt / decay_s))

    def _draw_drums(self, draw: ImageDraw.ImageDraw, t: float, loud: float, onset: float, flash: float) -> None:
        w, h = self.cfg.width, self.cfg.height

        # Onset history (bottom bars).
        k = self._env_index(t)
        n = 24
        hist = self.onset[max(0, k - n + 1) : k + 1]
        if hist.size < n:
            pad = np.zeros((n - hist.size,), dtype=np.float32)
            hist = np.concatenate([pad, hist], axis=0)
        bar_w = w / n
        for i in range(n):
            v = float(hist[i])
            bh = int((v**0.6) * (h * 0.42))
            x0 = int(i * bar_w)
            x1 = int((i + 1) * bar_w) - 2
            y0 = h - bh
            a = int(40 + 200 * v)
            draw.rectangle((x0, y0, x1, h), fill=(255, 255, 255, a))

        # "Drum head" ring + impact outline.
        cx, cy = w * 0.5, h * 0.45
        r = (min(w, h) * (0.14 + 0.22 * (loud**0.8)))
        ring_w = max(2, int(2 + 10 * onset))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(*self.c_accent, 220), width=ring_w)
        if onset > 0.15:
            r2 = r * (1.0 + 0.55 * onset)
            draw.ellipse((cx - r2, cy - r2, cx + r2, cy + r2), outline=(255, 255, 255, int(140 * onset)), width=2)

        # Beat flash: border pulse (more readable than full-screen strobe in a quadrant).
        if flash > 1e-3:
            a = int(220 * flash)
            draw.rectangle((3, 3, w - 3, h - 3), outline=(255, 255, 255, a), width=4)

    def _draw_bass(self, draw: ImageDraw.ImageDraw, t: float, loud: float, onset: float) -> None:
        w, h = self.cfg.width, self.cfg.height
        cx, cy = w * 0.5, h * 0.5

        amp = loud ** 0.9
        r = min(w, h) * (0.10 + 0.30 * amp)
        ow = max(2, int(2 + 10 * amp))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(*self.c_accent, 220), width=ow)
        draw.ellipse((cx - r * 0.55, cy - r * 0.55, cx + r * 0.55, cy + r * 0.55), outline=(255, 255, 255, 90), width=2)

        # Oscilloscope wave: slow, heavy motion.
        pts = []
        a = (h * 0.22) * amp
        freq = 1.4 + 1.6 * onset
        phase = t * 2.2
        for x in range(0, w + 1, 6):
            u = (x / max(1.0, w)) * (2.0 * math.pi)
            y = cy + math.sin(u * freq + phase) * a + math.sin(u * (freq * 0.5) + phase * 0.7) * (a * 0.35)
            pts.append((x, y))
        draw.line(pts, fill=(*self.c_accent, 220), width=max(2, int(3 + 8 * amp)))

        # VU bar on the left.
        meter_h = int(h * (0.10 + 0.80 * amp))
        draw.rectangle((10, h - 10 - meter_h, 22, h - 10), fill=(255, 255, 255, 160))

    def _draw_vocals(self, draw: ImageDraw.ImageDraw, t: float, loud: float, onset: float) -> None:
        w, h = self.cfg.width, self.cfg.height

        k = self._env_index(t)
        y_norm = 0.5
        voiced = 1.0
        if self.pitch_norm is not None and self.pitch_norm.size:
            y_norm = float(self.pitch_norm[min(k, int(self.pitch_norm.size - 1))])
            voiced = float(self.pitch_voiced[min(k, int(self.pitch_voiced.size - 1))]) if self.pitch_voiced is not None else 1.0

        x = (w * 0.5) + math.sin(t * 0.8) * (w * 0.18) * (0.3 + loud)
        y = (h * 0.85) - y_norm * (h * 0.70)

        self._trail.append((x, y))
        if len(self._trail) > 72:
            self._trail = self._trail[-72:]

        # Trail (older segments fade out).
        for i in range(1, len(self._trail)):
            a = int(12 + 170 * (i / len(self._trail)) ** 2 * (0.35 + 0.65 * voiced))
            draw.line((self._trail[i - 1], self._trail[i]), fill=(255, 255, 255, a), width=2)

        r = min(w, h) * (0.020 + 0.055 * (loud**0.9))
        fill_a = int(60 + 170 * loud) if voiced > 0.0 else int(30 + 90 * loud)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(*self.c_accent, fill_a), outline=(255, 255, 255, 200), width=2)
        if onset > 0.2:
            r2 = r * (1.0 + 1.8 * onset)
            draw.ellipse((x - r2, y - r2, x + r2, y + r2), outline=(255, 255, 255, int(140 * onset)), width=2)

    def _draw_other(self, draw: ImageDraw.ImageDraw, t: float, loud: float, onset: float) -> None:
        w, h = self.cfg.width, self.cfg.height
        cx, cy = w * 0.5, h * 0.52

        # Chroma spokes if available.
        vec = None
        if self.chroma_12.size:
            k = self._env_index(t)
            if self.chroma_12.ndim == 2 and self.chroma_12.shape[0] > 0:
                vec = self.chroma_12[min(k, int(self.chroma_12.shape[0] - 1))]

        ang0 = t * 0.25 * (2.0 * math.pi)
        base_r = min(w, h) * 0.08
        max_r = min(w, h) * 0.42

        if vec is None:
            # Fallback: a 12-spoke "spark" wheel driven by onset.
            vec = np.full((12,), float(onset), dtype=np.float32)

        for i in range(12):
            v = float(vec[i])
            ang = ang0 + (i / 12.0) * (2.0 * math.pi)
            r = base_r + v * (max_r - base_r)
            x2 = cx + math.cos(ang) * r
            y2 = cy + math.sin(ang) * r
            a = int(40 + 190 * v)
            draw.line((cx, cy, x2, y2), fill=(*self.c_accent, a), width=2)
            if v > 0.3:
                draw.ellipse((x2 - 2, y2 - 2, x2 + 2, y2 + 2), fill=(255, 255, 255, a))

        # Gentle orbiting dot field.
        n = int(12 + 60 * (onset**0.8))
        for _ in range(n):
            a = float(self.rng.random()) * (2.0 * math.pi)
            rr = (min(w, h) * 0.45) * (0.2 + 0.8 * float(self.rng.random()))
            x = cx + math.cos(a + t * 0.3) * rr
            y = cy + math.sin(a + t * 0.3) * rr
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 255, 255, int(35 + 120 * onset)))

        # Small loudness meter (top).
        meter_w = int((w - 20) * (0.08 + 0.92 * (loud**0.8)))
        draw.rectangle((10, 10, 10 + meter_w, 18), fill=(255, 255, 255, 120))

    def frame_rgb24(self, t: float) -> bytes:
        w, h = self.cfg.width, self.cfg.height
        loud, onset = self._interp_env(t)
        flash = self._beat_flash(t)

        bg = self.base.astype(np.float32)
        brightness = 0.48 + 0.72 * (loud**0.6)
        bg *= brightness
        bg += (self.grain * (28.0 * onset))
        bg = np.clip(bg, 0, 255).astype(np.uint8)

        img = Image.fromarray(bg, mode="RGB")
        draw = ImageDraw.Draw(img, mode="RGBA")

        if self.stem == "drums":
            self._draw_drums(draw, t, loud, onset, flash)
        elif self.stem == "bass":
            self._draw_bass(draw, t, loud, onset)
        elif self.stem == "vocals":
            self._draw_vocals(draw, t, loud, onset)
        else:
            self._draw_other(draw, t, loud, onset)

        return img.tobytes()


class StemGridVisualizer:
    """
    2x2 grid layout: one stem per quadrant.

    Expected stem names (Demucs default): drums, bass, vocals, other.
    """

    def __init__(self, stem_analyses: dict[str, dict[str, Any]], cfg: RenderConfig, *, labels: bool = True):
        if cfg.width % 2 != 0 or cfg.height % 2 != 0:
            raise ValueError("width and height must be even for stems4 layout")

        self.cfg = cfg
        self.labels = labels
        self.order = ["drums", "bass", "vocals", "other"]

        missing = [k for k in self.order if k not in stem_analyses]
        if missing:
            raise ValueError(f"Missing stem analyses: {missing} (expected: {self.order})")

        self.w_half = cfg.width // 2
        self.h_half = cfg.height // 2

        try:
            self.font = ImageFont.load_default()
        except Exception:
            self.font = None

        self._quads: dict[str, StemQuadVisualizer] = {}
        for i, name in enumerate(self.order):
            # Stable seed offsets so each quadrant has its own deterministic palette.
            qcfg = replace(cfg, width=self.w_half, height=self.h_half, seed=int(cfg.seed) + ((i + 1) * 1000))
            self._quads[name] = StemQuadVisualizer(name, stem_analyses[name], qcfg)

    def frame_rgb24(self, t: float) -> bytes:
        img = Image.new("RGB", (self.cfg.width, self.cfg.height))

        for i, name in enumerate(self.order):
            q = self._quads[name]
            qimg = Image.frombytes("RGB", (self.w_half, self.h_half), q.frame_rgb24(t))
            x = 0 if (i % 2 == 0) else self.w_half
            y = 0 if (i < 2) else self.h_half
            img.paste(qimg, (x, y))

        draw = ImageDraw.Draw(img)
        # Dividers
        draw.line((self.w_half, 0, self.w_half, self.cfg.height), fill=(235, 235, 235), width=2)
        draw.line((0, self.h_half, self.cfg.width, self.h_half), fill=(235, 235, 235), width=2)

        if self.labels and self.font is not None:
            pad = 8
            for i, name in enumerate(self.order):
                x0 = 0 if (i % 2 == 0) else self.w_half
                y0 = 0 if (i < 2) else self.h_half
                label = name.upper()
                bbox = draw.textbbox((0, 0), label, font=self.font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                draw.rectangle((x0 + pad - 4, y0 + pad - 4, x0 + pad + tw + 6, y0 + pad + th + 4), fill=(0, 0, 0))
                draw.text((x0 + pad, y0 + pad), label, fill=(255, 255, 255), font=self.font)

        return img.tobytes()


def _render_mp4_with_visualizer(
    *,
    audio_path: str | Path,
    out_path: str | Path,
    cfg: RenderConfig,
    duration_s: float,
    visualizer: Any,
) -> None:
    ffmpeg = require_ffmpeg()

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    n_frames = int(math.ceil(duration_s * cfg.fps))

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
            proc.stdin.write(visualizer.frame_rgb24(t))
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


def render_mp4(
    *,
    analysis: dict[str, Any],
    audio_path: str | Path,
    out_path: str | Path,
    cfg: RenderConfig,
) -> None:
    duration_s = float(analysis["meta"]["duration_s"])
    vis = Visualizer(analysis, cfg)
    _render_mp4_with_visualizer(
        audio_path=audio_path,
        out_path=out_path,
        cfg=cfg,
        duration_s=duration_s,
        visualizer=vis,
    )


def render_mp4_stems4(
    *,
    stem_analyses: dict[str, dict[str, Any]],
    duration_s: float,
    audio_path: str | Path,
    out_path: str | Path,
    cfg: RenderConfig,
) -> None:
    vis = StemGridVisualizer(stem_analyses, cfg)
    _render_mp4_with_visualizer(
        audio_path=audio_path,
        out_path=out_path,
        cfg=cfg,
        duration_s=float(duration_s),
        visualizer=vis,
    )
