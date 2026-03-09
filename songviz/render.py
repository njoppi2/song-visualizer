from __future__ import annotations

import math
import subprocess
import tempfile
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


# Single source of truth for all colour schemes: (top, bot, accent).
# Section cycling uses this order; stem→palette mapping via _STEM_PALETTE_INDEX.
_PALETTES: list[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = [
    ((12, 18, 38), (24, 70, 140), (240, 238, 232)),   # 0: deep blue  (drums)
    ((22, 14, 6), (180, 90, 18), (248, 240, 232)),     # 1: amber      (vocals)
    ((8, 22, 18), (22, 120, 88), (240, 246, 242)),     # 2: teal       (bass)
    ((18, 10, 24), (124, 24, 100), (245, 238, 250)),   # 3: magenta    (other)
]
_STEM_PALETTE_INDEX: dict[str, int] = {"drums": 0, "bass": 2, "vocals": 1, "other": 3}
_ACTIVE_WORD_COLOR = (255, 220, 60)  # bright gold

_SECTION_TRANSITION_PULSE_S = 0.6   # half-width of bell pulse around boundary
_SECTION_TRANSITION_PEAK = 0.85     # max intensity (0-1)


def _as_np_float(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _pick_palette(rng: np.random.Generator) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    return _PALETTES[int(rng.integers(0, len(_PALETTES)))]


def _make_gradient(width: int, height: int, top: tuple[int, int, int], bot: tuple[int, int, int]) -> np.ndarray:
    # uint8 [H,W,3]
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    top_v = np.asarray(top, dtype=np.float32)[None, :]
    bot_v = np.asarray(bot, dtype=np.float32)[None, :]
    col = (1.0 - t) * top_v + t * bot_v  # [H,3]
    img = np.repeat(col[:, None, :], width, axis=1)
    return np.clip(img, 0, 255).astype(np.uint8)


def _stem_palette(stem: str) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    return _PALETTES[_STEM_PALETTE_INDEX.get(stem, 3)]


def _make_lyrics_font(size: int = 22) -> ImageFont.ImageFont:
    """Return a readable font for lyric overlays."""
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _draw_lyric_overlay(
    draw: ImageDraw.ImageDraw,
    segment_text: str,
    active_word: str,
    confidence: float,
    *,
    cx: float,
    cy: float,
    font: ImageFont.ImageFont,
    c_accent: tuple[int, int, int] = (255, 255, 255),
    active_word_index: int = -1,
) -> None:
    """Render the current lyric line with the active word highlighted in gold.

    Shows the full segment text. When active_word is empty (before first word
    in a segment) the full line is shown dimmed. Falls back to centered
    active_word only when segment_text is missing.
    """
    alpha = max(80, int(220 * max(0.35, float(confidence))))

    display = segment_text.strip() if segment_text else active_word
    if not display:
        return

    words = display.split()
    if not words:
        return

    try:
        # Measure each word.
        space_bbox = draw.textbbox((0, 0), " ", font=font)
        space_w = max(4, space_bbox[2] - space_bbox[0])
    except Exception:
        space_w = 6

    word_widths: list[int] = []
    line_h = 0
    for w in words:
        try:
            bb = draw.textbbox((0, 0), w, font=font)
            word_widths.append(bb[2] - bb[0])
            line_h = max(line_h, bb[3] - bb[1])
        except Exception:
            word_widths.append(len(w) * 8)
            line_h = max(line_h, 20)

    total_w = sum(word_widths) + space_w * (len(words) - 1)

    # Background rect behind the whole line.
    px, py = 12, 5
    draw.rectangle(
        (cx - total_w / 2 - px, cy - line_h / 2 - py, cx + total_w / 2 + px, cy + line_h / 2 + py),
        fill=(0, 0, 0, int(alpha * 0.55)),
    )

    # Render each word individually, highlighting the active one in gold.
    x = cx - total_w / 2
    for word_i, (w, ww) in enumerate(zip(words, word_widths)):
        is_active = word_i == active_word_index
        if is_active:
            # Subtle highlight rect behind the active word.
            pad = 3
            draw.rectangle(
                (x - pad, cy - line_h / 2 - pad, x + ww + pad, cy + line_h / 2 + pad),
                fill=(*_ACTIVE_WORD_COLOR, int(alpha * 0.25)),
            )
            color: tuple[int, int, int, int] = (*_ACTIVE_WORD_COLOR, alpha)
        else:
            color = (200, 200, 200, int(alpha * 0.5))
        try:
            draw.text((x, cy - line_h / 2), w, fill=color, font=font)
        except Exception:
            pass
        x += ww + space_w


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _hz_to_note_name(hz: float) -> str | None:
    if not np.isfinite(hz) or hz <= 0.0:
        return None
    midi = 69.0 + 12.0 * math.log2(max(hz, 1e-6) / 440.0)
    n = int(round(midi))
    name = _NOTE_NAMES[n % 12]
    octv = (n // 12) - 1
    return f"{name}{octv}"


def _silence_gate01(x: float, *, thr: float = 0.035, knee: float = 0.035) -> float:
    if not np.isfinite(x):
        return 0.0
    if x <= thr:
        return 0.0
    if knee <= 1e-9:
        return 1.0
    t = (x - thr) / knee
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)  # smoothstep


class _VisualizerBase:
    """Shared state and helpers for all visualizer classes."""

    def __init__(self, analysis: dict[str, Any], cfg: RenderConfig, *, alignment: dict[str, Any] | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._alignment = alignment
        self._lyrics_font = _make_lyrics_font() if alignment is not None else None

        env = analysis["envelopes"]
        self.env_times = _as_np_float(env["times_s"])
        self.loudness = _as_np_float(env["loudness"])
        self.onset = _as_np_float(env["onset_strength"])

        beats = analysis.get("beats") or {}
        self.beat_times = _as_np_float(beats.get("beat_times_s", []))
        self._beat_idx = -1

        story = analysis.get("story") if isinstance(analysis, dict) else None
        self._story_sections: list[dict[str, Any]] = []
        self._tension_times = np.zeros((0,), dtype=np.float32)
        self._tension_val = np.zeros((0,), dtype=np.float32)
        self._drop_times = np.zeros((0,), dtype=np.float32)
        self._buildup_events: list[dict[str, Any]] = []

        if isinstance(story, dict):
            secs = story.get("sections")
            if isinstance(secs, list):
                self._story_sections = [s for s in secs if isinstance(s, dict)]
            ten = story.get("tension")
            if isinstance(ten, dict):
                self._tension_times = _as_np_float(ten.get("times_s", []))
                self._tension_val = _as_np_float(ten.get("value", []))
            events = story.get("events") or {}
            drop_list = events.get("drop_times_s") or []
            if drop_list:
                self._drop_times = np.sort(_as_np_float(drop_list))
            buildup_list = events.get("buildups") or []
            if isinstance(buildup_list, list):
                self._buildup_events = [e for e in buildup_list if isinstance(e, dict)]

        # Build label→palette index so recurring sections (same motif label)
        # share the same colour palette across the song.
        self._label_to_palette: dict[str, int] = {}
        for sec in self._story_sections:
            label = str(sec.get("label") or "")
            if label and label not in self._label_to_palette:
                self._label_to_palette[label] = len(self._label_to_palette)

    # ── Query helpers ───────────────────────────────────────────────────────

    def _section_index(self, t: float) -> int:
        if not self._story_sections:
            return 0
        for i, sec in enumerate(self._story_sections):
            try:
                s0 = float(sec.get("start_s", 0.0))
                s1 = float(sec.get("end_s", 0.0))
            except Exception:
                continue
            if s0 <= t < s1:
                return int(i)
        return int(max(0, len(self._story_sections) - 1))

    def _interp_tension(self, t: float) -> float:
        if self._tension_times.size == 0 or self._tension_val.size == 0:
            return 0.0
        return float(np.interp(t, self._tension_times, self._tension_val))

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

    def _drop_intensity(self, t: float, *, decay_s: float = 0.5) -> float:
        """0→1 value that peaks at each drop and decays exponentially."""
        if self._drop_times.size == 0:
            return 0.0
        idx = int(np.searchsorted(self._drop_times, t, side="right")) - 1
        if idx < 0:
            return 0.0
        dt = t - float(self._drop_times[idx])
        if dt < 0:
            return 0.0
        return float(math.exp(-dt / decay_s))

    def _buildup_fraction(self, t: float) -> float:
        """0→1 linear progress through the current buildup window, 0 if none."""
        for ev in self._buildup_events:
            bs = float(ev.get("buildup_start_s", -1.0))
            bp = float(ev.get("buildup_peak_s", -1.0))
            if bs >= 0 and bp > bs and bs <= t <= bp:
                return float((t - bs) / (bp - bs))
        return 0.0

    def _palette_for_section(self, sec_idx: int) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
        """Return the palette for a section, sharing palettes across same-label sections."""
        if sec_idx < len(self._story_sections):
            label = str(self._story_sections[sec_idx].get("label") or "")
            if label and label in self._label_to_palette:
                return _PALETTES[self._label_to_palette[label] % len(_PALETTES)]
        return _PALETTES[sec_idx % len(_PALETTES)]

    def _section_boundary_intensity(self, t: float) -> float:
        """0→1 raised-cosine pulse centered on each section boundary (skips t=0).

        Transitions are structural ("breathe"), so they use a symmetric bell
        rather than the exponential decay used by drops ("hit").
        """
        if not self._story_sections:
            return 0.0
        best = 0.0
        for sec in self._story_sections:
            start = float(sec.get("start_s", 0.0))
            if start < 1e-3:
                continue  # skip the very beginning
            dt = abs(t - start)
            if dt < _SECTION_TRANSITION_PULSE_S:
                # raised cosine: 1 at dt=0, 0 at dt=pulse_s
                phi = dt / _SECTION_TRANSITION_PULSE_S  # 0..1
                pulse = _SECTION_TRANSITION_PEAK * 0.5 * (1.0 + math.cos(math.pi * phi))
                if pulse > best:
                    best = pulse
        return best

    def _section_palette_blend(
        self, t: float, *, crossfade_s: float = 1.5
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Return gradient (top, bot) colours, crossfading at section boundaries."""
        sec_idx = self._section_index(t)
        top_a, bot_a = self._palette_for_section(sec_idx)[:2]

        if self._story_sections:
            sec = self._story_sections[min(sec_idx, len(self._story_sections) - 1)]
            sec_end = float(sec.get("end_s", 0.0))
            time_to_end = sec_end - t
            if 0.0 < time_to_end < crossfade_s:
                next_top, next_bot = self._palette_for_section(sec_idx + 1)[:2]
                blend = float(np.clip(1.0 - time_to_end / crossfade_s, 0.0, 1.0))
                blend = blend * blend * (3.0 - 2.0 * blend)  # smoothstep
                top_a = tuple(int((1.0 - blend) * a + blend * b) for a, b in zip(top_a, next_top))  # type: ignore[assignment]
                bot_a = tuple(int((1.0 - blend) * a + blend * b) for a, b in zip(bot_a, next_bot))  # type: ignore[assignment]

        return top_a, bot_a  # type: ignore[return-value]


class Visualizer(_VisualizerBase):
    def __init__(self, analysis: dict[str, Any], cfg: RenderConfig, *, alignment: dict[str, Any] | None = None):
        super().__init__(analysis, cfg, alignment=alignment)

        c_top, c_bot, c_accent = _pick_palette(self.rng)
        self.c_top = c_top
        self.c_bot = c_bot
        self.c_accent = c_accent
        self.base = _make_gradient(cfg.width, cfg.height, c_top, c_bot)

        # Static film grain; scaled per-frame by onset.
        grain = self.rng.random((cfg.height, cfg.width, 1), dtype=np.float32)
        self.grain = (grain - 0.5)  # [-0.5, 0.5]

    def frame_rgb24(self, t: float) -> bytes:
        w, h = self.cfg.width, self.cfg.height
        loud, onset = self._interp_env(t)
        flash = self._beat_flash(t)
        tension = self._interp_tension(t)
        drop = self._drop_intensity(t)
        buildup = self._buildup_fraction(t)
        boundary = self._section_boundary_intensity(t)

        # Section-aware gradient with 1.5s crossfade at boundaries.
        top2, bot2 = self._section_palette_blend(t)
        story_mix = float(np.clip(0.15 + 0.55 * (tension**0.9), 0.0, 0.85))

        # Background: gradient + loudness brightness + onset grain.
        base = self.base.astype(np.float32)
        story = _make_gradient(w, h, top2, bot2).astype(np.float32)
        bg = (1.0 - story_mix) * base + story_mix * story
        brightness = 0.50 + 0.65 * (loud**0.6) + 0.12 * (tension**0.9)
        bg *= brightness
        bg += (self.grain * (22.0 * onset + 18.0 * tension + 30.0 * boundary))
        bg = np.clip(bg, 0, 255).astype(np.uint8)

        img = Image.fromarray(bg, mode="RGB")
        draw = ImageDraw.Draw(img, mode="RGBA")

        # Motion center (deterministic, smooth).
        cx = (w * 0.5) + math.sin(t * 0.7) * (w * 0.05) * (0.3 + onset)
        cy = (h * 0.5) + math.cos(t * 0.9) * (h * 0.05) * (0.3 + onset)

        # Pulse orb: loudness-driven radius, spikes on drops and boundaries.
        r = min(w, h) * (0.08 + 0.22 * (loud**0.8) + 0.10 * drop + 0.06 * boundary + 0.12 * buildup)
        orb_alpha = int(min(255, 90 + 130 * loud + 80 * drop))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*self.c_accent, orb_alpha))

        # Onset rays: more rays during buildup (5B).
        ray_n = int(onset * (18 + 14 * buildup))
        if ray_n:
            base_len = min(w, h) * (0.18 + 0.22 * loud)
            for _ in range(ray_n):
                ang = float(self.rng.random()) * (2.0 * math.pi)
                j = (float(self.rng.random()) - 0.5) * 0.35
                length = base_len * (0.65 + 0.9 * float(self.rng.random()))
                x2 = cx + math.cos(ang + j) * length
                y2 = cy + math.sin(ang + j) * length
                a = int(40 + 120 * onset)
                draw.line((cx, cy, x2, y2), fill=(255, 255, 255, a), width=2)

        # Beat flash overlay: quick full-frame strobe.
        if flash > 1e-3:
            draw.rectangle((0, 0, w, h), fill=(255, 255, 255, int(140 * flash)))

        # Drop flash: sharp white spike + brief accent afterglow.
        drop_spike = self._drop_intensity(t, decay_s=0.06)
        drop_glow  = self._drop_intensity(t, decay_s=0.30)
        if drop_spike > 1e-3:
            draw.rectangle((0, 0, w, h), fill=(255, 255, 255, int(200 * drop_spike)))
        if drop_glow > 1e-3:
            draw.rectangle((0, 0, w, h), fill=(*self.c_accent, int(55 * drop_glow)))

        # Section boundary: accent-coloured outline rect that breathes in/out.
        if boundary > 0.02:
            border_w = max(2, int(2 + 4 * boundary))
            border_a = int(120 * boundary)
            draw.rectangle((0, 0, w - 1, h - 1), outline=(*self.c_accent, border_a), width=border_w)

        # Section cut: 0.15s color wash in incoming section's palette color.
        for _sec in self._story_sections:
            _start = float(_sec.get("start_s", 0.0))
            if _start < 1e-3:
                continue
            _dt = t - _start
            if 0.0 <= _dt < 0.15:
                _incoming_bot = self._palette_for_section(self._section_index(t))[1]
                _wash_a = int(90 * math.exp(-_dt / 0.05))
                draw.rectangle((0, 0, w, h), fill=(*_incoming_bot, _wash_a))
                break

        # Buildup intensity bar: thin rising bar at bottom edge (5B).
        if buildup > 0.05:
            bar_h = max(2, int(h * buildup * 0.10))
            draw.rectangle((0, h - bar_h, w, h), fill=(*self.c_accent, int(80 + 120 * buildup)))

        # Lyric line overlay: full segment with active word highlighted (5C).
        if self._alignment is not None and self._lyrics_font is not None:
            from .lyrics import lyric_activity_at as _lyric_at
            act = _lyric_at(self._alignment, t)
            if act.get("active_segment"):
                _draw_lyric_overlay(
                    draw,
                    act["active_segment"],
                    act["active_word"],
                    act.get("word_confidence", 0.8),
                    cx=w / 2,
                    cy=h * 0.84,
                    font=self._lyrics_font,
                    c_accent=self.c_accent,
                    active_word_index=act.get("active_word_index", -1),
                )

        return img.tobytes()


class StemQuadVisualizer(_VisualizerBase):
    """
    A per-stem visual grammar, so each quadrant "reads" differently.

    Inputs are still envelopes + beats, but mapped to different shapes:
    - drums: onset history bars + impact ring + beat border flash
    - bass: sub ring + oscilloscope wave
    - vocals: pitch trail (if available) + syllable pulses
    - other: chroma spokes wheel (if available) + texture
    """

    def __init__(self, stem: str, analysis: dict[str, Any], cfg: RenderConfig, *, alignment: dict[str, Any] | None = None):
        self.stem = str(stem)
        # Lyrics overlay only on vocals quadrant.
        effective_alignment = alignment if self.stem == "vocals" else None
        super().__init__(analysis, cfg, alignment=effective_alignment)

        try:
            self.font = ImageFont.load_default()
        except Exception:
            self.font = None

        feats = analysis.get("features") or {}
        self.pitch_hz = _as_np_float(feats.get("pitch_hz", []))
        self.chroma_12 = np.asarray(feats.get("chroma_12", []), dtype=np.float32)
        self.drums_bands_3 = np.asarray(feats.get("drums_bands_3", []), dtype=np.float32)
        self.note_events = feats.get("note_events") or []
        if not isinstance(self.note_events, list):
            self.note_events = []
        self._note_idx = -1

        # Pre-process pitch into a stable [0,1] range for drawing.
        self.pitch_norm = None
        self.pitch_voiced = None
        if self.pitch_hz.size:
            p = self.pitch_hz.astype(np.float32)
            voiced = np.isfinite(p) & (p > 0.0)
            p2 = np.where(voiced, p, np.nan).astype(np.float32)
            midi = 69.0 + 12.0 * np.log2(np.maximum(p2, 1e-6) / 440.0)
            midi_min, midi_max = 48.0, 84.0  # C3..C6
            norm = (np.clip(midi, midi_min, midi_max) - midi_min) / (midi_max - midi_min)
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

        grain = self.rng.random((cfg.height, cfg.width, 1), dtype=np.float32)
        self.grain = (grain - 0.5)

        self._trail: list[tuple[float, float]] = []

    def _env_index(self, t: float) -> int:
        if self.env_times.size == 0:
            return 0
        k = int(np.searchsorted(self.env_times, t, side="right") - 1)
        return max(0, min(int(self.env_times.size - 1), k))

    def _draw_band_meters(self, draw: ImageDraw.ImageDraw, bands: np.ndarray, *, x0: int, y0: int, w: int, h: int) -> None:
        labels = ["K", "S", "H"]
        pad = 10
        bw = int((w - pad * 2) / 3) - 6
        base_y = y0 + h - pad
        top_y = y0 + pad + 26
        for i in range(3):
            v = float(np.clip(bands[i], 0.0, 1.0))
            bh = int((base_y - top_y) * (v**0.65))
            bx0 = x0 + pad + i * (bw + 10)
            bx1 = bx0 + bw
            by0 = base_y - bh
            draw.rectangle((bx0, top_y, bx1, base_y), outline=(255, 255, 255, 80), width=2)
            if bh > 2:
                draw.rectangle((bx0 + 2, by0, bx1 - 2, base_y - 2), fill=(*self.c_accent, int(80 + 140 * v)))
            draw.text((bx0 + 2, y0 + pad), labels[i], fill=(255, 255, 255, 220), font=self.font if hasattr(self, "font") else None)

    def _draw_drums(self, draw: ImageDraw.ImageDraw, t: float, loud: float, onset: float, flash: float) -> None:
        w, h = self.cfg.width, self.cfg.height
        vis = _silence_gate01(loud)

        k = self._env_index(t)
        if self.drums_bands_3.size and self.drums_bands_3.ndim == 2 and self.drums_bands_3.shape[1] == 3:
            b = self.drums_bands_3[min(k, int(self.drums_bands_3.shape[0] - 1))]
            self._draw_band_meters(draw, b, x0=0, y0=0, w=w, h=h)
        else:
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

        cx, cy = w * 0.5, h * 0.45
        if vis > 1e-3:
            r = (min(w, h) * (0.14 + 0.22 * (loud**0.8))) * (0.25 + 0.75 * vis)
            ring_w = max(2, int(2 + 10 * onset))
            a = int(220 * vis)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(*self.c_accent, a), width=ring_w)
            if onset > 0.15:
                r2 = r * (1.0 + 0.55 * onset)
                draw.ellipse((cx - r2, cy - r2, cx + r2, cy + r2), outline=(255, 255, 255, int(140 * onset * vis)), width=2)

        if flash > 1e-3:
            a = int(220 * flash)
            draw.rectangle((3, 3, w - 3, h - 3), outline=(255, 255, 255, a), width=4)

    def _draw_bass(self, draw: ImageDraw.ImageDraw, t: float, loud: float, onset: float) -> None:
        w, h = self.cfg.width, self.cfg.height
        cx, cy = w * 0.5, h * 0.5
        vis = _silence_gate01(loud)

        amp = loud ** 0.9
        if vis <= 1e-3:
            return

        r = min(w, h) * (0.10 + 0.30 * amp) * (0.25 + 0.75 * vis)
        ow = max(2, int(2 + 10 * amp))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(*self.c_accent, int(220 * vis)), width=ow)
        draw.ellipse((cx - r * 0.55, cy - r * 0.55, cx + r * 0.55, cy + r * 0.55), outline=(255, 255, 255, 90), width=2)

        pts = []
        a = (h * 0.22) * amp
        freq = 1.4 + 1.6 * onset
        phase = t * 2.2
        for x in range(0, w + 1, 6):
            u = (x / max(1.0, w)) * (2.0 * math.pi)
            y = cy + math.sin(u * freq + phase) * a + math.sin(u * (freq * 0.5) + phase * 0.7) * (a * 0.35)
            pts.append((x, y))
        draw.line(pts, fill=(*self.c_accent, int(220 * vis)), width=max(2, int(3 + 8 * amp)))

        meter_h = int(h * (0.10 + 0.80 * amp))
        draw.rectangle((10, h - 10 - meter_h, 22, h - 10), fill=(255, 255, 255, 160))

        if self.pitch_hz.size:
            k = self._env_index(t)
            hz = float(self.pitch_hz[min(k, int(self.pitch_hz.size - 1))])
            note = _hz_to_note_name(hz)
            if note:
                draw.text((w - 10 - 60, 10), note, fill=(255, 255, 255, 230), anchor="ra", font=self.font if hasattr(self, "font") else None)

    def _draw_vocals(self, draw: ImageDraw.ImageDraw, t: float, loud: float, onset: float) -> None:
        w, h = self.cfg.width, self.cfg.height
        vis = _silence_gate01(loud)

        k = self._env_index(t)
        y_norm = 0.5
        voiced = 1.0
        if self.pitch_norm is not None and self.pitch_norm.size:
            y_norm = float(self.pitch_norm[min(k, int(self.pitch_norm.size - 1))])
            voiced = float(self.pitch_voiced[min(k, int(self.pitch_voiced.size - 1))]) if self.pitch_voiced is not None else 1.0

        x = (w * 0.5) + math.sin(t * 0.8) * (w * 0.18) * (0.3 + loud)
        y = (h * 0.85) - y_norm * (h * 0.70)

        grid_a = int(35 + 55 * voiced)
        for j in range(6):
            yy = (h * 0.20) + j * (h * 0.10)
            draw.line((20, yy, w - 20, yy), fill=(255, 255, 255, grid_a), width=1)

        if self.note_events:
            while self._note_idx + 1 < len(self.note_events):
                ne = self.note_events[self._note_idx + 1]
                if float(ne.get("start_s", 0.0)) <= t:
                    self._note_idx += 1
                else:
                    break

            win_s = 6.0
            left = 20
            top = int(h * 0.10)
            right = w - 20
            bottom = int(h * 0.62)
            draw.rectangle((left, top, right, bottom), outline=(255, 255, 255, 90), width=2)

            midi_min = 48.0
            midi_max = 84.0
            span = max(1e-6, (midi_max - midi_min))

            def _x(time_s: float) -> float:
                return left + (right - left) * ((time_s - (t - win_s)) / win_s)

            def _y(midi: float) -> float:
                return bottom - (bottom - top) * ((midi - midi_min) / span)

            for ne in self.note_events[max(0, self._note_idx - 200) : self._note_idx + 200]:
                s = float(ne.get("start_s", 0.0))
                e = float(ne.get("end_s", s))
                if e < t - win_s or s > t + 0.2:
                    continue
                m = float(ne.get("midi", np.nan))
                if not np.isfinite(m):
                    continue
                x0 = _x(s)
                x1 = _x(e)
                yy = _y(m)
                hh = 10
                a = int(90 + 120 * loud)
                draw.rectangle((x0, yy - hh / 2, x1, yy + hh / 2), fill=(*self.c_accent, a))

            px = _x(t)
            draw.line((px, top, px, bottom), fill=(255, 255, 255, 200), width=2)

            cur = None
            if 0 <= self._note_idx < len(self.note_events):
                ne = self.note_events[self._note_idx]
                s = float(ne.get("start_s", 0.0))
                e = float(ne.get("end_s", s))
                if s <= t <= e + 1e-3:
                    cur = float(ne.get("midi", np.nan))
            if cur is not None and np.isfinite(cur):
                hz = 440.0 * (2.0 ** ((cur - 69.0) / 12.0))
                note = _hz_to_note_name(hz)
                if note:
                    draw.text(
                        (w - 10, h - 16),
                        f"{note}",
                        fill=(255, 255, 255, 230),
                        anchor="rd",
                        font=self.font if hasattr(self, "font") else None,
                    )

        else:
            if vis > 1e-3:
                self._trail.append((x, y))
                if len(self._trail) > 72:
                    self._trail = self._trail[-72:]
            else:
                self._trail = []

            for i in range(1, len(self._trail)):
                a = int(12 + 170 * (i / len(self._trail)) ** 2 * (0.35 + 0.65 * voiced))
                draw.line((self._trail[i - 1], self._trail[i]), fill=(255, 255, 255, a), width=2)

            if vis > 1e-3:
                r = min(w, h) * (0.020 + 0.055 * (loud**0.9)) * (0.25 + 0.75 * vis)
                fill_a = (int(60 + 170 * loud) if voiced > 0.0 else int(30 + 90 * loud)) * vis
                draw.ellipse((x - r, y - r, x + r, y + r), fill=(*self.c_accent, int(fill_a)), outline=(255, 255, 255, int(200 * vis)), width=2)
                if onset > 0.2:
                    r2 = r * (1.0 + 1.8 * onset)
                    draw.ellipse((x - r2, y - r2, x + r2, y + r2), outline=(255, 255, 255, int(140 * onset * vis)), width=2)

                if self.pitch_hz.size:
                    hz = float(self.pitch_hz[min(k, int(self.pitch_hz.size - 1))])
                    note = _hz_to_note_name(hz)
                    if note:
                        draw.text((w - 10, h - 16), f"{note}  {hz:0.0f}Hz", fill=(255, 255, 255, int(160 + 70 * voiced)), anchor="rd", font=self.font if hasattr(self, "font") else None)

        if self._alignment is not None and self._lyrics_font is not None:
            from .lyrics import lyric_activity_at as _lyric_at
            act = _lyric_at(self._alignment, t)
            if act.get("active_segment"):
                _draw_lyric_overlay(
                    draw,
                    act["active_segment"],
                    act["active_word"],
                    act.get("word_confidence", 0.8),
                    cx=w / 2,
                    cy=h * 0.88,
                    font=self._lyrics_font,
                    c_accent=self.c_accent,
                    active_word_index=act.get("active_word_index", -1),
                )

    def _draw_other(self, draw: ImageDraw.ImageDraw, t: float, loud: float, onset: float) -> None:
        w, h = self.cfg.width, self.cfg.height
        cx, cy = w * 0.5, h * 0.52

        vec = None
        if self.chroma_12.size:
            k = self._env_index(t)
            if self.chroma_12.ndim == 2 and self.chroma_12.shape[0] > 0:
                vec = self.chroma_12[min(k, int(self.chroma_12.shape[0] - 1))]

        ang0 = t * 0.25 * (2.0 * math.pi)
        base_r = min(w, h) * 0.08
        max_r = min(w, h) * 0.42

        if vec is None:
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

        n = int(12 + 60 * (onset**0.8))
        for _ in range(n):
            a = float(self.rng.random()) * (2.0 * math.pi)
            rr = (min(w, h) * 0.45) * (0.2 + 0.8 * float(self.rng.random()))
            x = cx + math.cos(a + t * 0.3) * rr
            y = cy + math.sin(a + t * 0.3) * rr
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 255, 255, int(35 + 120 * onset)))

        meter_w = int((w - 20) * (0.08 + 0.92 * (loud**0.8)))
        draw.rectangle((10, 10, 10 + meter_w, 18), fill=(255, 255, 255, 120))

    def frame_rgb24(self, t: float) -> bytes:
        w, h = self.cfg.width, self.cfg.height
        loud, onset = self._interp_env(t)
        flash = self._beat_flash(t)
        tension = self._interp_tension(t)
        drop = self._drop_intensity(t)
        boundary = self._section_boundary_intensity(t)

        # Section crossfade background (5A).
        top2, bot2 = self._section_palette_blend(t)
        story_mix = float(np.clip(0.12 + 0.40 * (tension**0.9), 0.0, 0.65))

        base = self.base.astype(np.float32)
        story = _make_gradient(w, h, top2, bot2).astype(np.float32)
        bg = (1.0 - story_mix) * base + story_mix * story
        brightness = 0.46 + 0.72 * (loud**0.6) + 0.10 * (tension**0.9) + 0.08 * boundary
        bg *= brightness
        bg += (self.grain * (18.0 * onset + 16.0 * tension + 25.0 * boundary))
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

        # Drop flash: sharp white spike + brief accent afterglow.
        drop_spike = self._drop_intensity(t, decay_s=0.06)
        drop_glow  = self._drop_intensity(t, decay_s=0.30)
        if drop_spike > 1e-3:
            draw.rectangle((0, 0, w, h), fill=(255, 255, 255, int(170 * drop_spike)))
        if drop_glow > 1e-3:
            draw.rectangle((0, 0, w, h), fill=(*self.c_accent, int(40 * drop_glow)))

        # Section boundary: accent-coloured outline rect (slightly thinner than mix).
        if boundary > 0.02:
            border_w = max(1, int(1 + 3 * boundary))
            border_a = int(100 * boundary)
            draw.rectangle((0, 0, w - 1, h - 1), outline=(*self.c_accent, border_a), width=border_w)

        # Section cut: 0.15s color wash in incoming section's palette color.
        for _sec in self._story_sections:
            _start = float(_sec.get("start_s", 0.0))
            if _start < 1e-3:
                continue
            _dt = t - _start
            if 0.0 <= _dt < 0.15:
                _incoming_bot = self._palette_for_section(self._section_index(t))[1]
                _wash_a = int(60 * math.exp(-_dt / 0.05))
                draw.rectangle((0, 0, w, h), fill=(*_incoming_bot, _wash_a))
                break

        return img.tobytes()


class StemGridVisualizer:
    """
    2x2 grid layout: one stem per quadrant.

    Expected stem names (Demucs default): drums, bass, vocals, other.
    """

    def __init__(self, stem_analyses: dict[str, dict[str, Any]], cfg: RenderConfig, *, labels: bool = True, alignment: dict[str, Any] | None = None):
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
            qcfg = replace(cfg, width=self.w_half, height=self.h_half, seed=int(cfg.seed) + ((i + 1) * 1000))
            quad_alignment = alignment if name == "vocals" else None
            self._quads[name] = StemQuadVisualizer(name, stem_analyses[name], qcfg, alignment=quad_alignment)

        # Use drums quad as timing reference for boundary intensity (avoids duplicating story data).
        self._timing_ref = self._quads["drums"]

    def frame_rgb24(self, t: float) -> bytes:
        img = Image.new("RGB", (self.cfg.width, self.cfg.height))

        for i, name in enumerate(self.order):
            q = self._quads[name]
            qimg = Image.frombytes("RGB", (self.w_half, self.h_half), q.frame_rgb24(t))
            x = 0 if (i % 2 == 0) else self.w_half
            y = 0 if (i < 2) else self.h_half
            img.paste(qimg, (x, y))

        draw = ImageDraw.Draw(img, mode="RGBA")

        # Grid dividers: widen and blend toward accent colour at section boundaries.
        boundary = self._timing_ref._section_boundary_intensity(t)
        grid_w = max(2, int(2 + 2 * boundary))
        if boundary > 0.02:
            ref_accent = self._timing_ref.c_accent
            grid_r = int(235 * (1.0 - boundary) + ref_accent[0] * boundary)
            grid_g = int(235 * (1.0 - boundary) + ref_accent[1] * boundary)
            grid_b = int(235 * (1.0 - boundary) + ref_accent[2] * boundary)
            grid_color: tuple[int, ...] = (grid_r, grid_g, grid_b)
        else:
            grid_color = (235, 235, 235)
        draw.line((self.w_half, 0, self.w_half, self.cfg.height), fill=grid_color, width=grid_w)
        draw.line((0, self.h_half, self.cfg.width, self.h_half), fill=grid_color, width=grid_w)

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
    with tempfile.NamedTemporaryFile(
        prefix=f".{out_p.name}.",
        suffix=".tmp",
        dir=str(out_p.parent),
        delete=False,
    ) as tmp_f:
        tmp_p = Path(tmp_f.name)

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
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{cfg.width}x{cfg.height}",
        "-r", str(cfg.fps),
        "-i", "pipe:0",
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", cfg.preset,
        "-crf", str(cfg.crf),
        "-pix_fmt", "yuv420p",
        *audio_args,
        "-shortest",
        "-movflags", "+faststart",
        "-f", "mp4",
        str(tmp_p),
    ]

    # Capture stderr to a temp file so it doesn't deadlock the pipe and we can
    # include it in the error message on non-zero exit.
    with tempfile.TemporaryFile() as stderr_f:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=stderr_f)
        assert proc.stdin is not None
        pipe_error: RuntimeError | None = None
        try:
            for i in range(n_frames):
                t = i / cfg.fps
                proc.stdin.write(visualizer.frame_rgb24(t))
        except BrokenPipeError as e:
            pipe_error = RuntimeError("ffmpeg pipeline failed while writing video frames")
            pipe_error.__cause__ = e
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

        rc = proc.wait()

        # Read captured stderr for error reporting.
        stderr_f.seek(0)
        stderr_data = stderr_f.read()

    if pipe_error is not None:
        try:
            tmp_p.unlink(missing_ok=True)
        except Exception:
            pass
        raise pipe_error
    if rc != 0:
        try:
            tmp_p.unlink(missing_ok=True)
        except Exception:
            pass
        snippet = stderr_data[-2048:].decode("utf-8", errors="replace") if stderr_data else ""
        raise RuntimeError(f"ffmpeg exited with code {rc}\n{snippet}")

    tmp_p.replace(out_p)


class LyricsOnlyVisualizer:
    """Minimal lyrics-only visualizer: black background, centered lyrics, timestamp."""

    def __init__(self, alignment: dict[str, Any], cfg: RenderConfig):
        self.cfg = cfg
        self._alignment = alignment
        self._font = _make_lyrics_font(size=26)
        try:
            self._ts_font = ImageFont.load_default()
        except Exception:
            self._ts_font = None

    def frame_rgb24(self, t: float) -> bytes:
        w, h = self.cfg.width, self.cfg.height
        img = Image.new("RGB", (w, h), (0, 0, 0))
        draw = ImageDraw.Draw(img, mode="RGBA")

        from .lyrics import lyric_activity_at as _lyric_at

        act = _lyric_at(self._alignment, t)
        if act.get("active_segment"):
            _draw_lyric_overlay(
                draw,
                act["active_segment"],
                act["active_word"],
                act.get("word_confidence", 0.8),
                cx=w / 2,
                cy=h / 2,
                font=self._font,
                active_word_index=act.get("active_word_index", -1),
            )

        # Timestamp in bottom-left.
        mins = int(t) // 60
        secs = t % 60
        ts_text = f"{mins:02d}:{secs:05.2f}"
        draw.text((8, h - 20), ts_text, fill=(180, 180, 180, 200), font=self._ts_font)

        return img.tobytes()


def render_mp4_lyrics_only(
    *,
    alignment: dict[str, Any],
    audio_path: str | Path,
    out_path: str | Path,
    cfg: RenderConfig | None = None,
    duration_s: float,
) -> None:
    if cfg is None:
        cfg = RenderConfig(width=960, height=270, fps=15)
    vis = LyricsOnlyVisualizer(alignment, cfg)
    _render_mp4_with_visualizer(
        audio_path=audio_path,
        out_path=out_path,
        cfg=cfg,
        duration_s=duration_s,
        visualizer=vis,
    )


def render_mp4(
    *,
    analysis: dict[str, Any],
    audio_path: str | Path,
    out_path: str | Path,
    cfg: RenderConfig,
    alignment: dict[str, Any] | None = None,
) -> None:
    duration_s = float(analysis["meta"]["duration_s"])
    vis = Visualizer(analysis, cfg, alignment=alignment)
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
    alignment: dict[str, Any] | None = None,
) -> None:
    vis = StemGridVisualizer(stem_analyses, cfg, alignment=alignment)
    _render_mp4_with_visualizer(
        audio_path=audio_path,
        out_path=out_path,
        cfg=cfg,
        duration_s=float(duration_s),
        visualizer=vis,
    )
