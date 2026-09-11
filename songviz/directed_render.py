"""A small, deterministic renderer for an evidence-directed song passage.

The renderer intentionally consumes a plan rather than interpreting instruments
itself: a plan can put *any* layer through any of the four visual treatments.
This makes repeated motifs recognisable without making, for example, every
snare permanently a ring.
"""
from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from math import exp, sin, sqrt
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


class DirectedVisualizer:
    """Render RGB24 clip-local frames from an already validated direction plan.

    ``t`` is always local to the clip; signal timestamps and plan boundaries
    remain on the source-song clock.  No mutable playhead is kept, so seeking
    and rendering frames out of order produce exactly the same pixels.
    """

    LAYERS = ("pulse", "kick", "snare", "hh", "bass", "vocals", "other")
    _HIT_LAYERS = ("kick", "snare", "hh")
    _WINDOW = {"pulse": 0.16, "kick": 0.22, "snare": 0.26, "hh": 0.12}
    _DECAY = {"pulse": 0.070, "kick": 0.085, "snare": 0.095, "hh": 0.042}
    _SUPPORT_LANES = {
        "pulse": (0.16, 0.48), "kick": (0.50, 0.75), "snare": (0.80, 0.48),
        "hh": (0.18, 0.20), "bass": (0.50, 0.66), "vocals": (0.50, 0.30),
        "other": (0.82, 0.22),
    }
    _LAYER_TINT = {
        "pulse": (214, 224, 220), "kick": (90, 205, 224), "snare": (255, 151, 103),
        "hh": (247, 205, 99), "bass": (112, 220, 193), "vocals": (194, 144, 247),
        "other": (130, 174, 244),
    }

    def __init__(self, plan: dict[str, Any], signals: dict[str, Any], width: int = 960, height: int = 540):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        # The pipeline can reuse and enrich its plan after creating a renderer.
        # Snapshot nested segments so such changes cannot alter later frames.
        self.plan = dict(plan)
        self.width, self.height = int(width), int(height)
        self.start_s = float(plan["start_s"])
        self.end_s = float(plan["end_s"])
        self._segments = tuple(sorted(deepcopy(plan.get("segments", ())), key=lambda item: float(item["start_s"])))
        self._segment_starts = tuple(float(segment["start_s"]) for segment in self._segments)

        self._beats = tuple(sorted(float(t) for t in signals.get("beat_times_s", ())))
        hits = {name: [] for name in self._HIT_LAYERS}
        for order, hit in enumerate(signals.get("hits", ())):
            name = hit.get("component")
            if name in hits:
                hits[name].append((float(hit["t"]), max(0.0, min(1.0, float(hit.get("velocity", 0)))), order))
        self._hits = {name: tuple(sorted(values)) for name, values in hits.items()}
        self._hit_times = {name: tuple(value[0] for value in values) for name, values in self._hits.items()}
        energy = signals.get("energy", {})
        self._energy_times = tuple(float(t) for t in energy.get("times_s", ()))
        self._energy = {
            name: tuple(max(0.0, min(1.0, float(v))) for v in energy.get(name, ()))
            for name in ("bass", "vocals", "other")
        }
        self._base = self._make_stage()

    def _make_stage(self) -> Image.Image:
        """Build a composed, event-free dark stage once per renderer."""
        y, x = np.ogrid[: self.height, : self.width]
        cx, cy = self.width * .5, self.height * .48
        vignette = np.sqrt(((x - cx) / (self.width * .70)) ** 2 + ((y - cy) / (self.height * .75)) ** 2)
        glow = np.exp(-(((x - cx) / (self.width * .34)) ** 2 + ((y - cy) / (self.height * .36)) ** 2))
        shade = np.clip(1.0 - vignette, 0.0, 1.0)
        pixels = np.empty((self.height, self.width, 3), dtype=np.uint8)
        pixels[..., 0] = 7 + (5 * shade + 3 * glow).astype(np.uint8)
        pixels[..., 1] = 11 + (8 * shade + 4 * glow).astype(np.uint8)
        pixels[..., 2] = 20 + (13 * shade + 9 * glow).astype(np.uint8)
        image = Image.fromarray(pixels, "RGB").convert("RGBA")
        static = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(static)
        scale = min(self.width / 960.0, self.height / 540.0)
        horizon = int(self.height * .77)
        draw.line((self.width * .10, horizon, self.width * .90, horizon), fill=(75, 102, 129, 48), width=max(1, round(scale)))
        draw.line((self.width * .18, horizon, self.width * .38, self.height * .58), fill=(65, 91, 120, 25), width=1)
        draw.line((self.width * .82, horizon, self.width * .62, self.height * .58), fill=(65, 91, 120, 25), width=1)
        draw.ellipse((cx - 122 * scale, cy - 122 * scale, cx + 122 * scale, cy + 122 * scale), outline=(99, 126, 147, 19), width=max(1, round(scale)))
        return Image.alpha_composite(image, static).convert("RGB")

    @staticmethod
    def _envelope(age: float, velocity: float, window: float, decay: float) -> float:
        return velocity * exp(-age / decay) if 0.0 <= age < window and velocity > 0 else 0.0

    def _attack(self, layer: str, absolute: float) -> float:
        if layer == "pulse":
            times, values = self._beats, None
        else:
            times, values = self._hit_times[layer], self._hits[layer]
        last = bisect_right(times, absolute)
        total = 0.0
        for i in range(last - 1, -1, -1):
            age = absolute - times[i]
            if age >= self._WINDOW[layer]:
                break
            velocity = 1.0 if values is None else values[i][1]
            total += self._envelope(age, velocity, self._WINDOW[layer], self._DECAY[layer])
        return min(1.0, total)

    def _energy_at(self, layer: str, absolute: float) -> float:
        """Causal sample-and-hold energy, avoiding a look ahead at render time."""
        values = self._energy[layer]
        if not values or not self._energy_times:
            return 0.0
        i = bisect_right(self._energy_times, absolute) - 1
        return values[min(i, len(values) - 1)] if i >= 0 else 0.0

    def _signal_at(self, layer: str, absolute: float) -> float:
        return self._attack(layer, absolute) if layer in self._WINDOW else self._energy_at(layer, absolute)

    @staticmethod
    def _motif_color(motif: str, palette: str, layer: str) -> tuple[int, int, int]:
        """A stable motif fingerprint, warmed or cooled by its plan palette."""
        # Deliberately avoid Python's salted hash: motif returns must look alike
        # across processes too.
        fingerprint = sum((i + 1) * ord(char) for i, char in enumerate(str(motif))) % 31
        base = DirectedVisualizer._LAYER_TINT[layer]
        bias = (18 + fingerprint, 6 + fingerprint // 3, -8) if palette == "warm" else (-10, 3 + fingerprint // 4, 16 + fingerprint)
        return tuple(max(0, min(255, base[i] + bias[i])) for i in range(3))

    def _segment_state(self, absolute: float) -> tuple[dict[str, Any] | None, dict[str, Any] | None, float]:
        current_index = bisect_right(self._segment_starts, absolute) - 1
        if current_index < 0:
            return None, None, 0.0
        current = self._segments[current_index]
        if absolute >= float(current["end_s"]):
            return current, None, 1.0
        duration = max(0.0, min(2.0, float(current.get("transition_s", 0.0))))
        if current_index and duration and absolute < float(current["start_s"]) + duration:
            return self._segments[current_index - 1], current, (absolute - float(current["start_s"])) / duration
        return current, None, 1.0

    def state_at(self, t: float) -> dict[str, Any]:
        """Expose timing and crossfade state without making it part of pixels."""
        absolute = self.start_s + float(t)
        old, new, amount = self._segment_state(absolute)
        active = new if new is not None else old
        return {
            "t": float(t), "absolute_s": absolute,
            "segment_index": self._segments.index(active) if active is not None else None,
            "previous_segment_index": self._segments.index(old) if new is not None else None,
            "transition": amount if new is not None else 1.0,
            "signals": {layer: self._signal_at(layer, absolute) for layer in self.LAYERS},
        }

    def _draw_treatment(self, draw: ImageDraw.ImageDraw, treatment: str, layer: str, signal: float,
                        color: tuple[int, int, int], alpha: int, absolute: float,
                        lane: tuple[float, float], geometry_scale: float = 1.0) -> None:
        """Draw a treatment solely from generic layer/lane/signal inputs."""
        sx, sy = self.width / 960.0, self.height / 540.0
        scale = min(sx, sy) * geometry_scale
        x, y = self.width * lane[0], self.height * lane[1]
        signal = max(0.0, min(1.0, signal))
        if alpha <= 0 or signal <= 0:
            return
        strong = max(1, round((2.0 + 3.5 * signal) * scale))
        if treatment == "ring":
            radius = (30 + 105 * signal) * scale
            box = (x - radius * 1.18, y - radius * .72, x + radius * 1.18, y + radius * .72)
            draw.arc(box, 196, 344, fill=(*color, alpha), width=strong)
            draw.arc((box[0] + 10 * sx, box[1] + 7 * sy, box[2] - 10 * sx, box[3] - 7 * sy), 16, 164, fill=(*color, int(alpha * .58)), width=max(1, strong - 1))
            core = (5 + 12 * signal) * scale
            draw.ellipse((x - core, y - core, x + core, y + core), fill=(*color, int(alpha * .52)))
        elif treatment == "ribbon":
            # A smooth fixed waveform; only its amplitude is signal/energy led.
            span, amp = 190 * sx * geometry_scale, (8 + 36 * signal) * sy * geometry_scale
            points = [(x - span + i * (2 * span / 32), y + sin(i * .55 + absolute * .45) * amp) for i in range(33)]
            draw.line(points, fill=(*color, alpha), width=strong, joint="curve")
            draw.line([(px, py + 7 * sy * geometry_scale) for px, py in points], fill=(*color, int(alpha * .30)), width=max(1, strong - 1), joint="curve")
        elif treatment == "ticks":
            length = (8 + 26 * signal) * sy * geometry_scale
            for i in range(5):
                dx = (i - 2) * 16 * sx * geometry_scale
                top = y - length * (.55 + .10 * (i % 2))
                draw.line((x + dx, top, x + dx + 3 * sx * geometry_scale, y + length * .28), fill=(*color, int(alpha * (.56 + .08 * i))), width=strong)
        elif treatment == "rails":
            spread = (45 + 290 * signal) * sx * geometry_scale
            gap = (6 + 8 * signal) * sy * geometry_scale
            for offset, factor in ((-gap, 1.0), (gap, .64)):
                draw.line((x - spread, y + offset, x - 12 * sx * geometry_scale, y + offset), fill=(*color, int(alpha * factor)), width=strong)
                draw.line((x + 12 * sx * geometry_scale, y + offset, x + spread, y + offset), fill=(*color, int(alpha * factor)), width=strong)
            dot = (4 + 7 * signal) * scale
            draw.ellipse((x - dot, y - dot, x + dot, y + dot), fill=(*color, int(alpha * .7)))

    def _draw_segment(self, draw: ImageDraw.ImageDraw, segment: dict[str, Any], weight: float, absolute: float) -> None:
        if weight <= 0:
            return
        palette = segment.get("palette", "cool")
        motif = segment.get("motif", "")
        for layer in self.LAYERS:
            spec = segment.get("layers", {}).get(layer, {})
            if not spec.get("visible", False):
                continue
            gain = max(0.0, min(1.0, float(spec.get("gain", 0.0))))
            if not gain:
                continue
            treatment = spec.get("treatment", "ticks")
            signal = self._signal_at(layer, absolute)
            # Supporting layers stay intentionally quieter than the plan focus.
            primary = layer == segment.get("focus")
            importance = 1.0 if primary else .34
            # A plan can make a stem visible without making near-silence glow:
            # signal response remains strong at musical levels but fades away
            # rapidly near zero.
            alpha = int(235 * weight * gain * importance * sqrt(signal))
            # Every focus claims the same central stage.  All other layers use
            # distinct fixed support lanes, so a dense plan remains readable.
            lane = (0.50, 0.47) if primary else self._SUPPORT_LANES[layer]
            self._draw_treatment(
                draw, treatment, layer, signal, self._motif_color(motif, palette, layer), alpha,
                absolute, lane, 1.0 if primary else .55,
            )

    def _scene_for_segment(self, segment: dict[str, Any], absolute: float) -> Image.Image:
        """Composite one complete scene, ready for a true scene crossfade."""
        effects = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        self._draw_segment(ImageDraw.Draw(effects), segment, 1.0, absolute)
        return Image.alpha_composite(self._base.convert("RGBA"), effects).convert("RGB")

    def frame_rgb24(self, t: float) -> bytes:
        """Return one deterministic RGB24 frame for a clip-local timestamp."""
        absolute = self.start_s + float(t)
        old, new, amount = self._segment_state(absolute)
        if new is None:
            image = self._scene_for_segment(old, absolute) if old is not None else self._base
        else:
            # Drawing old and new into one transparent layer lets later pixels
            # replace earlier ones.  Blend fully composited scenes instead.
            image = Image.blend(self._scene_for_segment(old, absolute), self._scene_for_segment(new, absolute), amount)
        return image.tobytes()
