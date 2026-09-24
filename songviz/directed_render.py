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
        if self.plan.get("visual_policy", {}).get("kind") == "authored_source_vocabulary_v1":
            # The opt-in visual pass has a clean stage: construction guides
            # and a permanent ring would compete with the sound identities.
            return image.convert("RGB")
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
    def _layer_values(spec: dict[str, Any], absolute: float) -> tuple[float, float]:
        """Evaluate an optional v2 envelope; v1 stays its original constant."""
        envelope = spec.get("envelope")
        if not envelope:
            return max(0.0, min(1.0, float(spec.get("gain", 0.0)))), 1.0
        if absolute <= float(envelope[0]["t_s"]):
            key = envelope[0]
            return float(key["gain"]), float(key["emphasis"])
        if absolute >= float(envelope[-1]["t_s"]):
            key = envelope[-1]
            return float(key["gain"]), float(key["emphasis"])
        index = bisect_right([float(key["t_s"]) for key in envelope], absolute) - 1
        left, right = envelope[index], envelope[index + 1]
        amount = (absolute - float(left["t_s"])) / (float(right["t_s"]) - float(left["t_s"]))
        gain = float(left["gain"]) + amount * (float(right["gain"]) - float(left["gain"]))
        emphasis = float(left["emphasis"]) + amount * (float(right["emphasis"]) - float(left["emphasis"]))
        return gain, emphasis

    def _evaluated_layers(self, segment: dict[str, Any] | None, absolute: float) -> dict[str, dict[str, float | bool]]:
        if segment is None:
            return {name: {"visible": False, "gain": 0.0, "emphasis": 0.0} for name in self.LAYERS}
        return {name: {"visible": bool(segment.get("layers", {}).get(name, {}).get("visible", False)),
                       "gain": self._layer_values(segment.get("layers", {}).get(name, {}), absolute)[0],
                       "emphasis": self._layer_values(segment.get("layers", {}).get(name, {}), absolute)[1]}
                for name in self.LAYERS}

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
        evaluated = self._evaluated_layers(active, absolute)
        previous_evaluated = self._evaluated_layers(old if new is not None else None, absolute)
        incoming_evaluated = evaluated
        return {
            "t": float(t), "absolute_s": absolute,
            "segment_index": self._segments.index(active) if active is not None else None,
            "previous_segment_index": self._segments.index(old) if new is not None else None,
            "transition": amount if new is not None else 1.0,
            "signals": {layer: self._signal_at(layer, absolute) for layer in self.LAYERS},
            "evaluated_layers": evaluated,
            "gains": {layer: values["gain"] for layer, values in evaluated.items()},
            "emphasis": {layer: values["emphasis"] for layer, values in evaluated.items()},
            # During a scene crossfade, ``evaluated_layers`` is the incoming
            # plan. These fields expose the outgoing scene rather than hiding
            # it from diagnostics.
            "previous_evaluated_layers": previous_evaluated,
            "incoming_evaluated_layers": incoming_evaluated,
            "scene_weights": {"previous": (1.0 - amount) if new is not None else 0.0,
                              "incoming": amount if new is not None else 1.0},
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
        elif treatment == "filament":
            # A decorative upright veil: deliberately continuous and organic,
            # never a transcription of a vocal's pitch or formants.
            height = (135 + 135 * sqrt(signal)) * sy * geometry_scale
            spread = (38 + 36 * sqrt(signal)) * sx * geometry_scale
            for strand in range(7):
                offset = (strand - 3) * spread / 4.5
                points = []
                for step in range(41):
                    fraction = step / 40
                    yy = y - height / 2 + height * fraction
                    taper = sin(fraction * 3.141592653589793)
                    sway = sin(absolute * .75 + fraction * 6.1 + strand * .8)
                    xx = x + (offset + sway * (12 + 18 * signal) * sx * geometry_scale) * taper
                    points.append((xx, yy))
                strand_alpha = int(alpha * (.55 + .14 * (3 - abs(strand - 3))))
                width = max(1, round(2.5 * scale))
                draw.line(points, fill=(*color, int(strand_alpha * .12)), width=width + max(2, round(8 * scale)), joint="curve")
                draw.line(points, fill=(*color, strand_alpha), width=width, joint="curve")
        elif treatment == "shards":
            # Short discontinuous angular marks make a snare visually distinct
            # from the voice even in monochrome/small-screen viewing.
            reach = (22 + 46 * signal) * scale
            for i, angle in enumerate((-2.45, -1.54, -.55, .42, 1.45, 2.50)):
                phase = absolute * 7.0 + i * 1.91
                inner = 7 * scale + sin(phase) * 2 * scale
                outer = reach * (.70 + .24 * ((i + 1) % 3) / 2)
                dx, dy = np.cos(angle), np.sin(angle)
                bend_x, bend_y = np.cos(angle + .38), np.sin(angle + .38)
                p1 = (x + dx * inner, y + dy * inner)
                p2 = (x + dx * outer * .58 + bend_x * 5 * scale, y + dy * outer * .58 + bend_y * 5 * scale)
                p3 = (x + dx * outer, y + dy * outer)
                draw.line((p1, p2, p3), fill=(*color, int(alpha * (.80 + .12 * (i % 2)))), width=strong, joint="curve")
        elif treatment == "impact":
            # A low compact mass with a bounded halo, rather than a long rail.
            rx = (23 + 31 * signal) * sx * geometry_scale
            ry = (9 + 16 * signal) * sy * geometry_scale
            draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=(*color, int(alpha * .90)))
            halo = 1.0 + .42 * signal
            draw.ellipse((x - rx * halo, y - ry * halo, x + rx * halo, y + ry * halo), outline=(*color, int(alpha * .75)), width=strong)
            draw.arc((x - rx * 1.8, y - ry * 2.3, x + rx * 1.8, y + ry * 2.3), 202, 338, fill=(*color, int(alpha * .38)), width=max(1, strong - 1))
        elif treatment == "contour":
            # Broad, low layered fields.  These sit behind the voice through
            # authored draw order and keep a silhouette unlike vertical filaments.
            span = (180 + 85 * signal) * sx * geometry_scale
            amp = (12 + 26 * signal) * sy * geometry_scale
            for band in range(4):
                points = []
                for step in range(49):
                    fraction = step / 48
                    xx = x - span + fraction * span * 2
                    yy = y + (band - 1) * 10 * sy * geometry_scale + sin(fraction * 6.3 + absolute * (.28 + band * .07) + band) * amp * (.38 + .18 * band)
                    points.append((xx, yy))
                band_alpha = int(alpha * (.28 + .16 * band))
                draw.line(points, fill=(*color, int(band_alpha * .12)), width=max(3, strong + 5), joint="curve")
                draw.line(points, fill=(*color, band_alpha), width=max(1, strong - 1), joint="curve")

    def _draw_segment(self, draw: ImageDraw.ImageDraw, segment: dict[str, Any], weight: float, absolute: float) -> None:
        if weight <= 0:
            return
        palette = segment.get("palette", "cool")
        motif = segment.get("motif", "")
        # The authored atmospheric field goes down first.  Legacy plans retain
        # their original layer order and therefore their frozen pixels.
        authored = any("anchor" in segment.get("layers", {}).get(layer, {}) for layer in self.LAYERS)
        layers = ("other", "bass", "pulse", "kick", "snare", "hh", "vocals") if authored else self.LAYERS
        for layer in layers:
            spec = segment.get("layers", {}).get(layer, {})
            if not spec.get("visible", False):
                continue
            gain, emphasis = self._layer_values(spec, absolute)
            gain = max(0.0, min(1.0, gain))
            emphasis = max(0.0, min(1.0, emphasis))
            if not gain:
                continue
            treatment = spec.get("treatment", "ticks")
            signal = self._signal_at(layer, absolute)
            # Supporting layers stay intentionally quieter than the plan focus.
            primary = layer == segment.get("focus")
            importance = 1.0 if primary else (.34 if "envelope" not in spec else .15 + .85 * emphasis)
            # A plan can make a stem visible without making near-silence glow:
            # signal response remains strong at musical levels but fades away
            # rapidly near zero.
            alpha = int(235 * weight * gain * importance * sqrt(signal))
            if treatment in {"filament", "shards", "impact", "contour"}:
                # The new thin geometry needs a brighter ink response to stay
                # legible at review size; plan gain/clock remain unchanged.
                alpha = min(255, round(alpha * 1.6))
            anchor = spec.get("anchor")
            if anchor is not None:
                # Authored anchored layout: anchors never jump with focus. Focus is
                # expressed by a modest scale lift, preserving source identity.
                lane = (float(anchor[0]), float(anchor[1]))
                geometry = (.62 + .50 * emphasis) * (1.16 if primary else 1.0)
            # Every focus claims the same central stage.  All other layers use
            # distinct fixed support lanes, so a dense plan remains readable.
            elif primary or "envelope" not in spec:
                lane = (0.50, 0.47) if primary else self._SUPPORT_LANES[layer]
                geometry = (.55 + .45 * emphasis) if "envelope" in spec else (1.0 if primary else .55)
            else:
                support = self._SUPPORT_LANES[layer]
                # V2 emphasis moves a support treatment toward the center;
                # it never changes treatment identity or selects a new layer.
                pull = .72 * emphasis
                lane = (support[0] + ((.50 - support[0]) * pull), support[1] + ((.47 - support[1]) * pull))
                geometry = .55 + .45 * emphasis
            self._draw_treatment(
                draw, treatment, layer, signal, self._motif_color(motif, palette, layer), alpha,
                absolute, lane, geometry,
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
