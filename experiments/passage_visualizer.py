"""A restrained, deterministic opening-passage visual treatment.

``PassageVisualizer`` deliberately has no knowledge of the render pipeline.  It
only turns absolute beat and percussion-event times into RGB24 frames for one
short clip.  This keeps it useful for experiments and makes seeking frames in
any order safe.
"""
from __future__ import annotations

from bisect import bisect_right
from math import exp
import numpy as np
from PIL import Image, ImageDraw


class PassageVisualizer:
    """Minimal ink-stage visualizer for a song passage.

    ``t`` passed to :meth:`frame_rgb24` and :meth:`signals_at` is clip-local;
    events and beats supplied to the constructor remain on the song's absolute
    clock.  Inputs are read into private immutable tuples, so callers can reuse
    or modify their source containers after construction without changing a
    render already in progress.
    """

    _WINDOWS = {"beat": 0.120, "kick": 0.200, "snare": 0.250, "hh": 0.100}
    _DECAYS = {"beat": 0.055, "kick": 0.072, "snare": 0.085, "hh": 0.035}
    _COMPONENTS = ("kick", "snare", "hh")

    def __init__(self, beats, events, start=0.0, end=22.0, width=960, height=540):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self.start = float(start)
        self.end = float(end)
        self.width = int(width)
        self.height = int(height)

        # Sorting private copies gives the same answer for an unordered input
        # list without changing the caller's list or dictionaries.
        self._beats = tuple(sorted(float(value) for value in beats))
        copied = {name: [] for name in self._COMPONENTS}
        for index, hit in enumerate((events or {}).get("hits", ())):
            component = hit.get("component")
            if component not in copied:
                continue
            velocity = max(0.0, min(1.0, float(hit.get("velocity", 0.0))))
            copied[component].append((float(hit["t"]), velocity, index))
        self._hits = {
            name: tuple(sorted(values, key=lambda value: (value[0], value[2])))
            for name, values in copied.items()
        }
        self._times = {
            name: tuple(value[0] for value in values) for name, values in self._hits.items()
        }
        self._base = self._make_stage()

    def _make_stage(self):
        """Build the intentional, event-free ink stage once."""
        y, x = np.ogrid[: self.height, : self.width]
        cx, cy = self.width * 0.5, self.height * 0.49
        # A blue-black field with a small central lift and a gentle vignette.
        edge = np.sqrt(((x - cx) / (self.width * 0.66)) ** 2 + ((y - cy) / (self.height * 0.72)) ** 2)
        center = np.exp(-(((x - cx) / (self.width * 0.34)) ** 2 + ((y - cy) / (self.height * 0.38)) ** 2))
        shade = np.clip(1.0 - edge, 0.0, 1.0)
        pixels = np.empty((self.height, self.width, 3), dtype=np.uint8)
        pixels[..., 0] = 8 + (4 * shade + 3 * center).astype(np.uint8)
        pixels[..., 1] = 13 + (7 * shade + 4 * center).astype(np.uint8)
        pixels[..., 2] = 21 + (11 * shade + 9 * center).astype(np.uint8)
        image = Image.fromarray(pixels, "RGB")

        # Fixed architectural traces give the empty opening a composed rather
        # than merely blank feeling.  They never react to a hit.
        trace = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(trace)
        scale = min(self.width / 960.0, self.height / 540.0)
        horizon = int(self.height * 0.745)
        draw.line((int(self.width * .12), horizon, int(self.width * .88), horizon), fill=(71, 103, 122, 52), width=max(1, round(scale)))
        draw.line((int(self.width * .20), horizon + 1, int(self.width * .39), int(self.height * .58)), fill=(56, 87, 106, 26), width=1)
        draw.line((int(self.width * .80), horizon + 1, int(self.width * .61), int(self.height * .58)), fill=(56, 87, 106, 26), width=1)
        box = (int(cx - 108 * scale), int(cy - 108 * scale), int(cx + 108 * scale), int(cy + 108 * scale))
        draw.ellipse(box, outline=(87, 115, 129, 24), width=max(1, round(scale)))
        draw.arc((int(cx - 190 * scale), int(cy - 145 * scale), int(cx + 190 * scale), int(cy + 145 * scale)), 202, 338, fill=(70, 100, 120, 24), width=1)
        return Image.alpha_composite(image.convert("RGBA"), trace).convert("RGB")

    @staticmethod
    def _envelope(age, velocity, window, decay):
        if age < 0.0 or age >= window or velocity <= 0.0:
            return 0.0
        return velocity * exp(-age / decay)

    def _signal(self, name, absolute):
        """Sum only the recent attacks for one independent component."""
        if name == "beat":
            times, values = self._beats, None
        else:
            times, values = self._times[name], self._hits[name]
        last = bisect_right(times, absolute)
        total = 0.0
        window, decay = self._WINDOWS[name], self._DECAYS[name]
        # Only events inside the very short effect window matter.  Walking
        # backward also preserves separate closely-spaced attacks.
        for index in range(last - 1, -1, -1):
            age = absolute - times[index]
            if age >= window:
                break
            velocity = 1.0 if values is None else values[index][1]
            total += self._envelope(age, velocity, window, decay)
        return min(1.0, total)

    def signals_at(self, t):
        """Return current post-onset strengths, useful for timing checks."""
        absolute = self.start + float(t)
        return {name: self._signal(name, absolute) for name in ("beat", *self._COMPONENTS)}

    def _active_hits(self, name, absolute):
        values = self._hits[name]
        last = bisect_right(self._times[name], absolute)
        active = []
        for index in range(last - 1, -1, -1):
            age = absolute - values[index][0]
            if age >= self._WINDOWS[name]:
                break
            strength = self._envelope(age, values[index][1], self._WINDOWS[name], self._DECAYS[name])
            if strength:
                active.append((age, strength, values[index][2]))
        return active

    def frame_rgb24(self, t):
        """Render one RGB24 frame at a clip-local time, without temporal state."""
        absolute = self.start + float(t)
        image = self._base.copy().convert("RGBA")
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        sx, sy = self.width / 960.0, self.height / 540.0
        cx, cy = self.width * .5, self.height * .485

        # The accepted pulse is deliberately quiet: an inner breath, never a
        # frame-wide flash, and only after an actual beat onset.
        pulse = self._signal("beat", absolute)
        if pulse:
            radius = (25 + 13 * pulse) * min(sx, sy)
            alpha = int(18 + 52 * pulse)
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(174, 203, 203, alpha), width=max(1, round((1 + pulse) * min(sx, sy))))
            core = max(2, round(6 * pulse * min(sx, sy)))
            draw.ellipse((cx - core, cy - core, cx + core, cy + core), fill=(179, 206, 199, int(18 + 35 * pulse)))

        # Warm snare/clap rings take visual precedence at the centre.
        for age, strength, _ in self._active_hits("snare", absolute):
            progress = age / self._WINDOWS["snare"]
            radius = (30 + 128 * progress) * min(sx, sy)
            alpha = int(38 + 212 * strength)
            width = max(1, round((2 + 4 * strength) * min(sx, sy)))
            box = (cx - radius * 1.16, cy - radius * .74, cx + radius * 1.16, cy + radius * .74)
            draw.arc(box, 198, 342, fill=(255, 143, 93, alpha), width=width)
            draw.arc((box[0] + 9 * sx, box[1] + 6 * sy, box[2] - 9 * sx, box[3] - 6 * sy), 18, 162, fill=(255, 190, 128, int(alpha * .72)), width=width)
            core = (8 + 14 * strength) * min(sx, sy)
            draw.ellipse((cx - core, cy - core, cx + core, cy + core), fill=(255, 174, 113, int(38 + 190 * strength)))

        # Kick lives low and moves out across a grounded horizon, separate from
        # the central snare language.
        ground = self.height * .745
        for age, strength, _ in self._active_hits("kick", absolute):
            progress = age / self._WINDOWS["kick"]
            spread = (42 + 270 * progress) * sx
            alpha = int(30 + 205 * strength)
            line_width = max(1, round((2 + 4 * strength) * sy))
            for side in (-1, 1):
                x0 = self.width * .5 + side * 18 * sx
                x1 = self.width * .5 + side * spread
                draw.line((x0, ground, x1, ground - 4 * sy), fill=(79, 216, 235, alpha), width=line_width)
                draw.ellipse((x1 - 3 * sx, ground - 7 * sy, x1 + 3 * sx, ground - 1 * sy), fill=(102, 227, 239, int(alpha * .78)))
            draw.ellipse((self.width * .5 - 10 * sx, ground - 5 * sy, self.width * .5 + 10 * sx, ground + 5 * sy), fill=(74, 202, 224, int(alpha * .48)))

        # Hi-hats are small gold ticks in the upper periphery.  Their position
        # comes from the input order, not randomness, so seeking is repeatable.
        for age, strength, index in self._active_hits("hh", absolute):
            side = -1 if index % 2 == 0 else 1
            lane = (index % 5) - 2
            x = self.width * (.22 if side < 0 else .78) + lane * 18 * sx
            y = self.height * (.205 + (index % 3) * .055)
            lift = (age / self._WINDOWS["hh"]) * 20 * sy
            length = (4 + 13 * strength) * sy
            alpha = int(30 + 195 * strength)
            draw.line((x, y + lift - length, x + side * 3 * sx, y + lift + length), fill=(239, 195, 92, alpha), width=max(1, round(2 * sx)))
            draw.line((x - 4 * sx, y + lift, x + 4 * sx, y + lift), fill=(255, 216, 119, int(alpha * .7)), width=1)

        return Image.alpha_composite(image, layer).convert("RGB").tobytes()
