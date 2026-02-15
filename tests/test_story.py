from __future__ import annotations

import numpy as np

from songviz.analyze import analyze_audio
from songviz.story import compute_story


def test_story_shapes_and_keys() -> None:
    sr = 22050
    dur_s = 8.0
    t = np.linspace(0.0, dur_s, int(sr * dur_s), endpoint=False, dtype=np.float32)
    # Two-tone signal with a small amplitude ramp to create a tension change.
    y = (0.15 * np.sin(2 * np.pi * 220.0 * t) + 0.08 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    y *= np.linspace(0.2, 1.0, y.size, dtype=np.float32)

    a = analyze_audio(y, sr, hop_length=512, frame_length=2048)
    story = compute_story(y, sr, hop_length=512, frame_length=2048)

    assert "sections" in story
    assert "tension" in story
    assert isinstance(story["sections"], list)
    assert "times_s" in story["tension"]
    assert "value" in story["tension"]

    # Tension curve aligns with envelope frames.
    assert len(story["tension"]["times_s"]) == len(a["envelopes"]["times_s"])
    assert len(story["tension"]["value"]) == len(a["envelopes"]["times_s"])

