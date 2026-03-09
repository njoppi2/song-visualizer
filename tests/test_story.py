from __future__ import annotations

import numpy as np

from songviz.analyze import analyze_audio
from songviz.story import _merge_same_label_sections, compute_story


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


def test_merge_same_label_sections_unit() -> None:
    # Sections small enough to be merged (combined 20 s < default 45 s cap).
    sections = [
        {"label": "A", "start_s": 0.0, "end_s": 10.0},
        {"label": "A", "start_s": 10.0, "end_s": 20.0},
        {"label": "B", "start_s": 20.0, "end_s": 30.0},
        {"label": "A", "start_s": 30.0, "end_s": 40.0},
    ]
    merged = _merge_same_label_sections(sections)
    assert len(merged) == 3
    assert merged[0] == {"label": "A", "start_s": 0.0, "end_s": 20.0}
    assert merged[1] == {"label": "B", "start_s": 20.0, "end_s": 30.0}
    assert merged[2] == {"label": "A", "start_s": 30.0, "end_s": 40.0}


def test_merge_same_label_sections_respects_cap() -> None:
    # Sections too long to merge (combined 150 s > 120 s cap).
    sections = [
        {"label": "A", "start_s": 0.0, "end_s": 80.0},
        {"label": "A", "start_s": 80.0, "end_s": 150.0},
    ]
    merged = _merge_same_label_sections(sections)
    assert len(merged) == 2, "Should NOT merge when combined length exceeds cap"


def test_sections_no_adjacent_same_label() -> None:
    """compute_story must not emit consecutive sections with the same label."""
    sr = 22050
    dur_s = 60.0
    t = np.linspace(0.0, dur_s, int(sr * dur_s), endpoint=False, dtype=np.float32)
    y = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    story = compute_story(y, sr)
    sections = story["sections"]
    for i in range(1, len(sections)):
        assert sections[i]["label"] != sections[i - 1]["label"], (
            f"Adjacent same-label sections at index {i}: {sections[i-1]} / {sections[i]}"
        )


def test_drop_pretension_filter() -> None:
    """Flat-energy signal (tension always near 0) should produce no drops."""
    sr = 22050
    dur_s = 30.0
    # Very quiet, constant sine — tension stays low, no real drops
    t = np.linspace(0.0, dur_s, int(sr * dur_s), endpoint=False, dtype=np.float32)
    y = (0.01 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    story = compute_story(y, sr)
    assert story["events"]["drop_times_s"] == [], (
        f"Expected 0 drops on flat signal, got: {story['events']['drop_times_s']}"
    )

