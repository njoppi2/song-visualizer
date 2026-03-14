from __future__ import annotations

import numpy as np

from songviz.analyze import analyze_audio
from songviz.story import (
    _assign_role_based_labels,
    _assign_roles,
    _checkerboard_novelty,
    _compute_section_features,
    _detect_subsections,
    _merge_same_label_sections,
    _revise_roles_globally,
    compute_story,
)


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

    # Role-based fields present on every section.
    for sec in story["sections"]:
        assert "role" in sec, f"Missing 'role' on section {sec}"
        assert "visual_behavior" in sec
        assert "confidence" in sec
        assert "intensity" in sec
        assert "repetition_strength" in sec
        assert "novelty_to_prev" in sec
        assert "relative_intensity_rank" in sec


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


def test_sections_have_roles_and_labels() -> None:
    """Every section must carry a role and a label after role-based labeling."""
    sr = 22050
    dur_s = 60.0
    t = np.linspace(0.0, dur_s, int(sr * dur_s), endpoint=False, dtype=np.float32)
    y = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    story = compute_story(y, sr)
    sections = story["sections"]
    assert len(sections) >= 1
    for sec in sections:
        assert "label" in sec
        assert "role" in sec
        assert sec["role"] in ("intro", "build", "payoff", "valley", "contrast", "outro")


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


def test_subsections_present_and_cover_section() -> None:
    """Each section must have subsections that fully cover its range."""
    sr = 22050
    dur_s = 60.0
    t = np.linspace(0.0, dur_s, int(sr * dur_s), endpoint=False, dtype=np.float32)
    y = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    y *= np.linspace(0.2, 1.0, y.size, dtype=np.float32)
    story = compute_story(y, sr)
    for sec in story["sections"]:
        assert "subsections" in sec, f"Section {sec['label']} missing subsections"
        subs = sec["subsections"]
        assert len(subs) >= 1, f"Section {sec['label']} has no subsections"
        # First subsection starts at section start, last ends at section end.
        assert abs(subs[0]["start_s"] - sec["start_s"]) < 1e-3
        assert abs(subs[-1]["end_s"] - sec["end_s"]) < 1e-3
        # No gaps between subsections.
        for i in range(1, len(subs)):
            assert abs(subs[i]["start_s"] - subs[i - 1]["end_s"]) < 1e-3, (
                f"Gap between subsections {i-1} and {i} in section {sec['label']}"
            )
        # Each subsection has a valid energy descriptor.
        for sub in subs:
            assert sub["energy"] in ("low", "mid", "high", "rising", "falling")


def test_long_section_gets_subdivided() -> None:
    """A section longer than 25s should get at least 2 subsections."""
    tension = np.concatenate([
        np.linspace(0.2, 0.8, 50, dtype=np.float32),
        np.linspace(0.8, 0.1, 50, dtype=np.float32),
        np.linspace(0.1, 0.9, 50, dtype=np.float32),
    ])
    times_s = np.linspace(0.0, 60.0, tension.size, dtype=np.float32)
    section = {"label": "A", "start_s": 0.0, "end_s": 60.0}
    subs = _detect_subsections(section, tension, times_s)
    assert len(subs) >= 2, f"Expected >=2 subsections for 60s section, got {len(subs)}"


def test_short_section_single_subsection() -> None:
    """A section shorter than 20s should not be subdivided."""
    tension = np.linspace(0.3, 0.7, 30, dtype=np.float32)
    times_s = np.linspace(0.0, 15.0, tension.size, dtype=np.float32)
    section = {"label": "A", "start_s": 0.0, "end_s": 15.0}
    subs = _detect_subsections(section, tension, times_s)
    assert len(subs) == 1, f"Expected 1 subsection for 15s section, got {len(subs)}"


def test_checkerboard_novelty_detects_block_boundary() -> None:
    """Checkerboard novelty should peak near a block boundary in a block-diagonal SSM."""
    n = 100
    R = np.zeros((n, n), dtype=np.float64)
    R[:50, :50] = 1.0
    R[50:, 50:] = 1.0

    kw = 10
    novelty = _checkerboard_novelty(R, kernel_width=kw)
    peak_idx = int(np.argmax(novelty))
    assert 40 <= peak_idx <= 60, f"Expected peak near index 50, got {peak_idx}"


def test_ssm_segmentation_two_distinct_sections() -> None:
    """Two acoustically distinct halves should produce a boundary near 30s.

    First half: C4 (261.6 Hz) pure tone — chroma peaks at pitch class C.
    Second half: white noise — flat chroma, different MFCC envelope.
    These are acoustically very different, giving a clear block-diagonal SSM.
    """
    sr = 22050
    dur_s = 60.0
    n_samples = int(sr * dur_s)
    half = n_samples // 2

    t1 = np.linspace(0.0, dur_s / 2, half, endpoint=False, dtype=np.float32)
    rng = np.random.default_rng(42)
    y1 = (0.3 * np.sin(2 * np.pi * 261.6 * t1)).astype(np.float32)  # C4 tone
    y2 = (0.2 * rng.standard_normal(n_samples - half)).astype(np.float32)  # white noise
    y = np.concatenate([y1, y2])

    story = compute_story(y, sr)
    sections = story["sections"]

    # At least one internal boundary should fall within 20–40s
    internal_starts = [sec["start_s"] for sec in sections[1:]]
    assert any(20.0 <= t <= 40.0 for t in internal_starts), (
        f"Expected a boundary near 30s, got section starts: {internal_starts}"
    )


# ---------------------------------------------------------------------------
# Role-based labeling unit tests
# ---------------------------------------------------------------------------


def test_compute_section_features_basic() -> None:
    """Synthetic bounds + arrays → all 11 features present and in expected ranges."""
    n_frames = 200
    rms01 = np.linspace(0.1, 0.9, n_frames, dtype=np.float32)
    onset01 = np.linspace(0.2, 0.8, n_frames, dtype=np.float32)
    cent01 = np.linspace(0.3, 0.7, n_frames, dtype=np.float32)
    times_s = np.linspace(0.0, 20.0, n_frames, dtype=np.float32)
    beat_times = np.arange(0.0, 20.0, 0.5)
    bounds_s = [0.0, 10.0, 20.0]
    hop_s = 20.0 / n_frames

    rng = np.random.default_rng(42)
    section_means = [
        rng.standard_normal(34).astype(np.float32),
        rng.standard_normal(34).astype(np.float32),
    ]

    features = _compute_section_features(
        bounds_s,
        rms01=rms01, onset01=onset01, cent01=cent01,
        times_s=times_s, beat_times=beat_times,
        section_means=section_means,
        duration_s=20.0, hop_s=hop_s,
    )

    assert len(features) == 2
    expected_keys = {
        "mean_rms", "onset_density", "spectral_centroid_mean",
        "rms_slope", "rms_variance", "song_position",
        "repetition_strength", "novelty_to_prev", "novelty_to_next",
        "duration_beats", "relative_intensity_rank",
    }
    for f in features:
        assert set(f.keys()) == expected_keys
        for key in ("mean_rms", "onset_density", "spectral_centroid_mean",
                     "rms_slope", "rms_variance"):
            assert -1e-6 <= f[key] <= 1.0 + 1e-6, f"{key}={f[key]}"
        assert 0.0 <= f["relative_intensity_rank"] <= 1.0


def test_assign_roles_payoff_no_repetition() -> None:
    """High-intensity section gets payoff even with repetition_strength=0."""
    features = [
        {
            "mean_rms": 0.3, "onset_density": 0.3,
            "spectral_centroid_mean": 0.3, "rms_slope": 0.5,
            "rms_variance": 0.2, "song_position": 0.15,
            "repetition_strength": 0.0, "novelty_to_prev": 1.0,
            "novelty_to_next": 0.5, "duration_beats": 16.0,
            "relative_intensity_rank": 0.2,
        },
        {
            "mean_rms": 1.0, "onset_density": 0.9,
            "spectral_centroid_mean": 0.9, "rms_slope": 0.5,
            "rms_variance": 0.1, "song_position": 0.5,
            "repetition_strength": 0.0, "novelty_to_prev": 0.5,
            "novelty_to_next": 0.5, "duration_beats": 16.0,
            "relative_intensity_rank": 1.0,
        },
        {
            "mean_rms": 0.2, "onset_density": 0.2,
            "spectral_centroid_mean": 0.2, "rms_slope": 0.3,
            "rms_variance": 0.1, "song_position": 0.85,
            "repetition_strength": 0.0, "novelty_to_prev": 0.5,
            "novelty_to_next": 1.0, "duration_beats": 16.0,
            "relative_intensity_rank": 0.0,
        },
    ]
    roles = _assign_roles(features)
    assert roles[1]["role"] == "payoff"


def test_revise_roles_intro_only_first() -> None:
    """Intro cannot appear mid-song (after first 2 sections)."""
    sections = [
        {"start_s": 0, "end_s": 10, "role": "intro", "confidence": 0.8},
        {"start_s": 10, "end_s": 20, "role": "build", "confidence": 0.7},
        {"start_s": 20, "end_s": 30, "role": "intro", "confidence": 0.6},
        {"start_s": 30, "end_s": 40, "role": "payoff", "confidence": 0.9},
    ]
    features = [
        {"relative_intensity_rank": 0.2, "song_position": 0.125},
        {"relative_intensity_rank": 0.4, "song_position": 0.375},
        {"relative_intensity_rank": 0.3, "song_position": 0.625},
        {"relative_intensity_rank": 0.9, "song_position": 0.875},
    ]
    role_assignments = [
        {"role": "intro", "confidence": 0.8,
         "scores": {"intro": 0.8, "valley": 0.3, "build": 0.2,
                     "payoff": 0.1, "contrast": 0.1, "outro": 0.0},
         "second_best_role": "valley"},
        {"role": "build", "confidence": 0.7,
         "scores": {"build": 0.7, "valley": 0.3, "intro": 0.0,
                     "payoff": 0.2, "contrast": 0.1, "outro": 0.0},
         "second_best_role": "valley"},
        {"role": "intro", "confidence": 0.6,
         "scores": {"intro": 0.6, "valley": 0.5, "build": 0.3,
                     "payoff": 0.2, "contrast": 0.4, "outro": 0.0},
         "second_best_role": "valley"},
        {"role": "payoff", "confidence": 0.9,
         "scores": {"payoff": 0.9, "valley": 0.1, "intro": 0.0,
                     "build": 0.2, "contrast": 0.1, "outro": 0.0},
         "second_best_role": "build"},
    ]
    rng = np.random.default_rng(42)
    section_means = [rng.standard_normal(34) for _ in range(4)]

    _revise_roles_globally(sections, features, role_assignments, section_means)

    assert sections[0]["role"] == "intro"
    assert sections[2]["role"] != "intro"


def test_role_based_labels_same_role_same_letter() -> None:
    """Two payoff sections with high cosine similarity get the same label."""
    base = np.ones(34, dtype=np.float32)
    sections = [
        {"start_s": 0, "end_s": 10, "role": "intro"},
        {"start_s": 10, "end_s": 30, "role": "payoff"},
        {"start_s": 30, "end_s": 50, "role": "valley"},
        {"start_s": 50, "end_s": 70, "role": "payoff"},
    ]
    section_means = [
        base * 0.5,
        base * 1.0,
        base * 0.3,
        base * 1.01,
    ]

    _assign_role_based_labels(sections, section_means)

    assert sections[1]["label"] == sections[3]["label"]
    assert sections[0]["label"] != sections[1]["label"]

