from __future__ import annotations

import numpy as np

from songviz.analyze import analyze_audio
from songviz.story import (
    _assign_role_based_labels,
    _assign_roles,
    _bar_phase_similarity_diagnostic,
    _checkerboard_novelty,
    _compute_section_features,
    _cqt_similarity_curves,
    _detect_intro_onset_boundary,
    _detect_silence_events,
    _detect_stem_transitions,
    _detect_subsections,
    _first_stable_active_beat,
    _lag_matrix_novelty,
    _novelty_curves_from_lag,
    _onset_similarity_curves,
    _phrase_similarity_curves,
    _snap_anchor_to_bar_beat,
    _stem_block_anchor,
    _stem_block_offset,
    _stem_block_offsets,
    _merge_same_label_sections,
    _merge_short_segments,
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


# ---------------------------------------------------------------------------
# _detect_intro_onset_boundary
# ---------------------------------------------------------------------------

def test_intro_onset_detected_for_quiet_start() -> None:
    """A song with a quiet intro followed by a loud section should detect the boundary."""
    # 222s song, normalized tension: quiet 0-12.7s (≈0.05), loud after (≈0.65)
    sr = 22050
    hop = 512
    n_frames = int(222.0 * sr / hop)
    times = np.arange(n_frames) * (hop / sr)
    tension = np.where(times < 12.7, 0.05, 0.65).astype(np.float32)

    result = _detect_intro_onset_boundary(tension, times, duration_s=222.0)

    assert result is not None, "Expected intro boundary to be detected"
    # Should land within 2s of the actual onset at 12.7s
    assert abs(result - 12.7) <= 2.0, f"Expected ~12.7s, got {result:.2f}s"


def test_intro_onset_returns_none_for_loud_start() -> None:
    """A song that starts loud should not produce a false intro boundary."""
    sr = 22050
    hop = 512
    n_frames = int(180.0 * sr / hop)
    times = np.arange(n_frames) * (hop / sr)
    tension = np.full(n_frames, 0.70, dtype=np.float32)

    result = _detect_intro_onset_boundary(tension, times, duration_s=180.0)

    assert result is None, f"Expected None for loud-start song, got {result}"


def test_intro_onset_returns_none_for_uniform_quiet() -> None:
    """A uniformly quiet song has no onset to detect."""
    sr = 22050
    hop = 512
    n_frames = int(180.0 * sr / hop)
    times = np.arange(n_frames) * (hop / sr)
    tension = np.full(n_frames, 0.10, dtype=np.float32)

    result = _detect_intro_onset_boundary(tension, times, duration_s=180.0)

    assert result is None, f"Expected None for uniformly quiet song, got {result}"


# ---------------------------------------------------------------------------
# _detect_subsections — quiet-start detection
# ---------------------------------------------------------------------------

def test_subsections_quiet_start_adds_split() -> None:
    """A section that starts very quiet should get an extra subsection split."""
    # 28s section: quiet for first 6s, then rises, with a valley mid-way.
    sr = 22050
    hop = 512
    n_frames = int(240.0 * sr / hop)
    times = np.arange(n_frames) * (hop / sr)

    sec_start = 100.0
    sec_end = 128.0  # 28s section
    tension = np.zeros(n_frames, dtype=np.float32)
    for i, t in enumerate(times):
        if t < sec_start or t >= sec_end:
            tension[i] = 0.65  # surroundings are loud
        elif t < sec_start + 6.0:
            tension[i] = 0.08   # quiet opening
        elif t < sec_start + 16.0:
            tension[i] = 0.08 + (t - (sec_start + 6.0)) / 10.0 * 0.55  # rising
        elif t < sec_start + 20.0:
            tension[i] = 0.45   # valley dip
        else:
            tension[i] = 0.60   # resumes high

    section = {"label": "C", "start_s": sec_start, "end_s": sec_end}
    subs = _detect_subsections(section, tension, times)

    assert len(subs) >= 2, f"Expected ≥2 subsections for quiet-start section, got {len(subs)}"
    # First subsection must not span the whole section — quiet-start split was added
    sec_mid = (sec_start + sec_end) / 2.0
    assert subs[0]["end_s"] < sec_end, f"Only subsection spans the whole section"
    assert subs[0]["start_s"] == sec_start, "First subsection must start at section start"


def test_novelty_curves_peak_at_boundary_and_decay() -> None:
    """Novelty curves should be near-zero within a stable riff and spike at the boundary."""
    n_feat = 12
    n_beats = 64
    boundary = 32  # riff A: beats [0, 32), riff B: beats [32, 64)

    v = np.eye(n_feat)
    features = np.zeros((n_feat, n_beats))
    for i in range(boundary):
        features[:, i] = v[i % 4]            # period-4 riff A
    for i in range(boundary, n_beats):
        features[:, i] = v[4 + (i - boundary) % 4]  # period-4 riff B (different)

    beat_times = np.arange(n_beats, dtype=float)
    frame_times = np.arange(n_beats, dtype=float)

    L = _lag_matrix_novelty(features, beat_times, frame_times)
    curves = _novelty_curves_from_lag(L, L)   # chroma=mfcc=L for simplicity

    for name in ("short", "medium", "long"):
        assert name in curves, f"Missing curve '{name}'"
        assert curves[name].shape == (n_beats,), (
            f"Expected ({n_beats},), got {curves[name].shape}"
        )

    short = curves["short"]    # window S=4 beats
    medium = curves["medium"]  # window S=16 beats

    # Within riff A (after 4-beat warm-up): period-4 pattern → novelty near 0.
    inner_A = short[4:boundary]
    assert inner_A.max() < 0.1, f"short: expected ~0 in riff A, got max={inner_A.max():.4f}"

    # At boundary: both curves should spike.
    assert short[boundary] > 0.7, f"short: expected spike at boundary, got {short[boundary]:.4f}"
    assert medium[boundary] > 0.7, f"medium: expected spike at boundary, got {medium[boundary]:.4f}"

    # Short curve decays within 4 beats (new riff starts repeating at lag=4).
    short_settle = short[boundary + 4 : boundary + 8]
    assert short_settle.max() < 0.2, (
        f"short: expected decay by beat {boundary + 4}, got max={short_settle.max():.4f}"
    )

    # Nesting: short >= medium >= long (more history = more chance of a match).
    for t in range(32, n_beats):
        assert short[t] >= medium[t] - 0.01, (
            f"nesting violated at t={t}: short={short[t]:.3f} < medium={medium[t]:.3f}"
        )


def _synthetic_onset_inputs(
    blocks: list[list[tuple[float, int, int]]],
    *,
    lag: int = 8,
    frames_per_beat: int = 10,
    n_bins: int = 84,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Build Q_mag/rms inputs for onset block-comparison tests.

    Each event tuple is (relative beat, CQT bin, sustain_frames).
    """
    sr = 1000
    hop_length = 100
    n_beats = lag * len(blocks)
    n_frames = n_beats * frames_per_beat
    Q_mag = np.zeros((n_bins, n_frames), dtype=np.float32)
    rms = np.zeros(n_frames, dtype=np.float32)

    for block_idx, events in enumerate(blocks):
        block_start = block_idx * lag * frames_per_beat
        for rel_beat, cqt_bin, sustain_frames in events:
            frame = int(round(block_start + rel_beat * frames_per_beat))
            end = min(n_frames, frame + sustain_frames)
            if 0 <= frame < n_frames and end > frame:
                rms[frame:end] = 1.0
                Q_mag[cqt_bin, frame:end] = 1.0

    beat_times = np.arange(n_beats, dtype=float)
    beat_frames = np.arange(n_beats, dtype=int) * frames_per_beat
    frame_times = np.arange(n_frames, dtype=float) * hop_length / sr
    return Q_mag, rms, beat_frames, beat_times, frame_times, sr, hop_length


def test_onset_similarity_penalizes_shifted_attacks() -> None:
    base_events = [(0.0, 36, 3), (2.0, 40, 3), (4.0, 43, 3), (6.0, 40, 3)]
    shifted_events = [(0.45, 36, 3), (2.45, 40, 3), (4.45, 43, 3), (6.45, 40, 3)]

    same = _onset_similarity_curves(
        *_synthetic_onset_inputs([base_events, base_events]),
        lags=(8,),
        min_dist=1,
    )["onset_sim_8"]
    shifted = _onset_similarity_curves(
        *_synthetic_onset_inputs([base_events, shifted_events]),
        lags=(8,),
        min_dist=1,
    )["onset_sim_8"]

    same_score = float(same[80])
    shifted_score = float(shifted[80])
    assert same_score > 0.85
    assert shifted_score < same_score - 0.25


def test_onset_similarity_penalizes_different_activity_shape() -> None:
    attacks_only = [(0.0, 36, 2), (2.0, 40, 2), (4.0, 43, 2), (6.0, 40, 2)]
    sustained = [(0.0, 36, 10), (2.0, 40, 10), (4.0, 43, 10), (6.0, 40, 10)]

    same = _onset_similarity_curves(
        *_synthetic_onset_inputs([attacks_only, attacks_only]),
        lags=(8,),
        min_dist=1,
    )["onset_sim_8"]
    different_shape = _onset_similarity_curves(
        *_synthetic_onset_inputs([attacks_only, sustained]),
        lags=(8,),
        min_dist=1,
    )["onset_sim_8"]

    same_score = float(same[80])
    shape_score = float(different_shape[80])
    assert same_score > 0.85
    assert shape_score < same_score - 0.05


def _synthetic_phrase_inputs(
    blocks: list[list[tuple[float, int, int]]],
    *,
    lag: int = 8,
    frames_per_beat: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Q_mag, rms, beat_frames, beat_times, frame_times, _, _ = _synthetic_onset_inputs(
        blocks,
        lag=lag,
        frames_per_beat=frames_per_beat,
    )
    n_beats = lag * len(blocks)
    Q_beat = np.zeros((Q_mag.shape[0], n_beats), dtype=np.float32)
    for beat in range(n_beats):
        f0 = beat * frames_per_beat
        f1 = f0 + frames_per_beat
        Q_beat[:, beat] = Q_mag[:, f0:f1].mean(axis=1)
    return Q_beat, rms, beat_frames, beat_times, frame_times


def test_phrase_similarity_scores_identical_whole_blocks_high() -> None:
    events = [(0.0, 36, 3), (2.0, 40, 5), (4.0, 43, 3), (6.0, 40, 5)]

    phrase = _phrase_similarity_curves(
        *_synthetic_phrase_inputs([events, events]),
        lags=(8,),
    )["phrase_sim_8"]

    assert float(phrase[80]) > 0.95


def test_phrase_similarity_leaves_unscored_warmup_neutral() -> None:
    events = [(0.0, 36, 3), (2.0, 40, 5), (4.0, 43, 3), (6.0, 40, 5)]

    phrase = _phrase_similarity_curves(
        *_synthetic_phrase_inputs([events, events]),
        lags=(16,),
    )

    assert float(phrase["phrase_sim_16"][0]) == 1.0
    assert float(phrase["phrase_sim_16"][-1]) == 1.0
    assert float(phrase["phrase_spec_sim_16"][0]) == 1.0
    assert float(phrase["phrase_spec_sim_16"][-1]) == 1.0
    assert float(phrase["phrase_env_sim_16"][0]) == 1.0
    assert float(phrase["phrase_env_sim_16"][-1]) == 1.0
    assert float(phrase["phrase_accent_sim_16"][0]) == 1.0
    assert float(phrase["phrase_accent_sim_16"][-1]) == 1.0
    assert float(phrase["phrase_amp_sim_16"][0]) == 1.0
    assert float(phrase["phrase_amp_sim_16"][-1]) == 1.0
    assert float(phrase["phrase_aud_sim_16"][0]) == 1.0
    assert float(phrase["phrase_aud_sim_16"][-1]) == 1.0


def test_phrase_similarity_penalizes_spectral_and_activity_changes() -> None:
    base = [(0.0, 36, 10), (2.0, 40, 10), (4.0, 43, 10), (6.0, 40, 10)]
    different_spectrum = [(0.0, 60, 10), (2.0, 64, 10), (4.0, 67, 10), (6.0, 64, 10)]
    different_shape = [(0.0, 36, 20), (2.0, 40, 20), (4.0, 43, 20), (6.0, 40, 20)]

    same = _phrase_similarity_curves(
        *_synthetic_phrase_inputs([base, base]),
        lags=(8,),
    )
    spectral = _phrase_similarity_curves(
        *_synthetic_phrase_inputs([base, different_spectrum]),
        lags=(8,),
    )
    shape = _phrase_similarity_curves(
        *_synthetic_phrase_inputs([base, different_shape]),
        lags=(8,),
    )

    same_score = float(same["phrase_sim_8"][80])
    same_spec = float(same["phrase_spec_sim_8"][80])
    same_env = float(same["phrase_env_sim_8"][80])
    same_amp = float(same["phrase_amp_sim_8"][80])
    assert float(spectral["phrase_sim_8"][80]) < same_score - 0.25
    assert float(spectral["phrase_spec_sim_8"][80]) < same_spec - 0.5
    assert float(spectral["phrase_env_sim_8"][80]) > same_env - 0.05
    assert float(spectral["phrase_amp_sim_8"][80]) > same_amp - 0.05
    assert float(shape["phrase_sim_8"][80]) < same_score - 0.05
    assert float(shape["phrase_spec_sim_8"][80]) > float(spectral["phrase_spec_sim_8"][80]) + 0.5
    assert float(shape["phrase_env_sim_8"][80]) < same_env - 0.25
    assert float(shape["phrase_amp_sim_8"][80]) < same_amp - 0.20


def test_phrase_similarity_downweights_inaudible_residual_spectrum() -> None:
    lag = 8
    frames_per_beat = 10
    n_beats = lag * 3
    n_frames = n_beats * frames_per_beat
    Q_beat = np.zeros((84, n_beats), dtype=np.float32)
    rms = np.zeros(n_frames, dtype=np.float32)

    # Loud first block establishes the stem's audible scale.
    Q_beat[36, :lag] = 1.0
    rms[: lag * frames_per_beat] = 1.0

    # Two inaudible blocks have different residual spectral shapes. They should
    # still be treated as equivalent silence.
    Q_beat[12, lag : 2 * lag] = 1e-4
    Q_beat[60, 2 * lag : 3 * lag] = 1e-4
    rms[lag * frames_per_beat :] = 1e-4

    beat_frames = np.arange(n_beats, dtype=int) * frames_per_beat
    beat_times = np.arange(n_beats, dtype=float)
    frame_times = np.arange(n_frames, dtype=float) / frames_per_beat

    phrase = _phrase_similarity_curves(
        Q_beat,
        rms,
        beat_frames,
        beat_times,
        frame_times,
        lags=(lag,),
    )

    idx = 2 * lag * frames_per_beat
    assert float(phrase["phrase_sim_8"][idx]) < 0.5
    assert float(phrase["phrase_spec_sim_8"][idx]) > 0.90  # noise-floor blocks → equivalent silence
    assert float(phrase["phrase_aud_sim_8"][idx]) > 0.95


def test_phrase_env_penalizes_energy_peak_and_valley_changes() -> None:
    lag = 8
    frames_per_beat = 10
    n_beats = lag * 2
    n_frames = n_beats * frames_per_beat
    Q_beat = np.zeros((84, n_beats), dtype=np.float32)
    rms = np.zeros(n_frames, dtype=np.float32)

    Q_beat[36, :] = 1.0
    block_shape = np.array([1.0, 0.9, 0.2, 0.9, 1.0, 0.8, 0.2, 0.8])
    flat_shape = np.full(lag, 0.7)
    for beat, value in enumerate(block_shape):
        rms[beat * frames_per_beat : (beat + 1) * frames_per_beat] = value
    for beat, value in enumerate(flat_shape, start=lag):
        rms[beat * frames_per_beat : (beat + 1) * frames_per_beat] = value

    beat_frames = np.arange(n_beats, dtype=int) * frames_per_beat
    beat_times = np.arange(n_beats, dtype=float)
    frame_times = np.arange(n_frames, dtype=float) / frames_per_beat

    phrase = _phrase_similarity_curves(
        Q_beat,
        rms,
        beat_frames,
        beat_times,
        frame_times,
        lags=(lag,),
    )

    idx = lag * frames_per_beat
    assert float(phrase["phrase_spec_sim_8"][idx]) > 0.95
    assert float(phrase["phrase_env_sim_8"][idx]) < 0.75


def test_phrase_accent_matches_frame_level_energy_peaks() -> None:
    lag = 8
    frames_per_beat = 50
    n_beats = lag * 2
    n_frames = n_beats * frames_per_beat
    Q_beat = np.zeros((84, n_beats), dtype=np.float32)
    Q_beat[36, :] = 1.0

    def make_rms(second_offsets: list[int]) -> np.ndarray:
        rms = np.full(n_frames, 0.08, dtype=np.float32)
        base_offsets = [25, 70, 116, 165, 215, 260, 306, 350]
        for block, offsets in enumerate([base_offsets, second_offsets]):
            block_start = block * lag * frames_per_beat
            for off in offsets:
                i = block_start + off
                rms[max(block_start, i - 2) : min(block_start + lag * frames_per_beat, i + 3)] = 1.0
        return rms

    beat_frames = np.arange(n_beats, dtype=int) * frames_per_beat
    beat_times = np.arange(n_beats, dtype=float)
    frame_times = np.arange(n_frames, dtype=float) / frames_per_beat

    same = _phrase_similarity_curves(
        Q_beat,
        make_rms([25, 70, 116, 165, 215, 260, 306, 350]),
        beat_frames,
        beat_times,
        frame_times,
        lags=(lag,),
    )
    shifted = _phrase_similarity_curves(
        Q_beat,
        make_rms([40, 95, 145, 196, 245, 292, 338]),
        beat_frames,
        beat_times,
        frame_times,
        lags=(lag,),
    )

    idx = lag * frames_per_beat
    assert float(same["phrase_spec_sim_8"][idx]) > 0.95
    assert float(same["phrase_accent_sim_8"][idx]) > 0.85
    assert float(shifted["phrase_spec_sim_8"][idx]) > 0.95
    assert float(shifted["phrase_accent_sim_8"][idx]) < 0.5


def test_phrase_accent_tolerates_small_peak_drift_and_extra_events() -> None:
    lag = 8
    frames_per_beat = 50
    n_beats = lag * 2
    n_frames = n_beats * frames_per_beat
    Q_beat = np.zeros((84, n_beats), dtype=np.float32)
    Q_beat[36, :] = 1.0

    base_offsets = [25, 70, 116, 165, 215, 260, 306, 350]
    drifted_offsets = [30, 72, 112, 150, 166, 215, 260, 306, 345]
    different_offsets = [40, 95, 145, 196, 245, 292, 338]

    def make_rms(second_offsets: list[int]) -> np.ndarray:
        rms = np.full(n_frames, 0.08, dtype=np.float32)
        for block, offsets in enumerate([base_offsets, second_offsets]):
            block_start = block * lag * frames_per_beat
            for off in offsets:
                i = block_start + off
                rms[max(block_start, i - 2) : min(block_start + lag * frames_per_beat, i + 3)] = 1.0
        return rms

    beat_frames = np.arange(n_beats, dtype=int) * frames_per_beat
    beat_times = np.arange(n_beats, dtype=float)
    frame_times = np.arange(n_frames, dtype=float) / frames_per_beat

    drifted = _phrase_similarity_curves(
        Q_beat,
        make_rms(drifted_offsets),
        beat_frames,
        beat_times,
        frame_times,
        lags=(lag,),
    )
    different = _phrase_similarity_curves(
        Q_beat,
        make_rms(different_offsets),
        beat_frames,
        beat_times,
        frame_times,
        lags=(lag,),
    )

    idx = lag * frames_per_beat
    assert float(drifted["phrase_accent_sim_8"][idx]) > 0.6
    assert float(different["phrase_accent_sim_8"][idx]) < 0.5


def test_phrase_amp_detects_pulsed_vs_constant_energy_texture() -> None:
    lag = 8
    frames_per_beat = 50
    n_beats = lag * 2
    n_frames = n_beats * frames_per_beat
    Q_beat = np.zeros((84, n_beats), dtype=np.float32)
    Q_beat[36, :] = 1.0
    rms = np.full(n_frames, 0.2, dtype=np.float32)

    for off in [25, 70, 116, 165, 215, 260, 306, 350]:
        rms[max(0, off - 4) : min(lag * frames_per_beat, off + 8)] = 1.0
    rms[lag * frames_per_beat :] = 0.45

    beat_frames = np.arange(n_beats, dtype=int) * frames_per_beat
    beat_times = np.arange(n_beats, dtype=float)
    frame_times = np.arange(n_frames, dtype=float) / frames_per_beat

    phrase = _phrase_similarity_curves(
        Q_beat,
        rms,
        beat_frames,
        beat_times,
        frame_times,
        lags=(lag,),
    )

    idx = lag * frames_per_beat
    assert float(phrase["phrase_spec_sim_8"][idx]) > 0.95
    assert float(phrase["phrase_amp_sim_8"][idx]) < 0.65


def test_phrase_amp_tolerates_small_peak_drift() -> None:
    lag = 8
    frames_per_beat = 50
    n_beats = lag * 2
    n_frames = n_beats * frames_per_beat
    Q_beat = np.zeros((84, n_beats), dtype=np.float32)
    Q_beat[36, :] = 1.0

    base_offsets = [25, 70, 116, 165, 215, 260, 306, 350]
    drifted_offsets = [30, 72, 112, 150, 166, 215, 260, 306, 345]

    rms = np.full(n_frames, 0.12, dtype=np.float32)
    for block, offsets in enumerate([base_offsets, drifted_offsets]):
        block_start = block * lag * frames_per_beat
        for off in offsets:
            i = block_start + off
            rms[max(block_start, i - 4) : min(block_start + lag * frames_per_beat, i + 8)] = 1.0

    beat_frames = np.arange(n_beats, dtype=int) * frames_per_beat
    beat_times = np.arange(n_beats, dtype=float)
    frame_times = np.arange(n_frames, dtype=float) / frames_per_beat

    phrase = _phrase_similarity_curves(
        Q_beat,
        rms,
        beat_frames,
        beat_times,
        frame_times,
        lags=(lag,),
    )

    idx = lag * frames_per_beat
    assert float(phrase["phrase_spec_sim_8"][idx]) > 0.95
    assert float(phrase["phrase_amp_sim_8"][idx]) > 0.70


def test_phrase_similarity_counts_short_tail_without_dominating_silence() -> None:
    lag = 8
    frames_per_beat = 10
    n_beats = lag * 3
    n_frames = n_beats * frames_per_beat
    Q_beat = np.zeros((84, n_beats), dtype=np.float32)
    rms = np.zeros(n_frames, dtype=np.float32)

    # Loud first block establishes the stem scale.
    Q_beat[36, :lag] = 1.0
    rms[: lag * frames_per_beat] = 1.0

    # Second block has only a short leading decay/tail, then silence.
    tail_end = lag * frames_per_beat + 20
    Q_beat[36, lag : lag + 2] = 0.25
    rms[lag * frames_per_beat : tail_end] = np.linspace(0.25, 0.02, 20)

    # Third block is fully inaudible residual noise.
    Q_beat[60, 2 * lag : 3 * lag] = 1e-4
    rms[2 * lag * frames_per_beat :] = 1e-4

    beat_frames = np.arange(n_beats, dtype=int) * frames_per_beat
    beat_times = np.arange(n_beats, dtype=float)
    frame_times = np.arange(n_frames, dtype=float) / frames_per_beat

    phrase = _phrase_similarity_curves(
        Q_beat,
        rms,
        beat_frames,
        beat_times,
        frame_times,
        lags=(lag,),
    )

    idx = 2 * lag * frames_per_beat
    assert float(phrase["phrase_sim_8"][idx]) < 0.5
    score = float(phrase["phrase_aud_sim_8"][idx])
    assert 0.80 < score < 1.0


def test_phrase_audible_keeps_sustained_quiet_content_active() -> None:
    lag = 8
    frames_per_beat = 10
    n_beats = lag * 3
    n_frames = n_beats * frames_per_beat
    Q_beat = np.zeros((84, n_beats), dtype=np.float32)
    rms = np.zeros(n_frames, dtype=np.float32)

    # Loud first block establishes scale, then two quiet but fully present blocks differ.
    Q_beat[36, :lag] = 1.0
    rms[: lag * frames_per_beat] = 1.0
    Q_beat[36, lag : 2 * lag] = 0.2
    rms[lag * frames_per_beat : 2 * lag * frames_per_beat] = 0.2
    Q_beat[60, 2 * lag : 3 * lag] = 0.2
    rms[2 * lag * frames_per_beat :] = 0.2

    beat_frames = np.arange(n_beats, dtype=int) * frames_per_beat
    beat_times = np.arange(n_beats, dtype=float)
    frame_times = np.arange(n_frames, dtype=float) / frames_per_beat

    phrase = _phrase_similarity_curves(
        Q_beat,
        rms,
        beat_frames,
        beat_times,
        frame_times,
        lags=(lag,),
    )

    idx = 2 * lag * frames_per_beat
    assert float(phrase["phrase_aud_sim_8"][idx]) < 0.6


def test_stem_block_offset_tracks_first_stable_activity() -> None:
    delayed = np.zeros(64, dtype=np.float32)
    delayed[8:] = 1.0
    immediate = np.ones(64, dtype=np.float32)
    late_on_grid = np.zeros(64, dtype=np.float32)
    late_on_grid[16:] = 1.0

    assert _stem_block_offset(delayed, 16) == 8
    assert _stem_block_offset(delayed, 8) == 0
    assert _stem_block_offset(immediate, 16) == 0
    assert _stem_block_offset(late_on_grid, 16) == 0


def test_stem_block_anchor_snaps_nearby_activity_to_bar_boundary() -> None:
    assert _snap_anchor_to_bar_beat(7) == 8
    assert _snap_anchor_to_bar_beat(9) == 8
    assert _snap_anchor_to_bar_beat(10) == 8
    assert _snap_anchor_to_bar_beat(11) == 12
    assert _snap_anchor_to_bar_beat(8, bar_phase=1) == 9


def test_stem_block_offsets_use_stem_anchor_per_lag() -> None:
    delayed = np.zeros(64, dtype=np.float32)
    delayed[8:] = 1.0

    offsets = _stem_block_offsets(delayed, (8, 16, 32))

    assert _first_stable_active_beat(delayed) == 8
    assert _stem_block_anchor(delayed) == 8
    assert offsets == {8: 0, 16: 8, 32: 8}


def test_stem_block_offsets_use_estimated_bar_phase() -> None:
    delayed = np.zeros(64, dtype=np.float32)
    delayed[8:] = 1.0

    offsets = _stem_block_offsets(delayed, (8, 16, 32), bar_phase=1)

    assert _stem_block_anchor(delayed, bar_phase=1) == 9
    assert offsets == {8: 1, 16: 9, 32: 9}


def test_bar_phase_similarity_diagnostic_selects_clear_contrast_phase() -> None:
    n_beats = 96
    q = np.zeros((4, n_beats), dtype=np.float32)
    rms = np.ones(n_beats, dtype=np.float32)
    a = np.array([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32)
    b = np.array([[0.0], [1.0], [0.0], [0.0]], dtype=np.float32)
    c = np.array([[0.0], [0.0], [1.0], [0.0]], dtype=np.float32)
    d = np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float32)
    sequence = [(a, b), (a, b), (c, d), (c, d), (a, b)]
    for i, (first_half, second_half) in enumerate(sequence):
        start = 2 + 16 * i
        q[:, start:start + 8] = first_half
        q[:, start + 8:start + 16] = second_half

    diag = _bar_phase_similarity_diagnostic(
        q,
        rms,
        beats_per_bar=4,
        parent_beats=16,
        min_samples=3,
        min_margin=0.01,
    )

    assert diag["phase"] == 2
    assert diag["estimated_phase"] == 2
    assert diag["accepted"] is True


def test_bar_phase_similarity_diagnostic_falls_back_on_low_contrast() -> None:
    q = np.ones((4, 96), dtype=np.float32)
    rms = np.ones(96, dtype=np.float32)

    diag = _bar_phase_similarity_diagnostic(
        q,
        rms,
        beats_per_bar=4,
        parent_beats=16,
        min_samples=3,
        min_margin=0.01,
    )

    assert diag["phase"] == 0
    assert diag["accepted"] is False


def test_cqt_half_block_compares_same_position_in_previous_parent() -> None:
    q = np.zeros((2, 16), dtype=np.float32)
    q[0, 0:4] = 1.0
    q[1, 4:8] = 1.0
    q[0, 8:12] = 1.0
    q[0, 12:16] = 1.0
    beat_times = np.arange(16, dtype=float)
    frame_times = np.arange(16, dtype=float)

    curves = _cqt_similarity_curves(q, beat_times, frame_times, lags=(8,))

    assert float(curves["cqt_half_sim_8"][8]) > 0.99
    assert float(curves["cqt_half_sim_8"][12]) < 0.05


def test_detect_silence_events_finds_long_gap_only() -> None:
    sr = 22050
    hop = 512
    # 5-second audio: loud 0–1s, silent 1–1.3s, loud 1.3–3s, silent 3–4.5s, loud 4.5–5s
    n = int(sr * 5)
    y = np.zeros(n, dtype=np.float32)
    t = np.linspace(0, 5, n, endpoint=False)
    loud = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    # mask loud regions
    y[(t < 1.0) | ((t >= 1.3) & (t < 3.0)) | (t >= 4.5)] = \
        loud[(t < 1.0) | ((t >= 1.3) & (t < 3.0)) | (t >= 4.5)]

    import librosa
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop, center=True)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    events = _detect_silence_events(rms, times, min_s=0.8, frac=0.05)
    # Only the 1.5s gap (3–4.5s) should qualify; the 0.3s gap is too short
    assert len(events) == 1, f"Expected 1 silence event, got {len(events)}: {events}"
    ev = events[0]
    assert ev["end_s"] - ev["start_s"] >= 0.8, "Silence too short"
    assert ev["start_s"] >= 2.5, f"Expected silence after 2.5s, got start={ev['start_s']:.2f}"


def test_detect_stem_transitions_with_hysteresis() -> None:
    sr = 22050
    hop = 512
    frame_length = 2048

    def _make_stem(loud_mask: np.ndarray, n: int) -> np.ndarray:
        t = np.linspace(0, n / sr, n, endpoint=False)
        y = 0.4 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        y[~loud_mask] = 0.0
        return y

    n = int(sr * 10)
    sample_t = np.linspace(0, 10, n, endpoint=False)

    # Clean stem: silent 0–2s, loud 2–8s, silent 8–10s
    clean_loud = (sample_t >= 2.0) & (sample_t < 8.0)
    clean_y = _make_stem(clean_loud, n)

    # Flickery stem: loud with a 0.2s gap at 5–5.2s (shorter than _STEM_MIN_HOLD_S=0.6s)
    flick_loud = (sample_t >= 0.5) & (sample_t < 9.5) & ~((sample_t >= 5.0) & (sample_t < 5.2))
    flick_y = _make_stem(flick_loud, n)

    stems = {"clean": clean_y, "flickery": flick_y}
    events = _detect_stem_transitions(stems, sr, hop_length=hop, frame_length=frame_length)

    clean_evs = [e for e in events if e["stem"] == "clean"]
    flick_evs = [e for e in events if e["stem"] == "flickery"]

    # Clean: exactly one enter (~2s) and one exit (~8s)
    assert len(clean_evs) == 2, f"clean stem: expected 2 transitions, got {clean_evs}"
    kinds = [e["kind"] for e in clean_evs]
    assert kinds == ["enter", "exit"], f"clean stem: expected [enter, exit], got {kinds}"
    assert 1.0 <= clean_evs[0]["time_s"] <= 3.0, f"enter time off: {clean_evs[0]['time_s']}"
    assert 7.0 <= clean_evs[1]["time_s"] <= 9.0, f"exit time off: {clean_evs[1]['time_s']}"

    # Flickery: the 0.2s gap is below hold threshold → no spurious exit/enter pair
    kinds_flick = [e["kind"] for e in flick_evs]
    assert "exit" not in kinds_flick or all(
        e["time_s"] >= 9.0 for e in flick_evs if e["kind"] == "exit"
    ), f"flickery stem: spurious exit/enter pair detected: {flick_evs}"


def test_merge_short_segments_discarded_boundaries_recoverable() -> None:
    # [0, 5, 9, 30] with min_len_s=8 and duration=30.
    # Cluster {5, 9} (gap 4 < 8): best=9, gap_back=9≥8, gap_fwd=21≥8 → kept.
    # 5 is discarded.
    bounds_in = [0.0, 5.0, 9.0, 30.0]
    kept = _merge_short_segments(bounds_in, min_len_s=8.0, duration_s=30.0)
    kept_set = {round(b, 3) for b in kept}

    discarded = [
        b for b in bounds_in
        if 0.0 < b < 30.0 and round(b, 3) not in kept_set
    ]
    # Exactly one boundary discarded, one kept from the cluster
    assert len(discarded) == 1, f"Expected 1 discarded boundary, got {discarded}"
    assert discarded[0] == 5.0, f"Expected 5.0 discarded, got {discarded}"
    assert len([b for b in kept if 0.0 < b < 30.0]) == 1, (
        f"Expected 1 kept internal boundary, got {kept}"
    )
    # The kept boundary (9.0) must produce sections ≥ min_len_s=8s
    internal_kept = [b for b in kept if 0.0 < b < 30.0][0]
    assert internal_kept >= 8.0, f"Kept boundary too early: {internal_kept}"
    assert 30.0 - internal_kept >= 8.0, f"Kept boundary too late: {internal_kept}"
