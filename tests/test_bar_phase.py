"""Unit tests for songviz.bar_phase — pure functions, no audio I/O."""
from __future__ import annotations

import numpy as np
import pytest

from songviz.bar_phase import (
    _build_bar_novelty,
    _confidence_margin,
    _normalise_scores,
    _phase_groups,
    beats_to_bar_times,
    hybrid_score,
    madmom_downbeat_compare,
    multi_feature_metrical_score,
    onset_strength_phase_vote,
    phrase_grid_boundary_contrast,
    similarity_phrase_alignment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_beat_times(n: int = 64, bpm: float = 120.0) -> np.ndarray:
    """Uniform beat grid starting at t=0."""
    interval = 60.0 / bpm
    return np.arange(n, dtype=np.float64) * interval


def _make_kick_pattern(beat_times: np.ndarray, downbeat_phase: int, sr: int, hop: int) -> np.ndarray:
    """Synthetic kick-drum onset envelope: strong pulse every 4th beat at downbeat_phase."""
    n_frames = int(beat_times[-1] * sr / hop) + hop
    env = np.zeros(n_frames, dtype=np.float32)
    for i, t in enumerate(beat_times):
        frame = int(round(t * sr / hop))
        if frame < n_frames:
            strength = 1.0 if (i - downbeat_phase) % 4 == 0 else 0.2
            env[max(0, frame - 1) : frame + 2] = strength
    return env


def _stub_m2(phase: int = 0, beats_per_bar: int = 4) -> dict:
    """Minimal M2 result stub."""
    scores = {str(p): 1.0 if p == phase else 0.0 for p in range(beats_per_bar)}
    return {
        "method": "multi_feature_metrical_score",
        "phase": phase,
        "scores": scores,
        "scores_normalised": scores,
        "sub_scores": {},
        "confidence": 1.0,
        "beats_per_bar": beats_per_bar,
    }


def _stub_m3(phrase_length: int = 16, offset: int = 0, contrast: float = 0.5) -> dict:
    cands = [
        {"phrase_length": phrase_length, "offset": offset, "contrast": contrast, "n_phrases": 4,
         "boundary_mean_novelty": 0.6, "nonboundary_mean_novelty": 0.1},
        {"phrase_length": 8, "offset": 0, "contrast": 0.2, "n_phrases": 8,
         "boundary_mean_novelty": 0.3, "nonboundary_mean_novelty": 0.1},
    ]
    return {
        "method": "phrase_grid_boundary_contrast",
        "best_phrase_length": phrase_length, "best_offset": offset,
        "best_contrast": contrast, "candidates": cands,
        "best_per_length": {str(phrase_length): cands[0]},
        "n_bars": 32, "bar_phase": 0,
    }


def _stub_m4(phrase_length: int = 16, offset: int = 0) -> dict:
    cands = [
        {"phrase_length": phrase_length, "offset": offset, "n_phrases": 4,
         "adj_similarity_mean": 0.7, "adj_similarity_std": 0.15,
         "boundary_novelty": 0.3, "within_consistency": 0.8,
         "repeated_phrase_similarity": 0.6, "stability_cleanliness": 0.15},
        {"phrase_length": 8, "offset": 0, "n_phrases": 8,
         "adj_similarity_mean": 0.6, "adj_similarity_std": 0.08,
         "boundary_novelty": 0.4, "within_consistency": 0.75,
         "repeated_phrase_similarity": 0.5, "stability_cleanliness": 0.08},
    ]
    return {
        "method": "similarity_phrase_alignment",
        "candidates": cands,
        "all_candidates": cands,
    }


# ---------------------------------------------------------------------------
# _phase_groups
# ---------------------------------------------------------------------------

def test_phase_groups_basic():
    db, ob = _phase_groups(8, phase=0, beats_per_bar=4)
    assert list(db) == [0, 4]
    assert list(ob) == [1, 2, 3, 5, 6, 7]


def test_phase_groups_phase_2():
    db, ob = _phase_groups(8, phase=2, beats_per_bar=4)
    assert list(db) == [2, 6]
    assert set(ob) == {0, 1, 3, 4, 5, 7}


def test_phase_groups_returns_all_indices():
    n = 16
    for p in range(4):
        db, ob = _phase_groups(n, phase=p)
        assert len(db) + len(ob) == n
        assert set(db) | set(ob) == set(range(n))


# ---------------------------------------------------------------------------
# _confidence_margin
# ---------------------------------------------------------------------------

def test_confidence_margin_clear_winner():
    scores = {0: 1.0, 1: 0.3, 2: 0.2, 3: 0.1}
    assert pytest.approx(_confidence_margin(scores), abs=1e-6) == 0.7


def test_confidence_margin_tie():
    scores = {0: 0.5, 1: 0.5, 2: 0.3, 3: 0.1}
    assert _confidence_margin(scores) == 0.0


def test_confidence_margin_single():
    assert _confidence_margin({0: 1.0}) == 0.0


# ---------------------------------------------------------------------------
# _normalise_scores
# ---------------------------------------------------------------------------

def test_normalise_scores_range():
    scores = {0: 0.1, 1: 0.3, 2: 0.5, 3: 0.9}
    normed = _normalise_scores(scores)
    assert pytest.approx(min(normed.values())) == 0.0
    assert pytest.approx(max(normed.values())) == 1.0


def test_normalise_scores_flat():
    scores = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}
    normed = _normalise_scores(scores)
    assert all(v == 0.0 for v in normed.values())


# ---------------------------------------------------------------------------
# beats_to_bar_times
# ---------------------------------------------------------------------------

def test_beats_to_bar_times_phase0():
    beats = np.arange(16, dtype=float) * 0.5
    bar_times = beats_to_bar_times(beats, bar_phase=0, beats_per_bar=4)
    expected = beats[::4]
    np.testing.assert_allclose(bar_times, expected)


def test_beats_to_bar_times_phase2():
    beats = np.arange(16, dtype=float) * 0.5
    bar_times = beats_to_bar_times(beats, bar_phase=2, beats_per_bar=4)
    # indices where (i - 2) % 4 == 0 → i = 2, 6, 10, 14
    np.testing.assert_allclose(bar_times, beats[[2, 6, 10, 14]])


# ---------------------------------------------------------------------------
# Method 1 — onset_strength_phase_vote
# ---------------------------------------------------------------------------

def test_m1_detects_correct_phase():
    """Synthetic kick pattern: strong beat every 4 at phase=2 → M1 should pick phase 2."""
    sr, hop = 22050, 512
    beat_times = _make_beat_times(64, bpm=120.0)
    true_phase = 2
    onset_env = _make_kick_pattern(beat_times, true_phase, sr, hop)
    result = onset_strength_phase_vote(beat_times, onset_env, sr, hop, beats_per_bar=4)
    assert result["phase"] == true_phase
    assert result["confidence"] > 0.0
    assert set(result["scores"].keys()) == {"0", "1", "2", "3"}


def test_m1_output_keys():
    sr, hop = 22050, 512
    beat_times = _make_beat_times(32)
    onset_env = np.ones(int(beat_times[-1] * sr / hop) + 10, dtype=np.float32)
    result = onset_strength_phase_vote(beat_times, onset_env, sr, hop)
    for key in ("method", "phase", "scores", "scores_normalised", "confidence", "beats_per_bar"):
        assert key in result


def test_m1_normalised_range():
    sr, hop = 22050, 512
    beat_times = _make_beat_times(32)
    onset_env = _make_kick_pattern(beat_times, 0, sr, hop)
    result = onset_strength_phase_vote(beat_times, onset_env, sr, hop)
    norm_vals = list(result["scores_normalised"].values())
    assert min(norm_vals) >= -1e-9
    assert max(norm_vals) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Method 2 — multi_feature_metrical_score
# ---------------------------------------------------------------------------

def test_m2_no_features_returns_warning():
    beat_times = _make_beat_times(32)
    result = multi_feature_metrical_score(beat_times, {}, 22050, 512)
    assert "warning" in result
    assert result["phase"] == 0


def test_m2_kick_dominated_correct_phase():
    sr, hop = 22050, 512
    beat_times = _make_beat_times(64, bpm=120.0)
    true_phase = 1
    kick = _make_kick_pattern(beat_times, true_phase, sr, hop)
    features = {"kick_energy": kick, "onset_env": kick}
    result = multi_feature_metrical_score(beat_times, features, sr, hop)
    assert result["phase"] == true_phase


def test_m2_sub_scores_structure():
    sr, hop = 22050, 512
    beat_times = _make_beat_times(32)
    kick = _make_kick_pattern(beat_times, 0, sr, hop)
    result = multi_feature_metrical_score(beat_times, {"kick_energy": kick}, sr, hop)
    for p in ("0", "1", "2", "3"):
        assert p in result["sub_scores"]
        sub = result["sub_scores"][p]
        for k in ("downbeat_contrast", "backbeat_score", "pattern_consistency", "total"):
            assert k in sub


def test_m2_configurable_weights():
    sr, hop = 22050, 512
    beat_times = _make_beat_times(32)
    kick = _make_kick_pattern(beat_times, 0, sr, hop)
    features = {"kick_energy": kick}
    custom = {"downbeat_contrast": 1.0, "backbeat_score": 0.0, "pattern_consistency": 0.0}
    result = multi_feature_metrical_score(beat_times, features, sr, hop, weights=custom)
    assert result["weights"]["backbeat_score"] == 0.0


# ---------------------------------------------------------------------------
# Method 3 — phrase_grid_boundary_contrast
# ---------------------------------------------------------------------------

def test_m3_output_keys():
    sr, hop = 22050, 512
    beat_times = _make_beat_times(128, bpm=120.0)
    kick = _make_kick_pattern(beat_times, 0, sr, hop)
    result = phrase_grid_boundary_contrast(
        beat_times, {"kick_energy": kick}, sr, hop, bar_phase=0,
        phrase_lengths=[8, 16], beats_per_bar=4,
    )
    for key in ("method", "best_phrase_length", "best_offset", "candidates", "n_bars"):
        assert key in result


def test_m3_candidates_sorted_descending():
    sr, hop = 22050, 512
    beat_times = _make_beat_times(128)
    kick = _make_kick_pattern(beat_times, 0, sr, hop)
    result = phrase_grid_boundary_contrast(
        beat_times, {"kick_energy": kick}, sr, hop, bar_phase=0, phrase_lengths=[8, 16],
    )
    contrasts = [c["contrast"] for c in result["candidates"]]
    assert contrasts == sorted(contrasts, reverse=True)


def test_m3_no_features():
    beat_times = _make_beat_times(64)
    result = phrase_grid_boundary_contrast(
        beat_times, {}, 22050, 512, bar_phase=0, phrase_lengths=[8],
    )
    assert result["candidates"] == []


def test_m3_phrase_boundary_indices():
    """With L=8 offset=0, boundaries at bars 0, 8, 16, 24 …"""
    n_bars = 32
    L, o = 8, 0
    b_idx = np.arange(o, n_bars, L)
    expected = np.array([0, 8, 16, 24])
    np.testing.assert_array_equal(b_idx, expected)


# ---------------------------------------------------------------------------
# Method 4 — similarity_phrase_alignment
# ---------------------------------------------------------------------------

def test_m4_output_keys():
    sr, hop = 22050, 512
    beat_times = _make_beat_times(128)
    kick = _make_kick_pattern(beat_times, 0, sr, hop)
    result = similarity_phrase_alignment(
        beat_times, {"kick_energy": kick}, sr, hop, bar_phase=0, phrase_lengths=[8, 16],
    )
    assert "candidates" in result
    assert "all_candidates" in result


def test_m4_no_features():
    beat_times = _make_beat_times(64)
    result = similarity_phrase_alignment(beat_times, {}, 22050, 512, bar_phase=0)
    assert result["candidates"] == []


def test_m4_candidate_fields():
    sr, hop = 22050, 512
    beat_times = _make_beat_times(128)
    kick = _make_kick_pattern(beat_times, 0, sr, hop)
    result = similarity_phrase_alignment(
        beat_times, {"kick_energy": kick}, sr, hop, bar_phase=0, phrase_lengths=[8],
    )
    if result["candidates"]:
        cand = result["candidates"][0]
        for field in ("phrase_length", "offset", "adj_similarity_mean", "stability_cleanliness",
                      "within_consistency", "repeated_phrase_similarity"):
            assert field in cand


# ---------------------------------------------------------------------------
# Method 5 — hybrid_score
# ---------------------------------------------------------------------------

def test_hybrid_output_structure():
    m2 = _stub_m2(phase=0)
    m3 = _stub_m3(phrase_length=16, offset=0)
    m4 = _stub_m4(phrase_length=16, offset=0)
    result = hybrid_score(m2, m3, m4)
    assert "best" in result
    assert "candidates" in result
    assert "confidence" in result
    assert "bar_phase" in result


def test_hybrid_candidates_sorted():
    m2 = _stub_m2(phase=0)
    m3 = _stub_m3()
    m4 = _stub_m4()
    result = hybrid_score(m2, m3, m4)
    scores = [c["total_score"] for c in result["candidates"]]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_best_matches_top_candidate():
    m2 = _stub_m2(phase=0)
    m3 = _stub_m3()
    m4 = _stub_m4()
    result = hybrid_score(m2, m3, m4)
    if result["candidates"] and result["best"]:
        assert result["best"]["total_score"] == result["candidates"][0]["total_score"]


def test_hybrid_configurable_weights():
    m2 = _stub_m2(phase=0)
    m3 = _stub_m3()
    m4 = _stub_m4()
    custom = {"metrical_role_score": 1.0, "bar_pattern_consistency": 0.0,
              "phrase_boundary_contrast": 0.0, "repeated_phrase_similarity": 0.0,
              "stability_curve_cleanliness": 0.0}
    result = hybrid_score(m2, m3, m4, weights=custom)
    assert result["weights"]["bar_pattern_consistency"] == 0.0


def test_hybrid_component_scores_present():
    m2 = _stub_m2(phase=0)
    m3 = _stub_m3()
    m4 = _stub_m4()
    result = hybrid_score(m2, m3, m4)
    if result["candidates"]:
        comp = result["candidates"][0]["component_scores"]
        for k in ("metrical_role_score", "bar_pattern_consistency",
                  "phrase_boundary_contrast", "repeated_phrase_similarity",
                  "stability_curve_cleanliness"):
            assert k in comp


# ---------------------------------------------------------------------------
# Method 6 — madmom (optional dep)
# ---------------------------------------------------------------------------

def test_madmom_skips_gracefully_when_not_installed(monkeypatch):
    """Force ImportError to verify the skip path."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if "madmom" in name:
            raise ImportError("madmom not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    beat_times = _make_beat_times(32)
    result = madmom_downbeat_compare(beat_times, "/nonexistent/audio.wav")
    assert result["skipped"] is True
    assert "madmom" in result.get("reason", "").lower()


# ---------------------------------------------------------------------------
# _build_bar_novelty helper
# ---------------------------------------------------------------------------

def test_build_bar_novelty_shape():
    """Should return one scalar per complete bar."""
    beat_times = _make_beat_times(32)
    sr, hop = 22050, 512
    beat_frames = np.round(beat_times * sr / hop).astype(int)
    kick = _make_kick_pattern(beat_times, 0, sr, hop)
    from songviz.bar_phase import _beat_to_frame, _sample_at_beats
    bf = np.clip(_beat_to_frame(beat_times, sr, hop), 0, len(kick) - 1)
    beat_feats = {"kick": _sample_at_beats(kick.astype(np.float64), bf)}
    novelty = _build_bar_novelty(beat_feats, bar_phase=0, beats_per_bar=4)
    # 32 beats / 4 per bar = 8 bars
    assert len(novelty) == 8


def test_build_bar_novelty_empty_features():
    novelty = _build_bar_novelty({}, bar_phase=0, beats_per_bar=4)
    assert len(novelty) >= 1
