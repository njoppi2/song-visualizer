import numpy as np
import pytest

from experiments.change_decision import compute_change_decision


def _run(feature, energy=None, beats=None):
    n = feature.shape[1]
    return compute_change_decision({"stem": feature}, {"stem": np.ones(n) if energy is None else energy},
                                   np.arange(n + 1, dtype=float) if beats is None else beats)


def _sample(result, h, k):
    return next(item for item in result["scales"] if item["scale_beats"] == h)["samples"][k]["per_stem"]["stem"]


def test_constant_and_balanced_alternation_are_below_threshold_including_negative_scores():
    constant = _run(np.ones((2, 32)))
    assert _sample(constant, 8, 16)["score"] == pytest.approx(0)
    alternating = np.array([[1, 0] * 16, [0, 1] * 16], dtype=float)
    result = _run(alternating)
    assert _sample(result, 4, 16)["score"] < 0
    assert _sample(result, 8, 16)["score"] < .20
    assert result["anchors"][16]["pattern_signal"] is False


def test_persistent_orthogonal_step_qualifies_at_both_scales_and_adjacent_anchors():
    feature = np.array([[1] * 16 + [0] * 16, [0] * 16 + [1] * 16], dtype=float)
    result = _run(feature)
    assert all(_sample(result, h, k)["score"] >= .20 for h in (4, 8) for k in (16, 17))
    assert result["anchors"][16]["pattern_changes"] == ["stem"]
    assert result["anchors"][16]["classification"] == "pattern_shift"


def test_joint_gain_rescale_preserves_scores_and_log_cqt_representation():
    raw = np.array([[1] * 16 + [0] * 16, [0] * 16 + [1] * 16], dtype=float)
    result = _run(np.log1p(raw), np.ones(32))
    scaled = _run(np.log1p(raw * 17), np.ones(32) * 17)
    for h in (4, 8):
        assert _sample(result, h, 16)["score"] == pytest.approx(_sample(scaled, h, 16)["score"])


def test_tiny_nonzero_spectra_do_not_disappear_through_cancellation():
    result = _run(np.log1p(np.full((2, 32), 1e-20)))
    assert _sample(result, 8, 16)["usable_left"] == 8
    assert _sample(result, 8, 16)["score"] == pytest.approx(0)


def test_silence_and_zero_spectra_are_unknown_and_edges_are_null():
    result = _run(np.zeros((2, 32)), np.zeros(32))
    assert _sample(result, 8, 16)["score"] is None
    assert result["anchors"][16]["classification"] == "insufficient_pattern_evidence"
    assert result["anchors"][7] is None and result["anchors"][24] is None
    assert next(item for item in result["scales"] if item["scale_beats"] == 8)["samples"][7] is None


def test_timing_validation_input_preservation_and_level_intersection():
    feature = np.ones((2, 32))
    energy = np.array([1] * 16 + [4] * 16, dtype=float)
    beats = np.arange(33, dtype=float)
    original = (feature.copy(), energy.copy(), beats.copy())
    result = _run(feature, energy, beats)
    assert result["anchors"][16]["level_changes"] == [{"stem": "stem", "direction": "increase"}]
    assert (result["anchors"][16]["support_start_s"], result["anchors"][16]["support_end_s"],
            result["anchors"][16]["available_at_s"]) == (8, 25, 25)
    assert np.array_equal(feature, original[0]) and np.array_equal(energy, original[1]) and np.array_equal(beats, original[2])
    with pytest.raises(ValueError):
        _run(feature, energy, np.array([0, 1, 1] + list(range(3, 33)), dtype=float))
