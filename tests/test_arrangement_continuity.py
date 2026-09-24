import json

import numpy as np
import pytest

from experiments.arrangement_continuity import compute_arrangement_continuity


N_BEATS = 28
ANCHOR = 14


def _log_pattern(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.vstack((values, values * 0.6)))


def _inputs(*, background: str = "stable", uniform_gain: bool = False) -> tuple[dict, dict, np.ndarray]:
    beats = np.arange(N_BEATS + 1, dtype=float)
    base = 1.0 + (np.arange(N_BEATS) % 4) * 0.15
    changed = np.where(np.arange(N_BEATS) < ANCHOR, 1.0, 4.2)
    features = {"drums": _log_pattern(changed)}
    energy = {"drums": changed.copy()}
    for stem, multiplier in (("bass", 0.8), ("other", 1.2), ("vocals", 0.7)):
        values = base * multiplier
        cqt_values = values
        if background == "silence" and stem in {"bass", "other"}:
            values = np.zeros(N_BEATS)
            cqt_values = values
        elif background == "reordered" and stem in {"bass", "other"}:
            # Keep RMS matched while moving sparse spectral peaks to different
            # corresponding beat positions in each comparison window.
            cqt_values = np.ones(N_BEATS) * 0.01
            left = np.array([10.0, 0.01, 5.0, 0.01, 3.0, 0.01, 2.0, 0.01])
            cqt_values[ANCHOR - 8:ANCHOR] = left
            cqt_values[ANCHOR:ANCHOR + 8] = np.roll(left, 1)
        if uniform_gain:
            values = np.where(np.arange(N_BEATS) < ANCHOR, values, values * 4.2)
            cqt_values = np.where(np.arange(N_BEATS) < ANCHOR, cqt_values, cqt_values * 4.2)
        features[stem] = _log_pattern(cqt_values)
        energy[stem] = values.copy()
    return features, energy, beats


def _anchor(result: dict) -> dict:
    record = result["anchors"][ANCHOR]
    assert record is not None
    return record


def test_changed_layer_with_unchanged_patterned_background_is_candidate() -> None:
    result = compute_arrangement_continuity(*_inputs())
    record = _anchor(result)
    assert record["level_change"] is True
    assert record["continuity_support"] is True
    assert record["arrangement_candidate"] is True
    assert record["changed_layers"] == [{"stem": "drums", "direction": "increase"}]
    assert record["support_union_start_s"] == 6.0
    assert record["support_union_end_s"] == 23.0
    assert record["available_at_s"] == 23.0
    assert json.loads(json.dumps(result))["anchors"][ANCHOR]["arrangement_candidate"] is True


def test_uniform_gain_is_level_only_not_continuity_candidate() -> None:
    result = compute_arrangement_continuity(*_inputs(uniform_gain=True))
    record = _anchor(result)
    assert record["level_change"] is True
    assert record["continuity_support"] is False
    assert record["arrangement_candidate"] is False


def test_shared_silence_does_not_support_continuity() -> None:
    result = compute_arrangement_continuity(*_inputs(background="silence"))
    record = _anchor(result)
    assert record["level_change"] is True
    assert record["continuity_support"] is False
    bass = record["samples"]["anchor"]["4"]["per_stem"][1]
    assert bass["cosine_similarity"] is None
    assert bass["cosine_reason"] == "nonpositive_norm"


def test_reordered_background_with_matched_energy_rejects_continuity() -> None:
    result = compute_arrangement_continuity(*_inputs(background="reordered"))
    record = _anchor(result)
    assert record["level_change"] is True
    assert record["continuity_support"] is False
    bass = record["samples"]["anchor"]["8"]["per_stem"][1]
    assert bass["median_left"] == pytest.approx(bass["median_right"])
    # The test alters CQT ordering only: its RMS remains matched while ordered
    # spectral support is deliberately made dissimilar.
    assert bass["cosine_similarity"] < 0.90


def test_unsupported_edges_are_explicit_and_invalid_inputs_rejected() -> None:
    features, energy, beats = _inputs()
    result = compute_arrangement_continuity(features, energy, beats)
    assert result["scales"][1]["samples"][0] is None
    assert result["anchors"][0] is None
    bad = dict(features)
    bad["drums"] = bad["drums"].copy()
    bad["drums"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_arrangement_continuity(bad, energy, beats)
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_arrangement_continuity(features, energy, np.r_[beats[:3], beats[2:]])
    with pytest.raises(ValueError, match="identical stem names"):
        compute_arrangement_continuity(features, {"drums": energy["drums"]}, beats)
    huge = dict(features)
    huge["drums"] = huge["drums"].copy()
    huge["drums"][0, 0] = 1_000.0
    with pytest.raises(ValueError, match="expm1"):
        compute_arrangement_continuity(huge, energy, beats)


def test_midpoint_ties_do_not_satisfy_the_strict_level_side_rule() -> None:
    features, energy, beats = _inputs()
    # At anchor 14 the left median is 2, right median is 6, and the midpoint
    # ties occupy half of each side.  They must not count toward either side.
    tied = energy["drums"].copy()
    tied[ANCHOR - 8:ANCHOR] = [0.0, 0.0, 4.0, 4.0, 0.0, 0.0, 4.0, 4.0]
    tied[ANCHOR:ANCHOR + 8] = [8.0, 8.0, 4.0, 4.0, 8.0, 8.0, 4.0, 4.0]
    energy = dict(energy)
    energy["drums"] = tied
    result = compute_arrangement_continuity(features, energy, beats)
    record = _anchor(result)
    assert record["level_change"] is False
    drums = record["samples"]["anchor"]["4"]["per_stem"][0]
    assert drums["left_midpoint_fraction"] == drums["right_midpoint_fraction"] == 0.5
    assert drums["level_change"] is False
