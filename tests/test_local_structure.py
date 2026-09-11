import json

import numpy as np
import pytest

from songviz.local_structure import LocalStructureConfig, detect_local_structure


def _inputs(n=24, *, energy=None, feature=None):
    energy = np.ones(n) if energy is None else np.asarray(energy, dtype=float)
    feature = np.ones((2, n)) if feature is None else np.asarray(feature, dtype=float)
    return {"stem": feature}, {"stem": energy}, np.arange(n + 1, dtype=float)


def test_silence_has_no_candidates_and_null_curves():
    result = detect_local_structure(*_inputs(12, energy=np.zeros(12)))
    assert not result["changes"] and not result["transitions"]
    assert all(all(value is None for value in curve["combined_change"]) for curve in result["curves"])


def test_stationary_repeated_pattern_has_no_changes():
    assert not detect_local_structure(*_inputs())["changes"]


def test_level_step_is_arrangement_change_but_not_a_dip():
    result = detect_local_structure(*_inputs(24, energy=[1] * 12 + [2] * 12))
    assert any(change["beat_index"] == 12 and change["arrangement_change"] > 0 for change in result["changes"])
    assert not result["transitions"]


def test_spectral_step_with_unchanged_energy_is_pattern_change():
    feature = np.array([[1] * 12 + [0] * 12, [0] * 12 + [1] * 12], dtype=float)
    result = detect_local_structure(*_inputs(24, feature=feature))
    assert any(change["beat_index"] == 12 and change["pattern_change"] > 0 for change in result["changes"])


def test_exact_two_beat_energy_dip_is_bounded_span():
    energy = np.array([1] * 6 + [.1, .1] + [1] * 8, dtype=float)
    result = detect_local_structure(*_inputs(len(energy), energy=energy))
    assert [(x["start_beat"], x["end_beat"]) for x in result["transitions"]] == [(6, 8)]


def test_adjacent_dips_remain_separate():
    energy = np.array([1] * 5 + [.1, .1] + [1] + [.1, .1] + [1] * 6, dtype=float)
    result = detect_local_structure(
        *_inputs(len(energy), energy=energy),
        config=LocalStructureConfig(dip_flank_beats=1),
    )
    spans = [(x["start_beat"], x["end_beat"]) for x in result["transitions"]]
    assert (5, 7) in spans and (8, 10) in spans


def test_track_edges_and_ramps_do_not_fabricate_dips():
    for energy in ([.1, .1] + [1] * 12, [1] * 12 + [.1, .1], np.linspace(.1, 1, 20), np.linspace(1, .1, 20)):
        assert not detect_local_structure(*_inputs(len(energy), energy=energy))["transitions"]


def test_scale_edges_are_null_and_right_support_sets_availability():
    feature = np.array([[1] * 12 + [0] * 12, [0] * 12 + [1] * 12], dtype=float)
    result = detect_local_structure(*_inputs(24, feature=feature), config=LocalStructureConfig(contrast_scales=(4,)))
    curve = result["curves"][0]
    assert curve["combined_change"][:4] == [None] * 4 and curve["combined_change"][-4:] == [None] * 4
    change = next(x for x in result["changes"] if x["beat_index"] == 12)
    assert change["available_at_s"] == 16.0


def test_validation_immutability_and_strict_json():
    f, e, bt = _inputs(10)
    before = (f["stem"].copy(), e["stem"].copy(), bt.copy())
    result = detect_local_structure(f, e, bt)
    assert np.array_equal(f["stem"], before[0]) and np.array_equal(e["stem"], before[1]) and np.array_equal(bt, before[2])
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError):
        detect_local_structure({"stem": np.array([[np.nan] * 10])}, e, bt)
    with pytest.raises(ValueError):
        detect_local_structure(f, e, np.array([0, 1, 1] + list(range(3, 11)), dtype=float))
    with pytest.raises(ValueError):
        detect_local_structure(f, e, bt, config=LocalStructureConfig(contrast_scales=(True,)))
    with pytest.raises(ValueError):
        detect_local_structure(f, e, bt, config=LocalStructureConfig(dip_ratio=float("nan")))


def test_multiscale_support_includes_widest_evidence_not_only_primary_scale():
    feature = np.array([[1] * 24 + [0] * 24, [0] * 24 + [1] * 24], dtype=float)
    result = detect_local_structure(*_inputs(48, feature=feature),
                                    config=LocalStructureConfig(mad_multiplier=0))
    change = next(c for c in result['changes'] if c['beat_index'] == 24)
    assert change['primary_scale_beats'] == 2
    assert change['supporting_scales_beats'] == [2, 4, 8]
    assert change['support_start_s'] == 16
    assert change['support_end_s'] == change['available_at_s'] == 32
    assert {s['end_s'] for s in change['scale_support']} == {26, 28, 32}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('config', [LocalStructureConfig(contrast_scales=(2, 2)),
    LocalStructureConfig(dip_ratio=complex(1, 1)), LocalStructureConfig(dip_min_beats=13),
    LocalStructureConfig(peak_spacing_beats=True), LocalStructureConfig(threshold_floor=1.01)])
def test_invalid_custom_configs_rejected(config):
    with pytest.raises(ValueError):
        detect_local_structure(*_inputs(), config=config)


def test_empty_feature_bins_rejected():
    with pytest.raises(ValueError):
        detect_local_structure(*_inputs(12, feature=np.zeros((0, 12))))


def test_configured_dip_ratio_also_controls_affected_stem_evidence():
    energy = np.r_[np.ones(8), [.7, .7], np.ones(8)]
    result = detect_local_structure(*_inputs(len(energy), energy=energy),
                                    config=LocalStructureConfig(dip_ratio=.8))
    assert result['transitions'][0]['evidence']['affected_stems'] == ['stem']
