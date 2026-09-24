import json

import numpy as np
import pytest

from songviz.change_episodes import _peak_context
from songviz.role_context import RoleContextConfig, compute_role_context


def _inputs(n=12, *, stems=None, beats=None):
    stems = {"stem": (np.ones((2, n)), np.ones(n))} if stems is None else stems
    return (
        {name: np.asarray(feature, dtype=float) for name, (feature, _) in stems.items()},
        {name: np.asarray(rms, dtype=float) for name, (_, rms) in stems.items()},
        np.arange(n + 1, dtype=float) if beats is None else np.asarray(beats, dtype=float),
    )


def _sample(result, h, k):
    return next(curve for curve in result["curves"] if curve["scale_beats"] == h)["samples"][k]


def test_constant_and_silent_inputs_have_complete_context_but_unknown_silent_spectra():
    constant = compute_role_context(*_inputs(8), config=RoleContextConfig(scales=(2,)))
    sample = _sample(constant, 2, 4)
    stem = sample["stems"]["stem"]
    assert stem["changes"] == {
        "signed_rms_difference": 0.0,
        "signed_active_fraction_difference": 0.0,
        "signed_rms_power_share_difference": 0.0,
        "signed_spectral_concentration_difference": 0.0,
        "signed_adjacent_spectral_change_difference": 0.0,
    }
    assert _sample(constant, 2, 0) is None

    silent = compute_role_context(*_inputs(8, stems={"quiet": (np.ones((2, 8)), np.zeros(8))}),
                                  config=RoleContextConfig(scales=(2,)))
    quiet = _sample(silent, 2, 4)["stems"]["quiet"]
    assert quiet["left"]["active_fraction"] == quiet["right"]["active_fraction"] == 0
    assert quiet["left"]["spectral_concentration"] is quiet["right"]["spectral_concentration"] is None
    assert quiet["left"]["adjacent_spectral_change"] is quiet["right"]["adjacent_spectral_change"] is None
    assert quiet["left"]["rms_power_share"] is quiet["right"]["rms_power_share"] is None
    assert quiet["changes"]["signed_rms_power_share_difference"] is None
    assert quiet["changes"]["signed_spectral_concentration_difference"] is None
    assert quiet["changes"]["signed_adjacent_spectral_change_difference"] is None
    json.dumps(silent, allow_nan=False)


def test_hand_computed_power_shares_and_signed_descriptor_trends():
    features, energy, beats = _inputs(6, stems={
        "a": (np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]]), [1, 1, 3, 3, 3, 3]),
        "b": (np.ones((2, 6)), [2, 2, 1, 1, 1, 1]),
    })
    result = compute_role_context(features, energy, beats, config=RoleContextConfig(scales=(2,)))
    a = _sample(result, 2, 2)["stems"]["a"]
    b = _sample(result, 2, 2)["stems"]["b"]
    assert (a["left"]["mean_rms"], a["right"]["mean_rms"]) == (1, 3)
    assert (a["left"]["rms_power_share"], a["right"]["rms_power_share"]) == pytest.approx((.2, .9))
    assert (b["left"]["rms_power_share"], b["right"]["rms_power_share"]) == pytest.approx((.8, .1))
    assert a["changes"] == pytest.approx({
        "signed_rms_difference": 2, "signed_active_fraction_difference": 0,
        "signed_rms_power_share_difference": .7, "signed_spectral_concentration_difference": -1,
        "signed_adjacent_spectral_change_difference": 1,
    })
    assert b["changes"]["signed_rms_power_share_difference"] == pytest.approx(-.7)


def test_irregular_grid_support_and_edge_nulls_are_explicit():
    features, energy, beats = _inputs(8, beats=[0, .2, .7, 1.8, 3, 3.3, 4.5, 6, 9])
    result = compute_role_context(features, energy, beats, config=RoleContextConfig(scales=(2, 4)))
    assert _sample(result, 2, 0) is None and _sample(result, 2, 1) is None
    assert _sample(result, 4, 3) is None and _sample(result, 4, 4) is not None
    sample = _sample(result, 2, 4)
    assert sample["anchor_s"] == 3
    assert sample["support_start_s"] == .7 and sample["support_end_s"] == 4.5
    assert sample["available_at_s"] == 4.5
    assert sample["left_support"] == {"start_s": .7, "end_s": 3}
    assert sample["right_support"] == {"start_s": 3, "end_s": 4.5}


def test_proportional_gain_invariance_above_absolute_floor_and_absolute_floor_exception():
    stems = {
        "a": (np.array([[1] * 6, [0] * 6]), [1, 1, 2, 2, 2, 2]),
        "b": (np.ones((2, 6)), [2, 2, 1, 1, 1, 1]),
    }
    features, energy, beats = _inputs(6, stems=stems)
    original = compute_role_context(features, energy, beats, config=RoleContextConfig(scales=(2,)))
    scaled = compute_role_context(features, {name: values * 7 for name, values in energy.items()}, beats,
                                  config=RoleContextConfig(scales=(2,)))
    for k in range(2, 5):
        for name in ("a", "b"):
            original_stem = _sample(original, 2, k)["stems"][name]
            scaled_stem = _sample(scaled, 2, k)["stems"][name]
            for side in ("left", "right"):
                assert scaled_stem[side]["mean_rms"] == pytest.approx(original_stem[side]["mean_rms"] * 7)
                for key in ("active_fraction", "rms_power_share", "spectral_concentration", "adjacent_spectral_change"):
                    assert scaled_stem[side][key] == pytest.approx(original_stem[side][key])
            assert scaled_stem["changes"]["signed_rms_difference"] == pytest.approx(
                original_stem["changes"]["signed_rms_difference"] * 7)
            for key in ("signed_active_fraction_difference", "signed_rms_power_share_difference",
                        "signed_spectral_concentration_difference", "signed_adjacent_spectral_change_difference"):
                assert scaled_stem["changes"][key] == pytest.approx(original_stem["changes"][key])

    tiny = compute_role_context(*_inputs(6, stems={"tiny": (np.ones((2, 6)), [1e-9] * 6)}),
                                config=RoleContextConfig(scales=(2,)))
    amplified = compute_role_context(*_inputs(6, stems={"tiny": (np.ones((2, 6)), [1e-5] * 6)}),
                                     config=RoleContextConfig(scales=(2,)))
    tiny_stem = _sample(tiny, 2, 2)["stems"]["tiny"]
    amplified_stem = _sample(amplified, 2, 2)["stems"]["tiny"]
    assert tiny["audibility_floors"]["tiny"] == 1e-8
    assert tiny_stem["left"]["active_fraction"] == 0
    assert tiny_stem["left"]["spectral_concentration"] is None
    assert amplified["audibility_floors"]["tiny"] == pytest.approx(2e-7)
    assert amplified_stem["left"]["active_fraction"] == 1
    assert amplified_stem["left"]["spectral_concentration"] == 0


def test_absent_or_inaudible_spectral_evidence_leaves_differences_unknown():
    result = compute_role_context(*_inputs(6, stems={
        "entering": (np.ones((2, 6)), [0, 0, 1, 1, 1, 1]),
    }), config=RoleContextConfig(scales=(2,)))
    stem = _sample(result, 2, 2)["stems"]["entering"]
    assert stem["left"]["spectral_concentration"] is None
    assert stem["right"]["spectral_concentration"] == 0
    assert stem["changes"]["signed_spectral_concentration_difference"] is None
    assert stem["changes"]["signed_adjacent_spectral_change_difference"] is None


def test_candidate_selection_and_threshold_paths_are_never_called(monkeypatch):
    import songviz.change_episodes as episodes
    import songviz.local_structure as local

    def fail(*args, **kwargs):
        raise AssertionError("candidate-selection path must not be used")

    monkeypatch.setattr(episodes, "detect_change_episodes", fail)
    monkeypatch.setattr(episodes, "_threshold", fail)
    monkeypatch.setattr(episodes, "_episodes", fail)
    monkeypatch.setattr(local, "_threshold", fail)
    result = compute_role_context(*_inputs(8), config=RoleContextConfig(scales=(2,)))
    assert _sample(result, 2, 4) is not None


def test_inputs_unchanged_validation_and_peak_descriptor_parity():
    features, energy, beats = _inputs(8, stems={
        "z": (np.array([[1] * 4 + [0] * 4, [0] * 4 + [1] * 4]), [2] * 8),
        "a": (np.ones((2, 8)), [1] * 8),
    })
    before = ({name: value.copy() for name, value in features.items()},
              {name: value.copy() for name, value in energy.items()}, beats.copy())
    result = compute_role_context(features, energy, beats, config=RoleContextConfig(scales=(2,)))
    expected = _peak_context(features, energy, 4, 2)
    sample = _sample(result, 2, 4)
    assert result["stem_names"] == ["a", "z"]
    for name in result["stem_names"]:
        assert sample["stems"][name]["left"] == expected["stems"][name]["left"]
        assert sample["stems"][name]["right"] == expected["stems"][name]["right"]
        assert sample["stems"][name]["changes"]["signed_rms_difference"] == expected["stems"][name]["changes"]["signed_rms_difference"]
        assert sample["stems"][name]["musical_role"] is sample["stems"][name]["vocal_function"] is None
    assert all(np.array_equal(features[name], before[0][name]) for name in features)
    assert all(np.array_equal(energy[name], before[1][name]) for name in energy)
    assert np.array_equal(beats, before[2])

    with pytest.raises(ValueError):
        compute_role_context(features, energy, beats, config=RoleContextConfig(scales=()))
    with pytest.raises(ValueError):
        compute_role_context(features, energy, beats, config=RoleContextConfig(scales=(2, 2)))
    with pytest.raises(ValueError):
        compute_role_context(features, energy, beats, config=RoleContextConfig(scales=(True,)))
    with pytest.raises(ValueError):
        compute_role_context({"stem": np.array([[np.nan] * 8])}, {"stem": np.ones(8)}, np.arange(9))
