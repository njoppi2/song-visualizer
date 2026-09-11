import json

import numpy as np
import pytest

from songviz.local_structure import LocalStructureConfig, detect_local_structure
from songviz.local_structure_variants import LocalStructureVariantConfig, detect_local_structure_variant


def _inputs(n=32, *, energy=None, feature=None):
    energy = np.ones(n) if energy is None else np.asarray(energy, dtype=float)
    feature = np.ones((2, n)) if feature is None else np.asarray(feature, dtype=float)
    return {"stem": feature}, {"stem": energy}, np.arange(n + 1, dtype=float)


def test_control_is_exact_frozen_detector_passthrough_including_custom_base_config():
    inputs = _inputs(32, energy=[1] * 16 + [2] * 16)
    base = LocalStructureConfig(contrast_scales=(4,), mad_multiplier=0)
    expected = detect_local_structure(*inputs, config=base)
    actual = detect_local_structure_variant(*inputs, variant="control",
                                            config=LocalStructureVariantConfig(base_detector_config=base))
    assert actual == expected


@pytest.mark.parametrize("variant", ["separate_channels", "sustained_activity", "combined"])
def test_variants_retain_frozen_curves_and_dips_with_json_safe_nulls(variant):
    energy = [1] * 8 + [.1, .1] + [1] * 22
    inputs = _inputs(len(energy), energy=energy)
    control = detect_local_structure(*inputs)
    result = detect_local_structure_variant(*inputs, variant=variant)
    assert result["curves"] == control["curves"]
    assert result["transitions"] == control["transitions"]
    json.dumps(result, allow_nan=False)


def test_stationary_and_silence_are_null_or_zero_activity_with_no_changes():
    for inputs in (_inputs(), _inputs(24, energy=np.zeros(24))):
        result = detect_local_structure_variant(*inputs, variant="combined")
        assert not result["changes"]
        for record in result["activity_curves"]:
            if np.max(inputs[1]["stem"]) == 0:
                assert all(value is None for value in record["activity_change"])
            else:
                assert set(value for value in record["activity_change"] if value is not None) == {0.0}


def test_pattern_and_level_steps_are_independent_channel_proposals():
    pattern = np.array([[1] * 16 + [0] * 16, [0] * 16 + [1] * 16], dtype=float)
    pattern_result = detect_local_structure_variant(*_inputs(32, feature=pattern), variant="separate_channels")
    assert any(c["channel"] == "pattern" and c["beat_index"] == 16 for c in pattern_result["changes"])
    level_result = detect_local_structure_variant(*_inputs(32, energy=[1] * 16 + [2] * 16),
                                                  variant="separate_channels")
    assert any(c["channel"] == "arrangement" and c["beat_index"] == 16 for c in level_result["changes"])


def test_independently_computed_channel_records_match_frozen_raw_curves_exactly():
    feature = np.array([[1] * 16 + [0] * 16, [0] * 16 + [1] * 16], dtype=float)
    inputs = _inputs(32, feature=feature, energy=[1] * 12 + [2] * 20)
    control = detect_local_structure(*inputs)
    separate = detect_local_structure_variant(*inputs, variant="separate_channels")
    for frozen, independent in zip(control["curves"], separate["channel_curves"], strict=True):
        assert independent["scale_beats"] == frozen["scale_beats"]
        assert independent["pattern_change"] == frozen["pattern_change"]
        assert independent["arrangement_change"] == frozen["arrangement_change"]


def test_sustained_activity_accepts_step_and_return_but_rejects_blip_and_alternation():
    entrance_exit = [1] * 12 + [3] * 12 + [1] * 12
    result = detect_local_structure_variant(*_inputs(len(entrance_exit), energy=entrance_exit),
                                            variant="sustained_activity")
    proposed = [support for change in result["changes"] for support in change.get("variant_support", [])
                if support["source"] == "sustained_activity"]
    # Both configured windows retain their own leftmost plateaus: h=4 is one
    # beat early and h=8 is two beats early (one quarter of each window).
    assert {support["time_s"] for support in proposed} == {10.0, 11.0, 22.0, 23.0}
    assert {support["stem_evidence"]["stem"]["direction"] for support in proposed} == {"increase", "decrease"}
    for sequence in ([1] * 12 + [3] + [1] * 19, [1, 3] * 16):
        rejected = detect_local_structure_variant(*_inputs(len(sequence), energy=sequence), variant="sustained_activity")
        assert not [support for change in rejected["changes"] for support in change.get("variant_support", [])
                    if support["source"] == "sustained_activity"]


def test_activity_evidence_reports_primary_scale_medians_and_support_fractions():
    result = detect_local_structure_variant(*_inputs(32, energy=[1] * 16 + [4] * 16),
                                            variant="sustained_activity")
    activity = next(s for c in result["changes"] for s in c.get("variant_support", [])
                    if s["source"] == "sustained_activity" and s["scale_beats"] == 4)
    evidence = activity["stem_evidence"]["stem"]
    assert activity["scale_beats"] == 4
    assert evidence["left_median"] == 1 and evidence["right_median"] == 4
    assert evidence["left_support_fraction"] == 1
    assert evidence["right_support_fraction"] == .75
    assert evidence["direction"] == "increase"
    assert evidence["audibility_state"] == "continuous"


def test_audibility_entrance_is_reserved_for_an_actual_floor_crossing():
    result = detect_local_structure_variant(*_inputs(32, energy=[.01] * 16 + [1] * 16),
                                            variant="sustained_activity")
    evidence = result["activity_curves"][0]["stem_evidence"]["stem"][15]
    assert evidence["direction"] == "increase"
    assert evidence["audibility_state"] == "entrance"


def test_median_plateau_selects_leftmost_pre_step_window_and_retains_raw_support():
    # With h=4, a clean step at 16 has equally strong qualified median windows
    # at 15, 16 and 17. Plateau discipline picks 15: one beat (h/4) early.
    config = LocalStructureVariantConfig(base_detector_config=LocalStructureConfig(contrast_scales=(2,), mad_multiplier=99),
                                         activity_scales=(4,))
    result = detect_local_structure_variant(*_inputs(32, energy=[1] * 16 + [3] * 16),
                                            variant="sustained_activity", config=config)
    curve = result["activity_curves"][0]["activity_change"]
    assert curve[15:18] == [pytest.approx(2 / 3)] * 3
    activity_support = [s for c in result["changes"] for s in c.get("variant_support", [])
                        if s["source"] == "sustained_activity"]
    support = next(s for s in activity_support if s["time_s"] == 15)
    assert (support["start_s"], support["end_s"], support["available_at_s"]) == (11, 19, 19)


def test_nearby_activity_merges_into_control_timestamp_and_unions_support_availability():
    # Spectral change at beat 16 gives a frozen control timestamp; the activity
    # candidate is deliberately placed one beat later and must augment it.
    feature = np.array([[1] * 16 + [0] * 17, [0] * 16 + [1] * 17], dtype=float)
    energy = [1] * 17 + [4] * 16
    base = LocalStructureConfig(contrast_scales=(2,), mad_multiplier=0)
    config = LocalStructureVariantConfig(base_detector_config=base, activity_scales=(4,))
    result = detect_local_structure_variant(*_inputs(33, feature=feature, energy=energy),
                                            variant="sustained_activity", config=config)
    control = detect_local_structure(*_inputs(33, feature=feature, energy=energy), config=base)
    merged = next(c for c in result["changes"] if c["beat_index"] == 16)
    original = next(c for c in control["changes"] if c["beat_index"] == 16)
    assert merged["time_s"] == original["time_s"] and merged["id"] == original["id"]
    assert {item["source"] for item in merged["variant_support"]} == {"control_contrast", "sustained_activity"}
    assert merged["available_at_s"] == merged["support_end_s"] == 20


def test_combined_prefers_separate_contrast_timestamp_and_preserves_all_sources():
    feature = np.array([[1] * 16 + [0] * 17, [0] * 16 + [1] * 17], dtype=float)
    energy = [1] * 17 + [4] * 16
    config = LocalStructureVariantConfig(base_detector_config=LocalStructureConfig(contrast_scales=(2,), mad_multiplier=0),
                                         activity_scales=(4,))
    result = detect_local_structure_variant(*_inputs(33, feature=feature, energy=energy), variant="combined", config=config)
    change = next(c for c in result["changes"] if c["beat_index"] == 16)
    assert change["source"] == "separate_channels"
    assert {item["source"] for item in change["variant_support"]} == {"separate_channels", "sustained_activity"}


def test_deterministic_plateau_ties_and_inputs_are_unmodified():
    feature = np.array([[1] * 12 + [0] * 2 + [1] * 18, [0] * 12 + [1] * 2 + [0] * 18], dtype=float)
    f, e, bt = _inputs(32, feature=feature)
    before = (f["stem"].copy(), e["stem"].copy(), bt.copy())
    config = LocalStructureVariantConfig(base_detector_config=LocalStructureConfig(contrast_scales=(2,), mad_multiplier=0))
    first = detect_local_structure_variant(f, e, bt, variant="separate_channels", config=config)
    second = detect_local_structure_variant(f, e, bt, variant="separate_channels", config=config)
    assert first == second
    assert np.array_equal(f["stem"], before[0]) and np.array_equal(e["stem"], before[1]) and np.array_equal(bt, before[2])


@pytest.mark.parametrize("config", [
    LocalStructureVariantConfig(pattern_threshold_floor=float("nan")),
    LocalStructureVariantConfig(arrangement_threshold_floor=1.1),
    LocalStructureVariantConfig(mad_multiplier=-.1),
    LocalStructureVariantConfig(activity_scales=(4, 4)),
    LocalStructureVariantConfig(activity_scales=(True,)),
    LocalStructureVariantConfig(activity_scales=(4.0,)),
    LocalStructureVariantConfig(activity_scales=()),
    LocalStructureVariantConfig(activity_min_change=0),
    LocalStructureVariantConfig(activity_min_change=-.1),
    LocalStructureVariantConfig(activity_support_fraction=0),
    LocalStructureVariantConfig(peak_spacing_beats=True),
    LocalStructureVariantConfig(peak_spacing_beats=2.0),
    LocalStructureVariantConfig(base_detector_config=LocalStructureConfig(contrast_scales=(2, 2))),
])
def test_invalid_variant_configs_rejected(config):
    with pytest.raises(ValueError):
        detect_local_structure_variant(*_inputs(), variant="combined", config=config)


def test_invalid_variant_name_is_rejected():
    with pytest.raises(ValueError):
        detect_local_structure_variant(*_inputs(), variant="not-a-variant")
