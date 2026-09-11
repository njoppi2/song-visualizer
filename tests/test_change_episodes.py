import json

import numpy as np
import pytest

from songviz.change_episodes import ChangeEpisodeConfig, detect_change_episodes


def _inputs(n=32, *, stems=None):
    stems = {"stem": (np.ones((2, n)), np.ones(n))} if stems is None else stems
    return ({name: np.asarray(feature, dtype=float) for name, (feature, _) in stems.items()},
            {name: np.asarray(energy, dtype=float) for name, (_, energy) in stems.items()},
            np.arange(n + 1, dtype=float))


def _episodes_at(result, stem="stem", channel=None):
    return [episode for episode in result["episodes"]
            if episode["stem"] == stem and (channel is None or episode["channel"] == channel)]


def test_stationary_and_silence_have_no_false_events_and_json_safe_contract():
    for inputs in (_inputs(), _inputs(24, stems={"stem": (np.zeros((2, 24)), np.zeros(24))})):
        result = detect_change_episodes(*inputs)
        assert result["schema_version"] == 1 and result["kind"] == "songviz-change-episodes"
        assert not result["episodes"]
        json.dumps(result, allow_nan=False)


def test_level_and_spectral_steps_are_separate_channels_with_unknown_quiet_pattern():
    spectral = np.array([[1] * 16 + [0] * 16, [0] * 16 + [1] * 16])
    pattern = detect_change_episodes(*_inputs(32, stems={"stem": (spectral, np.ones(32))}))
    assert any(event["channel"] == "pattern" and event["peak_index"] == 16 for event in pattern["episodes"])
    level = detect_change_episodes(*_inputs(32, stems={"stem": (np.ones((2, 32)), [1] * 16 + [3] * 16)}))
    assert any(event["channel"] == "arrangement" and event["peak_index"] == 16 for event in level["episodes"])
    quiet = detect_change_episodes(*_inputs(32, stems={"stem": (np.ones((2, 32)), [0] * 16 + [1] * 16)}))
    curve = next(c for c in quiet["curves"] if c["channel"] == "pattern" and c["scale_beats"] == 2)
    assert curve["values"][16] is None


def test_hysteresis_retains_gradual_response_and_separate_dip_recovery():
    # The pattern curve stays above release across a gradual progression, while
    # two later level responses remain distinct episodes.
    feature = np.array([[1] * 10 + [.8, .6, .4, .2, 0, 0, .2, .4, .6, .8] + [1] * 12,
                        [0] * 10 + [.2, .4, .6, .8, 1, 1, .8, .6, .4, .2] + [0] * 12])
    energy = [1] * 18 + [.2] * 5 + [1] * 9
    result = detect_change_episodes(*_inputs(32, stems={"stem": (feature, energy)}),
                                    config=ChangeEpisodeConfig(scales=(2,), mad_multiplier=0))
    arrangement = _episodes_at(result, channel="arrangement")
    assert len(arrangement) >= 2
    assert {(event["start_s"], event["end_s"]) for event in arrangement} >= {(17, 20), (22, 25)}
    assert _episodes_at(result, channel="pattern")  # gradual spectral response remains inspectable


def test_nested_and_overlapping_stem_scale_episodes_are_not_suppressed():
    a = np.array([[1] * 16 + [0] * 16, [0] * 16 + [1] * 16])
    b = np.array([[1] * 17 + [0] * 15, [0] * 17 + [1] * 15])
    result = detect_change_episodes(*_inputs(32, stems={"a": (a, np.ones(32)), "b": (b, np.ones(32))}),
                                    config=ChangeEpisodeConfig(scales=(2, 4), mad_multiplier=0))
    events = [event for event in result["episodes"] if event["channel"] == "pattern"]
    assert {(event["stem"], event["scale_beats"]) for event in events} >= {
        ("a", 2), ("a", 4), ("b", 2), ("b", 4)}
    assert all(event["contributing_stems"] == [event["stem"]] for event in events)


def test_null_gap_splits_run_edges_are_censored_and_peak_plateau_is_earliest_with_support_availability():
    # Entrance gives null pattern evidence on the left and a thresholded
    # arrangement plateau. Its first maximum must be the deterministic peak.
    feature = np.ones((2, 24))
    energy = [0] * 6 + [1] * 18
    result = detect_change_episodes(*_inputs(24, stems={"stem": (feature, energy)}),
                                    config=ChangeEpisodeConfig(scales=(2,), mad_multiplier=0))
    event = next(event for event in _episodes_at(result, channel="arrangement") if event["peak_value"] == 1)
    assert event["peak_index"] == min(event["raw_indices"])
    assert event["left_censored"]
    assert event["left_censor_reason"] == "missing_curve_value"
    assert event["support_start_s"] == event["response_support_start_s"] == event["start_s"] - 2
    assert event["response_support_end_s"] == event["end_s"] + 1
    assert event["available_at_s"] == event["support_end_s"] == event["end_s"] + 2
    assert event["physical_onset_s"] is event["physical_settled_s"] is None


def test_fixed_sharp_step_response_blur_and_reversal_keep_numeric_rms_directions():
    n = 64
    result = detect_change_episodes(*_inputs(n, stems={"stem": (np.ones((2, n)), [1] * 32 + [4] * 32)}))
    responses = {(event["scale_beats"], event["start_s"], event["end_s"])
                 for event in _episodes_at(result, channel="arrangement")}
    assert {(2, 31, 34), (4, 29, 36), (8, 25, 39)} <= responses
    curve = next(curve for curve in result["curves"]
                 if curve["scale_beats"] == 2 and curve["channel"] == "arrangement")
    assert curve["raw_rms_directions"][32]["signed_rms_difference"] > 0
    sharp = next(event for event in _episodes_at(result, channel="arrangement")
                 if event["scale_beats"] == 2 and event["start_s"] == 31)
    # The above-release response alone needs beats [29, 35], but observing the
    # known release sample at 34 needs right context through 36.
    assert (sharp["response_support_start_s"], sharp["response_support_end_s"],
            sharp["support_start_s"], sharp["available_at_s"]) == (29, 35, 28, 36)

    reversal = detect_change_episodes(*_inputs(n, stems={"stem": (np.ones((2, n)),
                                                           [1] * 24 + [4] * 8 + [1] * 32)}),
                                       config=ChangeEpisodeConfig(scales=(4,), mad_multiplier=0))
    directions = next(curve["raw_rms_directions"] for curve in reversal["curves"]
                      if curve["channel"] == "arrangement")
    assert directions[24]["signed_rms_difference"] > 0
    assert directions[32]["signed_rms_difference"] < 0


def test_context_descriptors_scale_as_specified_and_do_not_assign_voice_function():
    n = 24
    spectral = np.array([[1] * 12 + [0] * 12, [0] * 12 + [1] * 12])
    stems = {"voice": (spectral, [2] * n), "other": (np.ones((2, n)), [1] * n)}
    result = detect_change_episodes(*_inputs(n, stems=stems), config=ChangeEpisodeConfig(scales=(2,), mad_multiplier=0))
    event = next(event for event in _episodes_at(result, "voice", "pattern") if event["peak_index"] == 12)
    context = event["peak_context"]["stems"]
    assert context["voice"]["left"]["active_fraction"] == context["voice"]["right"]["active_fraction"] == 1
    assert context["voice"]["left"]["adjacent_spectral_change"] == 0
    assert event["vocal_function"] is None
    for side in ("left", "right"):
        shares = sum(context[name][side]["rms_power_share"] for name in context)
        assert shares == pytest.approx(1)
    doubled = detect_change_episodes(*_inputs(n, stems={"voice": (spectral, [4] * n),
                                                          "other": (np.ones((2, n)), [2] * n)}),
                                     config=ChangeEpisodeConfig(scales=(2,), mad_multiplier=0))
    doubled_event = next(event for event in _episodes_at(doubled, "voice", "pattern") if event["peak_index"] == 12)
    assert doubled_event["peak_context"]["stems"]["voice"]["changes"]["signed_rms_difference"] == 0
    assert doubled_event["peak_value"] == event["peak_value"]


def test_inaudible_positive_features_have_unknown_spectral_concentration():
    n = 24
    feature = np.array([[1] * 12 + [0] * 12, [0] * 12 + [1] * 12])
    result = detect_change_episodes(*_inputs(n, stems={"loud": (feature, [1] * n),
                                                        "quiet": (np.ones((2, n)), [0] * n)}),
                                    config=ChangeEpisodeConfig(scales=(2,), mad_multiplier=0))
    event = next(event for event in _episodes_at(result, "loud", "pattern") if event["peak_index"] == 12)
    quiet = event["peak_context"]["stems"]["quiet"]
    assert quiet["left"]["active_fraction"] == quiet["right"]["active_fraction"] == 0
    assert quiet["left"]["spectral_concentration"] is quiet["right"]["spectral_concentration"] is None


@pytest.mark.parametrize("config", [
    ChangeEpisodeConfig(scales=()), ChangeEpisodeConfig(scales=(2, 2)),
    ChangeEpisodeConfig(scales=(True,)), ChangeEpisodeConfig(scales=(2.0,)),
    ChangeEpisodeConfig(threshold_floor=float("nan")), ChangeEpisodeConfig(mad_multiplier=-1),
    ChangeEpisodeConfig(release_ratio=1.1),
])
def test_invalid_config_and_nonfinite_inputs_are_rejected_without_mutation(config):
    features, energy, beats = _inputs()
    before = (features["stem"].copy(), energy["stem"].copy(), beats.copy())
    with pytest.raises(ValueError):
        detect_change_episodes(features, energy, beats, config=config)
    assert np.array_equal(features["stem"], before[0])
    with pytest.raises(ValueError):
        detect_change_episodes({"stem": np.array([[np.nan] * 8])}, {"stem": np.ones(8)}, np.arange(9))


def test_inputs_unchanged_and_ids_follow_stem_scale_channel_time_order():
    feature = np.array([[1] * 12 + [0] * 12, [0] * 12 + [1] * 12])
    features, energy, beats = _inputs(24, stems={"z": (feature, np.ones(24)), "a": (feature, np.ones(24))})
    before = ({name: value.copy() for name, value in features.items()},
              {name: value.copy() for name, value in energy.items()}, beats.copy())
    result = detect_change_episodes(features, energy, beats, config=ChangeEpisodeConfig(scales=(2, 4), mad_multiplier=0))
    ordering = [(event["stem"], event["scale_beats"], event["channel"], event["start_s"])
                for event in result["episodes"]]
    assert ordering == sorted(ordering)
    assert [event["id"] for event in result["episodes"]] == [f"episode-{i:04d}" for i in range(1, len(ordering) + 1)]
    assert all(np.array_equal(features[name], before[0][name]) for name in features)
    assert all(np.array_equal(energy[name], before[1][name]) for name in energy)
    assert np.array_equal(beats, before[2])
