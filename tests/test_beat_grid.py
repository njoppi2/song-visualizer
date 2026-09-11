import numpy as np

from songviz.beat_grid import fit_regular_pulse, prominent_attacks


def test_regular_fit_handles_missing_and_extra_attacks_without_chasing_them():
    expected = 0.23 + np.arange(150) * .86
    missing = np.delete(expected, [14, 22, 51, 83])
    observations = np.sort(np.r_[missing, [18.07, 38.51, 57.22]])
    result = fit_regular_pulse(observations, duration_s=130, fit_start_s=10, fit_end_s=90)
    assert result['status'] == 'candidate'
    assert abs(result['tempo_bpm'] - 120/.86) < .01
    beats = np.array(result['beat_times_s'])
    assert np.max(abs(np.diff(beats) - .43)) < 1e-9
    assert result['outside_fit_median_residual_ms'] < 1


def test_tempo_drift_is_rejected_instead_of_forcing_constant_grid():
    intervals = np.linspace(.7, 1.1, 140)
    anchors = np.cumsum(intervals)
    result = fit_regular_pulse(anchors, duration_s=130, fit_start_s=0, fit_end_s=125)
    assert result['status'] != 'candidate'
    assert result['beat_times_s'] == []


def test_reference_attack_time_uses_actual_sample_rate_not_rounded_hop_duration():
    sr = 44100
    y = np.zeros(sr * 24)
    for time in [1., 12., 23.]:
        n = round(.04 * sr)
        y[round(time*sr):round(time*sr)+n] = np.exp(-np.arange(n) / (sr*.012))
    attacks = prominent_attacks(y, sr)
    np.testing.assert_allclose(attacks, [1.,12.,23.], atol=.004)


def test_silence_or_too_few_anchors_does_not_fabricate_grid():
    assert prominent_attacks(np.zeros(44100), 44100).size == 0
    result = fit_regular_pulse(np.array([1., 2.]), duration_s=20, fit_start_s=0, fit_end_s=15)
    assert result['status'] == 'insufficient_evidence'
