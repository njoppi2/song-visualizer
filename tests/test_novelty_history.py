import numpy as np

from songviz.story import _lag_history_mask, _lag_matrix_novelty, _novelty_curves_from_lag


def curves(features):
    times = np.arange(features.shape[1], dtype=float)
    lag = _lag_matrix_novelty(features, times, times)
    valid = _lag_history_mask(times, times)
    return lag, _novelty_curves_from_lag(lag, lag, history_mask=valid), valid


def test_no_history_is_unknown_not_opening_surprise():
    _, nov, valid = curves(np.ones((3, 40)))
    assert not valid[:, 0].any()
    assert valid[0, 1]
    for value in nov.values():
        np.testing.assert_allclose(value, 0, atol=1e-6)


def test_unavailable_long_lags_do_not_hide_real_early_change():
    features = np.eye(2)[:, [0, 0, 1, 1, 1, 1]]
    _, nov, _ = curves(features)
    for value in nov.values():
        assert value[2] == 1
        assert value[3] == 0


def test_lag_similarity_is_raw_cosine_not_per_row_scaled():
    features = np.array([[1, 0.8, 1], [0, 0.6, 0]], dtype=float)
    lag, nov, _ = curves(features)
    np.testing.assert_allclose(lag[0, 1:], [0.8, 0.8], atol=1e-6)
    assert np.isclose(nov['short'][1], 0.2)


def test_silence_does_not_create_novelty_and_entry_does():
    features = np.zeros((2, 40))
    features[0, 20:] = 1
    _, nov, _ = curves(features)
    for value in nov.values():
        np.testing.assert_allclose(value[:20], 0)
        assert value[20] == 1
        assert value[21] == 0


def test_history_uses_supplied_timestamps_not_frame_indices():
    mask = _lag_history_mask(np.array([0., 2., 5.]), np.array([0., 1., 2., 4., 5.]), 4)
    assert mask.tolist() == [[False, False, True, True, True],
                             [False, False, False, False, True],
                             [False] * 5, [False] * 5]
