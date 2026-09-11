"""Regression tests for SSM/energy boundary consensus."""

import numpy as np

from songviz.story import _score_and_filter_boundaries


def _score(
    ssm: list[float],
    energy: list[float],
    *,
    novelty: np.ndarray | None = None,
    beat_times: np.ndarray | None = None,
) -> list[float]:
    return _score_and_filter_boundaries(
        ssm,
        energy,
        novelty=novelty,
        beat_times=beat_times,
        duration_s=100.0,
    )


def test_agreed_pair_emits_only_ssm_timestamp() -> None:
    """An agreed structural change must not become two downstream boundaries."""
    assert _score([76.93], [79.09]) == [76.93]


def test_separated_agreed_changes_remain_distinct() -> None:
    assert _score([20.0, 60.0], [21.0, 61.0]) == [20.0, 60.0]


def test_duplicate_and_endpoint_candidates_are_sanitized() -> None:
    assert _score([0.0, 10.0, 10.0, 100.0], [0.0, 11.0, 11.0, 100.0]) == [10.0]


def test_one_energy_peak_does_not_suppress_a_strong_second_ssm_peak() -> None:
    """Only one SSM peak can pair; the other remains subject to solo policy."""
    beat_times = np.array([0.0, 10.0, 11.0, 12.0, 13.0, 18.0, 19.0, 20.0, 100.0])
    novelty = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0])

    # Both SSM peaks are equally close to 14.0.  The lower timestamp wins the
    # deterministic tie, while the independent strong peak remains a solo.
    assert _score([10.0, 18.0], [14.0], novelty=novelty, beat_times=beat_times) == [10.0, 18.0]


def test_nontransitive_greedy_pairing_keeps_only_closest_weak_candidate() -> None:
    beat_times = np.array([0.0, 10.0, 11.0, 12.0, 13.0, 17.0, 18.0, 19.0, 100.0])
    novelty = np.zeros_like(beat_times)

    # 17.0 is closer to 14.0 than 10.0 is.  The unmatched weak SSM candidate
    # must not be retained merely because it is also inside the agreement window.
    assert _score([10.0, 17.0], [14.0], novelty=novelty, beat_times=beat_times) == [17.0]


def test_ssm_solo_is_retained_when_novelty_information_is_unavailable() -> None:
    assert _score([35.0], [], novelty=None, beat_times=None) == [35.0]
