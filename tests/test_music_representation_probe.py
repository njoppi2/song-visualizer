import numpy as np
import pytest

from experiments.probe_music_representation import chunk_geometry, nominal_frame_times_s, source_sample_bounds


def test_source_sample_bounds_requires_exact_clock() -> None:
    assert source_sample_bounds(16.0, 2.0, 44_100) == (705_600, 793_800)


def test_nominal_muq_frame_clock_is_25hz_from_chunk_start() -> None:
    assert nominal_frame_times_s(16.0, 3) == [16.0, 16.04, 16.08]


@pytest.mark.parametrize("offset", [0., 2.5])
def test_shifted_geometry_preserves_half_phase_frames_and_full_edge_coverage(offset):
    chunks = chunk_geometry(9753744, 44100, offset)
    times = [c['start_s'] + j / 25 for c in chunks for j in c['retained_frame_indices']]
    assert times[0] == 0
    assert np.all(np.diff(times) > 0)
    assert np.max(np.diff(times)) <= .060001
    assert 9753744 / 44100 - times[-1] <= .040001
    assert len(times) > 5500
    for c in chunks[3:-3]:
        assert c['retained_frame_count'] == 125
    for i, c in enumerate(chunks):
        for j in c['retained_frame_indices']:
            t = c['start_s'] + j / 25
            eligible = [(abs(t - other['center_s']), k) for k, other in enumerate(chunks)
                        if other['support_start_s'] <= t < other['support_end_s']]
            assert min(eligible)[1] == i


def test_geometry_rejects_unregistered_shift_and_negative_source_time():
    with pytest.raises(ValueError):
        chunk_geometry(44100, 44100, 1.)
    with pytest.raises(ValueError):
        source_sample_bounds(-1., 2., 44100)
