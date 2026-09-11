"""Protect the controlled comparison and its separation of pulse and hits."""
import numpy as np

from experiments.build_rhythm_review import PulseVisualizer


def frame(viz, time):
    return np.frombuffer(viz.frame_rgb24(time), dtype=np.uint8).reshape(540, 960, 3)


def test_changing_grid_leaves_percussion_render_identical():
    events = {"hits": [{"t": 130.25, "component": "snare", "velocity": 1.0}]}
    cached = PulseVisualizer([130.1, 130.75], events, 130, 148, "test")
    regular = PulseVisualizer([130.25, 130.68], events, 130, 148, "test")
    old, new = frame(cached, .25), frame(regular, .25)
    np.testing.assert_array_equal(old[170:], new[170:])
    assert not np.array_equal(old[75:150], new[75:150])


def test_off_grid_snare_uses_absolute_clock_and_does_not_flash_kick():
    events = {"hits": [{"t": 130.25, "component": "snare", "velocity": 1.0}]}
    viz = PulseVisualizer([130.0, 130.43], events, 130, 148, "test")
    before, at = frame(viz, .24), frame(viz, .25)
    np.testing.assert_array_equal(before[180:230, 25:90], at[180:230, 25:90])
    assert not np.array_equal(before[275:325, 25:90], at[275:325, 25:90])


def test_pulse_can_continue_without_any_drum_hits():
    viz = PulseVisualizer([158.25], {"hits": []}, 158, 178, "test")
    before, at = frame(viz, .24), frame(viz, .25)
    np.testing.assert_array_equal(before[170:450, 25:90], at[170:450, 25:90])
    assert not np.array_equal(before[80:145, 25:90], at[80:145, 25:90])
