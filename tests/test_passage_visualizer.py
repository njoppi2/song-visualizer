"""Contract checks for the opening visual experiment."""
import numpy as np

from experiments.passage_visualizer import PassageVisualizer


def _frame(visualizer, t):
    return np.frombuffer(visualizer.frame_rgb24(t), dtype=np.uint8)


def test_frames_are_deterministic_when_requested_out_of_order():
    events = {"hits": [{"t": 10.35, "component": "snare", "velocity": 0.9}]}
    visualizer = PassageVisualizer([10.0], events, start=10.0)
    expected = _frame(visualizer, .35)
    _frame(visualizer, .60)
    _frame(visualizer, .05)
    np.testing.assert_array_equal(expected, _frame(visualizer, .35))


def test_frame_is_exact_rgb24_at_requested_size_and_empty_input_is_calm():
    visualizer = PassageVisualizer([], {"hits": []}, width=73, height=41)
    data = visualizer.frame_rgb24(4.0)
    assert len(data) == 73 * 41 * 3
    assert np.frombuffer(data, dtype=np.uint8).reshape(41, 73, 3).shape == (41, 73, 3)
    assert data == visualizer.frame_rgb24(9.0)


def test_beat_does_not_invent_drum_strengths():
    visualizer = PassageVisualizer([101.0], {"hits": []}, start=100.0)
    signals = visualizer.signals_at(1.0)
    assert signals["beat"] == 1.0
    assert signals["kick"] == signals["snare"] == signals["hh"] == 0.0


def test_event_times_use_absolute_song_seconds_exactly():
    visualizer = PassageVisualizer([], {"hits": [{"t": 100.25, "component": "kick", "velocity": 1.0}]}, start=100.0)
    assert visualizer.signals_at(.249)["kick"] == 0.0
    assert visualizer.signals_at(.25)["kick"] == 1.0
    assert visualizer.signals_at(.45)["kick"] == 0.0


def test_components_are_independent_and_effects_never_precede_an_onset():
    visualizer = PassageVisualizer(
        [],
        {"hits": [
            {"t": 4.10, "component": "hh", "velocity": 1.0},
            {"t": 4.20, "component": "snare", "velocity": 0.7},
        ]},
        start=4.0,
    )
    before = visualizer.signals_at(.099)
    hat = visualizer.signals_at(.10)
    snare = visualizer.signals_at(.20)
    assert before == {"beat": 0.0, "kick": 0.0, "snare": 0.0, "hh": 0.0}
    assert hat["hh"] == 1.0 and hat["snare"] == hat["kick"] == 0.0
    assert snare["snare"] == 0.7 and snare["kick"] == 0.0
    assert _frame(visualizer, .099).tobytes() != _frame(visualizer, .10).tobytes()


def test_zero_velocity_event_is_not_a_hit_and_inputs_stay_unchanged():
    beats = [2.0]
    events = {"hits": [{"t": 2.0, "component": "snare", "velocity": 0.0}]}
    visualizer = PassageVisualizer(beats, events, start=2.0)
    assert visualizer.signals_at(0)["snare"] == 0.0
    assert beats == [2.0]
    assert events == {"hits": [{"t": 2.0, "component": "snare", "velocity": 0.0}]}
