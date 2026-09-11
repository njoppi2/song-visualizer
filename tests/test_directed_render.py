from PIL import Image

from songviz.directed_render import DirectedVisualizer


def _segment(start=130, end=140, *, focus="snare", motif="verse-a", palette="warm", transition=0, layers=None):
    empty = {name: {"visible": False, "treatment": "ticks", "gain": 0.0} for name in DirectedVisualizer.LAYERS}
    empty.update(layers or {})
    return {"start_s": start, "end_s": end, "focus": focus, "motif": motif, "palette": palette,
            "transition_s": transition, "layers": empty, "reason": "test", "evidence_ids": []}


def _plan(*segments):
    return {"schema_version": 1, "start_s": 130, "end_s": 178, "segments": list(segments)}


def _signals():
    return {"beat_times_s": [130.0, 132.0],
            "hits": [{"t": 130.0, "component": "snare", "velocity": 1.0}, {"t": 132.0, "component": "kick", "velocity": 1.0}],
            "energy": {"times_s": [129.0, 131.0], "bass": [.2, .9], "vocals": [.1, .6], "other": [.3, .4]}}


def test_rgb24_size_and_determinism_out_of_order():
    viz = DirectedVisualizer(_plan(_segment(layers={"snare": {"visible": True, "treatment": "ring", "gain": 1}})), _signals(), 96, 54)
    expected = viz.frame_rgb24(0.03)
    assert len(expected) == 96 * 54 * 3
    viz.frame_rgb24(2.0)
    assert viz.frame_rgb24(0.03) == expected


def test_source_clock_offset_is_applied_to_state_and_hit():
    plan = _plan(_segment(layers={"snare": {"visible": True, "treatment": "ring", "gain": 1}}))
    viz = DirectedVisualizer(plan, _signals(), 96, 54)
    assert viz.state_at(0)["absolute_s"] == 130
    assert viz.state_at(0)["signals"]["snare"] > 0
    assert viz.state_at(1)["signals"]["snare"] == 0


def test_same_plan_and_signals_make_identical_frames():
    plan = _plan(_segment(layers={"bass": {"visible": True, "treatment": "ribbon", "gain": .8}}))
    assert DirectedVisualizer(plan, _signals(), 96, 54).frame_rgb24(1.1) == DirectedVisualizer(plan, _signals(), 96, 54).frame_rgb24(1.1)


def test_hidden_drums_have_no_effect_after_transition():
    visible = _segment(130, 132, layers={"snare": {"visible": True, "treatment": "ring", "gain": 1}})
    hidden = _segment(132, 140, transition=1, layers={"bass": {"visible": True, "treatment": "ribbon", "gain": .5}})
    with_hit = {"beat_times_s": [],
                "hits": [{"t": 131.1, "component": "snare", "velocity": 1},
                         {"t": 133.1, "component": "snare", "velocity": 1}],
                "energy": {"times_s": [130], "bass": [.5], "vocals": [0], "other": [0]}}
    without_hit = {**with_hit, "hits": []}
    plan = _plan(visible, hidden)
    # The old segment actually responds to its snare fixture before the cut.
    assert DirectedVisualizer(plan, with_hit, 96, 54).frame_rgb24(1.1) != DirectedVisualizer(plan, without_hit, 96, 54).frame_rgb24(1.1)
    # At absolute 133.1 the one-second transition is complete and snare is
    # hidden, so even a fresh full-strength hit has zero visual effect.
    assert DirectedVisualizer(plan, with_hit, 96, 54).frame_rgb24(3.1) == DirectedVisualizer(plan, without_hit, 96, 54).frame_rgb24(3.1)


def test_treatment_selection_changes_pixels_and_zero_gain_draws_nothing():
    ring = _plan(_segment(layers={"snare": {"visible": True, "treatment": "ring", "gain": 1}}))
    rails = _plan(_segment(layers={"snare": {"visible": True, "treatment": "rails", "gain": 1}}))
    zero = _plan(_segment(layers={"snare": {"visible": True, "treatment": "ring", "gain": 0}}))
    blank = _plan(_segment())
    assert DirectedVisualizer(ring, _signals(), 96, 54).frame_rgb24(0) != DirectedVisualizer(rails, _signals(), 96, 54).frame_rgb24(0)
    assert DirectedVisualizer(zero, _signals(), 96, 54).frame_rgb24(0) == DirectedVisualizer(blank, _signals(), 96, 54).frame_rgb24(0)


def test_crossfade_starts_at_boundary_without_anticipation():
    old = _segment(130, 132, layers={"snare": {"visible": True, "treatment": "ring", "gain": 1}})
    new = _segment(132, 140, transition=1, layers={"kick": {"visible": True, "treatment": "rails", "gain": 1}})
    plan = _plan(old, new)
    viz = DirectedVisualizer(plan, _signals(), 96, 54)
    assert viz.state_at(1.999)["transition"] == 1.0
    assert viz.state_at(2.0)["transition"] == 0.0
    assert 0 < viz.state_at(2.5)["transition"] < 1


def test_crossfade_midpoint_is_a_blend_of_independently_composited_scenes():
    old_layers = {"bass": {"visible": True, "treatment": "ribbon", "gain": 1}}
    new_layers = {"vocals": {"visible": True, "treatment": "ring", "gain": 1}}
    old = _segment(130, 132, focus="bass", layers=old_layers)
    new = _segment(132, 140, focus="vocals", transition=1, layers=new_layers)
    signals = {"beat_times_s": [], "hits": [],
               "energy": {"times_s": [132.0], "bass": [.8], "vocals": [.7], "other": [0]}}
    mixed = DirectedVisualizer(_plan(old, new), signals, 96, 54).frame_rgb24(2.5)
    old_scene = DirectedVisualizer(_plan(_segment(130, 140, focus="bass", layers=old_layers)), signals, 96, 54).frame_rgb24(2.5)
    new_scene = DirectedVisualizer(_plan(_segment(130, 140, focus="vocals", layers=new_layers)), signals, 96, 54).frame_rgb24(2.5)
    expected = Image.blend(Image.frombytes("RGB", (96, 54), old_scene), Image.frombytes("RGB", (96, 54), new_scene), .5).tobytes()
    assert mixed == expected


def test_plan_mutation_after_construction_does_not_change_frames():
    plan = _plan(_segment(layers={"snare": {"visible": True, "treatment": "ring", "gain": 1}}))
    renderer = DirectedVisualizer(plan, _signals(), 96, 54)
    control = DirectedVisualizer(plan, _signals(), 96, 54)
    plan["segments"][0]["layers"]["snare"]["gain"] = 0
    plan["segments"][0]["motif"] = "caller-edited"
    assert renderer.frame_rgb24(0) == control.frame_rgb24(0)
