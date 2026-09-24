from PIL import Image, ImageDraw
import pytest

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


def test_v2_envelope_is_seek_deterministic_and_exposed_for_diagnostics():
    layer = {"visible": True, "treatment": "ribbon", "gain": .1,
             "envelope": [{"t_s": 130., "gain": .1, "emphasis": .1}, {"t_s": 132., "gain": .9, "emphasis": .9}, {"t_s": 140., "gain": .9, "emphasis": .9}]}
    viz = DirectedVisualizer(_plan(_segment(focus="bass", layers={"bass": layer})), _signals(), 96, 54)
    first = viz.frame_rgb24(1.)
    viz.frame_rgb24(5.)
    assert viz.frame_rgb24(1.) == first
    state = viz.state_at(1.)
    assert state["gains"]["bass"] == .5
    assert state["emphasis"]["bass"] == .5
    assert state["evaluated_layers"]["bass"]["visible"]


def test_v2_hidden_layer_remains_invisible_even_with_a_signal():
    hidden = {"visible": False, "treatment": "ring", "gain": 0,
              "envelope": [{"t_s": 130., "gain": 0., "emphasis": 0.}, {"t_s": 140., "gain": 0., "emphasis": 0.}]}
    plan = _plan(_segment(layers={"snare": hidden}))
    baseline = _plan(_segment())
    assert DirectedVisualizer(plan, _signals(), 96, 54).frame_rgb24(0) == DirectedVisualizer(baseline, _signals(), 96, 54).frame_rgb24(0)


def test_v2_crossfade_diagnostics_keep_the_outgoing_layer_visible():
    outgoing = {"visible": True, "treatment": "ring", "gain": 1,
                "envelope": [{"t_s": 130., "gain": 1., "emphasis": 1.}, {"t_s": 132., "gain": 1., "emphasis": 1.}]}
    incoming = {"visible": True, "treatment": "ribbon", "gain": .5,
                "envelope": [{"t_s": 132., "gain": .5, "emphasis": .2}, {"t_s": 140., "gain": .5, "emphasis": .2}]}
    old = _segment(130, 132, layers={"snare": outgoing})
    new = _segment(132, 140, transition=1, focus="bass", layers={"bass": incoming})
    state = DirectedVisualizer(_plan(old, new), _signals(), 96, 54).state_at(2.5)
    assert state["evaluated_layers"]["snare"]["visible"] is False
    assert state["previous_evaluated_layers"]["snare"]["visible"] is True
    assert state["scene_weights"] == {"previous": .5, "incoming": .5}
    settled = DirectedVisualizer(_plan(old, new), _signals(), 96, 54).state_at(3.5)
    assert settled["scene_weights"] == {"previous": 0., "incoming": 1.}


def test_v2_constant_envelope_has_no_pixel_cut_at_a_shared_knot():
    layer_a = {"visible": True, "treatment": "rails", "gain": .5,
               "envelope": [{"t_s": 130., "gain": .5, "emphasis": .5}, {"t_s": 132., "gain": .5, "emphasis": .5}]}
    layer_b = {"visible": True, "treatment": "rails", "gain": .5,
               "envelope": [{"t_s": 132., "gain": .5, "emphasis": .5}, {"t_s": 140., "gain": .5, "emphasis": .5}]}
    old = _segment(130, 132, focus="bass", layers={"bass": layer_a})
    new = _segment(132, 140, focus="bass", layers={"bass": layer_b})
    constant = {"beat_times_s": [], "hits": [],
                "energy": {"times_s": [130.], "bass": [.6], "vocals": [0.], "other": [0.]}}
    viz = DirectedVisualizer(_plan(old, new), constant, 96, 54)
    assert viz.frame_rgb24(1.999) == viz.frame_rgb24(2.)


def test_authored_treatments_have_distinct_silhouettes_and_stable_anchors():
    signals = {"beat_times_s": [130.],
               "hits": [{"t": 130., "component": "kick", "velocity": 1.}, {"t": 130., "component": "snare", "velocity": 1.}],
               "energy": {"times_s": [130.], "bass": [.7], "vocals": [.8], "other": [.8]}}
    def styled(name, treatment, anchor):
        return {"visible": True, "treatment": treatment, "gain": 1., "anchor": anchor,
                "envelope": [{"t_s": 130., "gain": 1., "emphasis": .6}, {"t_s": 140., "gain": 1., "emphasis": .6}]}
    frames = {}
    for name, treatment, anchor in (("vocals", "filament", [.50, .42]), ("snare", "shards", [.80, .47]),
                                    ("kick", "impact", [.50, .79]), ("other", "contour", [.50, .66])):
        frames[name] = DirectedVisualizer(_plan(_segment(focus=name, layers={name: styled(name, treatment, anchor)})), signals, 96, 54).frame_rgb24(0)
    assert len(set(frames.values())) == 4
    # Anchor-based focus no longer relocates the shape to the shared center.
    supporting = styled("snare", "shards", [.80, .47])
    plan = _plan(_segment(focus="bass", layers={"snare": supporting, "bass": styled("bass", "ribbon", [.50, .67])}))
    state = DirectedVisualizer(plan, signals, 96, 54).state_at(0)
    assert state["evaluated_layers"]["snare"]["visible"]


def test_new_silhouettes_differ_with_identical_color_signal_and_position():
    viz = DirectedVisualizer(_plan(_segment()), _signals(), 480, 270)
    silhouettes = {}
    for treatment in ('filament', 'shards', 'impact', 'contour'):
        canvas = Image.new('RGBA', (480, 270))
        viz._draw_treatment(ImageDraw.Draw(canvas), treatment, 'vocals', .8,
                            (255, 255, 255), 200, 130., (.5, .5), 1.)
        mask = canvas.getchannel('A')
        assert mask.getbbox() is not None
        silhouettes[treatment] = mask
    assert len({mask.tobytes() for mask in silhouettes.values()}) == 4
    voice = silhouettes['filament'].getbbox()
    other = silhouettes['contour'].getbbox()
    assert voice[3] - voice[1] > 2 * (voice[2] - voice[0])
    assert other[2] - other[0] > 3 * (other[3] - other[1])


@pytest.mark.parametrize('treatment', ['filament', 'shards', 'impact', 'contour'])
def test_new_treatments_are_seek_deterministic_and_silent_at_zero(treatment):
    layer = {'visible': True, 'treatment': treatment, 'gain': 1., 'anchor': [.5, .5]}
    plan = _plan(_segment(focus='vocals', layers={'vocals': layer}))
    signals = {'beat_times_s': [], 'hits': [], 'energy': {'times_s': [130.], 'bass': [0.], 'other': [0.], 'vocals': [.7]}}
    viz = DirectedVisualizer(plan, signals, 192, 108)
    frame = viz.frame_rgb24(.5)
    viz.frame_rgb24(3.)
    assert viz.frame_rgb24(.5) == frame
    signals['energy']['vocals'] = [0.]
    quiet = DirectedVisualizer(plan, signals, 192, 108)
    assert quiet.frame_rgb24(.5) == quiet._base.tobytes()


def _authored_vocal_emphasis_plans():
    def plan(curve):
        layers = {name: {"visible": False, "treatment": "ticks", "gain": 0.}
                  for name in DirectedVisualizer.LAYERS}
        layers["vocals"] = {"visible": True, "treatment": "filament", "anchor": [.5, .42], "gain": 1.,
                            "envelope": [{"t_s": t, "gain": gain, "emphasis": emphasis}
                                         for t, gain, emphasis in curve]}
        return {"schema_version": 2, "start_s": 119., "end_s": 132.,
                "segments": [{"start_s": 119., "end_s": 132., "focus": "vocals", "motif": "verse-ending",
                              "palette": "cool", "transition_s": 0., "layers": layers,
                              "reason": "authored test", "evidence_ids": []}],
                "visual_policy": {"kind": "authored_source_vocabulary_v1"}}
    steady_curve = [(119., 1., 1.), (123., 1., 1.), (125.5, 1., 1.), (132., 1., 1.)]
    reduced_curve = [(119., 1., 1.), (123., 1., 1.), (125.5, .45, .30), (132., .45, .30)]
    return plan(reduced_curve), plan(steady_curve)


def _authored_vocal_signals(vocal_level=.8):
    return {"beat_times_s": [], "hits": [],
            "energy": {"times_s": [119., 123., 124.25, 125.5, 131.],
                       "bass": [0.] * 5, "other": [0.] * 5, "vocals": [vocal_level] * 5}}


def test_authored_vocal_emphasis_evaluates_fixed_ramp_and_preserves_source_activity():
    reduced_plan, steady_plan = _authored_vocal_emphasis_plans()
    signals = _authored_vocal_signals()
    reduced = DirectedVisualizer(reduced_plan, signals, 192, 108)
    steady = DirectedVisualizer(steady_plan, signals, 192, 108)
    checks = {2.: (1., 1.), 4.: (1., 1.), 5.25: (.725, .65), 6.5: (.45, .30), 12.: (.45, .30)}
    for local, (gain, emphasis) in checks.items():
        reduced_state, steady_state = reduced.state_at(local), steady.state_at(local)
        assert reduced_state["evaluated_layers"]["vocals"]["visible"]
        assert steady_state["evaluated_layers"]["vocals"]["visible"]
        assert reduced_state["gains"]["vocals"] == pytest.approx(gain)
        assert reduced_state["emphasis"]["vocals"] == pytest.approx(emphasis)
        assert steady_state["gains"]["vocals"] == steady_state["emphasis"]["vocals"] == 1.
        # Envelope evaluation changes presentation only; the source clock is
        # identical in both authored variants.
        assert reduced_state["signals"] == steady_state["signals"]


def test_authored_vocal_emphasis_pixels_are_identical_before_ramp_then_diverge_and_seek():
    reduced_plan, steady_plan = _authored_vocal_emphasis_plans()
    reduced = DirectedVisualizer(reduced_plan, _authored_vocal_signals(), 192, 108)
    steady = DirectedVisualizer(steady_plan, _authored_vocal_signals(), 192, 108)
    before, at_ramp = 2., 4.
    assert reduced.frame_rgb24(before) == steady.frame_rgb24(before)
    assert reduced.frame_rgb24(at_ramp) == steady.frame_rgb24(at_ramp)
    middle = reduced.frame_rgb24(5.25)
    assert middle != steady.frame_rgb24(5.25)
    reduced.frame_rgb24(12.)
    assert reduced.frame_rgb24(5.25) == middle


def test_authored_vocal_emphasis_late_trace_requires_source_voice():
    reduced_plan, _ = _authored_vocal_emphasis_plans()
    audible = DirectedVisualizer(reduced_plan, _authored_vocal_signals(.8), 192, 108)
    silent = DirectedVisualizer(reduced_plan, _authored_vocal_signals(0.), 192, 108)
    # The authored late gain stays positive and visible, while the renderer
    # still lets source silence suppress pixels rather than inventing a trace.
    late = 12.
    assert audible.state_at(late)["gains"]["vocals"] == .45
    assert audible.frame_rgb24(late) != audible._base.tobytes()
    assert silent.frame_rgb24(late) == silent._base.tobytes()
