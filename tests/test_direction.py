import copy
import json
from pathlib import Path

import pytest

from songviz.direction import (content_hash, fixed_plan, make_gradual_plan, make_plan,
                               make_visual_plan, make_vocal_emphasis_plans,
                               validate_plan, validate_vocal_emphasis_comparison)


def signals(shift=0):
    return {"beat_times_s": [shift + i * .5 for i in range(41)],
            "hits": [{"t": shift + t, "component": "snare", "velocity": 1.} for t in [1., 2., 3., 4., 14., 15., 16., 17., 18.]],
            "energy": {"times_s": [shift + i for i in range(21)], "bass": [.2] * 21, "other": [.7] * 21, "vocals": [.1 if i < 14 else .9 for i in range(21)]}}


def test_automatic_gap_focus_and_return_without_hand_authored_boundaries():
    s = signals()
    p = make_plan(s, 2., 18.)
    assert [x['focus'] for x in p['segments']] == ['snare', 'other', 'vocals']
    assert [x['start_s'] for x in p['segments']] == [2., 4.35, 14.]
    assert p['segments'][0]['motif'] == p['segments'][2]['motif']
    assert not p['segments'][1]['layers']['snare']['visible']
    assert make_plan(s, 2., 18.) == p


def test_same_policy_translates_with_song_clock_and_does_not_mutate_signals():
    s = signals(); before = copy.deepcopy(s)
    a, b = make_plan(s, 2., 18.), make_plan(signals(100), 102., 118.)
    assert [x['focus'] for x in a['segments']] == [x['focus'] for x in b['segments']]
    assert [round(x['start_s'] - 100, 6) for x in b['segments']] == [x['start_s'] for x in a['segments']]
    assert s == before


@pytest.mark.parametrize('change', ['gap', 'overlap', 'treatment', 'gain', 'focus', 'evidence', 'hash', 'nan'])
def test_rejects_invalid_plan(change):
    s = signals();p = make_plan(s, 2., 18.)
    if change == 'gap': p['segments'][1]['start_s'] += .1
    elif change == 'overlap': p['segments'][1]['start_s'] -= .1
    elif change == 'treatment': p['segments'][0]['layers']['snare']['treatment'] = 'arbitrary-code'
    elif change == 'gain': p['segments'][0]['layers']['snare']['gain'] = 3
    elif change == 'focus': p['segments'][1]['focus'] = 'snare'
    elif change == 'evidence': p['segments'][0]['evidence_ids'] = ['invented']
    elif change == 'hash': p['signals_sha256'] = 'wrong'
    elif change == 'nan': p['end_s'] = float('nan')
    with pytest.raises(ValueError): validate_plan(p, s)


def test_fixed_comparison_reuses_evidence_but_not_focus_schedule():
    s = signals();p = make_plan(s, 2., 18.);f = fixed_plan(p, s)
    assert len(f['segments']) == 1 and len(p['segments']) == 3
    assert all(x['visible'] for x in f['segments'][0]['layers'].values())
    assert f['signals_sha256'] == content_hash(s)


def test_fixed_comparison_normalizes_a_v2_source_back_to_the_v1_contract():
    s = signals(); gradual = make_gradual_plan(s, make_plan(s, 2., 18.))
    fixed = fixed_plan(gradual, s)
    assert fixed["schema_version"] == 1
    assert "envelope" not in fixed["segments"][0]["layers"]["snare"]
    validate_plan(fixed, s)


def test_empty_drums_can_select_remaining_texture_without_inventing_hits():
    s = signals();s['hits'] = []
    p = make_plan(s, 2., 10.)
    assert len(p['segments']) == 1 and p['segments'][0]['focus'] == 'other'
    assert not p['segments'][0]['layers']['kick']['visible']


def test_missing_energy_and_out_of_range_intervals_fail():
    s = signals()
    with pytest.raises(ValueError): make_plan(s, 0., 22.)
    s['energy']['vocals'] = []
    with pytest.raises(ValueError): make_plan(s, 2., 18.)


def test_gradual_plan_keeps_coarse_comparison_vocabulary_but_evolves_sparse_visuals():
    s = signals()
    s["energy"]["vocals"] = [.1 if i < 7 else min(.9, .1 + .12 * (i - 7)) for i in range(21)]
    baseline = make_plan(s, 2., 18.)
    gradual = make_gradual_plan(s, baseline)
    assert gradual["schema_version"] == 2
    assert [(x["start_s"], x["end_s"], x["palette"], x["motif"]) for x in gradual["segments"]] == [(x["start_s"], x["end_s"], x["palette"], x["motif"]) for x in baseline["segments"]]
    sparse = gradual["segments"][1]
    vocal_curve = sparse["layers"]["vocals"]["envelope"]
    assert sparse["layers"]["vocals"]["visible"]
    assert vocal_curve[-1]["gain"] > vocal_curve[0]["gain"]
    assert vocal_curve[-1]["emphasis"] > vocal_curve[0]["emphasis"]
    assert not sparse["layers"]["snare"]["visible"]
    assert gradual["evidence"][1]["semantic_interpretation"]["vocal_leadership"] == "unknown"
    assert sparse["baseline_reason"] == baseline["segments"][1]["reason"]
    support = gradual["evidence"][1]["envelope_support"]
    # At 4.35s the last completed sample is 4s; its trailing mean uses
    # samples at 3s and 4s, covering blocks beginning at 2s.
    assert support["support_start_s"] == 2.0
    validate_plan(gradual, s)


@pytest.mark.parametrize("mutate", [
    lambda p: p["segments"][1]["layers"]["vocals"]["envelope"].pop(),
    lambda p: p["segments"][1]["layers"]["vocals"]["envelope"].__setitem__(1, {"t_s": float("nan"), "gain": .2, "emphasis": .2}),
    lambda p: p["segments"][1]["layers"]["vocals"]["envelope"].__setitem__(1, {"t_s": 4.35, "gain": .2, "emphasis": .2}),
    lambda p: p["segments"][1]["layers"]["vocals"]["envelope"].__setitem__(1, {"t_s": 5., "gain": 1.2, "emphasis": .2}),
    lambda p: p["segments"][1]["layers"]["pulse"]["envelope"].__setitem__(0, {"t_s": 4.35, "gain": .9, "emphasis": .9}),
])
def test_gradual_plan_rejects_malformed_or_discontinuous_curves(mutate):
    s = signals(); plan = make_gradual_plan(s, make_plan(s, 2., 18.))
    mutate(plan)
    with pytest.raises(ValueError):
        validate_plan(plan, s)


def test_gradual_plan_resolves_evidence_by_id_not_list_position():
    s = signals(); baseline = make_plan(s, 2., 18.)
    baseline["evidence"].reverse()
    gradual = make_gradual_plan(s, baseline)
    sparse = next(segment for segment in gradual["segments"] if segment["start_s"] == 4.35)
    assert sparse["layers"]["vocals"]["visible"]
    assert all(record["envelope_support"]["available_through_s"] <= record["end_s"] for record in gradual["evidence"])


def test_schema2_focus_uses_effective_envelope_not_legacy_scalar_gain():
    s = signals(); plan = make_gradual_plan(s, make_plan(s, 2., 18.))
    sparse = plan["segments"][1]
    # Frozen v2 plans retain this scalar for old readers.  The renderer and
    # validator instead use the curve, so a deliberately different scalar is
    # compatible and must not make a working focus disappear.
    sparse["layers"][sparse["focus"]]["gain"] = 0.0
    validate_plan(plan, s)
    for key in sparse["layers"][sparse["focus"]]["envelope"]:
        key["gain"] = 0.0
    with pytest.raises(ValueError, match="Focus"):
        validate_plan(plan, s)


def test_schema2_focus_allows_intentional_zero_endpoint_fades():
    s = signals(); plan = make_gradual_plan(s, make_plan(s, 2., 18.))
    sparse = plan["segments"][1]
    curve = sparse["layers"][sparse["focus"]]["envelope"]
    curve[0]["gain"] = curve[-1]["gain"] = 0.0
    # Interior effective gain remains positive: fades may intentionally touch
    # zero at a segment edge without invalidating the focused passage.
    assert any(key["gain"] > 0 for key in curve[1:-1])
    validate_plan(plan, s)


@pytest.mark.parametrize("mutate", [
    lambda support: support.pop("available_through_s"),
    lambda support: support.pop("sample_clock"),
    lambda support: support.__setitem__("window_s", 1.25),
    lambda support: support.__setitem__("envelope_interval_s", [4.0, 10.0]),
    lambda support: support.__setitem__("support_start_s", 2.1),
    lambda support: support.__setitem__("available_through_s", 10.1),
])
def test_generated_gradual_support_must_be_complete_and_on_its_frozen_clock(mutate):
    s = signals(); plan = make_gradual_plan(s, make_plan(s, 2., 18.))
    mutate(plan["evidence"][1]["envelope_support"])
    with pytest.raises(ValueError, match="envelope support"):
        validate_plan(plan, s)


def test_authored_v2_provenance_is_explicit_and_cannot_strip_generated_support():
    s = signals(); plan = make_gradual_plan(s, make_plan(s, 2., 18.))
    # A visual policy is presentation-only, not an authored envelope claim.
    plan["visual_policy"] = {"kind": "authored_source_vocabulary_v1"}
    for record in plan["evidence"]:
        record.pop("envelope_support")
    with pytest.raises(ValueError, match="envelope support"):
        validate_plan(plan, s)

    plan["planner"] = {"kind": "manual", "version": "study_v1"}
    with pytest.raises(ValueError, match="authored provenance"):
        validate_plan(plan, s)
    plan["envelope_provenance"] = {"kind": "authored", "description": "Hand-authored review curve."}
    validate_plan(plan, s)


@pytest.mark.parametrize("package", ["directed-review-02", "directed-replay-02"])
def test_frozen_directed_02_plans_remain_valid_compatibility_inputs(package):
    root = Path(__file__).resolve().parents[1] / "outputs" / "reviews" / package
    if not (root / 'plan.json').is_file():
        pytest.skip('Local frozen development package is not included in a fresh clone')
    plan = json.loads((root / "plan.json").read_text())
    signals_input = json.loads((root / "signals.json").read_text())
    validate_plan(plan, signals_input)


def test_generated_support_cannot_relabel_another_interval_as_its_segment():
    s = signals(); plan = make_gradual_plan(s, make_plan(s, 2., 18.))
    record = plan['evidence'][1]
    record['end_s'] = 21.
    record['envelope_support']['envelope_interval_s'][1] = 21.
    record['envelope_support']['available_through_s'] = 20.
    with pytest.raises(ValueError, match='envelope support'):
        validate_plan(plan, s)


def test_visual_plan_is_a_visual_only_deep_copy_of_a_schema2_plan():
    s = signals()
    gradual = make_gradual_plan(s, make_plan(s, 2., 18.))
    visual = make_visual_plan(s, gradual)
    assert visual is not gradual and visual["segments"] is not gradual["segments"]
    assert visual["visual_policy"]["kind"] == "authored_source_vocabulary_v1"
    expected = {"vocals": "filament", "snare": "shards", "kick": "impact", "other": "contour",
                "bass": "ribbon", "hh": "ticks", "pulse": "ring"}
    for original, styled in zip(gradual["segments"], visual["segments"]):
        keys = ("start_s", "end_s", "focus", "motif", "reason", "evidence_ids", "transition_s", "baseline_reason")
        assert {key: styled[key] for key in keys} == {key: original[key] for key in keys}
        for name in expected:
            assert styled["layers"][name]["treatment"] == expected[name]
            assert styled["layers"][name]["anchor"]
            assert {key: styled["layers"][name][key] for key in ("visible", "gain", "envelope")} == {key: original["layers"][name][key] for key in ("visible", "gain", "envelope")}
    assert visual["evidence"] == gradual["evidence"]
    validate_plan(visual, s)


def test_visual_plan_requires_gradual_schema2_and_validates_anchors():
    s = signals()
    with pytest.raises(ValueError):
        make_visual_plan(s, make_plan(s, 2., 18.))
    visual = make_visual_plan(s, make_gradual_plan(s, make_plan(s, 2., 18.)))
    visual["segments"][0]["layers"]["vocals"]["anchor"] = [1.1, .4]
    with pytest.raises(ValueError):
        validate_plan(visual, s)


def emphasis_signals():
    s = signals(118.)
    s["beat_times_s"] = [118. + i * .5 for i in range(31)]
    s["energy"] = {"times_s": [118. + i * .5 for i in range(31)],
                   "bass": [.2] * 31, "other": [.5] * 31, "vocals": [.6] * 31}
    s["hits"] = [{"t": 119. + i, "component": "snare", "velocity": 1.} for i in range(13)]
    return s


def emphasis_feedback():
    return {"file": {"path": "benchmark/feedback/listening-examples-01.json", "sha256": "a" * 64},
            "answer": {"example_id": "verse-ending", "perceived_change": "subtle",
                       "notes": "Complete qualified raw verse-ending note."}}


def emphasis_parent(s):
    return make_visual_plan(s, make_gradual_plan(s, make_plan(s, 119., 132.)))


def test_authored_vocal_emphasis_plans_are_one_scene_and_preserve_feedback():
    s = emphasis_signals(); feedback = emphasis_feedback()
    reduced, steady = make_vocal_emphasis_plans(s, emphasis_parent(s), feedback)
    assert reduced["start_s"] == 119. and reduced["end_s"] == 132.
    assert reduced["segments"][0]["focus"] == steady["segments"][0]["focus"] == "vocals"
    assert reduced["evidence"] == steady["evidence"] == [{"id": "verse-ending-feedback", "feedback_record": feedback}]
    assert reduced["evidence"][0]["feedback_record"] is not feedback
    vocal = reduced["segments"][0]["layers"]["vocals"]
    control = steady["segments"][0]["layers"]["vocals"]
    assert vocal["gain"] == control["gain"] == 1.0
    assert [(x["t_s"], x["gain"], x["emphasis"]) for x in vocal["envelope"]] == [(119., 1., 1.), (123., 1., 1.), (125.5, .45, .30), (132., .45, .30)]
    assert all(x["gain"] == x["emphasis"] == 1. for x in control["envelope"])
    validate_vocal_emphasis_comparison(reduced, steady, s)


@pytest.mark.parametrize("mutate", [
    lambda reduced, steady: reduced["segments"][0].__setitem__("focus", "snare"),
    lambda reduced, steady: reduced.__setitem__("end_s", 131.9),
    lambda reduced, steady: reduced["segments"][0]["layers"]["kick"]["envelope"][2].__setitem__("gain", .1),
    lambda reduced, steady: reduced["segments"][0]["layers"]["snare"].__setitem__("anchor", [.1, .1]),
    lambda reduced, steady: reduced["evidence"][0]["feedback_record"]["answer"].__setitem__("notes", "shortened"),
    lambda reduced, steady: reduced["visual_policy"].__setitem__("stage", "changed"),
    lambda reduced, steady: reduced["segments"][0]["layers"]["vocals"]["envelope"][2].__setitem__("t_s", 125.4),
])
def test_authored_vocal_emphasis_pair_rejects_shared_or_boundary_tampering(mutate):
    s = emphasis_signals(); reduced, steady = make_vocal_emphasis_plans(s, emphasis_parent(s), emphasis_feedback())
    mutate(reduced, steady)
    with pytest.raises(ValueError):
        validate_vocal_emphasis_comparison(reduced, steady, s)


def test_authored_vocal_emphasis_pair_rejects_same_mutated_shared_field_on_both_sides():
    s = emphasis_signals(); reduced, steady = make_vocal_emphasis_plans(s, emphasis_parent(s), emphasis_feedback())
    for plan in (reduced, steady):
        plan["segments"][0]["layers"]["other"]["envelope"][0]["emphasis"] = .9
    with pytest.raises(ValueError, match="accompaniment"):
        validate_vocal_emphasis_comparison(reduced, steady, s)


def test_authored_vocal_emphasis_pair_rejects_scalar_vocal_gain_change():
    s = emphasis_signals(); reduced, steady = make_vocal_emphasis_plans(s, emphasis_parent(s), emphasis_feedback())
    reduced["segments"][0]["layers"]["vocals"]["gain"] = .45
    with pytest.raises(ValueError, match="scalar"):
        validate_vocal_emphasis_comparison(reduced, steady, s)
