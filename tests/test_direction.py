import copy

import pytest

from songviz.direction import content_hash, fixed_plan, make_plan, validate_plan


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
