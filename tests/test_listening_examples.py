from copy import deepcopy
import json

import pytest

from experiments.build_listening_examples import encode, examples


@pytest.fixture
def review():
    events = [
        ('sustained_activity-change-26', 79.10341721880629, 'drums', 'increase'),
        ('sustained_activity-change-8', 20.630015882218938, 'other', 'decrease'),
        ('sustained_activity-change-38', 124.58272936948532, 'other', 'decrease'),
    ]
    return {
        'duration_s': 221.173333,
        'reference': {'layers': [{'spans': [{'id': 'human', 'start_s': 61.368411, 'end_s': 64.855320}]}]},
        'variants': {'sustained_activity': {'predictions': {
            'changes': [{'id': identity, 'time_s': t, 'primary_stem': stem,
                         'stem_evidence': {stem: {'qualified': True, 'direction': direction}}}
                        for identity, t, stem, direction in events],
            'transitions': [{'id': 'transition-1', 'start_s': 63.943646501913264, 'end_s': 65.24305542050409}],
        }}},
    }


def test_curated_copy_is_bound_to_evidence_and_preserves_prior_reference(review):
    original = deepcopy(review)
    result = examples(review)
    assert len(result) == len({item['id'] for item in result}) == 4
    assert review == original
    assert result[-1]['focus_start_s'] == review['reference']['layers'][0]['spans'][0]['start_s']
    assert result[-1]['evidence']['dip']['start_s'] != result[-1]['focus_start_s']


@pytest.mark.parametrize('damage', ['time', 'stem', 'direction', 'qualification', 'missing', 'reference', 'duration'])
def test_changed_evidence_cannot_silently_reuse_curated_explanation(review, damage):
    changes = review['variants']['sustained_activity']['predictions']['changes']
    if damage == 'time':
        changes[0]['time_s'] += 1
    elif damage == 'stem':
        changes[0]['primary_stem'] = 'vocals'
    elif damage == 'direction':
        changes[0]['stem_evidence']['drums']['direction'] = 'decrease'
    elif damage == 'qualification':
        changes[0]['stem_evidence']['drums']['qualified'] = False
    elif damage == 'missing':
        changes.pop()
    elif damage == 'reference':
        review['reference']['layers'][0]['spans'][0]['start_s'] = 62
    else:
        review['duration_s'] = 80
    with pytest.raises(ValueError):
        examples(review)


def test_embedded_feedback_text_cannot_close_script():
    data = {'notes': '</script><img onerror="oops()"> & ação'}
    encoded = encode(data, embedded=True)
    assert '<' not in encoded and '&' not in encoded
    assert json.loads(encoded) == data
