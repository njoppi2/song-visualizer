"""Checks for separate query labels, reference isolation and ordinal evidence."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'experiments' / f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


transfer = load('music_discovery_transfer')
negative = load('music_discovery_negative_reference')


def fixtures():
    rng = np.random.default_rng(17)
    refs = [dict(id=f'r{i}', isrc=f'ir{i}', sha256=f'hr{i}', label='favorite' if i < 5 else 'dislike') for i in range(9)]
    queries = [dict(id=f'q{i}', isrc=f'iq{i}', sha256=f'hq{i}') for i in range(3)]
    return rng.normal(size=(9, 8)), rng.normal(size=(9, 6)), refs, rng.normal(size=(3, 8)), rng.normal(size=(3, 6)), queries


def test_labels_cannot_change_reference_selection_or_scores():
    args = fixtures()
    before = transfer.score_new_queries(*args, negative.score_queries)
    args[-1][0]['label'] = 'favorite'
    args[-1][1]['label'] = 'dislike'
    after = transfer.score_new_queries(*args, negative.score_queries)
    assert before == after
    assert len(before['positive_references']) == 5
    assert len(before['negative_references']) == 4
    assert set(before['positive_references'] + before['negative_references']).isdisjoint({q['id'] for q in args[-1]})


def test_other_query_cannot_change_a_query_score_or_normalization():
    args = fixtures()
    before = transfer.score_new_queries(*args, negative.score_queries)
    args[3][-1] *= 10000
    args[4][-1] *= 10000
    after = transfer.score_new_queries(*args, negative.score_queries)
    for kind in ('muq', 'simple'):
        assert before['methods'][kind]['queries'][0] == after['methods'][kind]['queries'][0]


def test_duplicate_query_recording_of_a_reference_is_rejected():
    args = fixtures()
    args[-1][0]['isrc'] = args[2][0]['isrc']
    with pytest.raises(ValueError, match='duplicate'):
        transfer.score_new_queries(*args, negative.score_queries)


def test_only_explicit_strict_preferences_are_evaluated():
    scores = transfer.score_new_queries(*fixtures(), negative.score_queries)
    relations = [{'higher': 'q0', 'lower': 'q1'}, {'higher': 'q0', 'lower': 'q2'}]
    result = transfer.evaluate_orderings(scores, relations)
    for kinds in result.values():
        for method in kinds.values():
            assert method['total'] == 2
            assert all(r['higher'] == 'q0' for r in method['comparisons'])
