"""Reference/query separation and negative-reference behavior on synthetic data."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('negative_reference', Path(__file__).resolve().parents[1] / 'experiments/music_discovery_negative_reference.py')
negative = importlib.util.module_from_spec(spec)
spec.loader.exec_module(negative)


def data():
    rng = np.random.default_rng(100)
    tracks = [dict(id=str(i), isrc=f'isrc{i}', sha256=f'hash{i}', label='favorite' if i < 5 else 'dislike') for i in range(9)]
    return rng.normal(size=(9, 7)), rng.normal(size=(9, 4)), tracks


def test_both_queries_are_removed_from_both_reference_sets():
    for fold in negative.evaluate(*data())['folds']:
        queries = {fold['favorite'], fold['dislike']}
        assert queries.isdisjoint(fold['positive_references'])
        assert queries.isdisjoint(fold['negative_references'])
        assert len(fold['positive_references']) == 4
        assert len(fold['negative_references']) == 3


def test_heldout_rejection_cannot_change_other_query_scaling_or_scores():
    _, features, _ = data()
    before = negative.score_queries(features, [0, 1, 2, 3], [5, 6, 7], [4, 8], 'simple')
    features[8] *= 100000
    after = negative.score_queries(features, [0, 1, 2, 3], [5, 6, 7], [4, 8], 'simple')
    assert before[0] == after[0]


def test_negative_references_can_resolve_an_acoustic_false_positive():
    # Negative query is nearer a positive seed but virtually matches a rejection.
    vectors = np.array([[0.], [20.], [4.], [12.], [4.01]])
    favorite, dislike = negative.score_queries(vectors, [0, 1], [2], [3, 4], 'simple')
    assert favorite['positive_only'] < dislike['positive_only']
    assert favorite['negative_aware'] > dislike['negative_aware']


def test_query_recording_and_duplicate_isrc_are_rejected():
    muq, simple, tracks = data()
    with pytest.raises(ValueError, match='overlap'):
        negative.score_queries(muq, [0, 1], [5, 6], [0, 7], 'muq')
    tracks[8]['isrc'] = tracks[0]['isrc']
    with pytest.raises(ValueError, match='duplicate'):
        negative.evaluate(muq, simple, tracks)


def test_zero_vectors_nonfinite_values_and_numerical_ties():
    muq, _, _ = data()
    muq[0] = 0
    with pytest.raises(ValueError, match='zero'):
        negative.score_queries(muq, [0, 1], [5, 6], [4, 8], 'muq')
    muq[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        negative.score_queries(muq, [0, 1], [5, 6], [4, 8], 'muq')
    assert negative.verdict(1e-12) == 'tie'
    assert negative.verdict(-1e-12) == 'tie'
