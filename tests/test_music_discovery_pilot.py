"""Guards against recording and evaluation leakage in the tiny diagnostic."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('music_discovery_pilot', Path(__file__).resolve().parents[1] / 'experiments/music_discovery_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def fixture():
    tracks = [dict(id=str(i), sha256=f'hash{i}', isrc=f'isrc{i}', label='favorite' if i < 5 else 'dislike') for i in range(9)]
    rng = np.random.default_rng(41)
    return rng.normal(size=(9, 7)), rng.normal(size=(9, 6)), tracks


def test_heldout_and_rejected_tracks_never_enter_references():
    muq, simple, tracks = fixture()
    for fold in pilot.evaluate_vectors(muq, simple, tracks)['folds']:
        assert fold['heldout'] not in fold['references']
        assert len(fold['references']) == 4
        assert set(fold['references']) <= set('01234')
        for method in fold['methods'].values():
            assert len(method['candidates']) == 5
            assert {x['id'] for x in method['candidates']}.isdisjoint(fold['references'])


def test_altering_one_test_rejection_does_not_change_other_candidate_scores():
    muq, simple, tracks = fixture()
    before = pilot.evaluate_vectors(muq, simple, tracks)
    simple[8] *= 1000000
    after = pilot.evaluate_vectors(muq, simple, tracks)
    for a, b in zip(before['folds'], after['folds']):
        sa = {x['id']: x['score'] for x in a['methods']['simple']['candidates'] if x['id'] != '8'}
        sb = {x['id']: x['score'] for x in b['methods']['simple']['candidates'] if x['id'] != '8'}
        assert sa == sb


def test_duplicate_recordings_are_rejected_even_with_different_names():
    muq, simple, tracks = fixture()
    tracks[-1]['isrc'] = tracks[0]['isrc']
    with pytest.raises(ValueError, match='recording isrc'):
        pilot.evaluate_vectors(muq, simple, tracks)


def test_zero_or_nonfinite_embeddings_do_not_become_recommendation_scores():
    muq, simple, tracks = fixture()
    muq[0] = 0
    with pytest.raises(ValueError, match='zero embedding'):
        pilot.evaluate_vectors(muq, simple, tracks)
    muq[0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        pilot.evaluate_vectors(muq, simple, tracks)
