"""Meaningful reference separation, ordinal-label and validation checks."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('emotion_profile', Path(__file__).resolve().parents[1] / 'experiments/music_emotion_profile.py')
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def data():
    tracks = [dict(id=f's{i}', isrc=f'i{i}', sha256=f'h{i}', label='favorite' if i < 6 else 'dislike' if i < 10 else 'like') for i in range(12)]
    return np.random.default_rng(500).uniform(.01,1,(12,56)), tracks


def test_queries_never_enter_references_and_like_is_not_dislike():
    results = profile.evaluate(*data())
    assert results['summary']['favorite_vs_dislike']['total'] == 24
    assert results['summary']['favorite_vs_like']['total'] == 12
    for p in results['pairs']:
        assert len(p['references']) == 5
        assert {p['favorite'], p['other']}.isdisjoint(p['references'])
        assert set(p['references']) <= {f's{i}' for i in range(6)}


def test_changing_one_nonfavorite_cannot_change_other_pairs():
    features, tracks = data()
    before = profile.evaluate(features, tracks)
    features[10] *= 30
    after = profile.evaluate(features, tracks)
    assert [p for p in before['pairs'] if p['other'] != 's10'] == [p for p in after['pairs'] if p['other'] != 's10']


def test_duplicate_recording_and_bad_vectors_are_rejected():
    features, tracks = data()
    tracks[11]['isrc'] = tracks[0]['isrc']
    with pytest.raises(ValueError, match='duplicate'):
        profile.evaluate(features, tracks)
    features, tracks = data()
    features[0] = 0
    with pytest.raises(ValueError, match='zero'):
        profile.evaluate(features, tracks)
    features[0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        profile.evaluate(features, tracks)


def test_resampler_endpoint_rounding_preserved_but_bad_lengths_rejected():
    wave = np.zeros(160001, dtype=np.float32)
    profile.validate_resampled_window(wave)
    assert len(wave) == 160001
    with pytest.raises(ValueError, match='resampling'):
        profile.validate_resampled_window(np.zeros(160002))
    wave[0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        profile.validate_resampled_window(wave)
