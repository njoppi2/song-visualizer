import numpy as np
import pytest

from songviz.recurrence import compare_phrases


def test_ordered_return_matches_without_section_roles():
    a = np.eye(4)
    f = np.concatenate([a, a[:, ::-1], a], axis=1)
    result = compare_phrases({'mix': f}, {'mix': np.ones(12)}, np.arange(13.), scale_beats=4, stride_beats=4)
    best = result['pairs'][0]
    assert (best['a'], best['b']) == (0, 2)
    assert best['similarity'] == pytest.approx(1)
    assert result['pairs'][-1]['similarity'] == 0


def test_shared_silence_is_not_positive_recurrence_evidence():
    result = compare_phrases({'mix': np.zeros((2, 12))}, {'mix': np.zeros(12)}, np.arange(13.), scale_beats=4)
    assert result['pairs'] == []


def test_entry_in_additional_stem_reduces_full_arrangement_match():
    result = compare_phrases({'bass': np.ones((2, 8)), 'vocals': np.ones((2, 8))},
                             {'bass': np.ones(8), 'vocals': np.r_[np.zeros(4), np.ones(4)]},
                             np.arange(9.), scale_beats=4, stride_beats=4)
    pair = result['pairs'][0]
    assert pair['stem_similarities']['bass'] == pytest.approx(1)
    assert pair['stem_similarities']['vocals'] == 0
    assert pair['similarity'] == pytest.approx(0.5)


def test_diagnostics_separate_shared_pattern_from_scaled_arrangement_and_legacy_score():
    result = compare_phrases(
        {'bass': np.ones((2, 8))},
        {'bass': np.r_[np.ones(4), np.full(4, 0.5)]},
        np.arange(9.), scale_beats=4, stride_beats=4,
    )
    pair = result['pairs'][0]
    # The legacy product remains intact while diagnostics identify its causes.
    assert pair['similarity'] == pair['stem_similarities']['bass'] == pytest.approx(0.5)
    assert pair['pattern_similarity'] == pytest.approx(1)
    assert pair['arrangement_similarity'] == pytest.approx(0.5)
    assert pair['shared_active_stems'] == ['bass']
    assert pair['stem_evidence']['bass'] == {
        'pattern_similarity': pytest.approx(1),
        'level_similarity': pytest.approx(0.5),
        'activity': 'both',
    }


def test_entering_stem_does_not_suppress_shared_pattern_evidence():
    result = compare_phrases(
        {'bass': np.ones((2, 8)), 'vocals': np.ones((2, 8))},
        {'bass': np.ones(8), 'vocals': np.r_[np.zeros(4), np.ones(4)]},
        np.arange(9.), scale_beats=4, stride_beats=4,
    )
    pair = result['pairs'][0]
    assert pair['pattern_similarity'] == pytest.approx(1)
    assert pair['arrangement_similarity'] == pytest.approx(0.5)
    assert pair['shared_active_stems'] == ['bass']
    assert pair['stem_evidence']['vocals'] == {
        'pattern_similarity': None,
        'level_similarity': 0,
        'activity': 'b_only',
    }


def test_one_sided_activity_has_arrangement_not_pattern_evidence():
    result = compare_phrases(
        {'mix': np.ones((2, 8))},
        {'mix': np.r_[np.ones(4), np.zeros(4)]},
        np.arange(9.), scale_beats=4, stride_beats=4,
    )
    pair = result['pairs'][0]
    assert pair['pattern_similarity'] is None
    assert pair['arrangement_similarity'] == 0
    assert pair['stem_evidence']['mix']['activity'] == 'a_only'
    assert result['context'][1]['local_pattern_change'] is None
    assert result['context'][1]['local_arrangement_change'] == 1


def test_shared_silent_stem_has_neither_evidence_not_a_pattern_vote():
    result = compare_phrases(
        {'bass': np.ones((2, 8)), 'pad': np.zeros((2, 8))},
        {'bass': np.ones(8), 'pad': np.zeros(8)},
        np.arange(9.), scale_beats=4, stride_beats=4,
    )
    pair = result['pairs'][0]
    assert pair['pattern_similarity'] == pytest.approx(1)
    assert pair['shared_active_stems'] == ['bass']
    assert pair['stem_evidence']['pad'] == {
        'pattern_similarity': None,
        'level_similarity': None,
        'activity': 'neither',
    }


def test_diagnostics_use_bounded_cosine_so_ordered_reversal_is_visible():
    a = np.eye(4)
    result = compare_phrases(
        {'mix': np.concatenate([a, a[:, ::-1]], axis=1)},
        {'mix': np.ones(8)}, np.arange(9.), scale_beats=4, stride_beats=4,
    )
    pair = result['pairs'][0]
    assert pair['similarity'] == 0  # Legacy score remains clipped.
    assert pair['pattern_similarity'] == pytest.approx(0)


def test_pattern_and_context_values_are_bounded_and_aba_keeps_history_past_only():
    a = np.eye(4)
    result = compare_phrases(
        {'mix': np.concatenate([a, a[:, ::-1], a], axis=1)},
        {'mix': np.ones(12)}, np.arange(13.), scale_beats=4, stride_beats=4,
    )
    for pair in result['pairs']:
        assert 0 <= pair['pattern_similarity'] <= 1
    for item in result['context']:
        for key in ('local_pattern_change', 'historical_pattern_novelty'):
            if item[key] is not None:
                assert 0 <= item[key] <= 1

    final = result['context'][2]
    assert final['available_at_s'] == result['spans'][2]['end_s']
    assert final['previous_span'] == 1
    assert final['local_pattern_change'] == pytest.approx(1)
    assert final['best_prior_span'] == 0
    assert final['historical_pattern_novelty'] == pytest.approx(0)
    assert final['best_prior_span'] < final['span']


def test_context_is_past_nonoverlapping_and_marks_missing_silent_local_pair():
    # The immediate complete predecessor and current span are silent, so their
    # pair has no evidence; structural context must still retain predecessor 1.
    energy = np.r_[np.ones(4), np.zeros(4), np.zeros(4)]
    result = compare_phrases(
        {'mix': np.ones((2, 12))}, {'mix': energy}, np.arange(13.),
        scale_beats=4, stride_beats=4,
    )
    context = result['context']
    assert context[0]['previous_span'] is None
    assert context[0]['historical_pattern_novelty'] is None
    assert context[2]['previous_span'] == 1
    assert context[2]['local_pattern_change'] is None
    assert context[2]['local_arrangement_change'] is None
    assert context[2]['best_prior_span'] is None
    assert context[2]['comparable_prior_count'] == 0
    assert context[2]['historical_pattern_novelty'] is None
    assert all(item['best_prior_span'] is None or item['best_prior_span'] < item['span']
               for item in context)


def test_context_excludes_overlaps_and_never_looks_to_future():
    result = compare_phrases(
        {'mix': np.ones((2, 8))}, {'mix': np.ones(8)}, np.arange(9.),
        scale_beats=4, stride_beats=1,
    )
    # Span 4 begins at beat 4; spans 1--3 overlap it and cannot be context.
    item = result['context'][4]
    assert item['previous_span'] == item['best_prior_span'] == 0
    assert item['comparable_prior_count'] == 1
    assert result['context'][0]['historical_pattern_novelty'] is None
    for item in result['context']:
        if item['previous_span'] is not None:
            assert result['spans'][item['previous_span']]['end_beat'] <= result['spans'][item['span']]['start_beat']
        if item['best_prior_span'] is not None:
            assert result['spans'][item['best_prior_span']]['end_beat'] <= result['spans'][item['span']]['start_beat']


def test_no_partial_or_overlapping_pairs():
    bt = np.arange(12.) * 0.43 + 5
    result = compare_phrases({'mix': np.ones((2, 11))}, {'mix': np.ones(11)}, bt, scale_beats=4, stride_beats=1)
    for pair in result['pairs']:
        a, b = result['spans'][pair['a']], result['spans'][pair['b']]
        assert a['end_s'] <= b['start_s']
        assert b['end_s'] <= bt[-1]
    assert all(s['end_beat'] - s['start_beat'] == 4 for s in result['spans'])


def test_partial_track_has_no_complete_phrase_pairs():
    result = compare_phrases({'mix': np.ones((2, 3))}, {'mix': np.ones(3)}, np.arange(4.), scale_beats=16)
    assert result['spans'] == result['pairs'] == []


def test_invalid_feature_lengths_rejected():
    with pytest.raises(ValueError):
        compare_phrases({'mix': np.ones((2, 5))}, {'mix': np.ones(4)}, np.arange(6.))


@pytest.mark.parametrize('scale_beats,stride_beats', [(True, 4), (4, False), (4.0, 4), (4, 4.0)])
def test_non_integer_phrase_settings_rejected(scale_beats, stride_beats):
    with pytest.raises(ValueError):
        compare_phrases({'mix': np.ones((2, 8))}, {'mix': np.ones(8)}, np.arange(9.),
                        scale_beats=scale_beats, stride_beats=stride_beats)
