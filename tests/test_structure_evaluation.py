from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from songviz.recurrence import compare_phrases
from songviz.structure_annotations import normalize_annotations
from songviz.structure_evaluation import evaluate_structure
from songviz.ingest import sha256_file
from experiments.evaluate_structure_feedback import build, report_markdown, verify_legacy_scores

ROOT = Path(__file__).resolve().parents[1]
FEEDBACK = ROOT/'benchmark/feedback/section-editor-02.json'
MAPPING = ROOT/'benchmark/feedback/section-editor-02.interpretation.json'


def reference():
    return normalize_annotations(json.loads(FEEDBACK.read_text()), feedback_sha256=sha256_file(FEEDBACK),
                                 interpretations=json.loads(MAPPING.read_text()))


def test_real_feedback_dimensions_preserved_without_predicting_them():
    ref = reference()
    before = deepcopy(ref)
    sections = [{'start_s': 0, 'end_s': ref['source']['duration_s'], 'role': 'outro', 'label': 'A'}]
    result = evaluate_structure(ref, sections, [])
    layer = result['layers'][0]
    assert ref == before
    assert len(layer['boundary_evidence']) == 18
    assert len(layer['transition_intervals']) == 5
    assert min(s['duration_s'] for s in layer['transition_intervals']) == pytest.approx(.807118)
    assert all(s['prediction_status'] == 'not_implemented' for s in layer['transition_intervals'])
    chorus_change = next(b for b in layer['boundary_evidence'] if b['time_s'] == 78.889763)
    assert chorus_change['identity_relation'] == 'same_explicit_group'
    assert chorus_change['variation_change'] is True
    assert chorus_change['nearest_detector_boundary_s'] is None
    assert chorus_change['detector_minus_annotation_s'] is None
    assert layer['legacy_section_overlap'][-1]['candidate_overlaps'][0]['role'] == 'outro'
    assert result['capabilities']['musical_identity'] == 'acoustic_candidates_only'


def simple_reference():
    def span(i, a, b, identity, variation=None, transition=None):
        return {'id': str(i), 'start_s': a, 'end_s': b, 'identity_id': identity, 'variation': variation,
                'transition': transition, 'label': str(i), 'certainty': 'unspecified'}
    return {'source': {'duration_s': 16, 'song_title': 'test'}, 'layers': [
        {'id': 'one', 'name': 'parts', 'identity_groups': [{'id': 'A', 'name': 'chorus'}, {'id': 'B', 'name': 'verse'}],
         'spans': [span(0, 0, 4, 'A', 'low'), span(1, 4, 8, 'B'),
                   span(2, 8, 12, 'A', 'high'), span(3, 12, 13, None, transition=True), span(4, 13, 16, None)]}]}


def test_repeat_evaluation_uses_positive_groups_not_role_labels_or_certified_negatives():
    ref = simple_reference()
    features = {'mix': np.tile(np.eye(4), 4)}
    recurrence = compare_phrases(features, {'mix': np.ones(16)}, np.arange(17.), scale_beats=4, stride_beats=4)
    candidate = [{'start_s': 0, 'end_s': 8, 'role': 'outro', 'label': 'X'},
                 {'start_s': 8, 'end_s': 16, 'role': 'intro', 'label': 'Y'}]
    result = evaluate_structure(ref, candidate, [recurrence])['layers'][0]['phrase_evidence'][0]
    assert result['cross_boundary_window_count'] == 1
    assert [s['complete_windows'] for s in result['span_coverage']] == [1, 1, 1, 0, 0]
    positive = next(b for b in result['pair_summaries'] if b['relation'] == 'separated_return')
    assert positive['identity_id'] == 'A'
    assert positive['variation_relation'] == 'different'
    assert positive['pair_count'] == 1
    assert positive['pattern_similarity']['mean'] == pytest.approx(1)
    assert any(b['relation'] == 'unlabeled_other_group' for b in result['pair_summaries'])
    assert len(result['strongest_return_examples']) == 1
    assert all(b['identity_id'] is not None or b['relation'] == 'unlabeled_other_group' for b in result['pair_summaries'])


def test_adjacent_same_group_is_variation_not_a_distant_return():
    ref = simple_reference()
    ref['layers'][0]['spans'][1]['identity_id'] = 'A'
    result = compare_phrases({'mix': np.ones((2, 16))}, {'mix': np.ones(16)}, np.arange(17.), scale_beats=4, stride_beats=4)
    ev = evaluate_structure(ref, [{'start_s': 0, 'end_s': 16}], [result])
    pairs = ev['layers'][0]['phrase_evidence'][0]['pair_summaries']
    assert {p['relation'] for p in pairs} == {'within_contiguous_group'}


def test_identity_windows_can_cross_variations_without_merging_annotations():
    ref = simple_reference()
    # Two eight-second occurrences, each split into two four-second variations.
    base = ref['layers'][0]['spans'][0]
    ref['source']['duration_s'] = 20
    ref['layers'][0]['spans'] = [
        {**base, 'id': str(i), 'start_s': a, 'end_s': b, 'identity_id': identity, 'variation': variant}
        for i, (a, b, identity, variant) in enumerate([
            (0, 4, 'A', 'low'), (4, 8, 'A', 'high'), (8, 12, 'B', None),
            (12, 16, 'A', 'low'), (16, 20, 'A', 'high')])]
    before = deepcopy(ref)
    result = compare_phrases({'mix': np.ones((2, 20))}, {'mix': np.ones(20)},
                             np.arange(21.), scale_beats=8, stride_beats=4)
    ev = evaluate_structure(ref, [{'start_s': 0, 'end_s': 20}], [result])
    evidence = ev['layers'][0]['phrase_evidence'][0]
    assert ref == before
    assert evidence['identity_window_count'] == evidence['identity_windows_crossing_variations'] == 2
    assert all(s['complete_windows'] == 0 for s in evidence['span_coverage'])
    positive = next(p for p in evidence['pair_summaries'] if p['relation'] == 'separated_return')
    assert positive['pair_count'] == 1
    assert positive['variation_relation'] == 'unknown'
    assert positive['pattern_similarity']['mean'] == pytest.approx(1)
    example = evidence['strongest_return_examples'][0]
    assert example['a_span_id'] is example['b_span_id'] is None
    assert example['a_span_ids'] == ['0', '1']
    assert example['b_span_ids'] == ['3', '4']


def test_unavailable_history_and_silence_are_not_zeroed_into_known_scores():
    ref = simple_reference()
    result = compare_phrases({'mix': np.ones((2, 16))}, {'mix': np.r_[np.ones(4), np.zeros(12)]},
                             np.arange(17.), scale_beats=4, stride_beats=4)
    ev = evaluate_structure(ref, [{'start_s': 0, 'end_s': 16}], [result])
    pair = next(p for p in ev['layers'][0]['phrase_evidence'][0]['pair_summaries'] if p['relation'] == 'separated_return')
    assert pair['pattern_similarity']['count'] == 0
    assert pair['pattern_similarity']['unknown_count'] == 1
    assert pair['pattern_similarity']['mean'] is None
    assert pair['arrangement_similarity']['mean'] == 0


@pytest.mark.parametrize('sections', [[], [{'start_s': 1, 'end_s': 16}],
    [{'start_s': 0, 'end_s': 15}], [{'start_s': False, 'end_s': 16}], [{'start_s': 0, 'end_s': float('nan')}],
    [{'start_s': 0, 'end_s': 10}, {'start_s': 8, 'end_s': 16}]])
def test_malformed_or_wrong_duration_detector_output_rejected(sections):
    with pytest.raises(ValueError):
        evaluate_structure(simple_reference(), sections, [])


def test_report_describes_limits_and_does_not_upgrade_unknowns():
    ref = reference()
    ev = evaluate_structure(ref, [{'start_s': 0, 'end_s': ref['source']['duration_s']}], [])
    report = report_markdown(ref, ev, {'pair_count': 0, 'max_absolute_error': 0})
    assert 'not** claim improved section detection' in report
    assert 'not precision/recall matches' in report
    assert '5 interpreted transition intervals' in report
    assert 'unknown' in report


def test_regression_control_checks_every_pair_and_component():
    result = compare_phrases({'mix': np.ones((2, 12))}, {'mix': np.ones(12)}, np.arange(13.), scale_beats=4, stride_beats=4)
    old = deepcopy(result)
    assert verify_legacy_scores([result], [old])['pair_count'] == 3
    old['pairs'][-1]['stem_similarities']['mix'] = 0
    with pytest.raises(ValueError, match='changed'):
        verify_legacy_scores([result], [old])


def test_build_refuses_existing_and_nested_output_before_other_reads(tmp_path):
    kwargs = dict(feedback=tmp_path/'feedback.json', interpretations=None, parent=tmp_path/'parent', editor=tmp_path/'editor')
    with pytest.raises(FileExistsError):
        build(**kwargs, out=tmp_path)
    with pytest.raises(ValueError, match='separate'):
        build(**kwargs, out=tmp_path/'parent'/'new')
    assert not (tmp_path/'parent'/'new').exists()
