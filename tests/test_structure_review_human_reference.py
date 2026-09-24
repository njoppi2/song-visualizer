"""Focused tests for the frozen human-reference structure review overlay."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import build_structure_review as builder
from songviz.ingest import sha256_file


ROOT = Path(__file__).parents[1]
ANNOTATIONS = ROOT / 'benchmark/feedback/section-editor-02.json'
FEEDBACK = ROOT / 'benchmark/feedback/listening-examples-01.json'
LISTENING_REVIEW = ROOT / 'outputs/reviews/listening-examples-01/review.json'
PARENT_REVIEW = ROOT / 'outputs/reviews/structure-review-03/review.json'


def _prepared() -> tuple[dict, dict, dict, dict]:
    annotation = json.loads(ANNOTATIONS.read_text())
    feedback = json.loads(FEEDBACK.read_text())
    listening = json.loads(LISTENING_REVIEW.read_text())
    parent = json.loads(PARENT_REVIEW.read_text())
    data = builder.prepare_human_reference_data(
        annotation, annotation_sha256=builder.HUMAN_ANNOTATIONS_SHA256,
        listening_review=listening, feedback=feedback,
        source_audio_sha256=annotation['source']['source_audio_sha256'],
        audio_sha256=annotation['source']['audio_sha256'], duration_s=annotation['source']['duration_s'],
        candidate_sections=parent['sections']['candidate'],
    )
    return data, annotation, feedback, parent


def test_human_spans_and_raw_feedback_stay_source_matched_and_verbatim():
    data, annotation, feedback, parent = _prepared()

    _, _, source_record, audio_record, duration = builder._verify_overlay_parent(PARENT_REVIEW.parent)
    assert source_record['sha256'] == annotation['source']['source_audio_sha256']
    assert audio_record['sha256'] == annotation['source']['audio_sha256']
    assert duration == annotation['source']['duration_s']
    assert sha256_file(ANNOTATIONS) == builder.HUMAN_ANNOTATIONS_SHA256
    assert sha256_file(FEEDBACK) == builder.LISTENING_FEEDBACK_SHA256
    _, _, chained_source, chained_audio, chained_duration = builder._verify_overlay_parent(
        ROOT / 'outputs/reviews/structure-review-05'
    )
    assert (chained_source['sha256'], chained_audio['sha256'], chained_duration) == (
        source_record['sha256'], audio_record['sha256'], duration
    )

    human = data['human_sections']
    assert len(human) == 19
    assert all(item['certainty'] == 'unspecified' for item in human)
    assert human[0]['start_s'] == 0
    assert human[-1]['end_s'] == annotation['source']['duration_s']
    assert all(left['end_s'] == right['start_s'] for left, right in zip(human, human[1:]))
    assert human[0]['motif'] is None and human[0]['motif_id'] is None
    assert next(item for item in human if item['motif'] == 'chorus')['motif_id'] == 'layer-0-identity-1'

    cards = {card['id']: card for card in data['listening_feedback']}
    answers = {answer['example_id']: answer for answer in feedback['answers']}
    assert set(cards) == set(builder.LISTENING_EXAMPLE_IDS)
    assert {key: (card['start_s'], card['end_s']) for key, card in cards.items()} == {
        'within-passage': (16.0, 26.0), 'transition-extent': (57.0, 70.0),
        'drum-entry': (74.0, 85.0), 'verse-ending': (119.0, 132.0),
    }
    assert {key: card['notes'] for key, card in cards.items()} == {key: answer['notes'] for key, answer in answers.items()}
    assert {key: card['perceived_change'] for key, card in cards.items()} == {key: answer['perceived_change'] for key, answer in answers.items()}
    assert cards['within-passage']['perceived_change'] == 'none'
    comparison_terms = ('candidate', 'baseline', 'section')
    assert all(not any(term in key for term in comparison_terms) for card in cards.values() for key in card)


def test_declared_timing_diagnostic_has_all_13_expected_signed_offsets():
    data, _, _, _ = _prepared()

    disagreements = data['boundary_diagnostic']['disagreements']
    assert data['boundary_diagnostic']['threshold_s'] == 1.0
    assert len(disagreements) == 13
    actual = [(item['human_boundary_s'], item['candidate_boundary_s'], item['candidate_minus_human_s']) for item in disagreements]
    expected = [
        (5.35001, 6.989206349206349, 1.6391963492063488),
        (30.467096, 6.989206349206349, -23.477889650793652),
        (33.444519, 6.989206349206349, -26.45531265079365),
        (61.368411, 64.36571428571429, 2.997303285714287),
        (92.815909, 79.08716553287982, -13.72874346712018),
        (95.398393, 79.08716553287982, -16.31122746712018),
        (124.295069, 137.99619047619046, 13.701121476190454),
        (144.177619, 137.99619047619046, -6.18142852380955),
        (158.590911, 165.7208163265306, 7.129905326530603),
        (187.440852, 165.7208163265306, -21.720035673469385),
        (189.926103, 165.7208163265306, -24.205286673469388),
        (203.758902, 165.7208163265306, -38.0380856734694),
        (217.973095, 165.7208163265306, -52.2522786734694),
    ]
    assert len(actual) == len(expected)
    for found, wanted in zip(actual, expected):
        assert found == pytest.approx(wanted)
    assert all(abs(item['candidate_minus_human_s']) > 1.0 for item in disagreements)


def test_rejects_human_export_bound_to_another_source():
    _, annotation, feedback, parent = _prepared()
    changed = deepcopy(annotation)
    changed['source']['source_audio_sha256'] = '0' * 64

    with pytest.raises(ValueError, match='not bound to the parent source audio'):
        builder.prepare_human_reference_data(
            changed, annotation_sha256=builder.HUMAN_ANNOTATIONS_SHA256,
            listening_review=json.loads(LISTENING_REVIEW.read_text()), feedback=feedback,
            source_audio_sha256=annotation['source']['source_audio_sha256'],
            audio_sha256=annotation['source']['audio_sha256'], duration_s=annotation['source']['duration_s'],
            candidate_sections=parent['sections']['candidate'],
        )


def test_listening_window_lane_preserves_four_raw_ranges_without_event_or_section_derivations():
    data, _, feedback, _ = _prepared()

    lane = data['listening_window_lane']
    answers = {answer['example_id']: answer for answer in feedback['answers']}
    assert len(lane) == 4
    assert {window['listener_state'] for window in lane} == {'change_heard', 'none_control'}
    assert sum(window['listener_state'] == 'change_heard' for window in lane) == 3
    assert sum(window['listener_state'] == 'none_control' for window in lane) == 1
    assert {window['id']: (window['start_s'], window['end_s']) for window in lane} == {
        'within-passage': (16.0, 26.0), 'transition-extent': (57.0, 70.0),
        'drum-entry': (74.0, 85.0), 'verse-ending': (119.0, 132.0),
    }
    assert {window['id']: window['perceived_change'] for window in lane} == {key: answer['perceived_change'] for key, answer in answers.items()}
    assert {window['id']: window['notes'] for window in lane} == {key: answer['notes'] for key, answer in answers.items()}
    assert all(set(window) == {'id', 'start_s', 'end_s', 'perceived_change', 'notes', 'listener_state'} for window in lane)
    prohibited = ('event', 'point', 'extent', 'focus', 'candidate', 'baseline', 'section', 'agreement', 'certainty')
    assert all(not any(word in key for word in prohibited) for window in lane for key in window)
    cards = data['listening_feedback']
    assert all(not any(word in key for word in ('candidate', 'baseline', 'section')) for card in cards for key in card)


def test_page_declares_review_windows_as_non_events_and_keeps_listening_certainty_unrecorded():
    page = (ROOT / 'experiments/templates/structure_review.html').read_text()

    assert 'id="listening-window-lane"' in page
    assert 'These are listener review windows, not event boundaries or durations.' in page
    assert 'Remaining song time is not reviewed by these prompts.' in page
    assert 'Listener marked a change' in page and 'Listener marked none (control)' in page
    assert 'Not reviewed by these prompts' in page
    assert 'Certainty: not recorded in the listening export.' in page
    assert 'candidate_boundary_summary' not in page
    assert '.review-window{position:absolute;' in page
    assert 'border:0;border-radius:18px' in page
    assert 'linear-gradient(90deg,transparent 0' in page
