"""Dimension-separated development diagnostics, not a ground-truth scorer.

Annotations only select evaluation examples. They never enter acoustic extraction
or recurrence prediction. No timing tolerance, negative identity labels or formal
hierarchy is inferred from an editor layer.
"""
from __future__ import annotations

import math
import statistics


def _summary(values: list[float | None]) -> dict:
    known = [v for v in values if v is not None]
    return {"count": len(known), "unknown_count": len(values) - len(known),
            "mean": statistics.mean(known) if known else None,
            "median": statistics.median(known) if known else None,
            "min": min(known) if known else None, "max": max(known) if known else None}


def _validate_sections(sections: list[dict], duration: float) -> None:
    if not sections:
        raise ValueError("Expected detector sections")
    previous = 0.0
    for section in sections:
        a, b = section['start_s'], section['end_s']
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in [a, b])
                or a < 0 or b > duration + 1e-6 or b <= a or abs(a - previous) > 1e-6):
            raise ValueError("Detector sections must partition the annotated duration")
        previous = b
    if abs(previous - duration) > 1e-6:
        raise ValueError("Detector duration differs from annotated duration")


def evaluate_structure(reference: dict, sections: list[dict], recurrence: list[dict]) -> dict:
    """Compare unchanged cuts and acoustic evidence with explicit development groups.

    Identity phrases may cross adjacent spans in the same explicit motif group;
    variation phrases must lie inside one annotated span. Unsupported spans are
    reported, not silently resized. Scores
    across differently named groups are unlabeled contrasts, never negatives.
    """
    _validate_sections(sections, reference['source']['duration_s'])
    cuts = [s['start_s'] for s in sections[1:]]
    layers = []
    for layer in reference['layers']:
        spans = layer['spans']
        by_id = {s['id']: s for s in spans}
        boundaries, overlaps, transitions = [], [], []
        # Occurrence is only an evaluation grouping of adjacent explicit motifs;
        # this never merges spans or claims a parent/child musical hierarchy.
        occurrence, run, previous_identity = {}, 0, None
        for span in spans:
            identity = span['identity_id']
            if identity is None or identity != previous_identity:
                run += 1
            occurrence[span['id']] = run
            previous_identity = identity
            overlaps.append({'span_id': span['id'], 'label': span['label'],
                'candidate_overlaps': [
                    {'section_index': i, 'role': s.get('role'), 'legacy_label': s.get('label'),
                     'overlap_s': min(span['end_s'], s['end_s']) - max(span['start_s'], s['start_s'])}
                    for i, s in enumerate(sections)
                    if min(span['end_s'], s['end_s']) > max(span['start_s'], s['start_s'])]})
            if span['transition'] is True:
                transitions.append({'span_id': span['id'], 'start_s': span['start_s'], 'end_s': span['end_s'],
                    'duration_s': span['end_s'] - span['start_s'], 'certainty': span['certainty'],
                    'source': 'explicit_analyst_mapping', 'prediction_status': 'not_implemented'})
        for left, right in zip(spans, spans[1:]):
            known = left['identity_id'] is not None and right['identity_id'] is not None
            same = known and left['identity_id'] == right['identity_id']
            t = right['start_s']
            nearest = min(cuts, key=lambda c: (abs(c-t), c)) if cuts else None
            boundaries.append({'time_s': t, 'left_span_id': left['id'], 'right_span_id': right['id'],
                'identity_relation': 'same_explicit_group' if same else 'different_named_groups' if known else 'unknown',
                'variation_change': left['variation'] != right['variation']
                    if same and left['variation'] is not None and right['variation'] is not None else None,
                'transition_start': right['transition'] is True, 'transition_end': left['transition'] is True,
                'nearest_detector_boundary_s': nearest,
                'detector_minus_annotation_s': nearest-t if nearest is not None else None})
        scales = []
        identity_runs = []
        for span in spans:
            if span['identity_id'] is None:
                continue
            if identity_runs and identity_runs[-1]['occurrence'] == occurrence[span['id']]:
                identity_runs[-1]['end_s'] = span['end_s']
                identity_runs[-1]['span_ids'].append(span['id'])
            else:
                identity_runs.append({'identity_id': span['identity_id'], 'occurrence': occurrence[span['id']],
                    'start_s': span['start_s'], 'end_s': span['end_s'], 'span_ids': [span['id']]})
        for result in recurrence:
            owners = []
            identity_owners = []
            for window in result['spans']:
                owners.append(next((s['id'] for s in spans
                    if window['start_s'] >= s['start_s'] and window['end_s'] <= s['end_s']), None))
                identity_owners.append(next((i for i, run_span in enumerate(identity_runs)
                    if window['start_s'] >= run_span['start_s'] and window['end_s'] <= run_span['end_s']), None))
            buckets, examples = {}, []
            for pair in result['pairs']:
                a_id, b_id = owners[pair['a']], owners[pair['b']]
                a_run, b_run = identity_owners[pair['a']], identity_owners[pair['b']]
                if a_run is None or b_run is None:
                    continue
                a, b = identity_runs[a_run], identity_runs[b_run]
                same = a['identity_id'] == b['identity_id']
                relation = ('within_contiguous_group' if a_run == b_run
                            else 'separated_return') if same else 'unlabeled_other_group'
                a_variation = by_id[a_id]['variation'] if a_id is not None else None
                b_variation = by_id[b_id]['variation'] if b_id is not None else None
                variation = ('same' if a_variation == b_variation else 'different') if (
                    a_variation is not None and b_variation is not None) else 'unknown'
                group = a['identity_id'] if same else None
                key = (group, relation, variation)
                buckets.setdefault(key, []).append(pair)
                if same and relation == 'separated_return':
                    examples.append({'a_span_id': a_id, 'b_span_id': b_id,
                        'a_span_ids': [s for s in a['span_ids'] if by_id[s]['start_s'] < result['spans'][pair['a']]['end_s'] and by_id[s]['end_s'] > result['spans'][pair['a']]['start_s']],
                        'b_span_ids': [s for s in b['span_ids'] if by_id[s]['start_s'] < result['spans'][pair['b']]['end_s'] and by_id[s]['end_s'] > result['spans'][pair['b']]['start_s']],
                        'a_start_s': result['spans'][pair['a']]['start_s'],
                        'b_start_s': result['spans'][pair['b']]['start_s'],
                        'identity_id': group, 'variation_relation': variation,
                        'pattern_similarity': pair['pattern_similarity'],
                        'arrangement_similarity': pair['arrangement_similarity'],
                        'legacy_similarity': pair['similarity'],
                        'shared_active_stems': pair['shared_active_stems']})
            summaries = []
            for (group, relation, variation), pairs in buckets.items():
                summaries.append({'identity_id': group, 'relation': relation, 'variation_relation': variation,
                    'pair_count': len(pairs), **{key: _summary([p[key] for p in pairs])
                        for key in ['pattern_similarity', 'arrangement_similarity', 'similarity']}})
            # Show the strongest observed pattern example for each user span pair,
            # not just a top-N list dominated by one long section. This selection
            # is explicitly illustrative, not representative or an accuracy metric.
            best = {}
            for ex in examples:
                if ex['pattern_similarity'] is None:
                    continue
                key = (tuple(ex['a_span_ids']), tuple(ex['b_span_ids']))
                if key not in best or ex['pattern_similarity'] > best[key]['pattern_similarity']:
                    best[key] = ex
            context_summaries = []
            for span in spans:
                contexts = [c for c in result['context'] if owners[c['span']] == span['id']]
                context_summaries.append({'span_id': span['id'], 'label': span['label'],
                    'window_count': len(contexts), **{key: _summary([c[key] for c in contexts]) for key in
                    ['local_pattern_change', 'local_arrangement_change', 'historical_pattern_novelty']}})
            scales.append({'scale_beats': result['scale_beats'], 'stride_beats': result['stride_beats'],
                'window_count': len(owners), 'cross_boundary_window_count': owners.count(None),
                'identity_window_count': sum(i is not None for i in identity_owners),
                'identity_windows_crossing_variations': sum(i is not None and s is None for i, s in zip(identity_owners, owners)),
                'identity_run_coverage': [{**run_span, 'complete_windows': identity_owners.count(i)} for i, run_span in enumerate(identity_runs)],
                'span_coverage': [{'span_id': s['id'], 'label': s['label'], 'complete_windows': owners.count(s['id'])} for s in spans],
                'pair_summaries': summaries, 'strongest_return_examples': list(best.values()),
                'context_summaries': context_summaries})
        layers.append({'id': layer['id'], 'name': layer['name'], 'boundary_evidence': boundaries,
            'legacy_section_overlap': overlaps, 'transition_intervals': transitions, 'phrase_evidence': scales})
    return {'schema_version': 1, 'kind': 'songviz-structural-development-evaluation',
        'source': reference['source'], 'layers': layers,
        'capabilities': {'legacy_boundaries': 'compared_without_matching_threshold',
            'musical_identity': 'acoustic_candidates_only', 'variation': 'acoustic_level_and_activity_evidence',
            'transition_intervals': 'representation_only_no_detector',
            'novelty': 'separate_local_and_prior_phrase_context_not_calibrated_surprise'},
        'limitations': [
            'Development examples only; no holdout quality or accuracy claim.',
            'Raw motif assertions and analyst variation/transition interpretations are distinct sources.',
            'Different named groups are unlabeled contrasts, not certified negatives; blank motifs are unknown.',
            'Nearest cut distances are descriptive, many-to-one and have no accepted timing tolerance.',
            'All pairs are correlated sliding windows; counts are not independent sample sizes.',
            'Strongest return examples are selected maxima, not representative quality scores.',
            'Complete 16/32-beat windows cannot assess many short transitions or variations.',
            'Identity windows may cross adjacent spans with the same explicit motif; variation evidence is unknown for these cross-subdivision windows.',
            'Context compares only prior non-overlapping phrases, but uses offline whole-song audibility calibration and the entire target window; not a realtime onset signal.',
            'Legacy roles/letters are never treated as identity predictions; no automatic identity or transition classifier exists here.']}
