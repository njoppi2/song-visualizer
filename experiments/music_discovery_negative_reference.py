#!/usr/bin/env python3
"""Saved-vector positive/negative-reference diagnostic with paired exclusions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TIE_TOLERANCE = 1e-10


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def validate_tracks(tracks):
    for field in ('id', 'sha256', 'isrc'):
        entries = [t[field] for t in tracks]
        if any(not x for x in entries) or len(set(entries)) != len(entries):
            raise ValueError(f'duplicate or missing {field}')
    labels = [t['label'] for t in tracks]
    if set(labels) != {'favorite', 'dislike'} or labels.count('favorite') < 3 or labels.count('dislike') < 2:
        raise ValueError('need at least three favorites and two explicit dislikes')


def score_queries(features, positives, negatives, queries, kind):
    """References and queries are separate; descriptor scaling uses positives only.

    Scores: nearest-positive similarity, and nearest-positive similarity minus
    nearest-negative similarity. Descriptors use negative standardized distance.
    Equal weights and the scale floor are fixed, not optimized from outcomes.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError('expected finite feature matrix')
    if not positives or not negatives or not queries:
        raise ValueError('empty references or queries')
    if set(positives) & set(negatives) or set(queries) & (set(positives) | set(negatives)):
        raise ValueError('reference/query overlap')
    if any(i < 0 or i >= len(features) for i in [*positives, *negatives, *queries]):
        raise ValueError('invalid row index')
    if kind == 'muq':
        norms = np.linalg.norm(features, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError('zero embedding')
        vectors = features / norms[:, None]
    elif kind == 'simple':
        train = features[positives]
        vectors = (features - train.mean(axis=0)) / np.maximum(train.std(axis=0, ddof=0), 1e-8)
    else:
        raise ValueError('unknown feature kind')
    rows = []
    for query in queries:
        similarities = []
        for refs in (positives, negatives):
            values = vectors[refs] @ vectors[query] if kind == 'muq' else -np.linalg.norm(vectors[refs] - vectors[query], axis=1)
            best = int(np.argmax(values))
            similarities.append((float(values[best]), refs[best]))
        (sp, ip), (sn, inn) = similarities
        rows.append({'query_index': query, 'positive_similarity': sp, 'negative_similarity': sn,
                     'nearest_positive_index': ip, 'nearest_negative_index': inn,
                     'positive_only': sp, 'negative_aware': sp - sn})
    return rows


def verdict(margin):
    return 'favorite_higher' if margin > TIE_TOLERANCE else 'dislike_higher' if margin < -TIE_TOLERANCE else 'tie'


def evaluate(muq, simple, tracks):
    validate_tracks(tracks)
    if any(np.asarray(f).ndim != 2 or len(f) != len(tracks) for f in (muq, simple)):
        raise ValueError('track/feature alignment mismatch')
    positives = sorted((i for i, t in enumerate(tracks) if t['label'] == 'favorite'), key=lambda i: tracks[i]['id'])
    negatives = sorted((i for i, t in enumerate(tracks) if t['label'] == 'dislike'), key=lambda i: tracks[i]['id'])
    folds = []
    for positive in positives:
        for negative in negatives:
            pr = [i for i in positives if i != positive]
            nr = [i for i in negatives if i != negative]
            row = {'favorite': tracks[positive]['id'], 'dislike': tracks[negative]['id'],
                   'positive_references': [tracks[i]['id'] for i in pr],
                   'negative_references': [tracks[i]['id'] for i in nr], 'methods': {}}
            for kind, features in [('muq', muq), ('simple', simple)]:
                scored = score_queries(features, pr, nr, [positive, negative], kind)
                details = []
                for score in scored:
                    details.append({
                        'id': tracks[score['query_index']]['id'],
                        'positive_similarity': score['positive_similarity'],
                        'negative_similarity': score['negative_similarity'],
                        'nearest_positive': tracks[score['nearest_positive_index']]['id'],
                        'nearest_negative': tracks[score['nearest_negative_index']]['id'],
                        'positive_only': score['positive_only'],
                        'negative_aware': score['negative_aware'],
                    })
                row['methods'][kind] = {'queries': details, 'comparisons': {
                    method: {'margin': details[0][method] - details[1][method],
                             'result': verdict(details[0][method] - details[1][method])}
                    for method in ('positive_only', 'negative_aware')}}
            folds.append(row)
    summary = {}
    for kind in ('muq', 'simple'):
        summary[kind] = {}
        for method in ('positive_only', 'negative_aware'):
            results = [f['methods'][kind]['comparisons'][method]['result'] for f in folds]
            summary[kind][method] = {
                'favorite_higher': results.count('favorite_higher'), 'dislike_higher': results.count('dislike_higher'),
                'ties': results.count('tie'), 'comparisons': len(results),
                'by_favorite': {tracks[i]['id']: sum(f['favorite'] == tracks[i]['id'] and f['methods'][kind]['comparisons'][method]['result'] == 'favorite_higher' for f in folds) for i in positives},
                'by_dislike': {tracks[i]['id']: sum(f['dislike'] == tracks[i]['id'] and f['methods'][kind]['comparisons'][method]['result'] == 'favorite_higher' for f in folds) for i in negatives},
            }
        summary[kind]['paired_changes'] = {
            'improved': sum(f['methods'][kind]['comparisons']['positive_only']['result'] == 'dislike_higher' and f['methods'][kind]['comparisons']['negative_aware']['result'] == 'favorite_higher' for f in folds),
            'worsened': sum(f['methods'][kind]['comparisons']['positive_only']['result'] == 'favorite_higher' and f['methods'][kind]['comparisons']['negative_aware']['result'] == 'dislike_higher' for f in folds),
        }
    return {'tie_absolute_tolerance': TIE_TOLERANCE, 'summary': summary, 'folds': folds,
            'limits': ['Selected previously seen nine recordings; exploratory development analysis.',
                       'Repeated recordings and overlapping references mean pairwise comparisons are not independent.',
                       'No prospective discovery, calibrated emotional score or model/weight tuning.']}


def run(protocol_path, output):
    started = time.perf_counter()
    protocol = json.loads(protocol_path.read_text())
    if sha(__file__) != protocol['runner_sha256']:
        raise ValueError('runner differs from frozen protocol')
    for receipt in protocol['input_files']:
        if sha(ROOT / receipt['path']) != receipt['sha256']:
            raise ValueError(f'input hash mismatch: {receipt["path"]}')
    parent = ROOT / protocol['parent_directory']
    parent_manifest = json.loads((parent / 'manifest.json').read_text())
    for receipt in parent_manifest['files']:
        if sha(ROOT / receipt['path']) != receipt['sha256']:
            raise ValueError(f'parent manifest mismatch: {receipt["path"]}')
    tracks = json.loads((parent / 'protocol.json').read_text())['tracks']
    with np.load(parent / 'run/song-vectors.npz', allow_pickle=False) as archive:
        muq, simple = archive['muq'], archive['simple']
    output.mkdir(exist_ok=False)
    full = evaluate(muq, simple, tracks)
    save(output / 'evaluation.json', full)
    keep = [i for i, t in enumerate(tracks) if not t.get('version_caveat')]
    save(output / 'version-sensitivity.json', evaluate(muq[keep], simple[keep], [tracks[i] for i in keep]))
    # A matched positive-only control must reproduce the prior pair decisions.
    prior = json.loads((parent / 'run/evaluation.json').read_text())
    parity = {kind: full['summary'][kind]['positive_only']['favorite_higher'] == prior['summary'][kind]['pairwise_wins'] for kind in ('muq', 'simple')}
    if not all(parity.values()):
        raise ValueError('matched positive-only controls do not reproduce parent result')
    for receipt in protocol['input_files']:
        if sha(ROOT / receipt['path']) != receipt['sha256']:
            raise ValueError('source changed during saved-vector analysis')
    save(output / 'runtime.json', {'status': 'complete', 'protocol_sha256': sha(protocol_path),
         'runner_sha256': sha(__file__), 'numpy': np.__version__, 'device': 'cpu',
         'elapsed_seconds': time.perf_counter() - started,
         'audio_decoded': False, 'model_inference': False, 'positive_only_parent_count_parity': parity})
    print(json.dumps({'full': full['summary'], 'output': str(output)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.output)
