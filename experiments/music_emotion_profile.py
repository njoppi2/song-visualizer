#!/usr/bin/env python3
"""Pinned local mood/theme extraction and fixed reference-based comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def save(path, data):
    with Path(path).open('x') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def validate_resampled_window(wave):
    # The official Essentia resampler can include one extra endpoint sample.
    # Preserve its output without padding/trimming and record the actual extent.
    if wave.ndim != 1 or abs(len(wave) - 160000) > 1 or not np.isfinite(wave).all():
        raise ValueError('unexpected resampling extent or nonfinite values')


def evaluate(features, tracks):
    """Same nearest-favorite cosine rule for every representation; no fitting."""
    features = np.asarray(features, dtype=np.float64)
    for key in ('id', 'isrc', 'sha256'):
        values = [t[key] for t in tracks]
        if any(not v for v in values) or len(set(values)) != len(values):
            raise ValueError(f'duplicate/missing recording {key}')
    if features.ndim != 2 or len(features) != len(tracks) or not np.isfinite(features).all():
        raise ValueError('feature alignment or finite-value failure')
    norms = np.linalg.norm(features, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError('zero vector')
    unit = features / norms[:, None]
    favorites = [i for i, t in enumerate(tracks) if t['label'] == 'favorite']
    others = [i for i, t in enumerate(tracks) if t['label'] in ('like', 'dislike')]
    if len(favorites) < 3 or len(others) != len(tracks) - len(favorites):
        raise ValueError('invalid preference groups')
    pairs = []
    for favorite in favorites:
        refs = [i for i in favorites if i != favorite]
        for other in others:
            scores = []
            for target in (favorite, other):
                similarities = unit[refs] @ unit[target]
                k = int(np.argmax(similarities))
                scores.append({'id': tracks[target]['id'], 'score': float(similarities[k]), 'nearest_reference': tracks[refs[k]]['id']})
            margin = scores[0]['score'] - scores[1]['score']
            pairs.append({'favorite': tracks[favorite]['id'], 'other': tracks[other]['id'], 'other_label': tracks[other]['label'],
                          'references': [tracks[i]['id'] for i in refs], 'queries': scores,
                          'margin': margin, 'result': 'agrees' if margin > 1e-10 else 'disagrees' if margin < -1e-10 else 'tie'})
    summary = {}
    for label in ('dislike', 'like'):
        group = [p for p in pairs if p['other_label'] == label]
        summary[f'favorite_vs_{label}'] = {
            'agrees': sum(p['result'] == 'agrees' for p in group), 'ties': sum(p['result'] == 'tie' for p in group),
            'total': len(group), 'by_favorite': {tracks[i]['id']: sum(p['favorite'] == tracks[i]['id'] and p['result'] == 'agrees' for p in group) for i in favorites}}
    return {'summary': summary, 'pairs': pairs, 'tie_tolerance': 1e-10,
            'limit': 'Selected known development recordings; repeated pairs are not independent trials; no calibrated emotional-intensity score.'}


def score_transfer(features, tracks, reference_ids, query_ids):
    features = np.asarray(features, dtype=np.float64)
    ids = [t['id'] for t in tracks]
    if set(reference_ids) & set(query_ids):
        raise ValueError('query/reference overlap')
    refs = [ids.index(i) for i in reference_ids]
    norms = np.linalg.norm(features, axis=1)
    if not np.isfinite(features).all() or np.any(norms <= 1e-12):
        raise ValueError('invalid vector')
    unit = features / norms[:, None]
    scores = {tid: float(np.max(unit[refs] @ unit[ids.index(tid)])) for tid in query_ids}
    return {'references': reference_ids, 'scores': scores,
            'ranking': sorted(query_ids, key=lambda tid: (-scores[tid], tid)),
            'supported_comparisons': [{'higher': 'milk', 'lower': low,
                'margin': scores['milk'] - scores[low],
                'result': 'agrees' if scores['milk'] - scores[low] > 1e-10 else 'disagrees' if scores['milk'] - scores[low] < -1e-10 else 'tie'} for low in ('move', 'do-i-wanna-know')]}


def compare_representations(representations, tracks):
    results = {name: evaluate(features, tracks) for name, features in representations.items()}
    keep = [i for i, t in enumerate(tracks) if not t.get('version_caveat')]
    sensitivity = {name: evaluate(features[keep], [tracks[i] for i in keep]) for name, features in representations.items()}
    references = [t['id'] for t in tracks[:9] if t['label'] == 'favorite']
    transfer = {name: score_transfer(features, tracks, references, ['milk', 'move', 'do-i-wanna-know']) for name, features in representations.items()}
    return {'full': results, 'without_laurel': sensitivity, 'original_reference_transfer': transfer}


def run(protocol_path, output):
    import essentia
    import essentia.standard as es
    import soundfile as sf

    protocol = json.loads(protocol_path.read_text())
    if sha(__file__) != protocol['runner_sha256']:
        raise ValueError('runner differs from protocol')
    for row in protocol['input_files']:
        if sha(ROOT / row['path']) != row['sha256']:
            raise ValueError(f'input mismatch: {row["path"]}')
    for manifest in protocol['parent_manifests']:
        for row in json.loads((ROOT / manifest).read_text())['files']:
            if sha(ROOT / row['path']) != row['sha256']:
                raise ValueError('parent provenance mismatch')
    for row in protocol['models']:
        if sha(ROOT / row['path']) != row['sha256']:
            raise ValueError('model mismatch')
    tracks = protocol['tracks']
    for track in tracks:
        if sha(track['path']) != track['sha256']:
            raise ValueError('audio source mismatch')
    output.mkdir(exist_ok=False)
    begun = time.perf_counter()
    runtime = {'protocol_sha256': sha(protocol_path), 'runner_sha256': sha(__file__), 'essentia': essentia.__version__, 'numpy': np.__version__, 'device': 'cpu'}
    try:
        embedding = es.TensorflowPredictEffnetDiscogs(graphFilename=str(ROOT / protocol['models'][0]['path']),
            output='PartitionedCall:1', batchSize=64, patchSize=128, patchHopSize=62, lastPatchMode='repeat')
        mood = es.TensorflowPredict2D(graphFilename=str(ROOT / protocol['models'][1]['path']), input='model/Placeholder', output='model/Sigmoid')
        profile_vectors, backbone_vectors, times, profile_rows = [], [], [], []
        classes = json.loads((ROOT / protocol['mood_metadata']).read_text())['classes']
        if len(classes) != 56:
            raise ValueError('unexpected mood output vocabulary')
        for track in tracks:
            start = time.perf_counter()
            info = sf.info(track['path'])
            if (info.frames, info.samplerate, info.channels) != (track['frames'], track['sample_rate'], 2):
                raise ValueError('audio geometry changed')
            resample = es.Resample(inputSampleRate=info.samplerate, outputSampleRate=16000, quality=4)
            window_moods, window_embeddings, bounds, patch_counts, sample_counts = [], [], [], [], []
            with sf.SoundFile(track['path']) as handle:
                for j in range(10):
                    if time.perf_counter() - begun > 900:
                        raise TimeoutError('15 minute budget exceeded')
                    center = round(info.frames * (.05 + j * .1))
                    a, b = center - 5*info.samplerate, center + 5*info.samplerate
                    if not 0 <= a < b <= info.frames:
                        raise ValueError('invalid source support')
                    handle.seek(a)
                    pcm = handle.read(b-a, dtype='float32', always_2d=True)
                    if pcm.shape != (b-a, 2) or not np.isfinite(pcm).all():
                        raise ValueError('invalid PCM')
                    wave = resample(pcm.mean(axis=1, dtype=np.float32))
                    validate_resampled_window(wave)
                    features = embedding(wave)
                    probabilities = mood(features)
                    if features.ndim != 2 or features.shape[1] != 1280 or probabilities.shape != (len(features), 56) or not len(features):
                        raise ValueError(f'output shape mismatch: {features.shape}, {probabilities.shape}')
                    if not np.isfinite(features).all() or not np.isfinite(probabilities).all() or np.min(probabilities) < 0 or np.max(probabilities) > 1:
                        raise ValueError('nonfinite/out-of-range output')
                    window_moods.append(probabilities.mean(axis=0))
                    window_embeddings.append(features.mean(axis=0))
                    patch_counts.append(len(features))
                    sample_counts.append(len(wave))
                    bounds.append([a,b])
            wm, we = np.stack(window_moods), np.stack(window_embeddings)
            np.savez_compressed(output / f'{track["id"]}.npz', mood_windows=wm, backbone_windows=we,
                                source_frame_bounds=np.array(bounds, dtype=np.int64), patch_counts=np.array(patch_counts), resampled_sample_counts=np.array(sample_counts))
            profile_vectors.append(wm.mean(axis=0))
            backbone_vectors.append(we.mean(axis=0))
            top = np.argsort(profile_vectors[-1])[::-1][:5]
            profile_rows.append({'id': track['id'], 'label': track['label'], 'scores': dict(zip(classes, map(float, profile_vectors[-1]))),
                                 'top_five': [{'tag': classes[i], 'score': float(profile_vectors[-1][i])} for i in top]})
            times.append({'id': track['id'], 'seconds': time.perf_counter()-start, 'patch_counts': patch_counts})
            print(json.dumps({'completed': len(times), 'total': len(tracks), **times[-1]}), flush=True)
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 6*1024**3:
                raise MemoryError('6 GiB RSS limit exceeded')
        profiles, backbone = np.stack(profile_vectors), np.stack(backbone_vectors)
        np.savez_compressed(output / 'song-vectors.npz', mood=profiles, backbone=backbone)
        save(output / 'profiles.json', {'classes': classes, 'tracks': profile_rows,
             'meaning': 'Uncalibrated model mood/theme scores, not personal emotion intensity or probability of liking.'})
        parent = ROOT / 'outputs/reviews/music-discovery-pilot-01'
        with np.load(parent / 'run/song-vectors.npz', allow_pickle=False) as archive:
            muq_old = archive['muq']
        with np.load(ROOT / 'outputs/reviews/music-discovery-transfer-01/run/query-vectors.npz', allow_pickle=False) as archive:
            muq_new = archive['muq']
        muq = np.vstack([muq_old, muq_new])
        comparison = compare_representations({'muq': muq, 'effnet_backbone': backbone, 'mood_profile': profiles}, tracks)
        # Original positive-only MuQ control on nine recordings must remain 8/20.
        old_control = evaluate(muq[:9], tracks[:9])
        if old_control['summary']['favorite_vs_dislike']['agrees'] != 8:
            raise ValueError('original MuQ control failed')
        save(output / 'comparison.json', comparison)
        save(output / 'baseline-control.json', old_control)
        for track in tracks:
            if sha(track['path']) != track['sha256']:
                raise ValueError('source changed during run')
        for row in protocol['input_files']:
            if sha(ROOT / row['path']) != row['sha256']:
                raise ValueError('input changed during run')
        runtime.update(status='complete', seconds=time.perf_counter()-begun, track_times=times,
                       maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, source_hashes_verified_before_after=True)
    except Exception as exc:
        runtime.update(status='failed', error=f'{type(exc).__name__}: {exc}', seconds=time.perf_counter()-begun)
        raise
    finally:
        save(output / 'runtime.json', runtime)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.output)
