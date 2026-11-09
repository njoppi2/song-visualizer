#!/usr/bin/env python3
"""Offline nine-recording diagnostic; frozen features and saved-vector replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / '.songviz/representation-models/muq-large-msd-iter-0562a57814f6'
MODEL_PINS = {
    'config.json': '237335ee27d8fb951ce778701a12a79e06c51ae636dd786f97e45f51ce532543',
    'model.safetensors': '273febab2be02872c37d2c37e48a9d6c52c1c9392f3eeeabd498efa281ccb7a6',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    with Path(path).open('x') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def check_recordings(tracks):
    for key in ('id', 'sha256', 'isrc'):
        values = [t[key] for t in tracks]
        if any(not x for x in values) or len(set(values)) != len(values):
            raise ValueError(f'duplicate or missing recording {key}')
    if {t['label'] for t in tracks} != {'favorite', 'dislike'}:
        raise ValueError('diagnostic requires explicit favorite/dislike labels')


def evaluate_vectors(muq, simple, tracks):
    """Each held-out favorite competes with dislikes using four OTHER favorites.

    Descriptor scaling sees only reference favorites; all target candidates are
    excluded from the scaling fit. No model is trained and no labels are inferred.
    """
    check_recordings(tracks)
    if any(v.ndim != 2 or len(v) != len(tracks) or not np.isfinite(v).all() for v in (muq, simple)):
        raise ValueError('expected finite song-level feature matrices')
    norms = np.linalg.norm(muq, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError('zero embedding')
    unit = muq / norms[:, None]
    favorites = [i for i, t in enumerate(tracks) if t['label'] == 'favorite']
    dislikes = [i for i, t in enumerate(tracks) if t['label'] == 'dislike']
    if len(favorites) < 3 or not dislikes:
        raise ValueError('insufficient labeled tracks')
    folds = []
    for heldout in favorites:
        refs = [i for i in favorites if i != heldout]
        targets = [heldout, *dislikes]
        train = simple[refs]
        scale = np.maximum(train.std(axis=0, ddof=0), 1e-8)
        # Centering cancels for Euclidean distance. Standard deviations are fit
        # only to the four reference tracks, never the query or rejected songs.
        normalized = (simple - train.mean(axis=0)) / scale
        fold = {'heldout': tracks[heldout]['id'], 'references': [tracks[i]['id'] for i in refs], 'methods': {}}
        for method in ('muq', 'simple'):
            candidates = []
            for target in targets:
                scores = (unit[refs] @ unit[target]) if method == 'muq' else -np.linalg.norm(normalized[refs] - normalized[target], axis=1)
                best = int(np.argmax(scores))
                candidates.append({'id': tracks[target]['id'], 'label': tracks[target]['label'], 'score': float(scores[best]), 'nearest_reference': tracks[refs[best]]['id']})
            candidates.sort(key=lambda r: (-r['score'], r['id']))
            positive = next(r for r in candidates if r['id'] == tracks[heldout]['id'])
            negatives = [r for r in candidates if r['label'] == 'dislike']
            fold['methods'][method] = {
                'candidates': candidates,
                'favorite_rank': 1 + sum(r['score'] > positive['score'] for r in negatives),
                'pairwise_wins': sum(positive['score'] > r['score'] for r in negatives),
                'pairwise_ties': sum(abs(positive['score'] - r['score']) <= 1e-12 for r in negatives),
                'negative_count': len(negatives),
            }
        folds.append(fold)
    return {'folds': folds, 'summary': {method: {
        'favorite_ranks': [f['methods'][method]['favorite_rank'] for f in folds],
        'pairwise_wins': sum(f['methods'][method]['pairwise_wins'] for f in folds),
        'pairwise_ties': sum(f['methods'][method]['pairwise_ties'] for f in folds),
        'pairwise_comparisons': len(favorites) * len(dislikes),
    } for method in ('muq', 'simple')},
        'limitations': ['Exploratory selected known tracks; no prospective discovery evaluation.', 'The same disliked recordings recur across folds; pairwise comparisons are not independent trials.', 'Different favorite seeds are used in different folds. No tuning or model training.']}


def descriptors(wave):
    import librosa
    y = wave.numpy()[0]
    spectrum = np.abs(librosa.stft(y, n_fft=2048, hop_length=512, center=True))
    power = spectrum ** 2
    mel = librosa.feature.melspectrogram(S=power, sr=24000, n_mels=128)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel, ref=1.0, top_db=80.0), n_mfcc=13)
    features = [mfcc,
                librosa.feature.spectral_centroid(S=spectrum, sr=24000),
                librosa.feature.spectral_bandwidth(S=spectrum, sr=24000),
                librosa.feature.spectral_flatness(S=spectrum),
                20 * np.log10(np.maximum(librosa.feature.rms(S=spectrum, frame_length=2048), 1e-10)),
                librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)]
    vector = np.concatenate([stat for x in features for stat in (x.mean(axis=1), x.std(axis=1, ddof=0))])
    if vector.shape != (36,) or not np.isfinite(vector).all():
        raise ValueError('invalid simple descriptors')
    return vector.astype(np.float32)


def run(protocol_path, output):
    import importlib.metadata
    import muq
    import soundfile as sf
    import torch
    import torchaudio

    if any(os.environ.get(k) != '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')):
        raise ValueError('offline flags required')
    protocol = json.loads(protocol_path.read_text())
    tracks = protocol['tracks']
    check_recordings(tracks)
    if protocol['runner_sha256'] != sha(__file__):
        raise ValueError('runner differs from frozen protocol')
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    runtime = {'protocol_sha256': sha(protocol_path), 'runner_sha256': sha(__file__), 'started_at_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(), 'device': 'cuda', 'dtype': 'float32', 'threads': 4, 'versions': {name: importlib.metadata.version(name) for name in ('torch', 'torchaudio', 'muq', 'transformers', 'librosa', 'numpy', 'soundfile')}}
    try:
        for name, digest in MODEL_PINS.items():
            if sha(MODEL / name) != digest:
                raise ValueError(f'model hash mismatch: {name}')
        for track in tracks:
            if sha(track['path']) != track['sha256']:
                raise ValueError(f'input hash mismatch: {track["id"]}')
        torch.set_num_threads(4)
        torch.manual_seed(0)
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; stop rather than silently change runtime')
        torch.cuda.set_per_process_memory_fraction(0.75)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        model = muq.MuQ.from_pretrained(str(MODEL)).cuda().eval()
        if (model.config.encoder_depth, model.config.encoder_dim) != (12, 1024):
            raise ValueError('unexpected checkpoint architecture')
        runtime['model_parameters'] = sum(p.numel() for p in model.parameters())
        runtime['gpu'] = torch.cuda.get_device_name(0)
        pooled, baseline, windows, times = [], [], [], []
        with torch.inference_mode():
            for track in tracks:
                track_started = time.perf_counter()
                info = sf.info(track['path'])
                if info.samplerate != track['sample_rate'] or info.frames != track['frames'] or info.channels != 2:
                    raise ValueError('audio header changed')
                means, simple, bounds, inference_seconds = [], [], [], []
                with sf.SoundFile(track['path']) as handle:
                    for j in range(10):
                        if time.perf_counter() - started > 900:
                            raise TimeoutError('15 minute budget exceeded')
                        center = round(info.frames * (0.05 + j * 0.1))
                        begin = center - 5 * info.samplerate
                        end = begin + 10 * info.samplerate
                        if begin < 0 or end > info.frames:
                            raise ValueError('sampled chunk lies outside source')
                        handle.seek(begin)
                        pcm = handle.read(end - begin, dtype='float32', always_2d=True)
                        if pcm.shape != (end - begin, 2) or not np.isfinite(pcm).all():
                            raise ValueError('bad source PCM')
                        mono = pcm.mean(axis=1, dtype=np.float32)
                        wave = torchaudio.functional.resample(torch.from_numpy(mono)[None], info.samplerate, 24000)
                        if wave.shape != (1, 240000):
                            raise ValueError('resampling length mismatch')
                        simple.append(descriptors(wave))
                        torch.cuda.synchronize()
                        t = time.perf_counter()
                        result = model(wave.cuda(), output_hidden_states=False).last_hidden_state
                        torch.cuda.synchronize()
                        inference_seconds.append(time.perf_counter() - t)
                        vector = result.mean(dim=1).float().cpu().numpy()[0]
                        if vector.shape != (1024,) or not np.isfinite(vector).all():
                            raise ValueError('invalid MuQ feature')
                        means.append(vector)
                        bounds.append([begin, end])
                        del result
                np.savez_compressed(output / f'{track["id"]}.npz', muq_windows=np.stack(means), simple_windows=np.stack(simple), source_frame_bounds=np.array(bounds, dtype=np.int64))
                pooled.append(np.stack(means).mean(axis=0))
                baseline.append(np.stack(simple).mean(axis=0))
                windows.append(bounds)
                times.append({'id': track['id'], 'inference_seconds': inference_seconds, 'wall_seconds': time.perf_counter() - track_started})
                print(json.dumps({'completed': len(pooled), 'total': len(tracks), 'id': track['id'], 'wall_seconds': times[-1]['wall_seconds']}, ensure_ascii=False), flush=True)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 6 * 1024**3:
                    raise MemoryError('6 GiB host RSS guard exceeded')
        vectors = np.stack(pooled)
        simple_vectors = np.stack(baseline)
        np.savez_compressed(output / 'song-vectors.npz', muq=vectors, simple=simple_vectors)
        save_json(output / 'evaluation.json', evaluate_vectors(vectors, simple_vectors, tracks))
        retained = [i for i, t in enumerate(tracks) if not t.get('version_caveat')]
        save_json(output / 'version-sensitivity.json', evaluate_vectors(vectors[retained], simple_vectors[retained], [tracks[i] for i in retained]))
        unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        save_json(output / 'muq-cosine-matrix.json', {'ids': [t['id'] for t in tracks], 'matrix': (unit @ unit.T).tolist()})
        for track in tracks:
            if sha(track['path']) != track['sha256']:
                raise ValueError('source changed during extraction')
        runtime.update(status='complete', elapsed_seconds=time.perf_counter() - started, track_times=times, maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), source_hashes_verified_before_after=True)
    except Exception as exc:
        runtime.update(status='failed', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.perf_counter() - started)
        raise
    finally:
        save_json(output / 'runtime.json', runtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.output)


if __name__ == '__main__':
    main()
