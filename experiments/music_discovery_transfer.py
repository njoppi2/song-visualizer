#!/usr/bin/env python3
"""Apply existing reference rules to three separately rated recordings."""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def score_new_queries(reference_muq, reference_simple, references, query_muq, query_simple, queries, score_function):
    """Query labels are absent from this API and never select reference groups."""
    combined = [*references, *queries]
    for key in ('id', 'isrc', 'sha256'):
        values = [t[key] for t in combined]
        if any(not v for v in values) or len(set(values)) != len(values):
            raise ValueError(f'duplicate or missing recording {key}')
    count = len(references)
    if len(reference_muq) != count or len(reference_simple) != count or len(query_muq) != len(queries) or len(query_simple) != len(queries):
        raise ValueError('feature/recording alignment mismatch')
    positives = [i for i, t in enumerate(references) if t['label'] == 'favorite']
    negatives = [i for i, t in enumerate(references) if t['label'] == 'dislike']
    rows = list(range(count, len(combined)))
    result = {'positive_references': [combined[i]['id'] for i in positives],
              'negative_references': [combined[i]['id'] for i in negatives], 'methods': {}}
    for kind, ref_features, query_features in [('muq', reference_muq, query_muq), ('simple', reference_simple, query_simple)]:
        features = np.vstack([ref_features, query_features])
        scored = score_function(features, positives, negatives, rows, kind)
        candidates = []
        for score in scored:
            candidates.append({'id': combined[score['query_index']]['id'],
                'positive_only': score['positive_only'], 'negative_aware': score['negative_aware'],
                'positive_similarity': score['positive_similarity'], 'negative_similarity': score['negative_similarity'],
                'nearest_positive': combined[score['nearest_positive_index']]['id'],
                'nearest_negative': combined[score['nearest_negative_index']]['id']})
        result['methods'][kind] = {'queries': candidates,
            'rankings': {method: [r['id'] for r in sorted(candidates, key=lambda r: (-r[method], r['id']))] for method in ('positive_only', 'negative_aware')}}
    return result


def evaluate_orderings(scores, supported_orderings):
    """Only the user's explicit strict orders count; Like vs Like stays unknown."""
    results = {}
    for kind, data in scores['methods'].items():
        lookup = {r['id']: r for r in data['queries']}
        results[kind] = {}
        for method in ('positive_only', 'negative_aware'):
            pairs = []
            for order in supported_orderings:
                margin = lookup[order['higher']][method] - lookup[order['lower']][method]
                outcome = 'agrees' if margin > 1e-10 else 'disagrees' if margin < -1e-10 else 'tie'
                pairs.append({**order, 'margin': margin, 'result': outcome})
            results[kind][method] = {'comparisons': pairs,
                'agrees': sum(p['result'] == 'agrees' for p in pairs), 'total': len(pairs)}
    return results


def run(protocol_path, output):
    import torch
    import torchaudio
    import soundfile as sf

    protocol = json.loads(protocol_path.read_text())
    pilot = module(ROOT / protocol['extraction_helpers']['path'], 'fixed_pilot_helpers')
    negative = module(ROOT / protocol['scoring_helpers']['path'], 'fixed_negative_helpers')
    if pilot.sha(__file__) != protocol['runner_sha256']:
        raise ValueError('runner differs from protocol')
    for receipt in protocol['input_files']:
        if pilot.sha(ROOT / receipt['path']) != receipt['sha256']:
            raise ValueError(f'changed input: {receipt["path"]}')
    for parent in protocol['parent_manifests']:
        for receipt in json.loads((ROOT / parent).read_text())['files']:
            if pilot.sha(ROOT / receipt['path']) != receipt['sha256']:
                raise ValueError('parent provenance mismatch')
    for name, expected in pilot.MODEL_PINS.items():
        if pilot.sha(pilot.MODEL / name) != expected:
            raise ValueError('cached model pin mismatch')
    queries = protocol['queries']
    for q in queries:
        if pilot.sha(q['path']) != q['sha256']:
            raise ValueError('query audio changed')
    references = json.loads((ROOT / protocol['reference_protocol']).read_text())['tracks']
    with np.load(ROOT / protocol['reference_vectors'], allow_pickle=False) as archive:
        reference_muq, reference_simple = archive['muq'], archive['simple']
    if os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('TRANSFORMERS_OFFLINE') != '1':
        raise ValueError('offline flags required')
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    runtime = {'protocol_sha256': pilot.sha(protocol_path), 'runner_sha256': pilot.sha(__file__),
               'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'dtype': 'float32', 'device': 'cuda'}
    try:
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; stop without changing device')
        torch.set_num_threads(4)
        torch.manual_seed(0)
        torch.cuda.set_per_process_memory_fraction(.75)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        import muq
        model = muq.MuQ.from_pretrained(str(pilot.MODEL)).cuda().eval()
        if (model.config.encoder_depth, model.config.encoder_dim) != (12, 1024):
            raise ValueError('unexpected model architecture')
        runtime.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, numpy=np.__version__)
        pooled_muq, pooled_simple, timing = [], [], []
        with torch.inference_mode():
            for query in queries:
                begun = time.perf_counter()
                info = sf.info(query['path'])
                if (info.samplerate, info.frames, info.channels) != (query['sample_rate'], query['frames'], 2):
                    raise ValueError('changed audio geometry')
                vectors, simple, bounds = [], [], []
                with sf.SoundFile(query['path']) as handle:
                    for j in range(10):
                        if time.perf_counter() - started > 300:
                            raise TimeoutError('five minute extraction budget exceeded')
                        center = round(info.frames * (.05 + j * .1))
                        a, b = center - 5 * info.samplerate, center + 5 * info.samplerate
                        if not 0 <= a < b <= info.frames:
                            raise ValueError('window beyond source support')
                        handle.seek(a)
                        pcm = handle.read(b - a, dtype='float32', always_2d=True)
                        if pcm.shape != (b - a, 2) or not np.isfinite(pcm).all():
                            raise ValueError('invalid source PCM')
                        wave = torchaudio.functional.resample(torch.from_numpy(pcm.mean(axis=1, dtype=np.float32))[None], info.samplerate, 24000)
                        if wave.shape != (1, 240000):
                            raise ValueError('unexpected resampling extent')
                        simple.append(pilot.descriptors(wave))
                        features = model(wave.cuda(), output_hidden_states=False).last_hidden_state
                        vector = features.mean(dim=1).float().cpu().numpy()[0]
                        if vector.shape != (1024,) or not np.isfinite(vector).all():
                            raise ValueError('invalid embedding')
                        vectors.append(vector)
                        bounds.append([a, b])
                        del features
                np.savez_compressed(output / f'{query["id"]}.npz', muq_windows=np.stack(vectors), simple_windows=np.stack(simple), source_frame_bounds=np.array(bounds, dtype=np.int64))
                pooled_muq.append(np.stack(vectors).mean(axis=0))
                pooled_simple.append(np.stack(simple).mean(axis=0))
                timing.append({'id': query['id'], 'wall_seconds': time.perf_counter() - begun})
                print(json.dumps({'completed': len(timing), 'total': len(queries), **timing[-1]}), flush=True)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 6 * 1024**3:
                    raise MemoryError('6 GiB RSS guard exceeded')
        qm, qs = np.stack(pooled_muq), np.stack(pooled_simple)
        np.savez_compressed(output / 'query-vectors.npz', muq=qm, simple=qs)
        keep = [i for i, t in enumerate(references) if not t.get('version_caveat')]
        labels = json.loads((ROOT / protocol['labels']).read_text())
        for name, rm, rs, cohort in [('full', reference_muq, reference_simple, references), ('without-laurel', reference_muq[keep], reference_simple[keep], [references[i] for i in keep])]:
            # Remove labels from queries before crossing the scoring interface.
            query_identity = [{k: q[k] for k in ('id', 'isrc', 'sha256')} for q in queries]
            scores = score_new_queries(rm, rs, cohort, qm, qs, query_identity, negative.score_queries)
            pilot.save_json(output / f'scores-{name}.json', scores)
            pilot.save_json(output / f'evaluation-{name}.json', evaluate_orderings(scores, labels['supported_orderings']))
        for query in queries:
            if pilot.sha(query['path']) != query['sha256']:
                raise ValueError('query source changed during extraction')
        for receipt in protocol['input_files']:
            if pilot.sha(ROOT / receipt['path']) != receipt['sha256']:
                raise ValueError('registered input changed')
        runtime.update(status='complete', elapsed_seconds=time.perf_counter() - started, query_times=timing,
            source_hashes_verified_before_after=True, maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as exc:
        runtime.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        pilot.save_json(output / 'runtime.json', runtime)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.output)
