"""Create an isolated, audio-linked boundary/recurrence review; never replace caches."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.build_review import fingerprint, run
from songviz import story as current
from songviz.ingest import sha256_file
from songviz.recurrence import compare_phrases


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_records(records):
    for record in records:
        if sha256_file(ROOT / record['path']) != record['sha256']:
            raise ValueError(f"Changed input: {record['path']}")


def choose_repeats(results, limit=3):
    """Diverse distant returns plus one lower-similarity contrast; no role inputs."""
    pool = []
    for result in results:
        for pair in result['pairs']:
            a, b = result['spans'][pair['a']], result['spans'][pair['b']]
            if b['start_s'] - a['end_s'] >= 20:
                pool.append({**pair, 'a_span': a, 'b_span': b, 'scale_beats': result['scale_beats']})
    pool.sort(key=lambda p: (-p['similarity'], p['a_span']['start_s'], p['b_span']['start_s']))
    selected = []
    def distinct(p):
        return all(abs(p['b_span']['start_s'] - q['b_span']['start_s']) > 20 for q in selected)
    # Include both window lengths instead of allowing short, easy matches to
    # occupy every slot. The contrast is a check against automatic agreement.
    for scale in sorted({p['scale_beats'] for p in pool}):
        for p in pool:
            if p['scale_beats'] == scale and distinct(p):
                selected.append(p)
                break
        if len(selected) >= max(1, limit - 1):
            break
    for p in reversed(pool):
        if distinct(p) and len(selected) < limit:
            selected.append(p)
            break
    return [{"id": f"repeat-{i+1}", "a_start_s": p['a_span']['start_s'], "a_end_s": p['a_span']['end_s'],
             "b_start_s": p['b_span']['start_s'], "b_end_s": p['b_span']['end_s'],
             "scale_beats": p['scale_beats'], "similarity": p['similarity'],
             "prompt": "Is B the musical idea from A returning, a variation, or a different part?",
             "evidence": f"Ordered {p['scale_beats']}-beat comparison, not verified bars. Per-stem acoustic scores: " + json.dumps(p['stem_similarities'])}
            for i, p in enumerate(selected[:limit])]


def boundary_questions(baseline, candidate, duration):
    old = [s['start_s'] for s in baseline['sections'][1:]]
    new = [s['start_s'] for s in candidate['sections'][1:]]
    changes = sorted((min((abs(t - n) for n in new), default=duration), t) for t in old)
    chosen = []
    # Start with the biggest disagreement, then sample later boundaries to expose
    # potentially missed changes inside the long ending. No correctness labels.
    for _, t in reversed(changes):
        if not chosen or all(abs(t - q) > 12 for q in chosen):
            chosen.append(t)
        if len(chosen) == 2:
            break
    for t in reversed(new):
        if all(abs(t - q) > 12 for q in chosen):
            chosen.append(t)
        if len(chosen) == 4:
            break
    return [{"id": f"boundary-{i+1}", "time_s": t, "start_s": max(0., t-9), "end_s": min(duration, t+11),
             "prompt": "Does a new musical part begin here, or is this a change inside the same part?",
             "evidence": f"Previous boundaries: {', '.join(f'{v:.2f}s' for v in old if abs(v-t)<8) or 'none nearby'}; candidate: {', '.join(f'{v:.2f}s' for v in new if abs(v-t)<8) or 'none nearby'}. Neither is ground truth."}
            for i, t in enumerate(sorted(chosen))]


def plots(out, baseline, candidate, chroma, feature_times, stems):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 1, figsize=(15, 4), sharex=True)
    for ax, name, story in zip(axes, ['Previous logic', 'Candidate logic'], [baseline, candidate]):
        for i, s in enumerate(story['sections']):
            ax.axvspan(s['start_s'], s['end_s'], color=plt.get_cmap('tab10')(i % 10), alpha=.65)
            ax.text((s['start_s']+s['end_s'])/2, .5, s['label']+'\n'+s['role'], ha='center', va='center', fontsize=8)
        ax.set_ylabel(name); ax.set_yticks([]); ax.set_xlim(0, candidate['meta']['duration_s'])
    axes[-1].set_xlabel('Original song seconds'); fig.suptitle('Same reviewed pulse; predicted boundaries/roles, not ground truth')
    fig.tight_layout(); fig.savefig(out/'sections-comparison.png', dpi=120); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    for ax, scale in zip(axes, ['short', 'medium', 'long']):
        for name, story, color in [('Previous (rescaled)', baseline, '#ff938c'), ('Candidate (raw cosine)', candidate, '#80ffc0')]:
            ax.plot(story['novelties']['times_s'], story['novelties']['novelty_'+scale], lw=.8, color=color, label=name)
        ax.set_ylabel(scale); ax.set_ylim(0, 1); ax.legend(loc='upper right')
    axes[-1].set_xlabel('Original song seconds'); fig.suptitle('4/16/32-beat nearest-past novelty: missing history is now unknown\nDifferent score semantics; amplitude differences are not quality scores')
    fig.tight_layout(); fig.savefig(out/'novelty-comparison.png', dpi=120); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    axes[0].imshow(chroma, aspect='auto', origin='lower', extent=[feature_times[0], feature_times[-1], 0, 12], cmap='magma')
    axes[0].set_ylabel('Mix pitch class'); axes[0].set_title('Audio evidence: chroma and stem RMS (not listening ground truth)')
    for name, y in stems.items():
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        axes[1].plot(np.arange(len(rms))*512/22050, rms, label=name, lw=.7)
    for ax in axes:
        for sec in candidate['sections'][1:]:
            ax.axvline(sec['start_s'], color='white', alpha=.45, lw=.7)
    axes[1].set_ylabel('Absolute RMS'); axes[1].legend(); axes[1].set_xlabel('Original song seconds')
    fig.tight_layout(); fig.savefig(out/'evidence.png', dpi=120); plt.close(fig)


def build(out: Path, parent: Path):
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    manifest = json.loads((parent/'manifest.json').read_text())
    verify_records(manifest['sources'] + manifest['input_snapshots'] + manifest['outputs'])
    source = ROOT / manifest['sources'][0]['path']
    baseline = json.loads((parent/'reviewed-grid/story.json').read_text())
    grid = baseline['meta']['beat_grid']
    if not grid['explicit'] or grid['fallback'] is not None:
        raise ValueError('Expected explicit reviewed baseline grid')
    out.mkdir(parents=True)
    inputs = out/'inputs'; inputs.mkdir()
    for rel in ['songviz/story.py', 'songviz/structure_grid.py', 'songviz/recurrence.py', 'songviz/ingest.py',
                'experiments/build_structure_review.py', 'experiments/templates/structure_review.html']:
        dest = inputs/rel; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/rel, dest)
    shutil.copy2(parent/'inputs/songviz/story.py', inputs/'baseline-story.py')
    shutil.copy2(parent/'manifest.json', inputs/'parent-manifest.json')
    shutil.copy2(parent/'reviewed-grid/beat-grid.json', inputs/'baseline-beat-grid.json')
    print('Load original full song and cached stems', flush=True)
    y, sr = librosa.load(source, sr=22050, mono=True)
    stems = {name: librosa.load(source.parent.parent/'outputs'/source.stem/'stems'/f'{name}.wav', sr=sr, mono=True)[0]
             for name in ['drums', 'bass', 'vocals', 'other']}
    samples, original_sr = sf.read(source, dtype='float32', always_2d=True)
    # Browser audio support for IEEE-float WAV is inconsistent. PCM_24 is
    # widely decodable while retaining substantially more precision than the
    # source material requires.
    sf.write(out/'original.wav', samples, original_sr, subtype='PCM_24')
    decoded, decoded_sr = sf.read(out/'original.wav', dtype='float32', always_2d=True)
    assert decoded_sr == original_sr and np.allclose(samples, decoded, rtol=0, atol=1.2e-7)
    kwargs = dict(stems=stems, other_y=stems['other'], beat_times_s=grid['requested_times_s'], beat_grid_source=grid['source'])
    captured = {}
    scorer = current._score_and_filter_boundaries
    def capture(ssm, energy, **kw):
        captured.update({'ssm_times_s': ssm, 'energy_times_s': energy})
        result = scorer(ssm, energy, **kw)
        captured['fused_times_s'] = result
        return result
    with patch.object(librosa.beat, 'beat_track', side_effect=AssertionError('Unexpected tracking')), patch.object(current, '_score_and_filter_boundaries', capture):
        print('Compute candidate structure and novelty', flush=True)
        candidate = current.compute_story(y, sr, **kwargs)
    assert candidate['meta']['beat_grid'] == grid
    assert candidate['meta']['section_method'] == 'ssm'
    assert set(candidate['stem_novelties']) == set(stems)
    (out/'candidate-story.json').write_text(json.dumps(candidate, separators=(',', ':'), allow_nan=False)+'\n')
    write_json(out/'boundary-evidence.json', captured)

    # Single-change ablations: ensure each fix's measured effects are isolated.
    spec = importlib.util.spec_from_file_location('songviz._review_control', inputs/'baseline-story.py')
    control = importlib.util.module_from_spec(spec); spec.loader.exec_module(control)
    with patch.object(current, '_score_and_filter_boundaries', control._score_and_filter_boundaries):
        print('Check novelty-only ablation', flush=True)
        novelty_only = current.compute_story(y, sr, **kwargs)
    assert novelty_only['sections'] == baseline['sections'], 'Novelty fix unexpectedly changed sections'
    def old_novelty(a, b, scales=None, **kw):
        return control._novelty_curves_from_lag(a, b, scales)
    with patch.object(current, '_lag_matrix_novelty', control._lag_matrix_novelty), patch.object(current, '_novelty_curves_from_lag', old_novelty):
        print('Check boundary-only ablation', flush=True)
        boundary_only = current.compute_story(y, sr, **kwargs)
    assert boundary_only['sections'] == candidate['sections']
    for scale in ['short', 'medium', 'long']:
        key = 'novelty_'+scale
        np.testing.assert_allclose(boundary_only['novelties'][key], baseline['novelties'][key], atol=1e-6)
        np.testing.assert_allclose(novelty_only['novelties'][key], candidate['novelties'][key], atol=1e-6)
    write_json(out/'ablations.json', {'novelty_only_sections_equal_baseline': True,
        'boundary_only_sections_equal_candidate': True, 'boundary_only_global_novelty_equal_baseline': True,
        'novelty_only_global_novelty_equal_candidate': True, 'numerical_tolerance': 1e-6,
        'note': 'Controlled code comparisons, not independent musical validation.'})
    del novelty_only, boundary_only

    print('Compute role-independent 16/32-beat recurrence candidates', flush=True)
    bt = np.asarray(grid['requested_times_s']); bf = librosa.time_to_frames(bt, sr=sr, hop_length=512)
    features, energy = {}, {}
    for name, stem in stems.items():
        q = np.abs(librosa.cqt(stem, sr=sr, hop_length=512, n_bins=84, fmin=librosa.note_to_hz('C1')))
        rms = librosa.feature.rms(y=stem, frame_length=2048, hop_length=512)
        features[name] = np.log1p(librosa.util.sync(q, bf, aggregate=np.mean, pad=False))
        energy[name] = librosa.util.sync(rms, bf, aggregate=np.mean, pad=False)[0]
    results = [compare_phrases(features, energy, bt, scale_beats=n) for n in [16, 32]]
    write_json(out/'recurrence.json', {'beat_grid_sha256': grid['requested_sha256'], 'results': results})
    duration = len(y)/sr
    hop = max(1, int(sr*.1))
    wave = [{'time_s': i/sr, 'value': float(np.sqrt(np.mean(y[i:i+hop]**2)))} for i in range(0,len(y),hop)]
    review = {'schema_version': 1, 'song_title': source.stem, 'duration_s': duration, 'audio_path': 'original.wav',
        'sections': {'baseline': baseline['sections'], 'candidate': candidate['sections']},
        'boundary_questions': boundary_questions(baseline, candidate, duration), 'repeat_questions': choose_repeats(results),
        'waveform': wave, 'notes': [
            'Both timelines use the same reviewed pulse and original audio. Neither is a reference annotation.',
            'Roles and section letters remain heuristic. Repetition questions are generated independently from ordered spectral/stem evidence.',
            'The final long section may contain missed changes. Mark any missing boundary while listening anywhere in the song.',
            '16/32 beats are window lengths, not verified bars; acoustic scores are not confidence probabilities.',
            'Feel Good Inc is development data. No holdout musical improvement has been established.']}
    write_json(out/'review.json', review)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    plots(out, baseline, candidate, chroma, np.arange(chroma.shape[1])*512/sr, stems)
    verify_records(manifest['sources'])
    result_manifest = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'parent': fingerprint(parent/'manifest.json'), 'sources': manifest['sources'],
        'git_head': run('git','rev-parse','HEAD'), 'versions': {'librosa': librosa.__version__, 'numpy': np.__version__},
        'settings': {'sr': sr, 'hop_length': 512, 'frame_length': 2048, 'recurrence_scales_beats': [16,32], 'stride_beats': 4},
        'audio_validation': 'Review WAV is a PCM_24 decode of the original, with no gain or timing edits (maximum quantization error under 1.2e-7).',
        'input_snapshots': [fingerprint(p) for p in sorted(inputs.rglob('*')) if p.is_file()],
        'outputs': [fingerprint(p) for p in sorted(out.iterdir()) if p.is_file()],
        'page_integrity': 'index.html embeds this manifest hash; template and review.json are fingerprinted to avoid a circular hash.'}
    write_json(out/'manifest.json', result_manifest)
    embedded = json.dumps(review, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    html = (ROOT/'experiments/templates/structure_review.html').read_text().replace('{{REVIEW_JSON}}', embedded).replace('{{MANIFEST_SHA}}', sha256_file(out/'manifest.json'))
    (out/'index.html').write_text(html)
    print(f'Ready: {out}/index.html', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=ROOT/'outputs/reviews/structure-grid-01')
    args = parser.parse_args(); build(args.out.resolve(), args.parent.resolve())
