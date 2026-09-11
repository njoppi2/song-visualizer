"""Inspect label-free local-change and bounded energy-dip proposals on cached evidence.

Creates a new review only. Does not regenerate stems, change legacy sections,
train on feedback, or decide musical identity from a change score.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import quote

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from songviz.ingest import sha256_file
from songviz.local_structure import LocalStructureConfig, detect_local_structure
from songviz.local_structure_evaluation import evaluate_local_structure

CODE = ['songviz/local_structure.py', 'songviz/local_structure_evaluation.py',
        'experiments/build_local_structure_review.py', 'experiments/templates/local_structure_review.html',
        'experiments/check_local_structure_review.cjs', 'songviz/ingest.py']


def record(path: Path) -> dict:
    return {'path': str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            'sha256': sha256_file(path), 'bytes': path.stat().st_size}


def verify(path: Path, digest: str) -> None:
    if sha256_file(path) != digest:
        raise ValueError(f'Changed or mismatched input: {path}')


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def embedded_json(data: dict) -> str:
    return json.dumps(data, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def report_markdown(data: dict) -> str:
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    def num(value):
        return 'unknown' if value is None else f'{value:.3f}'
    predictions, evaluation = data['predictions'], data['evaluation']
    lines = ['# Local structure candidate review', '', data['song_title'], '',
        f"{len(predictions['changes'])} local-change candidates; {len(predictions['transitions'])} bounded energy-dip candidates. Legacy section cuts: {max(0, len(data['legacy_sections'])-1)}.", '',
        'These are acoustic proposals, **not** a new song partition, accepted musical transitions, or inferred identities.',
        'The fixed default policy was chosen before this run. Human annotations only enter evaluation.', '',
        '## Evidence and limits', '']
    lines += ['- ' + note for note in data['notes']]
    lines += ['', '## All proposed local changes', '',
        '| ID | Time (s) | Strength | Pattern change | Arrangement change | Evidence available at (s) |',
        '| --- | --- | --- | --- | --- | --- |']
    for p in predictions['changes']:
        lines.append(f"| {p['id']} | {num(p['time_s'])} | {num(p['strength'])} | {num(p['pattern_change'])} | {num(p['arrangement_change'])} | {num(p['available_at_s'])} |")
    lines += ['', '## All proposed energy-dip intervals', '',
        '| ID | Start–end (s) | Strength | Evidence available at (s) |',
        '| --- | --- | --- | --- |']
    for p in predictions['transitions']:
        lines.append(f"| {p['id']} | {p['start_s']:.3f}–{p['end_s']:.3f} | {num(p['strength'])} | {num(p['available_at_s'])} |")
    for layer in evaluation['layers']:
        lines += ['', f"## Reference layer: {cell(layer['name'])}", '',
            'Every annotated mark is listed. Nearest distances are descriptive/many-to-one, with no selected correctness tolerance.', '',
            '| Mark (s) | Identity relation | Variation change | Nearest new change: delta (s) | Nearest legacy cut: delta (s) |',
            '| --- | --- | --- | --- | --- |']
        for row in layer['boundary_rows']:
            new, old = row['nearest_change'], row['nearest_legacy']
            lines.append(f"| {num(row['time_s'])} | {row['identity_relation']} | {cell(row['variation_change'])} | {num(new['delta_s'] if new else None)} | {num(old['delta_s'] if old else None)} |")
        lines += ['', 'Interpreted transition intervals: best overlapping prediction (not a one-to-one accuracy metric).', '',
            '| User label | User interval (s) | Best predicted interval (s) | IoU | Start error (s) | End error (s) |',
            '| --- | --- | --- | --- | --- | --- |']
        for row in layer['transition_rows']:
            match = row['predicted_best_overlap']
            interval = f"{match['start_s']:.3f}–{match['end_s']:.3f}" if match else 'no overlap'
            lines.append(f"| {cell(row['label'])} | {row['start_s']:.3f}–{row['end_s']:.3f} | {interval} | {num(match['iou'] if match else None)} | {num(match['start_error_s'] if match else None)} | {num(match['end_error_s'] if match else None)} |")
        lines += ['', 'Reverse comparisons for every predicted change/interval are retained in evaluation.json. Lack of an annotation is not a certified false positive.']
    lines += ['', '## Configuration', '', '```json', json.dumps(predictions['config'], indent=2), '```', '',
        'Full per-scale curves, thresholds, support windows and per-stem evidence are in predictions.json.',
        'Existing 16/32-beat recurrence is attached separately in evaluation.json: a local change does not itself mean new musical identity.',
        'Use index.html for original-audio playback and inspect candidates with surrounding context.', '',
        'No human listening approval or holdout improvement is established by this artifact.', '']
    return '\n'.join(lines)


def build(*, parent: Path, audio_review: Path, out: Path) -> None:
    parent, audio_review, out = (p.resolve() for p in [parent, audio_review, out])
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    if any(p == out or p in out.parents for p in [parent, audio_review]):
        raise ValueError('Output must be separate from input packages')
    manifest_path = parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('kind') != 'songviz-structural-development-evaluation':
        raise ValueError('Expected the dimension-separated structural evaluation package')
    consumed = [manifest_path]
    outputs = {Path(r['path']).name: r for r in manifest['outputs']}
    for name in ['reference.json', 'features.npz', 'timing.json', 'recurrence.json', 'legacy-sections.json']:
        verify(parent/name, outputs[name]['sha256'])
        consumed.append(parent/name)
    reference = json.loads((parent/'reference.json').read_text())
    timing = json.loads((parent/'timing.json').read_text())
    recurrence = json.loads((parent/'recurrence.json').read_text())
    legacy = json.loads((parent/'legacy-sections.json').read_text())
    audio_manifest_path = audio_review/'manifest.json'
    records = [r for r in manifest['sources'] if (ROOT/r['path']).resolve() == audio_manifest_path]
    if len(records) != 1:
        raise ValueError('Audio review is not the fingerprinted structural source')
    verify(audio_manifest_path, records[0]['sha256'])
    audio_manifest = json.loads(audio_manifest_path.read_text())
    audio_outputs = {Path(r['path']).name: r for r in audio_manifest['outputs']}
    for name in ['original.wav', 'review.json']:
        verify(audio_review/name, audio_outputs[name]['sha256'])
        consumed.append(audio_review/name)
    consumed.append(audio_manifest_path)
    verify(audio_review/'original.wav', reference['source']['audio_sha256'])
    info = sf.info(audio_review/'original.wav')
    if abs(info.duration - reference['source']['duration_s']) > 1e-6:
        raise ValueError('Audio duration differs from reference')
    source_records = [r for r in manifest['sources'] if r['sha256'] == reference['source']['source_audio_sha256']]
    if len(source_records) != 1:
        raise ValueError('Original source fingerprint not unique')
    source = ROOT/source_records[0]['path']
    verify(source, source_records[0]['sha256'])
    consumed.append(source)
    review = json.loads((audio_review/'review.json').read_text())
    if abs(review['duration_s'] - info.duration) > 1e-6:
        raise ValueError('Review waveform duration differs from audio')
    # Read a numeric cache only (never pickle); no annotation-dependent windows.
    with np.load(parent/'features.npz', allow_pickle=False) as cache:
        if 'beat_times_s' not in cache.files:
            raise ValueError('Missing beat times')
        bt = cache['beat_times_s']
        names = sorted(k[:-9] for k in cache.files if k.endswith('_features'))
        expected = {'beat_times_s'} | {n+'_features' for n in names} | {n+'_rms' for n in names}
        if not names or set(cache.files) != expected:
            raise ValueError('Unexpected or incomplete feature cache schema')
        features = {n: cache[n+'_features'] for n in names}
        energy = {n: cache[n+'_rms'] for n in names}
    if not np.array_equal(bt, np.asarray(timing['requested_times_s'])):
        raise ValueError('Feature and timing grids differ')
    if bt.ndim != 1 or bt.size < 2 or bt[0] < 0 or bt[-1] > info.duration:
        raise ValueError('Beat grid lies outside the audio')
    for result in recurrence['results']:
        for span in result['spans']:
            a, b = span['start_beat'], span['end_beat']
            if (isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int) or not isinstance(b, int)
                    or not 0 <= a < b < len(bt) or span['start_s'] != bt[a] or span['end_s'] != bt[b]):
                raise ValueError('Recurrence windows differ from the feature beat grid')
    consumed += [ROOT/rel for rel in CODE]
    before = [record(p) for p in consumed]
    config = LocalStructureConfig()
    # Intentionally only numeric acoustic arrays. Recurrence and human labels
    # enter the following evaluation step, never candidate generation.
    predictions = detect_local_structure(features, energy, bt, config=config)
    evaluation = evaluate_local_structure(reference, predictions, legacy['sections'], recurrence['results'])
    relative_audio = quote(Path(os.path.relpath(audio_review/'original.wav', out)).as_posix(), safe='/')
    data = {'schema_version': 1, 'song_title': reference['source']['song_title'], 'duration_s': info.duration,
        'audio_path': relative_audio, 'audio_sha256': reference['source']['audio_sha256'],
        'reference': reference, 'predictions': predictions, 'evaluation': evaluation,
        'legacy_sections': legacy['sections'], 'waveform': review.get('waveform', []),
        'notes': [
            'Human motif/interval assertions and analyst variation/transition tags are separate; unspecified certainty is retained.',
            'Local change is not necessarily a new section or unfamiliar material; no verse/chorus label is predicted.',
            'Bounded energy-dip-and-recovery is one transition hypothesis. Fills, ramps and other non-dip transitions can be missed.',
            'Compare all proposals, including those away from annotations. More proposals can reduce nearest distance without improving precision.',
            'Predictions use 2/4/8-beat contrast and short dip intervals on the existing reviewed pulse. Bar/downbeat phase is unverified.',
            'Feature frames precede requested beat timestamps by less than one 512-sample hop at 22050 Hz, as in the parent cache.',
            'Analysis is offline: right-side context and whole-song audibility calibration use future audio. available_at_s records the end of local support, not causal readiness of all calibration.',
            'Recurrence context uses full 16/32-beat windows after a candidate and is not instantaneous surprise or automatic identity.',
            'No trained thresholds, fixed target section count, production cache update, or holdout quality claim.']}
    # Validate JSON serializability before any package writes.
    encoded = embedded_json(data)
    for r in before:
        verify(ROOT/r['path'], r['sha256'])
    out.mkdir(parents=True)
    for rel in CODE:
        dest = out/'inputs'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, dest)
    shutil.copy2(manifest_path, out/'inputs'/'parent-manifest.json')
    shutil.copy2(audio_manifest_path, out/'inputs'/'audio-manifest.json')
    write_json(out/'predictions.json', predictions)
    write_json(out/'evaluation.json', evaluation)
    write_json(out/'review.json', data)
    (out/'report.md').write_text(report_markdown(data))
    for r in before:
        verify(ROOT/r['path'], r['sha256'])
    new_manifest = {'schema_version': 1, 'kind': 'songviz-local-structure-review',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'sources': before,
        'settings': asdict(config), 'numpy_version': np.__version__,
        'scope': 'Label-free local-change and bounded-energy-dip candidates; human feedback for evaluation only.',
        'input_snapshots': [record(p) for p in sorted((out/'inputs').rglob('*')) if p.is_file()],
        'outputs': [record(p) for p in sorted(out.iterdir()) if p.is_file()],
        'page_integrity': 'index.html derives from snapshotted template, review.json and manifest hash; original audio is referenced, not copied.'}
    write_json(out/'manifest.json', new_manifest)
    template = (out/'inputs'/'experiments/templates/local_structure_review.html').read_text()
    (out/'index.html').write_text(template.replace('{{REVIEW_JSON}}', encoded).replace('{{MANIFEST_SHA}}', sha256_file(out/'manifest.json')))
    print(f'Ready: {out}/index.html', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=ROOT/'outputs/reviews/structure-evaluation-03')
    parser.add_argument('--audio-review', type=Path, default=ROOT/'outputs/reviews/structure-review-03')
    parser.add_argument('--out', type=Path, required=True)
    build(**vars(parser.parse_args()))
