"""Build isolated, dimension-separated development diagnostics from section feedback.

Does not re-separate stems, rerun section detection, change cached stories, or use
annotations to extract/select acoustic windows. Output is not a trained detector.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from songviz.ingest import sha256_file
from songviz.recurrence import compare_phrases
from songviz.structure_annotations import normalize_annotations
from songviz.structure_evaluation import evaluate_structure

STEMS = ('bass', 'drums', 'other', 'vocals')
CODE = ('experiments/evaluate_structure_feedback.py', 'songviz/structure_annotations.py',
        'songviz/structure_evaluation.py', 'songviz/recurrence.py', 'songviz/ingest.py')


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def record(path: Path) -> dict:
    return {'path': str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            'sha256': sha256_file(path), 'bytes': path.stat().st_size}


def verify(path: Path, digest: str) -> None:
    if sha256_file(path) != digest:
        raise ValueError(f'Changed or mismatched input: {path}')


def extract_features(stems: dict[str, Path], beat_times: np.ndarray) -> tuple[dict, dict, dict]:
    """Fixed full-song extraction; no labels, roles, boundaries or feedback input.

    Preserve the previous review's requested-grid/floor-frame feature convention
    so the legacy score is a same-input control, not a new baseline in disguise.
    Both requested and actual frame endpoints are recorded in the artifact.
    """
    sr, hop = 22050, 512
    frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop)
    if frames.size < 2 or np.any(np.diff(frames) <= 0):
        raise ValueError('Beat grid collapses at feature-frame resolution')
    features, energy = {}, {}
    for name, path in stems.items():
        print(f'Extract fixed-grid acoustic evidence: {name}', flush=True)
        y, _ = librosa.load(path, sr=sr, mono=True)
        q = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, n_bins=84, fmin=librosa.note_to_hz('C1')))
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)
        if frames[0] < 0 or frames[-1] >= min(q.shape[1], rms.shape[1]):
            raise ValueError('Requested grid exceeds stem feature support')
        features[name] = np.log1p(librosa.util.sync(q, frames, aggregate=np.mean, pad=False))
        energy[name] = librosa.util.sync(rms, frames, aggregate=np.mean, pad=False)[0]
    return features, energy, {'sample_rate': sr, 'hop_length': hop,
        'requested_times_s': beat_times.tolist(), 'feature_frame_indices': frames.tolist(),
        'feature_frame_times_s': librosa.frames_to_time(frames, sr=sr, hop_length=hop).tolist(),
        'timing_note': 'Window times use requested beats; features use floor-quantized frames (< one hop earlier), matching the frozen review.'}


def verify_legacy_scores(results: list[dict], previous: list[dict], *, atol: float = 1e-6) -> dict:
    """Compare every legacy pair and stem component, not just selected maxima."""
    if len(results) != len(previous):
        raise ValueError('Legacy recurrence scale count changed')
    maximum, count = 0.0, 0
    for result, old in zip(results, previous):
        if any(result[k] != old[k] for k in ['scale_beats', 'stride_beats', 'spans', 'method']):
            raise ValueError('Legacy recurrence grid or method changed')
        pairs = {(p['a'], p['b']): p for p in result['pairs']}
        control = {(p['a'], p['b']): p for p in old['pairs']}
        if pairs.keys() != control.keys():
            raise ValueError('Legacy recurrence pair support changed')
        for key, pair in pairs.items():
            old_pair = control[key]
            if pair['stem_similarities'].keys() != old_pair['stem_similarities'].keys():
                raise ValueError('Legacy recurrence stem support changed')
            comparisons = [(pair['similarity'], old_pair['similarity'])] + [
                (v, old_pair['stem_similarities'][name]) for name, v in pair['stem_similarities'].items()]
            for new_value, old_value in comparisons:
                if new_value is None or old_value is None:
                    if new_value is not old_value:
                        raise ValueError('Legacy recurrence null support changed')
                    continue
                delta = abs(new_value - old_value)
                if not np.isfinite(delta) or delta > atol:
                    raise ValueError(f'Legacy recurrence changed: delta={delta}')
                maximum = max(maximum, delta)
            count += 1
    return {'all_legacy_pairs_preserved': True, 'pair_count': count,
            'max_absolute_error': maximum, 'tolerance': atol}


def report_markdown(reference: dict, evaluation: dict, regression: dict) -> str:
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    def score(value):
        return 'unknown' if value is None else f'{value:.3f}'
    lines = ['# Structural development evaluation', '', reference['source']['song_title'], '',
        'This separates evidence dimensions; it does **not** claim improved section detection.', '',
        'Human motif groups are preserved exactly. Variation and transition tags are explicit analyst interpretations.',
        'Unknown certainty stays unknown. No new negative labels, hierarchy or timing tolerance is assumed.', '',
        f"Legacy recurrence control: {regression['pair_count']} pairs preserved; maximum score error {regression['max_absolute_error']:.3g}.", '',
        '## What the implementation can and cannot infer', '',
        '- Pattern similarity: ordered spectral resemblance across shared active stems; not a musical-identity classifier.',
        '- Arrangement similarity: RMS agreement and stem presence, kept separate from pattern.',
        '- Local change: comparison with the closest complete non-overlapping previous window.',
        '- Historical novelty: one minus the best comparable prior-window pattern score; no history gives null, not surprise.',
        '- Context uses the entire target phrase and offline whole-song audibility calibration; do not trigger visuals at the window start as if the score were an instantaneous onset.',
        '- Transitions: exact annotated intervals are representable; an automatic transition-interval detector is still absent.',
        '- Existing section roles/letters are retained only as a legacy comparison, never promoted to identity.', '']
    for ref_layer, layer in zip(reference['layers'], evaluation['layers']):
        names = {g['id']: g['name'] for g in ref_layer['identity_groups']}
        lines += [f"## Layer: {cell(layer['name'])}", '',
            '| Start–end (s) | User label | Explicit identity | Interpreted variation | Interpreted transition |',
            '| --- | --- | --- | --- | --- |']
        for span in ref_layer['spans']:
            lines.append(f"| {span['start_s']:.3f}–{span['end_s']:.3f} | {cell(span['label'])} | {cell(names.get(span['identity_id'], 'unknown'))} | {cell(span['variation'] or 'unknown')} | {'yes' if span['transition'] is True else 'no' if span['transition'] is False else 'unknown'} |")
        lines += ['', f"{len(layer['transition_intervals'])} interpreted transition intervals are retained without merging.", '',
            '### Boundary evidence', '',
            'Distances below are many-to-one nearest-cut descriptions, not precision/recall matches.', '',
            '| User time (s) | Identity relation | Variation change | Transition edge | Nearest detector cut (s) | Detector minus user (s) |',
            '| --- | --- | --- | --- | --- | --- |']
        for row in layer['boundary_evidence']:
            edges = ', '.join(name for name in ['start', 'end'] if row['transition_' + name]) or '—'
            variation = 'unknown' if row['variation_change'] is None else 'yes' if row['variation_change'] else 'no'
            lines.append(f"| {row['time_s']:.3f} | {row['identity_relation']} | {variation} | {edges} | {score(row['nearest_detector_boundary_s'])} | {score(row['detector_minus_annotation_s'])} |")
        for scale in layer['phrase_evidence']:
            lines += ['', f"### {scale['scale_beats']}-beat phrase evidence", '',
                f"{scale['window_count']} fixed-grid windows; {scale['cross_boundary_window_count']} cross a user mark and lack single-variation support. {scale['identity_window_count']} have explicit identity support, including {scale['identity_windows_crossing_variations']} that cross subdivisions within the same motif.", '',
                'Spans with no complete contained window: ' + ', '.join(cell(s['label']) for s in scale['span_coverage'] if not s['complete_windows']) + '.', '',
                'Return summaries use all eligible pairs. Sliding windows are correlated, not independent samples.', '',
                '| Explicit group | Relation | Variation relationship | Pairs | Mean pattern | Mean arrangement | Mean legacy combined |',
                '| --- | --- | --- | --- | --- | --- | --- |']
            for row in scale['pair_summaries']:
                lines.append(f"| {cell(names.get(row['identity_id'], 'unlabeled other groups'))} | {row['relation']} | {row['variation_relation']} | {row['pair_count']} | {score(row['pattern_similarity']['mean'])} | {score(row['arrangement_similarity']['mean'])} | {score(row['similarity']['mean'])} |")
            lines += ['', 'Strongest pattern examples per returning span pair (selected maxima, **not** representative scores):', '',
                '| Group | A → B window starts (s) | Variation relationship | Pattern | Arrangement | Shared active stems |',
                '| --- | --- | --- | --- | --- | --- |']
            for ex in scale['strongest_return_examples']:
                lines.append(f"| {cell(names[ex['identity_id']])} | {ex['a_start_s']:.2f} → {ex['b_start_s']:.2f} | {ex['variation_relation']} | {score(ex['pattern_similarity'])} | {score(ex['arrangement_similarity'])} | {', '.join(ex['shared_active_stems'])} |")
            if not scale['strongest_return_examples']:
                lines += ['', 'No fully contained, comparable return pairs at this scale. This is missing support, not evidence that the song has no returns.']
            lines += ['', 'Local versus historical context (means over contained windows; not boundary-event scores):', '',
                '| User span start (s) | User label | Windows | Local pattern change | Local arrangement change | Historical pattern novelty |',
                '| --- | --- | --- | --- | --- | --- |']
            starts = {s['id']: s['start_s'] for s in ref_layer['spans']}
            for row in scale['context_summaries']:
                lines.append(f"| {starts[row['span_id']]:.2f} | {cell(row['label'])} | {row['window_count']} | {score(row['local_pattern_change']['mean'])} | {score(row['local_arrangement_change']['mean'])} | {score(row['historical_pattern_novelty']['mean'])} |")
    lines += ['', '## Limits and next step', ''] + ['- ' + s for s in evaluation['limitations']]
    lines += ['', 'Next: test shorter-scale change/transition proposals alongside recurrence evidence, preserving returning identity across arrangement changes. Keep this song as development data and reserve new listening examples before any general-quality claim.', '']
    return '\n'.join(lines)


def build(*, feedback: Path, interpretations: Path | None, parent: Path, editor: Path, out: Path) -> None:
    feedback, parent, editor, out = (p.resolve() for p in [feedback, parent, editor, out])
    interpretations = interpretations.resolve() if interpretations else None
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    if any(p == out or p in out.parents for p in [parent, editor, feedback.parent]):
        raise ValueError('Output must be separate from input packages')
    raw = json.loads(feedback.read_text())
    mapped = json.loads(interpretations.read_text()) if interpretations else None
    reference = normalize_annotations(raw, feedback_sha256=sha256_file(feedback), interpretations=mapped)
    manifest_path = parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    verify(editor/'manifest.json', raw['manifest_sha256'])
    editor_data = json.loads((editor/'editor.json').read_text())
    editor_manifest = json.loads((editor/'manifest.json').read_text())
    editor_records = {Path(r['path']).name: r for r in editor_manifest['outputs']}
    verify(editor/'editor.json', editor_records['editor.json']['sha256'])
    for key in ['audio_sha256', 'source_audio_sha256', 'duration_s', 'song_title']:
        if editor_data[key] != raw['source'][key]:
            raise ValueError(f'Feedback differs from editor: {key}')
    records = {Path(r['path']).name: r for r in manifest['outputs']}
    consumed = [feedback, manifest_path, editor/'manifest.json', editor/'editor.json']
    if interpretations:
        consumed.append(interpretations)
    for name in ['original.wav', 'candidate-story.json', 'recurrence.json']:
        verify(parent/name, records[name]['sha256'])
        consumed.append(parent/name)
    verify(parent/'original.wav', raw['source']['audio_sha256'])
    if abs(sf.info(parent/'original.wav').duration - raw['source']['duration_s']) > 1e-6:
        raise ValueError('Audio duration differs from feedback')
    source_records = [r for r in manifest['sources'] if r['sha256'] == raw['source']['source_audio_sha256']]
    if len(source_records) != 1:
        raise ValueError('Original source fingerprint not found uniquely')
    verify(ROOT/source_records[0]['path'], source_records[0]['sha256'])
    consumed.append(ROOT/source_records[0]['path'])
    stems = {}
    for name in STEMS:
        matches = [r for r in manifest['sources'] if Path(r['path']).name == name+'.wav' and Path(r['path']).parent.name == 'stems']
        if len(matches) != 1:
            raise ValueError(f'Expected one cached {name} stem')
        path = ROOT/matches[0]['path']
        verify(path, matches[0]['sha256'])
        if abs(sf.info(path).duration - raw['source']['duration_s']) > .05:
            raise ValueError(f'Stem duration differs from feedback: {name}')
        stems[name] = path
        consumed.append(path)
    story = json.loads((parent/'candidate-story.json').read_text())
    if abs(story['meta']['duration_s'] - raw['source']['duration_s']) > 1e-6:
        raise ValueError('Story duration differs from feedback')
    grid = story['meta']['beat_grid']
    if not grid['explicit'] or grid['fallback'] is not None:
        raise ValueError('Expected the reviewed explicit beat grid without fallback')
    bt = np.asarray(grid['requested_times_s'], dtype=float)
    if bt.ndim != 1 or not np.isfinite(bt).all() or bt.size < 2 or bt[0] < 0 or bt[-1] > raw['source']['duration_s'] or np.any(np.diff(bt) <= 0):
        raise ValueError('Invalid requested beat grid')
    previous = json.loads((parent/'recurrence.json').read_text())
    if previous['beat_grid_sha256'] != grid['requested_sha256']:
        raise ValueError('Recurrence and story grids differ')
    consumed += [ROOT/rel for rel in CODE]
    before = [record(p) for p in consumed]
    features, energy, timing = extract_features(stems, bt)
    results = [compare_phrases(features, energy, bt, scale_beats=n, stride_beats=4) for n in [16, 32]]
    regression = verify_legacy_scores(results, previous['results'])
    evaluation = evaluate_structure(reference, story['sections'], results)
    # No writes until input validation, acoustic extraction and the control pass.
    out.mkdir(parents=True)
    for path in [feedback] + ([interpretations] if interpretations else []):
        dest = out/'inputs'/path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    for rel in CODE:
        dest = out/'inputs'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, dest)
    shutil.copy2(manifest_path, out/'inputs'/'parent-manifest.json')
    shutil.copy2(editor/'manifest.json', out/'inputs'/'editor-manifest.json')
    write_json(out/'reference.json', reference)
    write_json(out/'evaluation.json', evaluation)
    write_json(out/'recurrence.json', {'beat_grid_sha256': grid['requested_sha256'], 'results': results})
    write_json(out/'timing.json', timing)
    write_json(out/'legacy-sections.json', {'sections': story['sections'], 'control': regression})
    np.savez_compressed(out/'features.npz', beat_times_s=bt,
        **{name+'_features': v for name, v in features.items()}, **{name+'_rms': v for name, v in energy.items()})
    (out/'report.md').write_text(report_markdown(reference, evaluation, regression))
    for r in before:
        verify(ROOT/r['path'], r['sha256'])
    write_json(out/'manifest.json', {'schema_version': 1, 'kind': 'songviz-structural-development-evaluation',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'sources': before,
        'versions': {'librosa': librosa.__version__, 'numpy': np.__version__, 'soundfile': sf.__version__},
        'settings': {'scales_beats': [16, 32], 'stride_beats': 4, 'feedback_used_in_acoustic_extraction': False},
        'legacy_control': regression, 'capabilities': evaluation['capabilities'],
        'input_snapshots': [record(p) for p in sorted((out/'inputs').rglob('*')) if p.is_file()],
        'outputs': [record(p) for p in sorted(out.iterdir()) if p.is_file()]})
    print(f'Ready: {out}/report.md', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feedback', type=Path, default=ROOT/'benchmark/feedback/section-editor-02.json')
    parser.add_argument('--interpretations', type=Path,
                        help='Optional, fingerprint-bound analyst mapping; no label interpretation by default')
    parser.add_argument('--parent', type=Path, default=ROOT/'outputs/reviews/structure-review-03')
    parser.add_argument('--editor', type=Path, default=ROOT/'outputs/reviews/section-editor-02')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build(**vars(args))
