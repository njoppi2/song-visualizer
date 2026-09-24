"""Create an isolated, audio-linked boundary/recurrence review; never replace caches."""
from __future__ import annotations

import argparse
from copy import deepcopy
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
from songviz.structure_annotations import normalize_annotations


HUMAN_ANNOTATIONS_SHA256 = 'dd100b34d43ece3dbe500b297f0fb4320db71e3eebb9dc705959145e54aa8b2f'
LISTENING_FEEDBACK_SHA256 = 'f69b14f0343d0ebe4af12d37f66e88c76aa5d8e7673f7529345333589d5c47b6'
LISTENING_EXAMPLE_IDS = ('drum-entry', 'within-passage', 'verse-ending', 'transition-extent')
LISTENING_CHANGE_VALUES = {'none', 'subtle', 'local', 'broad'}
HUMAN_BOUNDARY_DIAGNOSTIC_THRESHOLD_S = 1.0


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_records(records):
    for record in records:
        if sha256_file(ROOT / record['path']) != record['sha256']:
            raise ValueError(f"Changed input: {record['path']}")


def embedded_json(value):
    """Serialize data safely for the self-contained HTML page."""
    return json.dumps(value, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def _finite_time(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f'{field} must be a finite number')
    return float(value)


def _same_time(left, right, *, tolerance=1e-6):
    return abs(float(left) - float(right)) <= tolerance


def _record_by_name(records, name):
    matches = [record for record in records if Path(record.get('path', '')).name == name]
    if len(matches) != 1:
        raise ValueError(f'Parent manifest does not uniquely bind {name}')
    return matches[0]


def _verify_overlay_parent(parent: Path):
    """Return verified frozen review inputs without recomputing any features."""
    manifest_path = parent / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    outputs = manifest.get('outputs')
    sources = manifest.get('sources')
    if not isinstance(outputs, list):
        raise ValueError('Parent manifest lacks output fingerprints')
    audio_record = _record_by_name(outputs, 'original.wav')
    review_record = _record_by_name(outputs, 'review.json')
    for path, record in ((parent / 'original.wav', audio_record), (parent / 'review.json', review_record)):
        if not path.is_file() or sha256_file(path) != record.get('sha256'):
            raise ValueError(f'Changed parent input: {path.name}')
    # The original review has source records; an overlay instead fingerprints
    # its original parent manifest in reference_inputs. Follow that verified
    # link so a later overlay can still prove the original-song hash.
    provenance_manifest = manifest
    expected_source_sha = manifest.get('source_audio_sha256')
    while not (isinstance(provenance_manifest.get('sources'), list) and provenance_manifest['sources']):
        parent_record = provenance_manifest.get('parent')
        if not isinstance(parent_record, dict) or Path(parent_record.get('path', '')).name != 'manifest.json':
            raise ValueError('Overlay parent does not uniquely bind its prior manifest')
        prior_manifest_path = ROOT / parent_record['path']
        if not prior_manifest_path.is_file() or sha256_file(prior_manifest_path) != parent_record.get('sha256'):
            raise ValueError('Changed prior manifest in overlay provenance')
        provenance_manifest = json.loads(prior_manifest_path.read_text())
        inherited_source_sha = provenance_manifest.get('source_audio_sha256')
        if expected_source_sha is not None and inherited_source_sha is not None and inherited_source_sha != expected_source_sha:
            raise ValueError('Overlay source hash does not match its verified parent')
    source_record = provenance_manifest['sources'][0]
    if expected_source_sha is not None and expected_source_sha != source_record.get('sha256'):
        raise ValueError('Overlay source hash does not match its verified original source')
    # Structure-review manifests also fingerprint stem WAVs as sources. The
    # first record is the immutable original-song record established by the
    # parent builder; later source records are cached analyses/stems.
    if not isinstance(source_record, dict) or Path(source_record.get('path', '')).suffix.lower() not in {'.flac', '.mp3', '.wav', '.ogg', '.m4a'}:
        raise ValueError('Parent manifest does not identify the original source audio first')
    source_path = ROOT / source_record['path']
    if not source_path.is_file() or sha256_file(source_path) != source_record.get('sha256'):
        raise ValueError('Changed source audio in parent manifest')
    review = json.loads((parent / 'review.json').read_text())
    duration = _finite_time(review.get('duration_s'), 'parent review duration_s')
    sections = review.get('sections')
    if not isinstance(sections, dict) or not isinstance(sections.get('candidate'), list):
        raise ValueError('Parent review has no candidate section timeline')
    return manifest, review, source_record, audio_record, duration


def _human_sections(annotation: dict, *, annotation_sha256: str, source_audio_sha256: str,
                    audio_sha256: str, duration_s: float):
    """Validate the raw editor export and retain only authored section fields."""
    normalized = normalize_annotations(annotation, feedback_sha256=annotation_sha256)
    source = normalized['source']
    if source['source_audio_sha256'] != source_audio_sha256 or source['audio_sha256'] != audio_sha256:
        raise ValueError('Human annotations are not bound to the parent source audio')
    if not _same_time(source['duration_s'], duration_s):
        raise ValueError('Human annotation duration does not match the parent review')
    layers = normalized.get('layers')
    if not isinstance(layers, list) or len(layers) != 1:
        raise ValueError('Expected one authored human annotation layer')
    spans = layers[0].get('spans')
    if not isinstance(spans, list) or len(spans) != 19:
        raise ValueError('Expected exactly 19 authored human spans')
    if any(span.get('certainty') != 'unspecified' for span in spans):
        raise ValueError('Human annotation certainty must remain unspecified')
    sections = []
    for span in spans:
        sections.append({
            'id': span['id'], 'start_s': span['start_s'], 'end_s': span['end_s'],
            'label': span['label'], 'motif': span['motif'] or None,
            'motif_id': span['identity_id'], 'certainty': span['certainty'],
        })
    return sections


def _listening_cards(listening_review: dict, feedback: dict, *, source_audio_sha256: str,
                     audio_sha256: str, duration_s: float):
    """Join four hash-bound raw answers to their immutable source-time excerpts."""
    for key, expected in (('source_audio_sha256', source_audio_sha256), ('audio_sha256', audio_sha256)):
        if listening_review.get(key) != expected or feedback.get(key) != expected:
            raise ValueError(f'Listening {key} does not match the parent source audio')
    if not _same_time(listening_review.get('duration_s'), duration_s):
        raise ValueError('Listening review duration does not match the parent review')
    examples = listening_review.get('examples')
    answers = feedback.get('answers')
    if not isinstance(examples, list) or not isinstance(answers, list):
        raise ValueError('Listening review lacks examples or answers')
    if {example.get('id') for example in examples} != set(LISTENING_EXAMPLE_IDS):
        raise ValueError('Listening review does not contain the four expected excerpts')
    answer_by_id = {answer.get('example_id'): answer for answer in answers if isinstance(answer, dict)}
    if set(answer_by_id) != set(LISTENING_EXAMPLE_IDS) or len(answer_by_id) != len(answers):
        raise ValueError('Listening feedback must contain each of the four excerpts exactly once')
    cards = []
    for example in examples:
        answer = answer_by_id[example['id']]
        start = _finite_time(example.get('start_s'), 'listening excerpt start_s')
        end = _finite_time(example.get('end_s'), 'listening excerpt end_s')
        if not 0 <= start < end <= duration_s:
            raise ValueError('Listening excerpt is outside the parent source duration')
        if answer.get('perceived_change') not in LISTENING_CHANGE_VALUES or not isinstance(answer.get('notes'), str):
            raise ValueError('Listening feedback has an invalid perceived-change value or note')
        cards.append({
            'id': example['id'], 'title': example.get('title', example['id']), 'start_s': start, 'end_s': end,
            'focus_start_s': _finite_time(example.get('focus_start_s'), 'listening focus_start_s'),
            'focus_end_s': _finite_time(example.get('focus_end_s'), 'listening focus_end_s'),
            'perceived_change': answer['perceived_change'], 'notes': answer['notes'],
        })
    return cards


def listening_window_lane(cards: list[dict]):
    """Prepare a display-only lane from raw guided-listening windows.

    The state is only a visual grouping of the listener's supplied
    ``perceived_change`` value.  It does not create an event point, a change
    duration, or a comparison with either section timeline.
    """
    if len(cards) != len(LISTENING_EXAMPLE_IDS) or {card.get('id') for card in cards} != set(LISTENING_EXAMPLE_IDS):
        raise ValueError('Expected the four raw guided-listening windows')
    windows = []
    for card in cards:
        perceived_change = card.get('perceived_change')
        if perceived_change not in LISTENING_CHANGE_VALUES or not isinstance(card.get('notes'), str):
            raise ValueError('Listening window must preserve raw change label and note text')
        start = _finite_time(card.get('start_s'), 'listening window start_s')
        end = _finite_time(card.get('end_s'), 'listening window end_s')
        if not start < end:
            raise ValueError('Listening window must have an increasing raw source range')
        windows.append({
            'id': card['id'], 'start_s': start, 'end_s': end,
            'perceived_change': perceived_change, 'notes': card['notes'],
            'listener_state': 'none_control' if perceived_change == 'none' else 'change_heard',
        })
    return windows


def human_candidate_disagreements(human_sections: list[dict], candidate_sections: list[dict], duration_s: float,
                                  threshold_s: float = HUMAN_BOUNDARY_DIAGNOSTIC_THRESHOLD_S):
    """List timing-only human/candidate boundary disagreements for review.

    The signed value is candidate time minus human time. It intentionally does
    not compare the unrelated user-label and heuristic-role taxonomies.
    """
    if threshold_s <= 0:
        raise ValueError('Diagnostic threshold must be positive')
    candidates = [_finite_time(section.get('start_s'), 'candidate boundary') for section in candidate_sections[1:]]
    records = []
    for section in human_sections[1:]:
        human_time = _finite_time(section.get('start_s'), 'human boundary')
        if not candidates:
            raise ValueError('Candidate timeline has no boundaries to compare')
        candidate_time = min(candidates, key=lambda value: (abs(value - human_time), value))
        signed_offset = candidate_time - human_time
        if abs(signed_offset) > threshold_s:
            records.append({
                'human_boundary_s': human_time, 'candidate_boundary_s': candidate_time,
                'candidate_minus_human_s': signed_offset,
                'context_start_s': max(0.0, human_time - 8.0),
                'context_end_s': min(duration_s, human_time + 8.0),
            })
    return records


def prepare_human_reference_data(annotation: dict, *, annotation_sha256: str, listening_review: dict,
                                 feedback: dict, source_audio_sha256: str, audio_sha256: str,
                                 duration_s: float, candidate_sections: list[dict]):
    """Pure preparation for the overlay page; it never computes audio features."""
    human_sections = _human_sections(annotation, annotation_sha256=annotation_sha256,
                                     source_audio_sha256=source_audio_sha256, audio_sha256=audio_sha256,
                                     duration_s=duration_s)
    cards = _listening_cards(listening_review, feedback, source_audio_sha256=source_audio_sha256,
                             audio_sha256=audio_sha256, duration_s=duration_s)
    return {
        'human_sections': human_sections,
        'listening_feedback': cards,
        'listening_window_lane': listening_window_lane(cards),
        'boundary_diagnostic': {
            'threshold_s': HUMAN_BOUNDARY_DIAGNOSTIC_THRESHOLD_S,
            'meaning': ('Declared review threshold only. It is not a universal timing tolerance, proof of a musical error, '
                        'or a semantic comparison of user labels with heuristic roles.'),
            'disagreements': human_candidate_disagreements(human_sections, candidate_sections, duration_s),
        },
    }


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


def build_human_reference_overlay(out: Path, parent: Path, annotations: Path, listening: Path, feedback: Path):
    """Create a new source-linked review page from frozen data only.

    This is deliberately separate from :func:`build`: it copies the already
    verified review audio/timelines and does not load audio, stems, or features.
    """
    out, parent, annotations, listening, feedback = (path.resolve() for path in (out, parent, annotations, listening, feedback))
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    if parent in out.parents or listening in out.parents:
        raise ValueError('Overlay output must be separate from frozen input packages')
    manifest, parent_review, source_record, audio_record, duration_s = _verify_overlay_parent(parent)
    page_asset_names = ('sections-comparison.png', 'novelty-comparison.png', 'evidence.png')
    page_assets = [_record_by_name(manifest.get('outputs', []), name) for name in page_asset_names]
    for asset in page_assets:
        source_path = parent / asset['path'].split('/')[-1]
        if not source_path.is_file() or sha256_file(source_path) != asset.get('sha256'):
            raise ValueError(f'Changed parent page asset: {source_path.name}')
    if sha256_file(annotations) != HUMAN_ANNOTATIONS_SHA256:
        raise ValueError('Human annotation SHA-256 does not match the frozen export')
    if sha256_file(feedback) != LISTENING_FEEDBACK_SHA256:
        raise ValueError('Listening feedback SHA-256 does not match the frozen export')
    listening_manifest_path = listening / 'manifest.json'
    listening_review_path = listening / 'review.json'
    if not listening_manifest_path.is_file() or not listening_review_path.is_file():
        raise FileNotFoundError('Listening package must contain manifest.json and review.json')
    listening_manifest = json.loads(listening_manifest_path.read_text())
    if listening_manifest.get('kind') != 'songviz-listening-examples':
        raise ValueError('Expected a frozen guided-listening package')
    listening_record = _record_by_name(listening_manifest.get('outputs', []), 'review.json')
    if sha256_file(listening_review_path) != listening_record.get('sha256'):
        raise ValueError('Changed guided-listening review')
    listening_review = json.loads(listening_review_path.read_text())
    feedback_data = json.loads(feedback.read_text())
    if feedback_data.get('kind') != 'songviz-listening-examples-feedback' or feedback_data.get('schema_version') != 1:
        raise ValueError('Feedback is not a guided-listening export')
    if feedback_data.get('review_sha256') != sha256_file(listening_review_path):
        raise ValueError('Listening feedback is not hash-bound to the guided-listening review')
    if feedback_data.get('example_set_id') != listening_review.get('example_set_id'):
        raise ValueError('Listening feedback example set does not match the guided-listening review')
    annotation_data = json.loads(annotations.read_text())
    reference = prepare_human_reference_data(
        annotation_data, annotation_sha256=HUMAN_ANNOTATIONS_SHA256,
        listening_review=listening_review, feedback=feedback_data,
        source_audio_sha256=source_record['sha256'], audio_sha256=audio_record['sha256'],
        duration_s=duration_s, candidate_sections=parent_review['sections']['candidate'],
    )
    review = deepcopy(parent_review)
    review['audio_path'] = 'original.wav'
    review['sections'] = {
        'human': reference['human_sections'],
        'baseline': parent_review['sections']['baseline'],
        'candidate': parent_review['sections']['candidate'],
    }
    review['human_reference'] = {
        'kind': 'source-matched-human-annotation', 'source': 'user marks',
        'certainty': 'unspecified', 'annotation_sha256': HUMAN_ANNOTATIONS_SHA256,
        'note': 'These labels are user marks, not external ground truth. Empty motifs remain unknown identities.',
    }
    review['listening_feedback'] = reference['listening_feedback']
    review['listening_window_lane'] = reference['listening_window_lane']
    review['boundary_diagnostic'] = reference['boundary_diagnostic']
    review['notes'] = [
        'Human annotations are user marks with certainty unspecified; they are not external ground truth.',
        'Baseline and candidate timelines are heuristic. The candidate is unvalidated; its role letters are not compared semantically with user labels.',
        'Raw listening judgments and optional acoustic/model material remain separate on this page.',
    ]
    snapshot_rels = [
        'experiments/build_structure_review.py', 'experiments/templates/structure_review.html',
        'songviz/structure_annotations.py', 'songviz/ingest.py', 'experiments/build_review.py',
    ]
    if any(not (ROOT / rel).is_file() for rel in snapshot_rels):
        raise FileNotFoundError('Required source snapshot is missing')
    out.mkdir(parents=True)
    inputs = out / 'inputs'; inputs.mkdir()
    for rel in snapshot_rels:
        destination = inputs / rel; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, destination)
    for source_path, name in (
        (parent / 'manifest.json', 'parent-manifest.json'), (parent / 'review.json', 'parent-review.json'),
        (annotations, 'human-annotations.json'), (listening_manifest_path, 'listening-manifest.json'),
        (listening_review_path, 'listening-review.json'), (feedback, 'listening-feedback.json'),
    ):
        shutil.copy2(source_path, inputs / name)
    shutil.copy2(parent / 'original.wav', out / 'original.wav')
    if sha256_file(out / 'original.wav') != audio_record['sha256']:
        raise ValueError('Copied review audio does not exactly match the verified frozen parent')
    for asset in page_assets:
        name = Path(asset['path']).name
        shutil.copy2(parent / name, out / name)
        if sha256_file(out / name) != asset['sha256']:
            raise ValueError(f'Copied parent page asset does not match its frozen hash: {name}')
    write_json(out / 'review.json', review)
    provenance = [
        fingerprint(parent / 'manifest.json'), fingerprint(parent / 'review.json'), fingerprint(annotations),
        fingerprint(listening_manifest_path), fingerprint(listening_review_path), fingerprint(feedback),
    ]
    result_manifest = {
        'schema_version': 1, 'kind': 'songviz-structure-review-human-reference',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'parent': fingerprint(parent / 'manifest.json'), 'source_audio_sha256': source_record['sha256'],
        'audio_sha256': audio_record['sha256'], 'reference_inputs': provenance,
        'input_snapshots': [fingerprint(path) for path in sorted(inputs.rglob('*')) if path.is_file()],
        'outputs': [fingerprint(out / name) for name in ('original.wav', 'review.json', *page_asset_names)],
        'scope': ('Frozen parent timelines plus source-matched human annotations and raw listening feedback. '
                  'No audio features, stories, boundaries, roles, or model outputs were recomputed.'),
        'page_integrity': 'index.html derives from the snapshotted template, review.json, and this manifest hash.',
    }
    write_json(out / 'manifest.json', result_manifest)
    template = (inputs / 'experiments/templates/structure_review.html').read_text()
    (out / 'index.html').write_text(template.replace('{{REVIEW_JSON}}', embedded_json(review)).replace('{{MANIFEST_SHA}}', sha256_file(out / 'manifest.json')))
    for record in provenance:
        path = ROOT / record['path'] if not Path(record['path']).is_absolute() else Path(record['path'])
        if sha256_file(path) != record['sha256']:
            raise ValueError(f'Input changed while building: {path}')
    print(f'Ready: {out}/index.html', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=ROOT/'outputs/reviews/structure-grid-01')
    parser.add_argument('--human-reference', action='store_true',
                        help='Copy a frozen structure review and add verified human/reference records without recomputing features.')
    parser.add_argument('--annotations', type=Path, default=ROOT/'benchmark/feedback/section-editor-02.json')
    parser.add_argument('--listening', type=Path, default=ROOT/'outputs/reviews/listening-examples-01')
    parser.add_argument('--feedback', type=Path, default=ROOT/'benchmark/feedback/listening-examples-01.json')
    args = parser.parse_args()
    if args.human_reference:
        build_human_reference_overlay(args.out, args.parent, args.annotations, args.listening, args.feedback)
    else:
        build(args.out.resolve(), args.parent.resolve())
