"""Create or replay a cached-evidence directing study, with no model/API calls.

Default: the reviewed Feel Good Inc transition/return at 130–178s. The planner
inspects song-wide percussion gaps and stem energy, not hardcoded section roles.
--replay uses a prior package's saved signals, plan and original PCM exclusively.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.build_review import cut_audio, fingerprint, run
from experiments.build_visual_passage import reviewed_inputs, verify_hash
from songviz.direction import make_plan, fixed_plan, validate_plan
from songviz.ingest import sha256_file
from songviz.render import RenderConfig, _render_mp4_with_visualizer


def stem_energy(path: Path) -> tuple[list, list, dict]:
    """Stereo-energy RMS in completed 100ms blocks, p95-normalized per stem."""
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    hop = round(sr * .1)
    values = np.array([np.sqrt(np.mean(chunk.astype(float) ** 2)) for chunk in (y[i:i+hop] for i in range(0, len(y), hop))])
    scale = max(float(np.quantile(values, .95)), 1e-7)
    times = np.minimum(np.arange(1, len(values)+1) * hop, len(y)) / sr
    return [0., *times.tolist()], [0., *np.minimum(values / scale, 1).tolist()], {"path": fingerprint(path), "normalization_rms_p95": scale, "hop_s": hop / sr, "timestamps": "completed-block end; initial zero"}


def prepare(review: Path, start: float, end: float) -> tuple[dict, dict, np.ndarray, int, dict]:
    data = reviewed_inputs(review)
    source = ROOT / data['manifest']['source_audio']['path']
    root = ROOT / 'outputs' / source.stem / 'stems'
    energy, provenance = {}, []
    for name in ['bass', 'vocals', 'other']:
        times, values, meta = stem_energy(root / (name + '.wav'))
        if energy and energy['times_s'] != times:
            raise ValueError('Stem energy timelines must match')
        energy['times_s'] = times
        energy[name] = values
        provenance.append(meta)
    signals = {'beat_times_s': data['timing']['candidate']['beat_times_s'], 'hits': data['events']['hits'], 'energy': energy}
    plan = make_plan(signals, start, end)
    pcm, sr = cut_audio(source, start, end)
    origins = {'source_audio': data['manifest']['source_audio'], 'parent_review': fingerprint(review / 'manifest.json'), 'energy_sources': provenance, 'limitations': 'Rule-based development study on previously reviewed source. P95 scales differ by stem. Separation bleed and musical focus correctness remain uncertain.'}
    return signals, plan, pcm, sr, origins


def replay_inputs(package: Path, override_plan: Path | None = None):
    manifest = json.loads((package / 'manifest.json').read_text())
    for name in ['signals.json', 'plan.json', 'original.wav']:
        _recorded_output(package, manifest, name)
    signals = json.loads((package / 'signals.json').read_text())
    saved = json.loads((package / 'plan.json').read_text())
    plan = json.loads(override_plan.read_text()) if override_plan else saved
    validate_plan(plan, signals)
    if (plan['start_s'], plan['end_s']) != (saved['start_s'], saved['end_s']):
        raise ValueError('Edited replay plan must retain the audio excerpt interval')
    pcm, sr = sf.read(package / 'original.wav', always_2d=True)
    if abs(len(pcm) / sr - (plan['end_s'] - plan['start_s'])) > 1 / sr:
        raise ValueError('Replay audio duration does not match plan')
    origins = {**manifest['origins'], 'replayed_from': fingerprint(package / 'manifest.json')}
    if override_plan:
        origins['edited_plan'] = fingerprint(override_plan)
    return signals, plan, pcm, sr, origins


def _recorded_output(package: Path, manifest: dict, name: str) -> dict:
    """Return one verified output record, never selecting by unchecked metadata."""
    matches = [record for record in manifest.get('outputs', []) if Path(record.get('path', '')).name == name]
    if len(matches) != 1:
        raise ValueError(f'Package must record exactly one {name}')
    record = matches[0]
    path = package / name
    if Path(record['path']).resolve() != path.resolve():
        raise ValueError(f'Recorded {name} path does not bind this package')
    verify_hash(path, record['sha256'])
    return record


def _recorded_snapshot(package: Path, manifest: dict, name: str) -> dict:
    """Return one verified input snapshot by basename, rejecting ambiguity."""
    matches = [record for record in manifest.get('input_snapshots', [])
               if Path(record.get('path', '')).name == name]
    if len(matches) != 1:
        raise ValueError(f'Package must record exactly one input snapshot {name}')
    record = matches[0]
    path = package / 'inputs' / name
    if Path(record['path']).resolve() != path.resolve():
        raise ValueError(f'Recorded input snapshot {name} does not bind this package')
    verify_hash(path, record['sha256'])
    return record


def _verify_frozen_manifest(package: Path, manifest: dict) -> None:
    """Verify every recorded artifact and snapshot against the artifact itself."""
    for record in manifest.get('outputs', []):
        path = Path(record.get('path', ''))
        if path.resolve().parent != package.resolve() and package.resolve() not in path.resolve().parents:
            raise ValueError('Recorded output path does not bind this package')
        verify_hash(path, record['sha256'])
    for record in manifest.get('input_snapshots', []):
        path = Path(record.get('path', ''))
        if package.resolve() not in path.resolve().parents:
            raise ValueError('Snapshot path does not bind this package')
        verify_hash(path, record['sha256'])

    def verify_origins(value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                path = ROOT / value['path'] if not Path(value['path']).is_absolute() else Path(value['path'])
                verify_hash(path, value['sha256'])
            for child in value.values(): verify_origins(child)
        elif isinstance(value, list):
            for child in value: verify_origins(child)
    verify_origins(manifest.get('origins', {}))


def gradual_inputs(package: Path):
    """Load the frozen three-way source without analysis or planning.

    The parent plan is deliberately retained alongside the new gradual plan: it
    is a control, not merely metadata from which a replacement may be inferred.
    """
    manifest_path = package / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    _verify_frozen_manifest(package, manifest)
    for name in ('signals.json', 'plan.json', 'fixed-plan.json', 'original.wav'):
        _recorded_output(package, manifest, name)
    signals = json.loads((package / 'signals.json').read_text())
    baseline = json.loads((package / 'plan.json').read_text())
    fixed = json.loads((package / 'fixed-plan.json').read_text())
    validate_plan(baseline, signals)
    validate_plan(fixed, signals)
    if (fixed['start_s'], fixed['end_s']) != (baseline['start_s'], baseline['end_s']):
        raise ValueError('Source package comparison plans must retain one audio interval')
    pcm, sr = sf.read(package / 'original.wav', always_2d=True)
    if abs(len(pcm) / sr - (baseline['end_s'] - baseline['start_s'])) > 1 / sr:
        raise ValueError('Source package audio duration does not match baseline plan')
    origins = {'gradual_from': fingerprint(manifest_path), 'source_manifest_origins': manifest.get('origins'),
               'limitations': 'Saved-plan development comparison. Artistic acceptance remains pending; rule-based proxies do not establish musical roles.'}
    return signals, baseline, fixed, pcm, sr, origins


def three_way_replay_inputs(package: Path, override_plan: Path | None = None):
    """Replay an already-built gradual package exactly; do not call a planner."""
    manifest_path = package / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('comparison_kind') not in {'gradual_three_way', 'gradual_three_way_replay'}:
        raise ValueError('Replay package is not a frozen three-way comparison')
    _verify_frozen_manifest(package, manifest)
    for name in ('signals.json', 'plan.json', 'baseline-plan.json', 'fixed-plan.json', 'original.wav'):
        _recorded_output(package, manifest, name)
    signals = json.loads((package / 'signals.json').read_text())
    gradual = json.loads(override_plan.read_text()) if override_plan else json.loads((package / 'plan.json').read_text())
    if gradual.get('schema_version') != 2:
        raise ValueError('Three-way replay requires a version 2 gradual plan')
    baseline = json.loads((package / 'baseline-plan.json').read_text())
    fixed = json.loads((package / 'fixed-plan.json').read_text())
    for schedule in (gradual, baseline, fixed):
        validate_plan(schedule, signals)
        if (schedule['start_s'], schedule['end_s']) != (gradual['start_s'], gradual['end_s']):
            raise ValueError('Three-way replay plans must retain one audio interval')
    pcm, sr = sf.read(package / 'original.wav', always_2d=True)
    if abs(len(pcm) / sr - (gradual['end_s'] - gradual['start_s'])) > 1 / sr:
        raise ValueError('Replay audio duration does not match plan')
    origins = {'replayed_from': fingerprint(manifest_path), 'source_manifest_origins': manifest.get('origins')}
    if override_plan: origins['edited_plan'] = fingerprint(override_plan)
    return signals, gradual, baseline, fixed, pcm, sr, origins


def validate_visual_comparison(plan: dict, baseline: dict) -> None:
    """A visual comparison keeps the complete direction schedule fixed."""
    for key in ('schema_version', 'start_s', 'end_s', 'signals_sha256', 'evidence'):
        if plan.get(key) != baseline.get(key):
            raise ValueError(f'Visual comparison must preserve {key}')
    def schedule(value):
        segments = copy.deepcopy(value['segments'])
        for segment in segments:
            for layer in segment['layers'].values():
                layer.pop('treatment', None)
                layer.pop('anchor', None)
        return segments
    if schedule(plan) != schedule(baseline):
        raise ValueError('Visual comparison must preserve the direction schedule')


def visual_inputs(package: Path, *, replay: bool = False, override_plan: Path | None = None):
    """Use the frozen gradual plan, or replay its saved visual-only revision."""
    manifest = json.loads((package / 'manifest.json').read_text())
    _verify_frozen_manifest(package, manifest)
    for name in ('signals.json', 'plan.json', 'original.wav'):
        _recorded_output(package, manifest, name)
    signals, saved, pcm, sr, _ = replay_inputs(package)
    if saved.get('schema_version') != 2:
        raise ValueError('Visual pass requires a version 2 gradual plan')
    if replay:
        _recorded_output(package, manifest, 'baseline-plan.json')
        baseline = json.loads((package / 'baseline-plan.json').read_text())
        plan = json.loads(override_plan.read_text()) if override_plan else saved
    else:
        from songviz.direction import make_visual_plan
        baseline = saved
        plan = make_visual_plan(signals, baseline)
    validate_plan(plan, signals)
    validate_plan(baseline, signals)
    validate_visual_comparison(plan, baseline)
    origins = {'replayed_from' if replay else 'visual_from': fingerprint(package / 'manifest.json'),
               'source_manifest_origins': manifest.get('origins')}
    if override_plan:
        origins['edited_plan'] = fingerprint(override_plan)
    return signals, plan, baseline, pcm, sr, origins


_VOCAL_PARENT_MANIFEST = '95c47b36345d25aa66bb16adfb1a0a7a7289e2998e2851114b05c28d6fa7cc60'
_VOCAL_SIGNALS = 'a0e5d277967f6e2a27735243fa8766dc168dbc24b25691012c7595bea771286f'
_VOCAL_SOURCE = '657af9333edafbb6cf8cc573446650773d1169b0117c784106d1895cf802cb44'
_VOCAL_FEEDBACK = 'f69b14f0343d0ebe4af12d37f66e88c76aa5d8e7673f7529345333589d5c47b6'


def _vocal_feedback_record() -> tuple[Path, dict]:
    path = ROOT / 'benchmark/feedback/listening-examples-01.json'
    record = fingerprint(path)
    if record['sha256'] != _VOCAL_FEEDBACK:
        raise ValueError('Authored vocal comparison requires the bound raw feedback file')
    raw = json.loads(path.read_text())
    answers = [answer for answer in raw.get('answers', []) if answer.get('example_id') == 'verse-ending']
    if len(answers) != 1:
        raise ValueError('Raw feedback must contain exactly one verse-ending answer')
    return path, {'file': record, 'answer': copy.deepcopy(answers[0])}


def vocal_emphasis_inputs(visual_parent: Path):
    """Build the authored two-way study from a verified visual parent.

    Unlike normal replay, this deliberately cuts a new 119--132s PCM excerpt
    from the hash-bound FLAC; the visual parent WAV starts at 130s.
    """
    manifest_path = visual_parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    _verify_frozen_manifest(visual_parent, manifest)
    for name in ('signals.json', 'plan.json', 'original.wav'):
        _recorded_output(visual_parent, manifest, name)
    if sha256_file(manifest_path) != _VOCAL_PARENT_MANIFEST:
        raise ValueError('Authored vocal comparison parent manifest hash does not match the reviewed visual parent')
    if sha256_file(visual_parent / 'signals.json') != _VOCAL_SIGNALS:
        raise ValueError('Authored vocal comparison parent signals hash does not match the reviewed visual parent')
    signals = json.loads((visual_parent / 'signals.json').read_text())
    visual_plan = json.loads((visual_parent / 'plan.json').read_text())
    validate_plan(visual_plan, signals)
    source = ROOT / 'songs/Gorillaz - Feel Good Inc (featuring De La Soul).flac'
    if sha256_file(source) != _VOCAL_SOURCE:
        raise ValueError('Authored vocal comparison source FLAC hash does not match')
    pcm, sr = cut_audio(source, 119., 132.)
    if sr != 44100 or pcm.shape != (573300, 2):
        raise ValueError('Authored vocal comparison must freshly cut 573300 stereo frames at 44.1kHz')
    feedback_path, feedback_record = _vocal_feedback_record()
    origins = {
        'vocal_emphasis_from': fingerprint(manifest_path),
        'source_audio': fingerprint(source),
        'feedback_record': feedback_record,
        'limitations': 'Authored two-plan vocal-emphasis experiment. It does not infer vocal function, laughter, leadership, or musical importance.',
    }
    return signals, visual_plan, feedback_path, feedback_record, pcm, sr, origins


def vocal_emphasis_replay_inputs(package: Path, override_plan: Path | None = None):
    """Replay paired authored plans exactly, without planning, cutting, or extraction."""
    manifest_path = package / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('comparison_kind') != 'vocal_emphasis_two_way':
        raise ValueError('Replay package is not an authored vocal-emphasis comparison')
    _verify_frozen_manifest(package, manifest)
    for name in ('signals.json', 'plan.json', 'baseline-plan.json', 'original.wav'):
        _recorded_output(package, manifest, name)
    for name in ('raw-listening-feedback.json', 'parent-manifest.json', 'parent-plan.json', 'parent-signals.json'):
        _recorded_snapshot(package, manifest, name)
    signals = json.loads((package / 'signals.json').read_text())
    reduced = json.loads(override_plan.read_text()) if override_plan else json.loads((package / 'plan.json').read_text())
    steady = json.loads((package / 'baseline-plan.json').read_text())
    validate_plan(reduced, signals)
    validate_plan(steady, signals)
    from songviz.direction import validate_vocal_emphasis_comparison
    validate_vocal_emphasis_comparison(reduced, steady, signals)
    pcm, sr = sf.read(package / 'original.wav', always_2d=True)
    if sr != 44100 or pcm.shape != (573300, 2):
        raise ValueError('Authored vocal replay PCM must be the saved 119--132s stereo excerpt')
    origins = {'replayed_from': fingerprint(manifest_path), 'source_manifest_origins': manifest.get('origins')}
    if override_plan:
        origins['edited_plan'] = fingerprint(override_plan)
    return signals, reduced, steady, pcm, sr, origins


def save_evidence_plot(path: Path, plan: dict, signals: dict) -> None:
    """Diagnostic evidence, separate from the artistic video and its claims."""
    img = Image.new('RGB', (1200, 900 if plan.get('schema_version') == 2 else 650), '#101a24')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    start, end = plan['start_s'], plan['end_s']
    def x(t): return 140 + (t-start)/(end-start)*1030
    draw.text((24, 15), 'Activity evidence and planned focus — not verified section labels', font=font, fill='white')
    draw.text((24, 42), 'Energy is normalized within each stem; curves do not compare absolute loudness.', font=font, fill='#a9bac0')
    for i, name in enumerate(['bass', 'vocals', 'other']):
        top = 90 + i*110
        draw.text((24, top+22), name, font=font, fill='#dce5ec')
        draw.line((140, top+80, 1170, top+80), fill='#344957')
        points = [(x(t), top+80-v*75) for t, v in zip(signals['energy']['times_s'], signals['energy'][name]) if start <= t <= end]
        if len(points)>1: draw.line(points, fill=['#70dcc1', '#c290f7', '#82aef4'][i], width=2)
    for i, name in enumerate(['kick', 'snare', 'hh']):
        y = 445+i*35
        draw.text((24,y-9),name,font=font,fill='#dce5ec')
        for h in signals['hits']:
            if h['component']==name and start<=h['t']<end:
                draw.line((x(h['t']),y-8,x(h['t']),y+8),fill='#f6c781',width=1)
    for s in plan['segments']:
        left, right = x(s['start_s']), x(s['end_s'])
        draw.line((left,80,left,560),fill='#526a7a')
        draw.rectangle((left,560,right-1,595),fill='#29475b' if s['palette']=='cool' else '#513831')
        draw.text((left+8,568),s['focus'],font=font,fill='white')
    if plan.get('schema_version') == 2:
        draw.text((24, 650), 'Planned gain (separate development control)', font=font, fill='#dce5ec')
        for index, name in enumerate(('bass', 'vocals', 'other')):
            gain_points, emphasis_points = [], []
            for segment in plan['segments']:
                for keyframe in segment['layers'][name]['envelope']:
                    gain_points.append((x(keyframe['t_s']), 745 - keyframe['gain'] * 65))
                    emphasis_points.append((x(keyframe['t_s']), 870 - keyframe['emphasis'] * 65))
            if len(gain_points) > 1:
                color = ('#70dcc1', '#c290f7', '#82aef4')[index]
                draw.line(gain_points, fill=color, width=3)
                draw.line(emphasis_points, fill=color, width=3)
        draw.text((24, 775), 'Planned emphasis (recorded separately; not a musical-role claim)', font=font, fill='#dce5ec')
    for t in np.linspace(start,end,7):
        draw.text((x(t)-22,612),f'{t:.1f}s',font=font,fill='#dce5ec')
    img.save(path)


def build(out: Path, *, review: Path, start: float = 130., end: float = 178., replay: Path | None = None, override_plan: Path | None = None, gradual_from: Path | None = None, visual_from: Path | None = None, vocal_emphasis_from: Path | None = None, fps: int = 60):
    from songviz.directed_render import DirectedVisualizer

    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    if override_plan and not replay:
        raise ValueError('--plan requires --replay to bind it to saved signal/audio evidence')
    if gradual_from and (replay or override_plan):
        raise ValueError('--gradual-from is a new comparison build and cannot be combined with replay or --plan')
    if visual_from and (gradual_from or replay or override_plan or vocal_emphasis_from):
        raise ValueError('--visual-from cannot be combined with other input modes')
    if vocal_emphasis_from and (gradual_from or replay or override_plan):
        raise ValueError('--vocal-emphasis-from is a new comparison build and cannot be combined with other input modes')
    if fps not in (30, 60):
        raise ValueError('Use 30 or 60 FPS')
    replay_manifest = json.loads((replay / 'manifest.json').read_text()) if replay else None
    replay_visual = bool(replay_manifest and replay_manifest.get('comparison_kind') == 'visual_two_way')
    visual_pass = visual_from is not None or replay_visual
    replay_three_way = bool(replay_manifest and replay_manifest.get('comparison_kind') in {'gradual_three_way', 'gradual_three_way_replay'})
    replay_vocal = bool(replay_manifest and replay_manifest.get('comparison_kind') == 'vocal_emphasis_two_way')
    if replay_three_way and not (replay / 'baseline-plan.json').is_file():
        raise ValueError('Three-way replay package is missing its frozen baseline plan')
    vocal_emphasis = vocal_emphasis_from is not None or replay_vocal
    if vocal_emphasis and fps != 60:
        raise ValueError('Authored vocal-emphasis comparison requires 60 FPS')
    if vocal_emphasis_from:
        from songviz.direction import make_vocal_emphasis_plans, validate_vocal_emphasis_comparison
        signals, visual_parent, feedback_path, feedback_record, pcm, sr, origins = vocal_emphasis_inputs(vocal_emphasis_from)
        plan, baseline = make_vocal_emphasis_plans(signals, visual_parent, feedback_record)
        validate_vocal_emphasis_comparison(plan, baseline, signals)
        comparison_kind = 'vocal_emphasis_two_way'
    elif replay_vocal:
        signals, plan, baseline, pcm, sr, origins = vocal_emphasis_replay_inputs(replay, override_plan)
        comparison_kind = 'vocal_emphasis_two_way'
    elif visual_pass:
        signals, plan, baseline, pcm, sr, origins = visual_inputs(visual_from or replay, replay=replay_visual, override_plan=override_plan)
        fixed = fixed_plan(plan, signals)
        comparison_kind = 'visual_two_way'
    elif gradual_from:
        from songviz.direction import make_gradual_plan
        signals, baseline, fixed, pcm, sr, origins = gradual_inputs(gradual_from)
        plan = make_gradual_plan(signals, baseline)
        comparison_kind = 'gradual_three_way'
    elif replay_three_way:
        signals, plan, baseline, fixed, pcm, sr, origins = three_way_replay_inputs(replay, override_plan)
        comparison_kind = 'gradual_three_way_replay'
    else:
        signals, plan, pcm, sr, origins = replay_inputs(replay, override_plan) if replay else prepare(review, start, end)
        baseline = plan
        fixed = fixed_plan(plan, signals)
        comparison_kind = 'two_way'
    if vocal_emphasis:
        fixed = None
    validate_plan(plan, signals)
    validate_plan(baseline, signals)
    if fixed is not None:
        validate_plan(fixed, signals)
    start, end = plan['start_s'], plan['end_s']
    out.mkdir(parents=True, exist_ok=False)
    saved_plans = [('signals', signals), ('plan', plan), ('baseline-plan', baseline)]
    if not visual_pass and not vocal_emphasis:
        saved_plans.append(('fixed-plan', fixed))
    if replay_vocal:
        # A replay is evidence reproduction: keep the saved bytes, not merely
        # equivalent decoded JSON/PCM values.  An allowed override is the sole
        # exception and is copied verbatim after pair validation above.
        shutil.copy2(replay / 'signals.json', out / 'signals.json')
        shutil.copy2(override_plan or replay / 'plan.json', out / 'plan.json')
        shutil.copy2(replay / 'baseline-plan.json', out / 'baseline-plan.json')
        shutil.copy2(replay / 'original.wav', out / 'original.wav')
    else:
        for name, data in saved_plans:
            if vocal_emphasis and name == 'signals':
                # Full-track evidence is intentionally reused byte-for-byte.
                shutil.copy2(vocal_emphasis_from / 'signals.json', out / 'signals.json')
            else:
                (out / (name + '.json')).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
        sf.write(out / 'original.wav', pcm, sr, subtype='PCM_24')
    save_evidence_plot(out/'evidence.png', plan, signals)
    inputs = out / 'inputs'; inputs.mkdir()
    for rel in ['songviz/direction.py', 'songviz/directed_render.py', 'songviz/render.py', 'experiments/build_directed_review.py', 'experiments/templates/directed_review.html']:
        path = inputs / rel; path.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / rel, path)
    if gradual_from:
        for name in ('manifest.json', 'plan.json', 'fixed-plan.json'):
            shutil.copy2(gradual_from / name, inputs / ('parent-' + name))
    if visual_pass:
        parent = visual_from or replay
        for name in ('manifest.json', 'plan.json'):
            shutil.copy2(parent / name, inputs / ('parent-' + name))
    if vocal_emphasis_from:
        for name in ('manifest.json', 'plan.json', 'signals.json'):
            shutil.copy2(vocal_emphasis_from / name, inputs / ('parent-' + name))
        shutil.copy2(feedback_path, inputs / 'raw-listening-feedback.json')
    if replay_vocal:
        # These source snapshots were hash-verified by vocal_emphasis_replay_inputs.
        for name in ('raw-listening-feedback.json', 'parent-manifest.json', 'parent-plan.json', 'parent-signals.json'):
            shutil.copy2(replay / 'inputs' / name, inputs / name)
        for source, target in (('manifest.json', 'replay-manifest.json'),
                               ('plan.json', 'saved-plan.json'),
                               ('baseline-plan.json', 'saved-baseline-plan.json')):
            shutil.copy2(replay / source, inputs / target)
    cfg = RenderConfig(width=960, height=540, fps=fps, audio_codec='aac', audio_bitrate='192k')
    frames = out / 'frames'; frames.mkdir()
    diagnostics = []
    three_way = gradual_from is not None or replay_three_way
    schedules = ([( 'directed', plan), ('steady', baseline)] if vocal_emphasis else
                 [('directed', plan), ('previous', baseline)] if visual_pass else
                 [('directed', plan), ('coarse', baseline), ('fixed', fixed)] if three_way else
                 [('directed', plan), ('fixed', fixed)])
    for name, schedule in schedules:
        print(f'Rendering {name}: {start}–{end}s', flush=True)
        visualizer = DirectedVisualizer(schedule, signals, width=960, height=540)
        if (three_way or visual_pass or vocal_emphasis) and name == 'directed':
            for t in np.linspace(0, end-start, 17):
                diagnostics.append(visualizer.state_at(float(t)))
        _render_mp4_with_visualizer(audio_path=out/'original.wav', out_path=out/(name+'.mp4'), cfg=cfg, duration_s=end-start, visualizer=visualizer)
        for i, segment in enumerate(plan['segments']):
            t = min(segment['start_s'] + 1., (segment['start_s'] + segment['end_s']) / 2)
            Image.frombytes('RGB', (960, 540), visualizer.frame_rgb24(t-start)).save(frames/f'{name}-{i}.png')
    if diagnostics:
        (out / 'gradual-diagnostics.json').write_text(json.dumps({'schema_version': 1, 'sampled_states': diagnostics}, indent=2, allow_nan=False) + '\n')
        keyframes = [{'segment_start_s': segment['start_s'], 'segment_end_s': segment['end_s'], 'layer': layer, 'keyframes': values['envelope']}
                     for segment in plan['segments'] for layer, values in segment['layers'].items()]
        (out / 'gradual-keyframes.json').write_text(json.dumps({'schema_version': 1, 'source_plan': 'plan.json', 'layers': keyframes}, indent=2, allow_nan=False) + '\n')
    shutil.copy2(frames/'directed-0.png', out/'poster.png')
    manifest = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(), 'git_head': run('git', 'rev-parse', 'HEAD'), 'origins': origins,
                'start_s': start, 'end_s': end, 'render_config': asdict(cfg), 'maximum_frame_quantization_ms': 1000/fps,
                'comparison_kind': comparison_kind,
                'comparison': ('same audio, signals, visual identities, focus and accompaniment envelopes; only authored vocal gain/emphasis differ' if vocal_emphasis else ('same audio, signals, visibility, focus and emphasis schedule; only treatment and layout differ' if visual_pass else 'same audio, signal evidence and treatment vocabulary; only saved direction plans differ')),
                'production_pipeline_changed': False, 'llm_called': False, 'user_review': 'pending',
                'input_snapshots': [fingerprint(p) for p in sorted(inputs.rglob('*')) if p.is_file()],
                'outputs': [fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file() and inputs not in p.parents]}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    rows = '\n'.join(f"<tr><td>{s['start_s']:.2f}–{s['end_s']:.2f}s</td><td>{html.escape(s['focus'])}</td><td>{html.escape(s['reason'])}</td></tr>" for s in plan['segments'])
    page = (ROOT/'experiments/templates/directed_review.html').read_text()
    for key, value in {'MANIFEST_SHA': sha256_file(out/'manifest.json'), 'START_S': str(start), 'END_S': str(end), 'PLAN_ROWS': rows, 'THREE_WAY': 'true' if three_way else 'false', 'VISUAL_PASS': 'true' if visual_pass else 'false', 'VOCAL_EMPHASIS': 'true' if vocal_emphasis else 'false', 'PLAN_LABEL': 'Reduced vocal-emphasis plan' if vocal_emphasis else ('Visual plan' if visual_pass else ('Gradual plan' if three_way else 'Plan')), 'BASELINE_LINK': '<a href="baseline-plan.json">Steady vocal-emphasis plan</a> · ' if vocal_emphasis else ('<a href="baseline-plan.json">Earlier plan</a> · ' if three_way or visual_pass else ''), 'FIXED_LINK': '' if visual_pass or vocal_emphasis else '<a href="fixed-plan.json">Steady plan</a> · ', 'KEYFRAME_LINK': '<a href="gradual-keyframes.json">Attention keyframes</a> · <a href="gradual-diagnostics.json">Sampled details</a> · ' if three_way or visual_pass or vocal_emphasis else ''}.items():
        page = page.replace('{{'+key+'}}', value)
    (out/'index.html').write_text(page)
    (out/'page-derivation.json').write_text(json.dumps({'schema_version': 1, 'manifest_sha256': sha256_file(out/'manifest.json'), 'index_html_sha256': sha256_file(out/'index.html'), 'derivation': 'index.html is derived after manifest.json; it is intentionally excluded from manifest outputs to avoid a self-hash cycle.'}, indent=2) + '\n')
    print(f'Ready: {out / "index.html"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--review', type=Path, default=ROOT/'outputs/reviews/rhythm-review-01')
    parser.add_argument('--start', type=float, default=130.)
    parser.add_argument('--end', type=float, default=178.)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--gradual-from', type=Path, help='Build a new three-way gradual comparison from a frozen directed package')
    parser.add_argument('--visual-from', type=Path, help='Change visual identities/layout while retaining a frozen gradual schedule')
    parser.add_argument('--vocal-emphasis-from', type=Path, help='Build a two-way authored 119–132s vocal-emphasis comparison from a reviewed visual parent')
    parser.add_argument('--plan', type=Path, help='Edited plan with --replay; never edits its source package')
    parser.add_argument('--fps', type=int, choices=[30, 60], default=60)
    args = parser.parse_args()
    build(args.out.resolve(), review=args.review.resolve(), start=args.start, end=args.end,
          replay=args.replay.resolve() if args.replay else None, override_plan=args.plan.resolve() if args.plan else None,
          gradual_from=args.gradual_from.resolve() if args.gradual_from else None,
          visual_from=args.visual_from.resolve() if args.visual_from else None,
          vocal_emphasis_from=args.vocal_emphasis_from.resolve() if args.vocal_emphasis_from else None, fps=args.fps)
