"""Create or replay a cached-evidence directing study, with no model/API calls.

Default: the reviewed Feel Good Inc transition/return at 130–178s. The planner
inspects song-wide percussion gaps and stem energy, not hardcoded section roles.
--replay uses a prior package's saved signals, plan and original PCM exclusively.
"""
from __future__ import annotations

import argparse
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
        record = next(r for r in manifest['outputs'] if Path(r['path']).name == name)
        verify_hash(package / name, record['sha256'])
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


def save_evidence_plot(path: Path, plan: dict, signals: dict) -> None:
    """Diagnostic evidence, separate from the artistic video and its claims."""
    img = Image.new('RGB', (1200, 650), '#101a24')
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
    for t in np.linspace(start,end,7):
        draw.text((x(t)-22,612),f'{t:.1f}s',font=font,fill='#dce5ec')
    img.save(path)


def build(out: Path, *, review: Path, start: float = 130., end: float = 178., replay: Path | None = None, override_plan: Path | None = None, fps: int = 60):
    from songviz.directed_render import DirectedVisualizer

    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    if override_plan and not replay:
        raise ValueError('--plan requires --replay to bind it to saved signal/audio evidence')
    if fps not in (30, 60):
        raise ValueError('Use 30 or 60 FPS')
    signals, plan, pcm, sr, origins = replay_inputs(replay, override_plan) if replay else prepare(review, start, end)
    validate_plan(plan, signals)
    baseline = fixed_plan(plan, signals)
    start, end = plan['start_s'], plan['end_s']
    out.mkdir(parents=True, exist_ok=False)
    for name, data in [('signals', signals), ('plan', plan), ('fixed-plan', baseline)]:
        (out / (name + '.json')).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    sf.write(out / 'original.wav', pcm, sr, subtype='PCM_24')
    save_evidence_plot(out/'evidence.png', plan, signals)
    inputs = out / 'inputs'; inputs.mkdir()
    for rel in ['songviz/direction.py', 'songviz/directed_render.py', 'songviz/render.py', 'experiments/build_directed_review.py', 'experiments/templates/directed_review.html']:
        path = inputs / rel; path.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / rel, path)
    cfg = RenderConfig(width=960, height=540, fps=fps, audio_codec='aac', audio_bitrate='192k')
    frames = out / 'frames'; frames.mkdir()
    for name, schedule in [('directed', plan), ('fixed', baseline)]:
        print(f'Rendering {name}: {start}–{end}s', flush=True)
        visualizer = DirectedVisualizer(schedule, signals, width=960, height=540)
        _render_mp4_with_visualizer(audio_path=out/'original.wav', out_path=out/(name+'.mp4'), cfg=cfg, duration_s=end-start, visualizer=visualizer)
        for i, segment in enumerate(plan['segments']):
            t = min(segment['start_s'] + 1., (segment['start_s'] + segment['end_s']) / 2)
            Image.frombytes('RGB', (960, 540), visualizer.frame_rgb24(t-start)).save(frames/f'{name}-{i}.png')
    shutil.copy2(frames/'directed-0.png', out/'poster.png')
    manifest = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(), 'git_head': run('git', 'rev-parse', 'HEAD'), 'origins': origins,
                'start_s': start, 'end_s': end, 'render_config': asdict(cfg), 'maximum_frame_quantization_ms': 1000/fps,
                'comparison': 'same audio, signal evidence and treatment vocabulary; only direction plan differs',
                'production_pipeline_changed': False, 'llm_called': False, 'user_review': 'pending',
                'input_snapshots': [fingerprint(p) for p in sorted(inputs.rglob('*')) if p.is_file()],
                'outputs': [fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file() and inputs not in p.parents]}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    rows = '\n'.join(f"<tr><td>{s['start_s']:.2f}–{s['end_s']:.2f}s</td><td>{html.escape(s['focus'])}</td><td>{html.escape(s['reason'])}</td></tr>" for s in plan['segments'])
    page = (ROOT/'experiments/templates/directed_review.html').read_text()
    for key, value in {'MANIFEST_SHA': sha256_file(out/'manifest.json'), 'START_S': str(start), 'END_S': str(end), 'PLAN_ROWS': rows}.items():
        page = page.replace('{{'+key+'}}', value)
    (out/'index.html').write_text(page)
    print(f'Ready: {out / "index.html"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--review', type=Path, default=ROOT/'outputs/reviews/rhythm-review-01')
    parser.add_argument('--start', type=float, default=130.)
    parser.add_argument('--end', type=float, default=178.)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--plan', type=Path, help='Edited plan with --replay; never edits its source package')
    parser.add_argument('--fps', type=int, choices=[30, 60], default=60)
    args = parser.parse_args()
    build(args.out.resolve(), review=args.review.resolve(), start=args.start, end=args.end,
          replay=args.replay.resolve() if args.replay else None, override_plan=args.plan.resolve() if args.plan else None, fps=args.fps)
