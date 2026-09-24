"""Opt-in frozen-study verification (not a unit test requiring ignored media).

Run from the repository root after both packages exist:
    .songviz/venv/bin/python tests/test_directed_emphasis_artifacts.py
"""
from __future__ import annotations

import copy
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.build_directed_review import _verify_frozen_manifest
from songviz.direction import validate_vocal_emphasis_comparison
from songviz.directed_render import DirectedVisualizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    candidate = ROOT / 'outputs/reviews/directed-vocal-emphasis-02'
    replay = ROOT / 'outputs/reviews/directed-vocal-emphasis-replay-02'
    parent = ROOT / 'outputs/reviews/directed-visual-02'
    raw_path = ROOT / 'benchmark/feedback/listening-examples-01.json'
    raw_answer = next(a for a in json.loads(raw_path.read_text())['answers']
                      if a['example_id'] == 'verse-ending')
    for package in (candidate, replay):
        manifest = json.loads((package / 'manifest.json').read_text())
        _verify_frozen_manifest(package, manifest)
        plan = json.loads((package / 'plan.json').read_text())
        baseline = json.loads((package / 'baseline-plan.json').read_text())
        signals = json.loads((package / 'signals.json').read_text())
        validate_vocal_emphasis_comparison(plan, baseline, signals)
        evidence = plan['evidence'][0]['feedback_record']
        assert evidence['answer'] == raw_answer
        assert evidence['file']['sha256'] == sha(raw_path)
        assert (package / 'inputs/raw-listening-feedback.json').read_bytes() == raw_path.read_bytes()
        assert sha(package / 'signals.json') == sha(parent / 'signals.json')
        for rel in ['songviz/direction.py', 'songviz/directed_render.py', 'songviz/render.py',
                    'experiments/build_directed_review.py', 'experiments/templates/directed_review.html']:
            assert sha(package / 'inputs' / rel) == sha(ROOT / rel), rel
        derivation = json.loads((package / 'page-derivation.json').read_text())
        assert derivation['manifest_sha256'] == sha(package / 'manifest.json')
        assert derivation['index_html_sha256'] == sha(package / 'index.html')
        rows = '\n'.join(f"<tr><td>{s['start_s']:.2f}–{s['end_s']:.2f}s</td><td>{html.escape(s['focus'])}</td><td>{html.escape(s['reason'])}</td></tr>" for s in plan['segments'])
        values = {'MANIFEST_SHA': sha(package / 'manifest.json'), 'START_S': '119.0', 'END_S': '132.0',
                  'PLAN_ROWS': rows, 'THREE_WAY': 'false', 'VISUAL_PASS': 'false', 'VOCAL_EMPHASIS': 'true',
                  'PLAN_LABEL': 'Reduced vocal-emphasis plan',
                  'BASELINE_LINK': '<a href="baseline-plan.json">Steady vocal-emphasis plan</a> · ',
                  'FIXED_LINK': '',
                  'KEYFRAME_LINK': '<a href="gradual-keyframes.json">Attention keyframes</a> · <a href="gradual-diagnostics.json">Sampled details</a> · '}
        page = (package / 'inputs/experiments/templates/directed_review.html').read_text()
        for key, value in values.items():
            page = page.replace('{{' + key + '}}', value)
        assert page == (package / 'index.html').read_text()
        print(package.name, len(manifest['outputs']) + len(manifest['input_snapshots']),
              'snapshot/output hashes + origins + raw note + exact page verified;', sha(package / 'manifest.json'))
    for name in ('signals.json', 'plan.json', 'baseline-plan.json', 'original.wav', 'directed.mp4', 'steady.mp4'):
        assert sha(candidate / name) == sha(replay / name), name
    audio = []
    for name in ('directed.mp4', 'steady.mp4'):
        video_path = candidate / name
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(video_path)]))
        stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        assert stream['nb_frames'] == '780' and stream['r_frame_rate'] == '60/1'
        assert float(probe['format']['duration']) == 13.0
        audio.append(subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(video_path), '-map', '0:a:0', '-f', 'f32le', '-acodec', 'pcm_f32le', '-']))
    assert audio[0] == audio[1]
    pcm, sr = sf.read(candidate / 'original.wav', always_2d=True)
    original, source_sr = sf.read(ROOT / 'songs/Gorillaz - Feel Good Inc (featuring De La Soul).flac',
                                  start=119 * 44100, frames=573300, always_2d=True)
    assert sr == source_sr == 44100 and pcm.shape == (573300, 2)
    assert np.array_equal(pcm, original)
    print('Exact paired replay; both videos 13s/60FPS/780 frames, identical decoded AAC; WAV equals source PCM.')

    reduced = DirectedVisualizer(plan, signals, 480, 270)
    steady = DirectedVisualizer(baseline, signals, 480, 270)
    silent_signals = copy.deepcopy(signals)
    silent_signals['energy']['vocals'] = [0.] * len(signals['energy']['vocals'])
    silent = DirectedVisualizer(plan, silent_signals, 480, 270)
    times = [121.2, 124.25, 126.5, 129.2, 131.1]
    sheet = Image.new('RGB', (960, len(times) * 300), '#101a24')
    draw = ImageDraw.Draw(sheet)
    for row, absolute in enumerate(times):
        t = absolute - 119.
        for col, (label, filename) in enumerate((('Steady', 'steady.mp4'), ('Reduced', 'directed.mp4'))):
            decoded = subprocess.check_output(['ffmpeg', '-v', 'error', '-ss', str(t), '-i',
                                               str(candidate / filename), '-frames:v', '1',
                                               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
            frame = Image.frombytes('RGB', (960, 540), decoded).resize((480, 270))
            sheet.paste(frame, (col * 480, row * 300 + 30))
            draw.text((col * 480 + 12, row * 300 + 8), f'{label} at {absolute}s', fill='white')
        assert (reduced.frame_rgb24(t) == steady.frame_rgb24(t)) == (absolute <= 123.)
        if absolute > 125.5:
            assert reduced.state_at(t)['signals']['vocals'] > 0
            assert reduced.frame_rgb24(t) != silent.frame_rgb24(t)
    sheet.save('/tmp/songviz-directed-emphasis-verified-frames.png')
    print('Before-ramp equality, late vocal pixels, and 10 decoded-video frame inspection sheet verified.')


if __name__ == '__main__':
    verify()
