"""Compare structural analysis on two explicit grids using the same current code.

Uses the verified rhythm review's cached and reviewed grids and existing stems.
Writes a new diagnostic directory only; never updates production analysis caches.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.build_review import fingerprint, run
from experiments.build_visual_passage import reviewed_inputs
from songviz.story import compute_story
from songviz.structure_grid import grid_hash


def summary(story):
    return {'sections': [{k: s[k] for k in ['start_s', 'end_s', 'label', 'role']} for s in story['sections']],
            'section_method': story['meta']['section_method'], 'section_error': story['meta']['section_error'],
            'requested_grid_sha256': story['meta']['beat_grid']['requested_sha256'],
            'effective_grid_sha256': story['meta']['beat_grid']['effective_sha256'],
            'grid_fallback': story['meta']['beat_grid']['fallback'],
            'stem_diagnostics': sorted(story['stem_novelties'])}


def build(out: Path, review: Path):
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    reviewed = reviewed_inputs(review)
    source = ROOT / reviewed['manifest']['source_audio']['path']
    cache = ROOT/'outputs'/source.stem/'analysis'
    out.mkdir(parents=True, exist_ok=False)
    inputs = out/'inputs'; inputs.mkdir()
    for rel in ['songviz/story.py', 'songviz/structure_grid.py', 'experiments/regenerate_structure.py']:
        dest = inputs/rel; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/rel, dest)
    for path in [cache/'analysis.json', cache/'story.json', review/'timing.json']:
        shutil.copy2(path, inputs/path.name)
    source_records = [fingerprint(source), fingerprint(cache/'analysis.json'), fingerprint(cache/'story.json')]
    print('Load source and existing stems; no separation or beat tracking', flush=True)
    y, sr = librosa.load(source, sr=22050, mono=True)
    stems = {}
    for name in ['drums', 'bass', 'vocals', 'other']:
        path = cache.parent/'stems'/(name+'.wav')
        stems[name], _ = librosa.load(path, sr=sr, mono=True)
        source_records.append(fingerprint(path))
    results = {}
    # Guard the experiment against an accidental hidden tracker call.
    tracker = librosa.beat.beat_track
    def forbidden_tracker(*args, **kwargs):
        raise AssertionError('Explicit-grid experiment must not run a beat tracker')
    librosa.beat.beat_track = forbidden_tracker
    try:
        for variant, key in [('cached-grid', 'baseline'), ('reviewed-grid', 'candidate')]:
            beats = reviewed['timing'][key]['beat_times_s']
            dest = out/variant; dest.mkdir()
            print(f'Compute {variant}: {len(beats)} supplied beats', flush=True)
            story = compute_story(y, sr, stems=stems, other_y=stems['other'],
                                  beat_times_s=beats, beat_grid_source=f'rhythm-review-01/timing.json:{key}')
            grid = story['meta']['beat_grid']
            assert grid['requested_times_s'] == beats
            assert grid['requested_sha256'] == grid_hash(beats)
            assert grid['fallback'] is None and grid['explicit']
            if set(story['stem_novelties']) != set(stems):
                raise RuntimeError(f'Missing stem diagnostics: {set(stems)-set(story["stem_novelties"])}')
            (dest/'story.json').write_text(json.dumps(story, separators=(',', ':'), allow_nan=False)+'\n')
            (dest/'beat-grid.json').write_text(json.dumps(grid, indent=2)+'\n')
            results[variant] = {'summary': summary(story), 'novelties': story['novelties']}
            print(f'{variant}: {len(story["sections"])} sections; method={story["meta"]["section_method"]}', flush=True)
    finally:
        librosa.beat.beat_track = tracker
    comparison = {'scope': 'same current code/source/stems; only requested beat grid differs',
                  'historical_cache': 'preserved, not used as a controlled current-code baseline',
                  'musical_quality': 'not judged; changed boundaries/novelty are not automatically improvements',
                  'variants': {name: result['summary'] for name, result in results.items()}}
    (out/'comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    for ax, scale in zip(axes, ['short', 'medium', 'long']):
        for variant, color in [('cached-grid', '#ff938c'), ('reviewed-grid', '#80ffc0')]:
            nov = results[variant]['novelties']
            ax.plot(nov['times_s'], nov['novelty_'+scale], color=color, linewidth=.8, label=variant)
        ax.set_ylabel(scale+' novelty'); ax.legend(loc='upper right')
    axes[-1].set_xlabel('Original song seconds')
    fig.suptitle('Same-code grid comparison: 4 / 16 / 32-beat novelty\nEach curve is independently normalized; not calibrated surprise or quality')
    fig.tight_layout(); fig.savefig(out/'novelty-comparison.png', dpi=120); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(15, 4), sharex=True)
    for ax, (name, result) in zip(axes, results.items()):
        for i, s in enumerate(result['summary']['sections']):
            ax.axvspan(s['start_s'], s['end_s'], color=plt.get_cmap('tab10')(i%10), alpha=.65)
            ax.text((s['start_s']+s['end_s'])/2, .5, s['label']+'\n'+s['role'], ha='center', va='center', fontsize=8)
        ax.set_ylabel(name); ax.set_yticks([])
    axes[-1].set_xlabel('Original song seconds'); fig.suptitle('Predicted sections — neither row is ground truth')
    fig.tight_layout(); fig.savefig(out/'sections-comparison.png', dpi=120); plt.close(fig)
    manifest = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'git_head': run('git', 'rev-parse', 'HEAD'), 'sources': source_records,
                'parent_review': fingerprint(review/'manifest.json'), 'librosa_version': librosa.__version__,
                'settings': {'sr': sr, 'hop_length': 512, 'frame_length': 2048, 'stem_names': list(stems)},
                'input_snapshots': [fingerprint(p) for p in sorted(inputs.rglob('*')) if p.is_file()],
                'outputs': [fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file() and inputs not in p.parents]}
    # Recheck original caches after the run.
    for record in source_records:
        assert fingerprint(ROOT/record['path'])['sha256'] == record['sha256']
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Ready: {out}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--review', type=Path, default=ROOT/'outputs/reviews/rhythm-review-01')
    args = p.parse_args(); build(args.out.resolve(), args.review.resolve())
