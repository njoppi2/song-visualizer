"""Package a blank, local section editor using verified original review audio."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.build_review import fingerprint
from songviz.ingest import sha256_file


def editor_data(review: dict, audio_sha: str, source_sha: str, duration: float) -> dict:
    """Deliberately discard predictions, question prompts and algorithm labels."""
    return {'schema_version': 1, 'song_title': review['song_title'],
            'duration_s': duration, 'audio_path': 'original.wav',
            'audio_sha256': audio_sha, 'source_audio_sha256': source_sha,
            'waveform': review.get('waveform', [])}


def embedded_json(value: dict) -> str:
    return json.dumps(value, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def build(parent: Path, out: Path) -> None:
    parent, out = parent.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    if parent in out.parents:
        raise ValueError('Editor output must not be inside the parent review')
    manifest_path = parent/'manifest.json'
    original_manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    records = {Path(r['path']).name: r for r in manifest['outputs']}
    for name in ['original.wav', 'review.json']:
        if name not in records or sha256_file(parent/name) != records[name]['sha256']:
            raise ValueError(f'Changed parent input: {name}')
    source_sha = next(r['sha256'] for r in manifest['sources'] if Path(r['path']).suffix.lower() in {'.flac','.mp3','.wav','.ogg','.m4a'})
    review = json.loads((parent/'review.json').read_text())
    audio_info = sf.info(parent/'original.wav')
    if abs(audio_info.duration - review['duration_s']) > 1 / audio_info.samplerate:
        raise ValueError('Review duration differs from source audio')
    rels = ['experiments/build_section_editor.py', 'experiments/section_editor_state.js',
            'experiments/templates/section_editor.html', 'songviz/review_server.py']
    for rel in rels:
        if not (ROOT/rel).is_file():
            raise FileNotFoundError(ROOT/rel)
    out.mkdir(parents=True)
    inputs = out/'inputs'; inputs.mkdir()
    for rel in rels:
        dest = inputs/rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, dest)
    shutil.copy2(manifest_path, inputs/'parent-manifest.json')
    shutil.copy2(parent/'original.wav', out/'original.wav')
    shutil.copy2(ROOT/'experiments/section_editor_state.js', out/'annotation.js')
    data = editor_data(review, records['original.wav']['sha256'], source_sha, audio_info.duration)
    (out/'editor.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    new_manifest = {'schema_version': 1, 'kind': 'songviz-section-editor',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'parent_review': fingerprint(manifest_path),
        'source_audio_sha256': source_sha, 'audio_sha256': data['audio_sha256'],
        'scope': 'Blank human annotation workspace. No predicted sections, labels or boundaries supplied. Independent layers do not enforce hierarchy.',
        'draft_storage': 'Browser localStorage is a convenience only; download JSON to share/back up. Nothing is posted to a server.',
        'input_snapshots': [fingerprint(p) for p in sorted(inputs.rglob('*')) if p.is_file()],
        'outputs': [fingerprint(out/name) for name in ['original.wav','annotation.js','editor.json']],
        'page_integrity': 'HTML derives from the snapshotted section_editor.html, editor.json and this manifest hash.'}
    if sha256_file(out/'original.wav') != data['audio_sha256'] or sha256_file(manifest_path) != original_manifest_hash:
        raise ValueError('Input integrity changed while building')
    (out/'manifest.json').write_text(json.dumps(new_manifest, indent=2)+'\n')
    template = (inputs/'experiments/templates/section_editor.html').read_text()
    (out/'index.html').write_text(template.replace('{{EDITOR_JSON}}', embedded_json(data)).replace('{{MANIFEST_SHA}}', sha256_file(out/'manifest.json')))
    print(f'Ready: {out}/index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=ROOT/'outputs/reviews/structure-review-03')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); build(args.parent, args.out)
