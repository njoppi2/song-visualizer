"""Repackage an existing structural review with updated playback, no reanalysis."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.build_review import fingerprint
from songviz.ingest import sha256_file


def repair(parent: Path, out: Path) -> None:
    parent, out = parent.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    if parent in out.parents:
        raise ValueError('Repair output must not be inside its parent review')
    manifest = json.loads((parent/'manifest.json').read_text())
    for group in ('outputs', 'input_snapshots'):
        for record in manifest[group]:
            path = ROOT/record['path']
            if not path.is_relative_to(parent) or sha256_file(path) != record['sha256']:
                raise ValueError(f'Changed or out-of-package input: {path}')
    review = json.loads((parent/'review.json').read_text())
    shutil.copytree(parent, out)
    inputs = out/'inputs'
    # Retain the original derivation snapshots; new delivery code has its own
    # directory so a UI repair cannot masquerade as a new analysis run.
    delivery = inputs/'playback-repair'; delivery.mkdir()
    for rel in ['songviz/review_server.py', 'experiments/templates/structure_review.html',
                'experiments/repair_structure_review.py', 'experiments/check_structure_review.cjs']:
        dest = delivery/rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, dest)
    unchanged = {}
    for name in ['original.wav', 'review.json', 'candidate-story.json', 'recurrence.json', 'boundary-evidence.json', 'ablations.json']:
        old_hash = sha256_file(parent/name)
        if sha256_file(out/name) != old_hash:
            raise ValueError(f'Repair changed analysis/audio: {name}')
        unchanged[name] = old_hash
    new_manifest = {**manifest, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'delivery_revision': 'native_audio_range_server_v1', 'parent_review': fingerprint(parent/'manifest.json'),
        'repair_scope': 'Playback/template/server only; questions, predictions and audio unchanged.',
        'page_integrity': 'index.html derives from inputs/playback-repair/experiments/templates/structure_review.html, review.json and this manifest hash.',
        'unchanged_from_parent': unchanged,
        'input_snapshots': [fingerprint(p) for p in sorted(inputs.rglob('*')) if p.is_file()],
        'outputs': [fingerprint(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in {'manifest.json','index.html'}]}
    (out/'manifest.json').write_text(json.dumps(new_manifest, indent=2)+'\n')
    template = (delivery/'experiments/templates/structure_review.html').read_text()
    embedded = json.dumps(review, allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    (out/'index.html').write_text(template.replace('{{REVIEW_JSON}}', embedded).replace('{{MANIFEST_SHA}}', sha256_file(out/'manifest.json')))
    print(f'Ready: {out}/index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); repair(args.parent, args.out)
