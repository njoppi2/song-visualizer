"""Build a small guided listening review from an existing verified comparison."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import quote, unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from songviz.ingest import sha256_file

CODE = (
    'experiments/build_listening_examples.py',
    'experiments/templates/listening_examples.html',
    'experiments/check_listening_examples.cjs',
    'songviz/ingest.py',
)


def record(path: Path) -> dict:
    return {'path': str(path.resolve()), 'sha256': sha256_file(path)}


def verify_records(records: list[dict]) -> None:
    for row in records:
        path = ROOT / row['path']
        if not path.is_file() or sha256_file(path) != row['sha256']:
            raise ValueError(f'Changed or missing input: {path}')


def encode(data: dict, *, embedded: bool = False) -> str:
    text = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=None if embedded else 2)
    return text.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026') if embedded else text + '\n'


def examples(review: dict) -> list[dict]:
    """Curated development cases, never an automatic musical classifier.

    Bind explanatory copy to specific inspected evidence. Refuse a different
    prediction set rather than silently attaching this text to a nearest event.
    Focus windows are listening prompts, not inferred transition durations.
    """
    predictions = review['variants']['sustained_activity']['predictions']
    specifications = [
        ('drum-entry', 'Perto de 1:19', 74., 85., 78.5, 80.,
         'sustained_activity-change-26', 79.10341721880629, 'drums', 'increase',
         'Os sinais sugerem que a bateria passa de muito fraca para uma presença sustentada por volta de 1:19. A voz e outras camadas continuam. Isso pode justificar dar mais destaque visual à bateria.',
         'Tu já marcaste uma mudança de intensidade do refrão aqui. Este exemplo mostra qual evidência ajuda a explicá-la; não precisas marcar a seção novamente.'),
        ('within-passage', 'Perto de 0:21', 16., 26., 20., 21.5,
         'sustained_activity-change-8', 20.630015882218938, 'other', 'decrease',
         'Os sinais apontam uma redução em parte do acompanhamento por volta de 0:21. Essa camada reúne sons que a separação não colocou em voz, baixo ou bateria. Ainda não sabemos se tu percebes isso como um detalhe ou como uma mudança relevante.',
         'Uma marca do detector pode corresponder a um detalhe dentro de uma passagem. O tamanho da mudança visual ainda precisa de julgamento musical.'),
        ('verse-ending', 'Perto de 2:04', 119., 132., 123.5, 126.,
         'sustained_activity-change-38', 124.58272936948532, 'other', 'decrease',
         'Os sinais sugerem uma redução em parte do acompanhamento perto de 2:04–2:05. Esse indício está perto da subdivisão que tu marcaste no verso, mas ainda não explica toda a mudança musical.',
         'Este caso ajuda a pensar em mudanças que se desenvolvem ao longo de um trecho. A faixa destacada é apenas uma região para ouvir com atenção.'),
    ]
    result = []
    for identity, title, start, end, focus_a, focus_b, event_id, time_s, stem, direction, explanation, reason in specifications:
        matches = [item for item in predictions['changes'] if item['id'] == event_id]
        if len(matches) != 1:
            raise ValueError(f'Missing curated event: {event_id}')
        event = matches[0]
        evidence = event.get('stem_evidence', {}).get(stem, {})
        if (abs(event['time_s'] - time_s) > 1e-6 or event.get('primary_stem') != stem
                or evidence.get('direction') != direction or not evidence.get('qualified')):
            raise ValueError(f'Curated explanation no longer matches: {event_id}')
        result.append(dict(id=identity, title=title, start_s=start, end_s=end,
                           focus_start_s=focus_a, focus_end_s=focus_b, explanation=explanation,
                           reason=reason, evidence={'origin': 'acoustic_candidate', 'event': event}))
    spans = [span for layer in review['reference']['layers'] for span in layer['spans']
             if abs(span['start_s'] - 61.368411) < 1e-5 and abs(span['end_s'] - 64.855320) < 1e-5]
    dips = [item for item in predictions['transitions'] if item['id'] == 'transition-1']
    if (len(spans) != 1 or len(dips) != 1
            or abs(dips[0]['start_s'] - 63.943646501913264) > 1e-6
            or abs(dips[0]['end_s'] - 65.24305542050409) > 1e-6):
        raise ValueError('Missing curated transition/reference evidence')
    result.append(dict(
        id='transition-extent', title='De 1:01 a 1:05', start_s=57., end_s=70.,
        focus_start_s=spans[0]['start_s'], focus_end_s=spans[0]['end_s'],
        explanation='Tu marcaste uma transição de aproximadamente 1:01,4 a 1:04,9. A queda de energia encontrada automaticamente cobre só uma parte, por volta de 1:03,9 a 1:05,2. Uma transição pode incluir preparação e recuperação além do momento mais vazio.',
        reason='Este é um limite conhecido do detector. A faixa destacada usa tua marcação anterior. O próximo trabalho vai investigar como representar mudanças com durações e escalas diferentes.',
        evidence={'origin': 'human_interval_and_acoustic_dip', 'human_span': spans[0], 'dip': dips[0]}))
    for item in result:
        if not 0 <= item['start_s'] <= item['focus_start_s'] < item['focus_end_s'] <= item['end_s'] <= review['duration_s']:
            raise ValueError('Curated listening range is outside the source audio')
    return result


def build(*, parent: Path, out: Path) -> None:
    parent, out = parent.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    if parent in out.parents:
        raise ValueError('Output must be separate from parent package')
    manifest_path = parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('kind') != 'songviz-local-structure-comparison':
        raise ValueError('Expected verified local-structure comparison')
    # Frozen output/snapshot hashes remain meaningful when current source code
    # changes. Do not require historic source code to match the current checkout.
    verify_records(manifest['outputs'] + manifest['input_snapshots'])
    reviews = [row for row in manifest['outputs'] if (ROOT / row['path']).resolve() == parent / 'review.json']
    if len(reviews) != 1:
        raise ValueError('Review payload is not bound to the parent manifest')
    review = json.loads((parent / 'review.json').read_text())
    source = review['reference']['source']
    audio = (parent / unquote(review['audio_path'])).resolve()
    if audio.parent in out.parents:
        raise ValueError('Output must be separate from the original audio package')
    if sha256_file(audio) != source['audio_sha256']:
        raise ValueError('Original review audio fingerprint differs')
    source_records = [row for row in manifest['sources'] if row['sha256'] == source['source_audio_sha256']]
    if len(source_records) != 1:
        raise ValueError('Original source fingerprint is not unique')
    verify_records(source_records)
    data = dict(schema_version=1, kind='songviz-listening-examples', example_set_id=out.name,
                song_title=review['song_title'], duration_s=review['duration_s'],
                audio_path=quote(Path(os.path.relpath(audio, out)).as_posix(), safe='/'),
                audio_sha256=source['audio_sha256'], source_audio_sha256=source['source_audio_sha256'],
                examples=examples(review), selection='Lead-curated development examples; focus windows are listening prompts, not inferred extents.')
    inputs = [record(manifest_path), record(parent / 'review.json'), record(audio), *source_records,
              *[record(ROOT / rel) for rel in CODE]]
    verify_records(inputs)
    out.mkdir(parents=True)
    for rel in CODE:
        dest = out / 'inputs' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    shutil.copy2(manifest_path, out / 'inputs' / 'parent-manifest.json')
    (out / 'review.json').write_text(encode(data))
    template = (out / 'inputs/experiments/templates/listening_examples.html').read_text()
    (out / 'index.html').write_text(template.replace('{{REVIEW_JSON}}', encode(data, embedded=True))
                                  .replace('{{REVIEW_SHA}}', sha256_file(out / 'review.json')))
    verify_records(inputs)
    manifest = dict(schema_version=1, kind='songviz-listening-examples', sources=inputs,
                    input_snapshots=[record(path) for path in sorted((out / 'inputs').rglob('*')) if path.is_file()],
                    outputs=[record(out / name) for name in ('review.json', 'index.html')])
    (out / 'manifest.json').write_text(encode(manifest))
    print(f'Ready: {out}/index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=ROOT / 'outputs/reviews/local-structure-comparison-02')
    parser.add_argument('--out', type=Path, required=True)
    build(**vars(parser.parse_args()))
