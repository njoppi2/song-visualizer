import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import experiments.build_directed_review as builder
from experiments.build_review import fingerprint
from songviz.direction import fixed_plan, make_gradual_plan, make_plan, make_visual_plan


def test_stem_energy_uses_stereo_power_for_antiphase_signal(tmp_path: Path) -> None:
    """Opposite stereo channels must add as energy, not cancel to silence."""
    wav = tmp_path / "antiphase.wav"
    mono = np.tile(np.array([1.0, -1.0], dtype=np.float32), 10)
    sf.write(wav, np.column_stack((mono, -mono)), 100)

    times, values, metadata = builder.stem_energy(wav)

    assert times == [0.0, 0.1, 0.2]
    assert all(0.0 <= value <= 1.0 for value in values)
    assert max(values) > 0.0
    assert metadata["timestamps"] == "completed-block end; initial zero"
    assert metadata["hop_s"] == pytest.approx(0.1)


def test_stem_energy_timestamps_and_normalized_bounds_include_partial_block(tmp_path: Path) -> None:
    wav = tmp_path / "partial.wav"
    samples = np.ones((25, 2), dtype=np.float32)
    sf.write(wav, samples, 100)

    times, values, metadata = builder.stem_energy(wav)

    assert times == [0.0, 0.1, 0.2, 0.25]
    assert len(times) == len(values)
    assert values[0] == 0.0
    assert all(0.0 <= value <= 1.0 for value in values)
    assert metadata["normalization_rms_p95"] == pytest.approx(1.0, rel=1e-4)


def _signals() -> dict:
    return {
        "beat_times_s": [0.0, 1.0, 2.0],
        "hits": [
            {"component": "kick", "t": 0.25, "velocity": 0.8},
            {"component": "snare", "t": 2.25, "velocity": 0.7},
        ],
        "energy": {
            "times_s": [0.0, 1.0, 2.0, 3.0],
            "bass": [0.2, 0.3, 0.4, 0.2],
            "vocals": [0.1, 0.7, 0.8, 0.2],
            "other": [0.2, 0.2, 0.3, 0.4],
        },
    }


def _write_replay_package(tmp_path: Path) -> tuple[Path, dict, dict]:
    package = tmp_path / "replay"
    package.mkdir()
    signals = _signals()
    plan = make_plan(signals, 1.0, 3.0)
    fixed = fixed_plan(plan, signals)
    (package / "signals.json").write_text(json.dumps(signals, indent=2) + "\n")
    (package / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (package / "fixed-plan.json").write_text(json.dumps(fixed, indent=2) + "\n")
    sf.write(package / "original.wav", np.zeros((20, 2), dtype=np.float32), 10)
    manifest = {
        "schema_version": 1,
        "origins": {"fixture": True},
        "outputs": [fingerprint(package / name) for name in ("signals.json", "plan.json", "fixed-plan.json", "original.wav")],
    }
    (package / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return package, signals, plan


def test_replay_inputs_loads_saved_fixture_without_prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package, signals, plan = _write_replay_package(tmp_path)
    monkeypatch.setattr(builder, "prepare", lambda *args, **kwargs: pytest.fail("prepare must not run during replay"))

    loaded_signals, loaded_plan, pcm, sr, origins = builder.replay_inputs(package)

    assert loaded_signals == signals
    assert loaded_plan == plan
    assert pcm.shape == (20, 2)
    assert sr == 10
    assert origins["fixture"] is True
    assert origins["replayed_from"]["sha256"] == builder.fingerprint(package / "manifest.json")["sha256"]


@pytest.mark.parametrize("filename", ["signals.json", "plan.json", "original.wav"])
def test_replay_inputs_rejects_changed_saved_evidence(tmp_path: Path, filename: str) -> None:
    package, _, _ = _write_replay_package(tmp_path)
    path = package / filename
    if filename.endswith(".json"):
        payload = json.loads(path.read_text())
        if filename == "signals.json":
            payload["beat_times_s"].append(2.5)
        else:
            payload["uncertainty"] = "changed"
        path.write_text(json.dumps(payload) + "\n")
    else:
        path.write_bytes(path.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="changed"):
        builder.replay_inputs(package)


@pytest.mark.parametrize("mutation, message", [
    ("duplicate", "exactly one"),
    ("misbound", "does not bind"),
    ("missing", "exactly one"),
])
def test_replay_refuses_ambiguous_or_misbound_required_output_before_creating_output(
    tmp_path: Path, mutation: str, message: str,
) -> None:
    package, _, _ = _write_replay_package(tmp_path)
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record_index = next(i for i, record in enumerate(manifest["outputs"])
                        if Path(record["path"]).name == "signals.json")
    if mutation == "duplicate":
        manifest["outputs"].append(dict(manifest["outputs"][record_index]))
    elif mutation == "misbound":
        manifest["outputs"][record_index]["path"] = str(tmp_path / "unrelated" / "signals.json")
    else:
        manifest["outputs"].pop(record_index)
    manifest_path.write_text(json.dumps(manifest) + "\n")

    out = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match=message):
        builder.build(out, review=tmp_path / "unused", replay=package)
    assert not out.exists()


def test_replay_inputs_rejects_override_with_wrong_audio_interval(tmp_path: Path) -> None:
    package, _, plan = _write_replay_package(tmp_path)
    override = tmp_path / "override-plan.json"
    edited = dict(plan)
    edited["end_s"] = 2.5
    edited["segments"] = [dict(edited["segments"][0], end_s=2.5)]
    override.write_text(json.dumps(edited) + "\n")

    with pytest.raises(ValueError, match="audio excerpt interval"):
        builder.replay_inputs(package, override)


def test_build_refuses_overwrite_before_prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "existing"
    out.mkdir()
    monkeypatch.setattr(builder, "prepare", lambda *args, **kwargs: pytest.fail("prepare ran before overwrite check"))

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        builder.build(out, review=tmp_path / "review")


def test_gradual_inputs_retains_verified_baseline_without_planning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package, signals, baseline = _write_replay_package(tmp_path)
    monkeypatch.setattr(builder, "prepare", lambda *args, **kwargs: pytest.fail("analysis must not run"))
    loaded_signals, loaded_baseline, fixed, pcm, sr, origins = builder.gradual_inputs(package)

    assert loaded_signals == signals
    assert loaded_baseline == baseline
    assert fixed == fixed_plan(baseline, signals)
    assert pcm.shape == (20, 2)
    assert sr == 10
    assert origins["gradual_from"]["sha256"] == fingerprint(package / "manifest.json")["sha256"]


@pytest.mark.parametrize("filename", ["signals.json", "plan.json", "fixed-plan.json", "original.wav"])
def test_gradual_inputs_rejects_tampered_frozen_source(tmp_path: Path, filename: str) -> None:
    package, _, _ = _write_replay_package(tmp_path)
    path = package / filename
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="changed"):
        builder.gradual_inputs(package)


def test_review_template_uses_native_video_urls_and_three_way_switching() -> None:
    page = (builder.ROOT / "experiments/templates/directed_review.html").read_text()

    assert "coarse.mp4" in page
    assert "directed.mp4" in page
    assert "fixed.mp4" in page
    assert "fetch(" not in page
    assert "response.blob" not in page


def test_gradual_build_and_replay_render_three_saved_views_without_replanning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, _, _ = _write_replay_package(tmp_path)
    rendered: list[str] = []

    def fake_render(*, out_path: Path, **kwargs) -> None:
        rendered.append(out_path.name)
        out_path.write_bytes(b"mp4")

    monkeypatch.setattr(builder, "_render_mp4_with_visualizer", fake_render)
    monkeypatch.setattr(builder, "run", lambda *args: "test-head")
    built = tmp_path / "gradual"
    builder.build(built, review=tmp_path / "unused", gradual_from=source, fps=30)

    assert rendered == ["directed.mp4", "coarse.mp4", "fixed.mp4"]
    assert json.loads((built / "baseline-plan.json").read_text()) == json.loads((source / "plan.json").read_text())
    assert (built / "gradual-keyframes.json").is_file()
    rendered.clear()
    monkeypatch.setattr("songviz.direction.make_gradual_plan", lambda *args: pytest.fail("replay must not replan"))
    replay = tmp_path / "replayed"
    builder.build(replay, review=tmp_path / "unused", replay=built, fps=30)

    assert rendered == ["directed.mp4", "coarse.mp4", "fixed.mp4"]
    for name in ("plan.json", "baseline-plan.json", "fixed-plan.json"):
        assert json.loads((replay / name).read_text()) == json.loads((built / name).read_text())


def _gradual_fixture(tmp_path: Path) -> Path:
    source, signals, baseline = _write_replay_package(tmp_path)
    plan = make_gradual_plan(signals, baseline)
    (source / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    manifest = json.loads((source / 'manifest.json').read_text())
    manifest['outputs'] = [fingerprint(source / name) for name in ('signals.json', 'plan.json', 'fixed-plan.json', 'original.wav')]
    (source / 'manifest.json').write_text(json.dumps(manifest))
    return source


def test_visual_build_replays_the_same_two_plans_without_restyling(tmp_path, monkeypatch):
    source = _gradual_fixture(tmp_path)
    rendered = []
    def fake_render(*, out_path, **kwargs):
        rendered.append(out_path.name)
        out_path.write_bytes(b'mock-video')
    monkeypatch.setattr(builder, '_render_mp4_with_visualizer', fake_render)
    monkeypatch.setattr(builder, 'prepare', lambda *args: pytest.fail('visual pass must not analyze'))
    monkeypatch.setattr(builder, 'run', lambda *args: 'test-head')
    built = tmp_path / 'visual'
    builder.build(built, review=tmp_path / 'unused', visual_from=source)
    assert rendered == ['directed.mp4', 'previous.mp4']
    assert (built / 'baseline-plan.json').read_bytes() == (source / 'plan.json').read_bytes()
    assert not (built / 'fixed-plan.json').exists()
    page = (built / 'index.html').read_text()
    assert 'const visualPass=true;' in page and '{{' not in page
    assert 'href="fixed-plan.json"' not in page
    rendered.clear()
    monkeypatch.setattr('songviz.direction.make_visual_plan', lambda *args: pytest.fail('replay must not restyle'))
    replay = tmp_path / 'visual-replay'
    builder.build(replay, review=tmp_path / 'unused', replay=built)
    assert rendered == ['directed.mp4', 'previous.mp4']
    for name in ('signals.json', 'plan.json', 'baseline-plan.json', 'original.wav'):
        assert (built / name).read_bytes() == (replay / name).read_bytes()


def test_visual_comparison_rejects_an_emphasis_change():
    signals = _signals()
    baseline = make_gradual_plan(signals, make_plan(signals, 1., 3.))
    visual = make_visual_plan(signals, baseline)
    builder.validate_visual_comparison(visual, baseline)
    visual['segments'][0]['layers']['vocals']['envelope'][0]['emphasis'] = .123
    with pytest.raises(ValueError, match='direction schedule'):
        builder.validate_visual_comparison(visual, baseline)


def test_visual_inputs_rejects_tampered_parent_plan(tmp_path):
    source = _gradual_fixture(tmp_path)
    (source / 'plan.json').write_bytes((source / 'plan.json').read_bytes() + b' ')
    with pytest.raises(ValueError, match='changed'):
        builder.visual_inputs(source)


def _vocal_fixture(tmp_path: Path) -> tuple[dict, dict, Path, dict]:
    """A self-contained schema-v2 visual parent covering the fixed study."""
    times = np.round(np.arange(0., 132.1, .1), 3).tolist()
    signals = {
        'beat_times_s': np.arange(0., 132.1, .5).tolist(),
        'hits': [{'component': component, 't': float(t), 'velocity': .7}
                 for component in ('kick', 'snare', 'hh') for t in np.arange(119., 132., .5)],
        'energy': {'times_s': times, **{name: [.5] * len(times) for name in ('bass', 'vocals', 'other')}},
    }
    visual_parent = make_visual_plan(signals, make_gradual_plan(signals, make_plan(signals, 119., 132.)))
    feedback_path = tmp_path / 'raw-feedback.json'
    feedback_path.write_text('{ "answers" : [ { "example_id" : "verse-ending", "perceived_change" : "subtle", "notes" : "the complete qualified fixture note" } ] }\n')
    feedback = {'file': fingerprint(feedback_path), 'answer': json.loads(feedback_path.read_text())['answers'][0]}
    return signals, visual_parent, feedback_path, feedback


def test_vocal_emphasis_build_and_replay_keep_the_two_saved_plans_without_replanning(tmp_path, monkeypatch):
    """The authored comparison is a strict two-plan replay, not a new analysis mode."""
    signals, visual_parent, feedback_path, feedback_record = _vocal_fixture(tmp_path)
    parent = tmp_path / 'parent'; parent.mkdir()
    # The fresh build copies these bytes after vocal_emphasis_inputs verifies them.
    parent_signals = json.dumps(signals, sort_keys=True, separators=(',', ': ')) + '\n'
    (parent / 'signals.json').write_text(parent_signals)
    (parent / 'manifest.json').write_text('{ "fixture" : "parent" }\n')
    (parent / 'plan.json').write_text(json.dumps(visual_parent, separators=(',', ':')) + '\n')
    pcm = np.zeros((573300, 2), dtype=np.float32)
    origins = {'fixture': True, 'feedback_record': feedback_record}
    monkeypatch.setattr(builder, 'vocal_emphasis_inputs',
                        lambda unused: (signals, visual_parent, feedback_path, feedback_record, pcm, 44100, origins))
    rendered = []
    def fake_render(*, out_path, **kwargs):
        rendered.append(out_path.name)
        out_path.write_bytes(b'mock-video')
    monkeypatch.setattr(builder, '_render_mp4_with_visualizer', fake_render)
    monkeypatch.setattr(builder, 'run', lambda *args: 'test-head')

    built = tmp_path / 'vocal'
    builder.build(built, review=tmp_path / 'unused', vocal_emphasis_from=parent)

    assert rendered == ['directed.mp4', 'steady.mp4']
    assert json.loads((built / 'manifest.json').read_text())['comparison_kind'] == 'vocal_emphasis_two_way'
    assert (built / 'inputs/raw-listening-feedback.json').read_bytes() == feedback_path.read_bytes()
    assert (built / 'signals.json').read_text() == parent_signals
    assert not (built / 'fixed-plan.json').exists()
    reduced = json.loads((built / 'plan.json').read_text())
    steady = json.loads((built / 'baseline-plan.json').read_text())
    from songviz.direction import validate_vocal_emphasis_comparison
    validate_vocal_emphasis_comparison(reduced, steady, signals)

    rendered.clear()
    monkeypatch.setattr('songviz.direction.make_vocal_emphasis_plans',
                        lambda *args: pytest.fail('vocal replay must not plan'))
    replay = tmp_path / 'vocal-replay'
    builder.build(replay, review=tmp_path / 'unused', replay=built)
    assert rendered == ['directed.mp4', 'steady.mp4']
    for name in ('signals.json', 'plan.json', 'baseline-plan.json', 'original.wav'):
        assert (built / name).read_bytes() == (replay / name).read_bytes()
    assert (replay / 'inputs/raw-listening-feedback.json').read_bytes() == feedback_path.read_bytes()


def test_vocal_emphasis_inputs_bind_the_reviewed_parent_and_fresh_source_cut():
    parent = builder.ROOT / 'outputs/reviews/directed-visual-02'
    source = builder.ROOT / 'songs/Gorillaz - Feel Good Inc (featuring De La Soul).flac'
    if not parent.is_dir() or not source.is_file():
        pytest.skip('reviewed development evidence is not included in this checkout')
    signals, visual_parent, feedback_path, feedback, pcm, sr, origins = builder.vocal_emphasis_inputs(parent)

    assert sr == 44100 and pcm.shape == (573300, 2)
    assert origins['vocal_emphasis_from']['sha256'] == builder._VOCAL_PARENT_MANIFEST
    assert origins['source_audio']['sha256'] == builder._VOCAL_SOURCE
    assert feedback_path.name == 'listening-examples-01.json'
    assert feedback['file']['sha256'] == builder._VOCAL_FEEDBACK
    assert feedback['answer']['example_id'] == 'verse-ending'
    assert visual_parent['schema_version'] == 2 and signals['energy']['times_s'][0] == 0.0


def test_vocal_emphasis_replay_rejects_misbound_or_changed_plan(tmp_path, monkeypatch):
    signals, visual_parent, feedback_path, feedback_record = _vocal_fixture(tmp_path)
    parent = tmp_path / 'parent'; parent.mkdir()
    (parent / 'signals.json').write_text(json.dumps(signals))
    (parent / 'manifest.json').write_text('{}\n')
    (parent / 'plan.json').write_text(json.dumps(visual_parent))
    monkeypatch.setattr(builder, 'vocal_emphasis_inputs', lambda unused: (
        signals, visual_parent, feedback_path, feedback_record, np.zeros((573300, 2), dtype=np.float32), 44100,
        {'fixture': True, 'feedback_record': feedback_record}))
    monkeypatch.setattr(builder, '_render_mp4_with_visualizer', lambda *, out_path, **kwargs: out_path.write_bytes(b'mock-video'))
    monkeypatch.setattr(builder, 'run', lambda *args: 'test-head')
    package = tmp_path / 'vocal'
    builder.build(package, review=tmp_path / 'unused', vocal_emphasis_from=parent)
    edited = json.loads((package / 'plan.json').read_text())
    edited['segments'][0]['layers']['other']['envelope'][0]['gain'] = .01
    override = tmp_path / 'bad-plan.json'
    override.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match='fixed and constant'):
        builder.vocal_emphasis_replay_inputs(package, override)


def _built_vocal_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict, Path]:
    signals, visual_parent, feedback_path, feedback_record = _vocal_fixture(tmp_path)
    parent = tmp_path / 'parent'; parent.mkdir()
    (parent / 'signals.json').write_text(json.dumps(signals, separators=(',', ':')))
    (parent / 'manifest.json').write_text('{}\n')
    (parent / 'plan.json').write_text(json.dumps(visual_parent))
    monkeypatch.setattr(builder, 'vocal_emphasis_inputs', lambda unused: (
        signals, visual_parent, feedback_path, feedback_record, np.zeros((573300, 2), dtype=np.float32), 44100,
        {'fixture': True, 'feedback_record': feedback_record}))
    monkeypatch.setattr(builder, '_render_mp4_with_visualizer',
                        lambda *, out_path, **kwargs: out_path.write_bytes(b'mock-video'))
    monkeypatch.setattr(builder, 'run', lambda *args: 'test-head')
    package = tmp_path / 'vocal'
    builder.build(package, review=tmp_path / 'unused', vocal_emphasis_from=parent)
    return package, signals, feedback_path


def test_vocal_replay_never_cuts_or_extracts_and_retains_nonstandard_override_bytes(tmp_path, monkeypatch):
    package, _, _ = _built_vocal_package(tmp_path, monkeypatch)
    override = tmp_path / 'formatted-override.json'
    override.write_text(json.dumps(json.loads((package / 'plan.json').read_text()), sort_keys=True, separators=(',', ': ')) + '\n')
    for name in ('cut_audio', 'prepare', 'stem_energy'):
        monkeypatch.setattr(builder, name, lambda *args, **kwargs: pytest.fail(f'{name} must not run during vocal replay'))
    monkeypatch.setattr('songviz.direction.make_vocal_emphasis_plans',
                        lambda *args: pytest.fail('planner must not run during vocal replay'))
    replay = tmp_path / 'replay'
    builder.build(replay, review=tmp_path / 'unused', replay=package, override_plan=override)
    assert (replay / 'plan.json').read_bytes() == override.read_bytes()
    for name in ('signals.json', 'baseline-plan.json', 'original.wav'):
        assert (replay / name).read_bytes() == (package / name).read_bytes()


@pytest.mark.parametrize('mutation, message', [('duplicate', 'exactly one'), ('misbound', 'does not bind')])
def test_vocal_replay_rejects_duplicate_or_misbound_required_output(tmp_path, monkeypatch, mutation, message):
    package, _, _ = _built_vocal_package(tmp_path, monkeypatch)
    manifest_path = package / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    index = next(i for i, record in enumerate(manifest['outputs']) if Path(record['path']).name == 'plan.json')
    if mutation == 'duplicate':
        manifest['outputs'].append(dict(manifest['outputs'][index]))
    else:
        manifest['outputs'][index]['path'] = str(tmp_path / 'unbound' / 'plan.json')
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        builder.vocal_emphasis_replay_inputs(package)


@pytest.mark.parametrize('kind', ['snapshot', 'origin'])
def test_vocal_replay_rejects_tampered_snapshot_or_origin(tmp_path, monkeypatch, kind):
    package, _, _ = _built_vocal_package(tmp_path, monkeypatch)
    if kind == 'snapshot':
        (package / 'inputs/raw-listening-feedback.json').write_text('tampered')
    else:
        manifest_path = package / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['origins']['feedback_record']['file']['sha256'] = '0' * 64
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='changed'):
        builder.vocal_emphasis_replay_inputs(package)


@pytest.mark.parametrize('edit, message', [
    ('curve', 'fixed authored curve'), ('focus', 'fixed vocal focus'), ('treatment', 'shared field'),
    ('evidence', 'shared field'), ('provenance', 'authored provenance'),
])
def test_vocal_replay_rejects_every_unauthorized_override(tmp_path, monkeypatch, edit, message):
    package, _, _ = _built_vocal_package(tmp_path, monkeypatch)
    plan = json.loads((package / 'plan.json').read_text())
    if edit == 'curve':
        plan['segments'][0]['layers']['vocals']['envelope'][2]['gain'] = .5
    elif edit == 'focus':
        plan['segments'][0]['focus'] = 'bass'
    elif edit == 'treatment':
        plan['segments'][0]['layers']['bass']['treatment'] = 'ring'
    elif edit == 'evidence':
        plan['evidence'][0]['feedback_record']['answer']['notes'] = 'changed'
    else:
        plan['envelope_provenance']['kind'] = 'generated'
    override = tmp_path / f'{edit}.json'; override.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match=message):
        builder.vocal_emphasis_replay_inputs(package, override)
