from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evidence_timeline_builder", ROOT / "experiments/build_evidence_timeline.py")
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(builder)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def rec(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": builder.sha256_file(path), "bytes": path.stat().st_size}


def manifest(package: Path, kind: str | None, files: list[Path]) -> None:
    value = {"schema_version": 1, "outputs": [rec(p) for p in files]}
    if kind:
        value["kind"] = kind
    write_json(package / "manifest.json", value)


def fixture(tmp_path: Path) -> dict[str, Path]:
    role, evaluation, listening, audio = (tmp_path / name for name in ("role", "evaluation", "listening", "audio"))
    feedback = tmp_path / "feedback.json"
    write_json(feedback, {"answers": [{"example_id": "one", "notes": "raw"}]})
    role_json = {"schema_version": 1, "kind": "songviz-role-context", "times_s": [0., 1., 2.], "stem_names": ["bass", "vocals"],
                 "audibility_floors": {"bass": .1, "vocals": .1}, "curves": [{"scale_beats": 1, "samples": [None, {"anchor_s": 1., "stems": {"bass": {"left": {"mean_rms": .1}, "right": {"mean_rms": .2}}, "vocals": {"left": {"active_fraction": 1}, "right": {"active_fraction": 1}}}}, None]}]}
    write_json(role / "role-context.json", role_json)
    manifest(role, "songviz-role-context-review", [role / "role-context.json"])
    role_manifest = json.loads((role / "manifest.json").read_text())
    role_manifest["sources"] = [rec(feedback)]
    write_json(role / "manifest.json", role_manifest)
    recurrence = {"results": [{"scale_beats": 16, "stride_beats": 4, "spans": [{"start_s": 0., "end_s": 1.}, {"start_s": 1., "end_s": 2.}], "context": [{"local_pattern_change": None}], "pairs": [{"a": 0, "b": 1, "similarity": .8, "pattern_similarity": .9, "arrangement_similarity": .7, "shared_active_stems": ["vocals"]}]}]}
    write_json(evaluation / "recurrence.json", recurrence)
    manifest(evaluation, "songviz-structural-development-evaluation", [evaluation / "recurrence.json"])
    (audio / "original.wav").parent.mkdir(parents=True, exist_ok=True)
    (audio / "original.wav").write_bytes(b"wav")
    manifest(audio, None, [audio / "original.wav"])
    review = {"song_title": "Test", "duration_s": 2., "audio_path": "original.wav", "audio_sha256": builder.sha256_file(audio / "original.wav"),
              "examples": [{"id": "one", "start_s": 0., "end_s": 2.}]}
    write_json(listening / "review.json", review)
    manifest(listening, "songviz-listening-examples", [listening / "review.json"])
    return {"role_context": role, "structure_evaluation": evaluation, "listening": listening, "audio_review": audio, "feedback": feedback}


def build(tmp_path: Path, paths: dict[str, Path]) -> Path:
    out = tmp_path / "out"
    builder.build(**paths, out=out)
    return out


def test_build_maps_nulls_audio_and_paired_recurrence(tmp_path: Path):
    out = build(tmp_path, fixture(tmp_path))
    data = json.loads((out / "evidence-timeline.json").read_text())
    assert data["local_context"]["scales"][0]["samples"][0] is None
    assert data["unavailable_probes"] == [{"time_s": 0.}, {"time_s": 2.}]
    assert data["audio_path"] == "../audio/original.wav"
    pair = data["paired_listening"][0]
    assert pair["a"]["start_s"] == pair["prior"]["start_s"] == 0.
    assert pair["b"]["end_s"] == pair["target"]["end_s"] == 2.
    assert data["recurrence"]["scales"][0]["pairs"][0]["a"] == 0
    manifest_data = json.loads((out / "manifest.json").read_text())
    assert builder.sha256_file(out / "evidence-timeline.json") == manifest_data["outputs"][0]["sha256"]
    page = (out / "index.html").read_text()
    assert "{{TIMELINE_JSON}}" not in page and "{{TIMELINE_SHA}}" not in page
    assert builder.sha256_file(out / "evidence-timeline.json") in page


def test_refuses_existing_output_and_tampered_frozen_record(tmp_path: Path):
    paths = fixture(tmp_path)
    out = build(tmp_path, paths)
    with pytest.raises(FileExistsError):
        builder.build(**paths, out=out)
    (paths["role_context"] / "role-context.json").write_text("{}")
    with pytest.raises(ValueError, match="Changed or missing frozen input"):
        builder.build(**paths, out=tmp_path / "new-out")


def test_rejects_invalid_recurrence_pair_and_unbound_audio(tmp_path: Path):
    paths = fixture(tmp_path)
    recurrence = paths["structure_evaluation"] / "recurrence.json"
    data = json.loads(recurrence.read_text())
    data["results"][0]["pairs"][0]["b"] = 9
    write_json(recurrence, data)
    manifest(paths["structure_evaluation"], "songviz-structural-development-evaluation", [recurrence])
    with pytest.raises(ValueError, match="missing span"):
        builder.build(**paths, out=tmp_path / "bad")
