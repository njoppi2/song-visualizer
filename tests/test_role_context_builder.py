from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from experiments import build_role_context_review as builder
from songviz.ingest import sha256_file
from test_local_structure_builder import _package, _record, _refresh_record, _write_json


def _manifest(path: Path, kind: str, *, sources: list[Path], snapshots: list[Path], outputs: list[Path]) -> None:
    _write_json(path, {"kind": kind, "sources": [_record(p) for p in sources], "input_snapshots": [_record(p) for p in snapshots], "outputs": [_record(p) for p in outputs]})


def _ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    package = _package(tmp_path)
    source = json.loads(package["reference"].read_text())["source"]
    listening = tmp_path / "listening"; listening.mkdir()
    review = listening / "review.json"
    examples = [{"id": key, "start_s": start, "end_s": end, "focus_start_s": start + 1.2, "focus_end_s": start + 2.7}
                for key, start, end in (("drum-entry", 8., 19.), ("within-passage", 16., 26.), ("verse-ending", 24., 36.), ("transition-extent", 0., 12.))]
    _write_json(review, {"example_set_id": "listening-examples-01", "source_audio_sha256": source["source_audio_sha256"], "audio_sha256": source["audio_sha256"], "duration_s": source["duration_s"], "examples": examples})
    snapshot = listening / "snapshot.html"; snapshot.write_text("frozen\n")
    _manifest(listening / "manifest.json", "songviz-listening-examples", sources=[package["source"]], snapshots=[snapshot], outputs=[review])
    feedback = tmp_path / "feedback.json"
    _write_json(feedback, {"schema_version": 1, "kind": "songviz-listening-examples-feedback", "review_sha256": sha256_file(review), "example_set_id": "listening-examples-01", "source_audio_sha256": source["source_audio_sha256"], "audio_sha256": source["audio_sha256"], "answers": [{"example_id": key, "perceived_change": value, "notes": f"exact frozen note for {key}"} for key, value in zip(builder.ANSWER_IDS, ("local", "none", "subtle", "broad"))]})
    monkeypatch.setattr(builder, "FEEDBACK_SHA256", sha256_file(feedback))
    return {**package, "listening": listening, "feedback": feedback}


def _build(p: dict[str, Path], out: Path) -> None:
    builder.build(parent=p["parent"], listening=p["listening"], feedback=p["feedback"], audio_review=p["audio_review"], out=out)


def test_builds_full_context_and_bounded_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _ready(tmp_path, monkeypatch); out = tmp_path / "out"; _build(p, out)
    assert {x.name for x in out.iterdir()} >= {"role-context.json", "evaluation.json", "review.json", "report.md", "index.html", "manifest.json"}
    full, page, evaluation = (json.loads((out / name).read_text()) for name in ("role-context.json", "review.json", "evaluation.json"))
    assert len(full["curves"]) == 3 and all(len(c["samples"]) == 49 for c in full["curves"])
    assert len(page["guided_cases"]) == 4 and "curves" not in page
    assert page["guided_cases"][1]["raw_response_notes"] == "exact frozen note for within-passage"
    assert len(evaluation["cases"][0]["per_scale"]) == 3
    for group in ("sources", "input_snapshots", "outputs"):
        for row in json.loads((out / "manifest.json").read_text())[group]:
            assert sha256_file(Path(row["path"])) == row["sha256"]
    template = (out / "inputs/experiments/templates/role_context_review.html").read_text()
    assert (out / "index.html").read_text() == template.replace("{{REVIEW_JSON}}", builder.embedded_json(page)).replace("{{REVIEW_SHA}}", sha256_file(out / "review.json"))


def test_refuses_overwrite_tamper_and_bad_support(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _ready(tmp_path, monkeypatch); out = tmp_path / "out"; _build(p, out)
    with pytest.raises(FileExistsError): _build(p, out)
    p = _ready(tmp_path / "tamper", monkeypatch); p["feedback"].write_text("{}\n")
    with pytest.raises(ValueError, match="SHA-256"): _build(p, tmp_path / "bad")
    p = _ready(tmp_path / "support", monkeypatch)
    native = builder._detector()
    def malformed(*args, **kwargs):
        result = native(*args, **kwargs); result["curves"][0]["samples"][2]["support_end_s"] += .1; return result
    monkeypatch.setattr(builder, "_detector", lambda: malformed)
    with pytest.raises(ValueError, match="support does not match"): _build(p, tmp_path / "bad-support")
    assert not (tmp_path / "bad-support").exists()


def test_labels_notes_and_episode_candidates_cannot_move_detector_or_anchor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _ready(tmp_path, monkeypatch); first, second = tmp_path / "first", tmp_path / "second"; _build(p, first)
    initial = json.loads((first / "role-context.json").read_text())
    reference = json.loads(p["reference"].read_text()); reference["layers"][0]["spans"][0]["label"] = "new human label"; _write_json(p["reference"], reference); _refresh_record(p["parent_manifest"], p["reference"])
    feedback = json.loads(p["feedback"].read_text()); feedback["answers"][1]["notes"] = "new exact note"; _write_json(p["feedback"], feedback); monkeypatch.setattr(builder, "FEEDBACK_SHA256", sha256_file(p["feedback"]))
    _build(p, second)
    assert json.loads((second / "role-context.json").read_text()) == initial
    assert json.loads((second / "review.json").read_text())["guided_cases"][1]["raw_response_notes"] == "new exact note"
    assert "episode" not in inspect.signature(builder.build).parameters
    case = json.loads((second / "evaluation.json").read_text())["cases"][0]
    focus = json.loads((p["listening"] / "review.json").read_text())["examples"][0]
    times = initial["times_s"]; midpoint = (focus["focus_start_s"] + focus["focus_end_s"]) / 2
    assert case["fixed_anchor_index"] == min(range(len(times)), key=lambda i: (abs(times[i] - midpoint), i))
