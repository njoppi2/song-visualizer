from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import build_change_episode_review as builder
from songviz.ingest import sha256_file
from test_local_structure_builder import _package, _record, _refresh_record, _write_json


def _manifest(path: Path, kind: str, *, sources: list[Path], snapshots: list[Path], outputs: list[Path]) -> None:
    _write_json(path, {"kind": kind, "sources": [_record(item) for item in sources],
                       "input_snapshots": [_record(item) for item in snapshots], "outputs": [_record(item) for item in outputs]})


def _episode(episodes: list[dict] | None = None):
    default_episode = {"id": "episode-0001", "stem": "vocals", "scale_beats": 2, "channel": "pattern", "start_s": 10., "end_s": 18., "peak_s": 14., "peak_index": 14, "support_start_s": 8., "support_end_s": 20., "available_at_s": 20., "physical_onset_s": None, "physical_settled_s": None, "perceived_importance": None, "vocal_function": None,
                       "peak_context": {"stems": {"bass": {"left": {"mean_rms": .2, "rms_power_share": .4}, "right": {"mean_rms": .4, "rms_power_share": .6}},
                                                  "vocals": {"left": {"mean_rms": .3, "rms_power_share": .6}, "right": {"mean_rms": .1, "rms_power_share": .4}}}}}
    def detect(features, energy, bt, *, config=None):
        assert set(features) == set(energy) == {"bass", "vocals"}
        return {"schema_version": 1, "method": "synthetic-numeric-only", "config": {}, "times_s": bt.tolist(),
                "curves": [{"stem": "vocals", "scale_beats": 2, "channel": "pattern", "values": [None] + [.3] * (len(bt)-1), "high_threshold": .2, "low_threshold": .1}],
                "episodes": episodes if episodes is not None else [default_episode]}
    return detect


def _ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    package = _package(tmp_path)
    reference = json.loads(package["reference"].read_text())
    reference["layers"][0]["spans"] = [{"id": f"t{i}", "start_s": float(i * 8), "end_s": float(i * 8 + 3), "label": f"transition {i}", "certainty": "unspecified", "identity_id": None, "variation": None, "transition": True} for i in range(5)]
    _write_json(package["reference"], reference); _refresh_record(package["parent_manifest"], package["reference"])
    comparison = tmp_path / "comparison"; comparison.mkdir(); control = comparison / "variants/control"; control.mkdir(parents=True)
    dips = [{"id": "dip-1", "start_s": 8., "end_s": 10.}, {"id": "dip-2", "start_s": 24., "end_s": 26.}]
    times = json.loads(package["timing"].read_text())["requested_times_s"]
    control_path = control / "predictions.json"; _write_json(control_path, {"times_s": times, "transitions": dips})
    source_fields = json.loads(package["reference"].read_text())["source"]
    review_path = comparison / "review.json"; _write_json(review_path, {"audio_sha256": source_fields["audio_sha256"], "duration_s": source_fields["duration_s"], "variants": {"control": {"predictions": {"transitions": dips}}}, "summary": {"variants": {"control": {"candidate_counts": {"changes": {"count": 16}, "transitions": {"count": 2}}}}}})
    snapshot = comparison / "snapshot.py"; snapshot.write_text("# frozen\n")
    _manifest(comparison / "manifest.json", "songviz-local-structure-comparison", sources=[package["parent_manifest"]], snapshots=[snapshot], outputs=[review_path, control_path])
    listening = tmp_path / "listening"; listening.mkdir(); listen_review = listening / "review.json"
    examples = [{"id": key, "start_s": a, "end_s": b, "focus_start_s": a + 1, "focus_end_s": a + 2} for key, a, b in (("drum-entry", 8., 19.), ("within-passage", 16., 26.), ("verse-ending", 24., 36.), ("transition-extent", 0., 12.))]
    _write_json(listen_review, {"example_set_id": "listening-examples-01", "source_audio_sha256": source_fields["source_audio_sha256"], "audio_sha256": source_fields["audio_sha256"], "duration_s": source_fields["duration_s"], "examples": examples})
    listen_snapshot = listening / "snapshot.html"; listen_snapshot.write_text("snapshot\n")
    _manifest(listening / "manifest.json", "songviz-listening-examples", sources=[package["source"]], snapshots=[listen_snapshot], outputs=[listen_review])
    feedback = tmp_path / "feedback.json"; _write_json(feedback, {"schema_version": 1, "kind": "songviz-listening-examples-feedback", "review_sha256": sha256_file(listen_review), "example_set_id": "listening-examples-01", "source_audio_sha256": source_fields["source_audio_sha256"], "audio_sha256": source_fields["audio_sha256"], "answers": [{"example_id": key, "perceived_change": value, "notes": f"verbatim notes for {key}"} for key, value in zip(builder.ANSWER_IDS, ("local", "none", "subtle", "broad"))]})
    monkeypatch.setattr(builder, "FEEDBACK_SHA256", sha256_file(feedback)); monkeypatch.setattr(builder, "_detector", lambda: _episode())
    return {**package, "comparison": comparison, "listening": listening, "feedback": feedback}


def _build(package: dict[str, Path], out: Path) -> None:
    builder.build(parent=package["parent"], comparison=package["comparison"], listening=package["listening"], feedback=package["feedback"], audio_review=package["audio_review"], out=out)


def test_builds_complete_immutable_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = _ready(tmp_path, monkeypatch); out = tmp_path / "out"; _build(package, out)
    assert {item.name for item in out.iterdir()} >= {"episodes.json", "evaluation.json", "review.json", "report.md", "index.html", "manifest.json"}
    manifest = json.loads((out / "manifest.json").read_text())
    for group in ("sources", "input_snapshots", "outputs"):
        assert manifest[group]
        for item in manifest[group]: assert sha256_file(Path(item["path"])) == item["sha256"]
    assert (out / "inputs/listening-feedback.json").read_text() == package["feedback"].read_text()
    evaluation = json.loads((out / "evaluation.json").read_text())
    assert len(evaluation["evaluation"]["human_transition_response_interval_overlap"]) == 5
    assert all(len(row["against_all_five_transition_spans"]) == 5 for row in evaluation["evaluation"]["frozen_two_dip_overlap"])
    first_dip = evaluation["evaluation"]["frozen_two_dip_overlap"][0]["against_all_five_transition_spans"][0]
    assert first_dip["transition_id"] == "t0" and first_dip["start_error_s"] == 8.0 and first_dip["end_error_s"] == 7.0
    assert evaluation["guided_cases"][1]["perceived_change"] == "none"
    assert "verbatim notes for within-passage" == evaluation["guided_cases"][1]["raw_response_notes"]
    assert "response intervals" in (out / "report.md").read_text()
    page = json.loads((out / "review.json").read_text())
    assert page["cases"][0]["chosen_episode_context"]["peak_windows_s"]["left"] == {"start_s": 12.0, "end_s": 14.0}
    template = (out / "inputs/experiments/templates/change_episode_review.html").read_text()
    assert (out / "index.html").read_text() == template.replace("{{REVIEW_JSON}}", builder.embedded_json(page)).replace("{{REVIEW_SHA}}", sha256_file(out / "review.json"))


def test_refuses_overwrite_feedback_tamper_and_missing_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = _ready(tmp_path, monkeypatch); out = tmp_path / "out"; _build(package, out)
    with pytest.raises(FileExistsError): _build(package, out)
    package = _ready(tmp_path / "tampered", monkeypatch); package["feedback"].write_text("{}\n")
    with pytest.raises(ValueError, match="SHA-256"): _build(package, tmp_path / "bad")
    package = _ready(tmp_path / "binding", monkeypatch); feedback = json.loads(package["feedback"].read_text()); feedback["review_sha256"] = "not-the-review"; _write_json(package["feedback"], feedback); monkeypatch.setattr(builder, "FEEDBACK_SHA256", sha256_file(package["feedback"]))
    with pytest.raises(ValueError, match="hash-bound"): _build(package, tmp_path / "binding-out")
    package = _ready(tmp_path / "missing", monkeypatch); feedback = json.loads(package["feedback"].read_text()); feedback["answers"].pop(); _write_json(package["feedback"], feedback); monkeypatch.setattr(builder, "FEEDBACK_SHA256", sha256_file(package["feedback"]))
    with pytest.raises(ValueError, match="four guided example IDs"): _build(package, tmp_path / "missing-out")


def test_labels_do_not_enter_detector_and_changed_numeric_grid_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = _ready(tmp_path, monkeypatch); first, second = tmp_path / "first", tmp_path / "second"; _build(package, first)
    reference = json.loads(package["reference"].read_text()); reference["layers"][0]["spans"][0]["label"] = "changed human label"; _write_json(package["reference"], reference); _refresh_record(package["parent_manifest"], package["reference"]); _build(package, second)
    assert json.loads((first / "episodes.json").read_text()) == json.loads((second / "episodes.json").read_text())
    package = _ready(tmp_path / "grid", monkeypatch); timing = json.loads(package["timing"].read_text()); timing["requested_times_s"][4] += .25; _write_json(package["timing"], timing); _refresh_record(package["parent_manifest"], package["timing"])
    with pytest.raises(ValueError, match="Feature and timing grids differ"): _build(package, tmp_path / "grid-out")
    package = _ready(tmp_path / "cross-grid", monkeypatch); control = package["comparison"] / "variants/control/predictions.json"; prediction = json.loads(control.read_text()); prediction["times_s"][4] += .25; _write_json(control, prediction); _refresh_record(package["comparison"] / "manifest.json", control)
    with pytest.raises(ValueError, match="Comparison control times_s"): _build(package, tmp_path / "cross-grid-out")
