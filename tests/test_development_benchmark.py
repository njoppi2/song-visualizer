import hashlib
import json
import shutil
from pathlib import Path

import pytest

from experiments.build_development_benchmark import RAW_HASH, build


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_minimal_repo(destination):
    for relative in (
        "benchmark/feedback/listening-examples-01.json",
        "benchmark/feedback/section-editor-02.json",
        "outputs/reviews/role-context-02/manifest.json",
        "outputs/reviews/role-context-02/review.json",
        "outputs/reviews/role-context-02/evaluation.json",
        "outputs/reviews/structure-evaluation-03/manifest.json",
        "outputs/reviews/structure-evaluation-03/reference.json",
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)


def test_build_preserves_notes_unknowns_and_blank_template(tmp_path):
    output = tmp_path / "package"
    build(ROOT, output)
    benchmark = json.loads((output / "benchmark.json").read_text())
    raw = json.loads((ROOT / "benchmark/feedback/listening-examples-01.json").read_text())
    expected = {x["example_id"]: x["notes"] for x in raw["answers"]}
    assert benchmark["provenance"]["accepted_source_hashes"]["benchmark/feedback/listening-examples-01.json"] == RAW_HASH
    assert [x["case_id"] for x in benchmark["cases"]] == ["case-01", "case-02", "case-03", "case-04"]
    for case in benchmark["cases"]:
        assert case["listener_feedback"]["raw_notes_exact"] == expected[case["evaluation_reference_id"]]
        assert len(case["acoustic_evidence"]["fixed_neighbor_scale_observations"]) == 15
        assert "null" in json.dumps(case["acoustic_evidence"])
        assert "Missing, ambiguous, or abstaining" in case["rubric"]["unresolved"]
        assert "Contradicts" in case["rubric"]["fail"]
    template = json.loads((output / "future_candidate_output_template.json").read_text())
    assert "rubric" not in json.dumps(template)
    assert [item["case_id"] for item in template["cases"]] == ["case-01", "case-02", "case-03", "case-04"]
    assert all(value is None for item in template["cases"] for key, value in item.items() if key != "case_id")


def test_rejects_tampered_provenance(tmp_path):
    repo = tmp_path / "repo"
    copy_minimal_repo(repo)
    raw = repo / "benchmark/feedback/listening-examples-01.json"
    raw.write_text(raw.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="raw listening feedback"):
        build(repo, tmp_path / "package")


def test_rejects_tampered_source_and_matching_mutable_manifest(tmp_path):
    repo = tmp_path / "repo"
    copy_minimal_repo(repo)
    review = repo / "outputs/reviews/role-context-02/review.json"
    review.write_text(review.read_text() + " ", encoding="utf-8")
    manifest_path = repo / "outputs/reviews/role-context-02/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for record in manifest["outputs"]:
        if record["path"].endswith("role-context-02/review.json"):
            record["sha256"] = sha(review)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="accepted role manifest"):
        build(repo, tmp_path / "package")


def test_refuses_overwrite_and_replays_deterministically(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    build(ROOT, first)
    with pytest.raises(FileExistsError):
        build(ROOT, first)
    build(ROOT, second)
    for name in ("benchmark.json", "future_candidate_output_template.json", "report.md"):
        assert sha(first / name) == sha(second / name)
