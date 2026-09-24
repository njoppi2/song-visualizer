#!/usr/bin/env python3
"""Build the small, provenance-bound development benchmark from frozen inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


RAW_HASH = "f69b14f0343d0ebe4af12d37f66e88c76aa5d8e7673f7529345333589d5c47b6"
SECTION_HASH = "dd100b34d43ece3dbe500b297f0fb4320db71e3eebb9dc705959145e54aa8b2f"
ROLE_MANIFEST_HASH = "129eab2556bc43bcef9a351015eed4847ca4b81b3555fe1afbeb615943af6cbe"
STRUCTURE_MANIFEST_HASH = "f7d3290b1ed33f7f66a469dedd6bd7714280dc5d3667023d9461ae4a3137192d"
CASE_DISTINCTIONS = {
    "within-passage": "within-passage acoustic differences without a perceived development",
    "drum-entry": "fuller drum arrangement within a continuing idea",
    "verse-ending": "continuing vocal sound with a reported change in behavior or role",
    "transition-extent": "broad breakdown and recovery extent rather than an exact boundary",
}
RUBRICS = {
    "within-passage": {
        "pass": "Explicitly and supportably concludes that there is no meaningful development at the reviewed focus despite observed acoustic variation.",
        "fail": "Contradicts the requirement by declaring meaningful development or importance solely because an acoustic descriptor changes.",
        "unresolved": "Missing, ambiguous, or abstaining output; identifying acoustic variation without addressing meaningful development is unresolved.",
    },
    "drum-entry": {
        "pass": "Supportably identifies an added drum layer and continuing material; it need not choose a verse/chorus name or make a bass claim.",
        "fail": "Contradicts the requirement by asserting no drum-entry development or discontinuous material without candidate evidence.",
        "unresolved": "Missing, ambiguous, or abstaining output; recognizing change without its continuity distinction is unresolved.",
    },
    "verse-ending": {
        "pass": "Produces evidence for changed vocal behavior while voice continues. Role/leadership remains unresolved unless independently supported; an actual supported laughter detector is allowed.",
        "fail": "Contradicts the requirement by asserting voice is fully absent or behavior unchanged without candidate evidence, or by treating RMS alone as behavior evidence.",
        "unresolved": "Missing, ambiguous, or abstaining output; acoustic change alone leaves vocal behavior/role unresolved.",
    },
    "transition-extent": {
        "pass": "Represents broad breakdown/recovery extent. It may propose time bounds when it supplies evidence and uncertainty, but must not present them as exact validated truth.",
        "fail": "Contradicts the requirement by asserting no broad breakdown/recovery or by treating audition bounds or a narrow dip as exact transition truth.",
        "unresolved": "Missing, ambiguous, or abstaining output; a local dip without broad extent is unresolved.",
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_hash(manifest: dict, suffix: str) -> str | None:
    for group in ("sources", "input_snapshots", "outputs"):
        for record in manifest.get(group, []):
            if record["path"].replace("\\", "/").endswith(suffix):
                return record["sha256"]
    return None


def verify(path: Path, expected: str, description: str) -> None:
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"provenance failure for {description}: expected {expected}, got {actual}")


def intersecting_spans(reference: dict, bounds: dict) -> list[dict]:
    layer = reference["layers"][0]
    groups = {item["id"]: item for item in layer["identity_groups"]}
    selected = []
    for span in layer["spans"]:
        if span["start_s"] < bounds["end_s"] and span["end_s"] > bounds["start_s"]:
            selected.append({
                "span": span,
                "label_status": "existing section-editor label; not benchmark-created truth",
                "analyst_fields": {
                    key: span[key] for key in ("identity_id", "variation", "transition", "interpretation_rationale")
                },
                "identity_group": groups.get(span["identity_id"]),
            })
    return selected


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build(repo: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    raw_path = repo / "benchmark/feedback/listening-examples-01.json"
    section_path = repo / "benchmark/feedback/section-editor-02.json"
    role_dir = repo / "outputs/reviews/role-context-02"
    structure_dir = repo / "outputs/reviews/structure-evaluation-03"
    role_manifest = load(role_dir / "manifest.json")
    structure_manifest = load(structure_dir / "manifest.json")
    verify(role_dir / "manifest.json", ROLE_MANIFEST_HASH, "accepted role manifest")
    verify(structure_dir / "manifest.json", STRUCTURE_MANIFEST_HASH, "accepted structure manifest")
    required = [
        (raw_path, RAW_HASH, "raw listening feedback"),
        (section_path, SECTION_HASH, "raw section feedback"),
        (role_dir / "review.json", manifest_hash(role_manifest, "role-context-02/review.json"), "role review"),
        (role_dir / "evaluation.json", manifest_hash(role_manifest, "role-context-02/evaluation.json"), "role evaluation"),
        (structure_dir / "reference.json", manifest_hash(structure_manifest, "structure-evaluation-03/reference.json"), "structure reference"),
    ]
    for path, expected, description in required:
        if not expected:
            raise ValueError(f"accepted manifest lacks hash for {description}")
        verify(path, expected, description)
    if manifest_hash(role_manifest, "listening-feedback.json") != RAW_HASH:
        raise ValueError("role manifest does not bind the accepted raw feedback hash")
    if manifest_hash(structure_manifest, "section-editor-02.json") != SECTION_HASH:
        raise ValueError("structure manifest does not bind the accepted section feedback hash")

    raw = load(raw_path)
    review = load(role_dir / "review.json")
    evaluation = load(role_dir / "evaluation.json")
    reference = load(structure_dir / "reference.json")
    answers = {answer["example_id"]: answer for answer in raw["answers"]}
    guided = {case["id"]: case for case in review["guided_cases"]}
    measured = {case["id"]: case for case in evaluation["cases"]}
    order = ["within-passage", "drum-entry", "verse-ending", "transition-extent"]
    cases = []
    for case_id in order:
        answer, context, acoustic = answers[case_id], guided[case_id], measured[case_id]
        if context["raw_response_notes"] != answer["notes"] or context["perceived_change"] != answer["perceived_change"]:
            raise ValueError(f"{case_id} review feedback does not match raw feedback")
        observations = []
        if [scale["scale_beats"] for scale in acoustic["per_scale"]] != [2, 4, 8]:
            raise ValueError(f"{case_id} scales are not the fixed 2/4/8 grid")
        for scale in acoustic["per_scale"]:
            expected_indices = list(range(acoustic["fixed_anchor_index"] - 2, acoustic["fixed_anchor_index"] + 3))
            if [sample["anchor_index"] for sample in scale["anchor_samples"]] != expected_indices:
                raise ValueError(f"{case_id} has non-neighbor anchor observations")
            # This is intentionally the five fixed neighbour anchors per scale: 15 total.
            observations.extend(scale["anchor_samples"])
        if len(observations) != 15:
            raise ValueError(f"{case_id} does not have exactly 15 fixed observations")
        cases.append({
            "case_id": f"case-{len(cases) + 1:02d}",
            "evaluation_reference_id": case_id,
            "development_distinction": CASE_DISTINCTIONS[case_id],
            "listener_feedback": {
                "perceived_change": answer["perceived_change"],
                "raw_notes_exact": answer["notes"],
                "uncertainty_and_scope": "Development listening feedback; it is not independently verified transcription, stem assignment, endpoint truth, or training target.",
            },
            "context": {
                "excerpt_bounds_s": context["excerpt_bounds_s"],
                "focus_bounds_s": context["focus_bounds_s"],
                "fixed_anchor_index": context["fixed_anchor_index"],
                "fixed_anchor_s": context["fixed_anchor_s"],
            },
            "acoustic_evidence": {
                "source": "role-context-02/evaluation.json",
                "observation_count": 15,
                "fixed_neighbor_scale_observations": observations,
                "interpretation_limit": "Acoustic descriptors and their null/unknown values do not establish identity, behavior, or importance.",
            },
            "identity_and_sections": {
                "source": "structure-evaluation-03/reference.json",
                "intersecting_existing_spans": intersecting_spans(reference, context["excerpt_bounds_s"]),
                "limit": "Labels and analyst interpretations remain distinct. Unlabeled times are unknown, not negative identity examples.",
            },
            "behavior": {"status": "evaluation-only claim; unknown unless stated by the raw listener note"},
            "importance": {"status": "evaluation-only listener judgment; no acoustic aggregate may substitute for it"},
            "rubric": RUBRICS[case_id],
        })
    provenance = {str(path.relative_to(repo)): digest(path) for path, _, _ in required}
    package = {
        "schema_version": 1,
        "kind": "songviz-development-benchmark",
        "development_only": True,
        "held_out": False,
        "tuning_disclosure": "These four previously seen listening cases are development data. Any method tuning on them must be disclosed; they cannot support held-out performance claims.",
        "scoring": "manual, per-case, unscored evaluation: no aggregate score, automatic promotion, or ranking of subtle versus local/broad labels; disagreement may expose a benchmark limit rather than universal truth",
        "separation": "Acoustic change, identity, behavior, and perceived importance are separate fields; no automatic semantic baseline prediction is inferred.",
        "provenance": {"accepted_source_hashes": provenance, "manifests": {"role_context": ROLE_MANIFEST_HASH, "structure": STRUCTURE_MANIFEST_HASH}},
        "cases": cases,
    }
    template = {
        "schema_version": 1,
        "purpose": "blank future candidate OUTPUT schema. It is not an inference input and does not itself prove blind evaluation.",
        "cases": [{"case_id": item["case_id"], "time_support": None, "transition_extent": None, "acoustic_change": None, "identity": None, "behavior": None, "importance": None, "evidence": None, "abstention": None, "rationale": None} for item in cases],
    }
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output / "build_development_benchmark.py")
    write_json(output / "benchmark.json", package)
    write_json(output / "future_candidate_output_template.json", template)
    report = "# Development benchmark\n\nFour previously seen development cases for manual future-candidate evaluation. This package is unscored and is not held out; tuning must be disclosed. Missing or abstaining output is unresolved, while contradiction is failure.\n\n"
    for item in cases:
        report += f"## {item['case_id']}\n\nRequired distinction: {item['development_distinction']}. Current acoustic evidence: 15 fixed neighbor/scale observations. Unknown: identity, behavior, and importance remain separate from acoustic descriptors.\n\nPass: {item['rubric']['pass']}\n\nFailure: {item['rubric']['fail']}\n\nUnresolved: {item['rubric']['unresolved']}\n\n"
    report += "## Appendix: exact raw listener notes\n\n"
    for item in cases:
        report += f"### {item['case_id']} ({item['evaluation_reference_id']}; `{item['listener_feedback']['perceived_change']}`)\n\n{item['listener_feedback']['raw_notes_exact']}\n\n"
    report += "Rubrics are evaluation-only manual criteria. Inspect assertions and cited evidence; do not use a keyword or text-based automatic grader. The blank output schema is not an inference input and does not prove blind evaluation.\n"
    (output / "report.md").write_text(report, encoding="utf-8")
    write_json(output / "manifest.json", {
        "schema_version": 1,
        "kind": "songviz-development-benchmark-package",
        "source_hashes": provenance,
        "source_manifest_hashes": package["provenance"]["manifests"],
        "code_snapshot": {"path": "build_development_benchmark.py", "sha256": digest(output / "build_development_benchmark.py")},
        "outputs": {name: digest(output / name) for name in ("benchmark.json", "future_candidate_output_template.json", "report.md")},
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        build(args.repo.resolve(), (args.output or args.repo / "outputs/reviews/development-benchmark-02").resolve())
    except (OSError, ValueError, KeyError) as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
