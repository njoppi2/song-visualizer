from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from songviz.structure_annotations import normalize_annotations


ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_PATH = ROOT / "benchmark" / "feedback" / "section-editor-02.json"
FEEDBACK_SHA256 = "dd100b34d43ece3dbe500b297f0fb4320db71e3eebb9dc705959145e54aa8b2f"


def _feedback() -> dict:
    return json.loads(FEEDBACK_PATH.read_text())


def _normalized(*, interpretations: dict | None = None) -> dict:
    return normalize_annotations(
        _feedback(), feedback_sha256=FEEDBACK_SHA256, interpretations=interpretations
    )


def test_real_feedback_preserves_19_spans_and_four_explicit_motif_groups() -> None:
    result = _normalized()

    assert result["provenance"]["feedback_sha256"] == hashlib.sha256(FEEDBACK_PATH.read_bytes()).hexdigest()
    layer = result["layers"][0]
    assert len(layer["spans"]) == 19
    assert [(group["name"], len(group["span_ids"])) for group in layer["identity_groups"]] == [
        ("verse 1", 1),
        ("chorus", 4),
        ("verse 2", 4),
        ("bridge", 2),
    ]
    assert layer["spans"][0]["identity_id"] is None
    assert layer["spans"][0]["certainty"] == "unspecified"
    assert layer["spans"][5]["transition"] is None  # Labels never create interpretation.


def test_explicit_map_preserves_short_transition_interval() -> None:
    feedback = _feedback()
    segment = feedback["annotations"]["layers"][0]["segments"][1]
    interpretations = {
        "schema_version": 1,
        "feedback_sha256": FEEDBACK_SHA256,
        "segments": {
            segment["id"]: {
                "transition": True,
                "rationale": "The analyst marks the supplied interval as a transition.",
            }
        },
    }

    result = _normalized(interpretations=interpretations)
    span = result["layers"][0]["spans"][1]
    assert span["end_s"] - span["start_s"] == pytest.approx(0.807118)
    assert span["transition"] is True
    assert span["interpretation_rationale"] == interpretations["segments"][segment["id"]]["rationale"]


def test_foreign_interpretation_fingerprint_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match"):
        _normalized(
            interpretations={"schema_version": 1, "feedback_sha256": "0" * 64, "segments": {}}
        )


def test_same_motif_in_different_layers_is_not_linked() -> None:
    feedback = _feedback()
    original = feedback["annotations"]["layers"][0]
    second = deepcopy(original)
    second["id"] = "second-layer"
    second["name"] = "Independent layer"
    second["segments"] = deepcopy(original["segments"])
    for index, segment in enumerate(second["segments"]):
        segment["id"] = f"second-{index}"
    second["segments"][0]["motif"] = "chorus"
    feedback["annotations"]["layers"].append(second)

    result = normalize_annotations(feedback, feedback_sha256=FEEDBACK_SHA256)
    first_chorus = next(group for group in result["layers"][0]["identity_groups"] if group["name"] == "chorus")
    second_chorus = next(group for group in result["layers"][1]["identity_groups"] if group["name"] == "chorus")
    assert first_chorus["id"] != second_chorus["id"]
    assert all(span_id.startswith("second-") for span_id in second_chorus["span_ids"])


@pytest.mark.parametrize("change", ["gap", "nan", "bool", "duplicate"])
def test_invalid_partitions_and_ids_are_rejected(change: str) -> None:
    feedback = _feedback()
    segments = feedback["annotations"]["layers"][0]["segments"]
    if change == "gap":
        segments[1]["start_s"] += 0.01
    elif change == "nan":
        segments[1]["end_s"] = float("nan")
    elif change == "bool":
        segments[1]["start_s"] = False
    else:
        segments[1]["id"] = segments[0]["id"]
    with pytest.raises(ValueError):
        normalize_annotations(feedback, feedback_sha256=FEEDBACK_SHA256)


@pytest.mark.parametrize(
    "interpretations",
    [
        {"schema_version": 1, "feedback_sha256": FEEDBACK_SHA256, "segments": {"missing": {"transition": True, "rationale": "x"}}},
        {"schema_version": 1, "feedback_sha256": FEEDBACK_SHA256, "segments": {}, "new_key": True},
        {"schema_version": 1, "feedback_sha256": FEEDBACK_SHA256, "segments": {}},
        {"schema_version": 1, "feedback_sha256": FEEDBACK_SHA256, "segments": {"PLACEHOLDER": {"transition": 1, "rationale": "x"}}},
    ],
)
def test_invalid_interpretations_are_rejected(interpretations: dict) -> None:
    feedback = _feedback()
    first_id = feedback["annotations"]["layers"][0]["segments"][0]["id"]
    rendered = json.loads(json.dumps(interpretations).replace("PLACEHOLDER", first_id))
    # The third parametrized value is valid by itself; make it invalid by using
    # an empty rationale on a known segment.
    if rendered["segments"] == {} and "new_key" not in rendered:
        rendered["segments"] = {first_id: {"transition": True, "rationale": " "}}
    with pytest.raises(ValueError):
        normalize_annotations(feedback, feedback_sha256=FEEDBACK_SHA256, interpretations=rendered)


def test_input_is_immutable_and_unknown_motifs_remain_unidentified() -> None:
    feedback = _feedback()
    feedback["source"]["extra_source_metadata"] = {"kept": ["as supplied"]}
    feedback["annotations"]["layers"][0]["segments"][0]["motif"] = ""
    before = deepcopy(feedback)

    result = normalize_annotations(feedback, feedback_sha256=FEEDBACK_SHA256)

    assert feedback == before
    assert result["source"]["extra_source_metadata"] == {"kept": ["as supplied"]}
    assert result["layers"][0]["spans"][0]["motif"] == ""
    assert result["layers"][0]["spans"][0]["identity_id"] is None
