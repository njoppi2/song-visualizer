"""Normalize validated human section annotations for structural development work.

This module deliberately keeps the human annotation and an analyst's reading of
it separate.  In particular, section labels are opaque text: they are never
used to infer recurrence, variation, or transition status.
"""

from __future__ import annotations

from copy import deepcopy
import math
import re
from typing import Any


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CERTAINTIES = {"unspecified", "clear", "uncertain"}
_PARTITION_TOLERANCE_S = 1e-6


def _fail(message: str) -> None:
    raise ValueError(f"Invalid structural annotation feedback: {message}")


def _record(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{field} must be an object")
    return value


def _string(value: Any, field: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        _fail(f"{field} must be a string")
    if nonempty and not value:
        _fail(f"{field} must not be empty")
    return value


def _sha256(value: Any, field: str) -> str:
    value = _string(value, field)
    if not _SHA256.fullmatch(value):
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _finite_number(value: Any, field: str) -> float | int:
    # bool is a subclass of int, but is never a valid timestamp or duration.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{field} must be a finite number")
    return value


def _require_fields(value: dict[str, Any], fields: tuple[str, ...], path: str) -> None:
    for field in fields:
        if field not in value:
            _fail(f"{path}.{field} is required")


def _validate_interpretations(
    interpretations: dict[str, Any] | None,
    feedback_sha256: str,
    span_ids: set[str],
) -> dict[str, dict[str, Any]]:
    if interpretations is None:
        return {}
    interpretation = _record(interpretations, "interpretations")
    allowed_top = {"schema_version", "feedback_sha256", "segments"}
    unknown_top = set(interpretation) - allowed_top
    if unknown_top:
        _fail(f"interpretations has unknown keys: {', '.join(sorted(unknown_top))}")
    _require_fields(interpretation, ("schema_version", "feedback_sha256", "segments"), "interpretations")
    if type(interpretation["schema_version"]) is not int or interpretation["schema_version"] != 1:
        _fail("interpretations.schema_version must be 1")
    mapped_digest = _sha256(interpretation["feedback_sha256"], "interpretations.feedback_sha256")
    if mapped_digest != feedback_sha256:
        _fail("interpretations.feedback_sha256 does not match feedback_sha256")
    segments = _record(interpretation["segments"], "interpretations.segments")

    result: dict[str, dict[str, Any]] = {}
    allowed_segment = {"variation", "transition", "rationale"}
    for span_id, raw_mapping in segments.items():
        _string(span_id, "interpretations.segments key", nonempty=True)
        if span_id not in span_ids:
            _fail(f"interpretations refers to unknown segment id: {span_id}")
        mapping = _record(raw_mapping, f"interpretations.segments.{span_id}")
        unknown = set(mapping) - allowed_segment
        if unknown:
            _fail(
                f"interpretations.segments.{span_id} has unknown keys: {', '.join(sorted(unknown))}"
            )
        if "rationale" not in mapping:
            _fail(f"interpretations.segments.{span_id}.rationale is required")
        rationale = _string(mapping["rationale"], f"interpretations.segments.{span_id}.rationale")
        if not rationale.strip():
            _fail(f"interpretations.segments.{span_id}.rationale must not be empty")
        if "variation" not in mapping and "transition" not in mapping:
            _fail(f"interpretations.segments.{span_id} must state variation or transition")
        normalized: dict[str, Any] = {"rationale": rationale}
        if "variation" in mapping:
            variation = _string(mapping["variation"], f"interpretations.segments.{span_id}.variation")
            if not variation.strip():
                _fail(f"interpretations.segments.{span_id}.variation must not be empty")
            normalized["variation"] = variation
        if "transition" in mapping:
            if type(mapping["transition"]) is not bool:
                _fail(f"interpretations.segments.{span_id}.transition must be a bool")
            normalized["transition"] = mapping["transition"]
        result[span_id] = normalized
    return result


def normalize_annotations(
    feedback: dict,
    *,
    feedback_sha256: str,
    interpretations: dict | None = None,
) -> dict:
    """Validate and normalize a raw section-editor export.

    ``feedback_sha256`` identifies the byte-preserved raw export.  It is not
    recomputed from a Python dictionary, since parsing JSON loses the original
    byte representation.  Optional interpretations must carry this same
    fingerprint, making their analyst assertions explicitly attributable.
    """
    raw_feedback = _record(feedback, "feedback")
    supplied_digest = _sha256(feedback_sha256, "feedback_sha256")
    if type(raw_feedback.get("schema_version")) is not int or raw_feedback["schema_version"] != 1:
        _fail("schema_version must be 1")
    if raw_feedback.get("kind") != "songviz-section-annotations":
        _fail("kind must be songviz-section-annotations")
    manifest_sha256 = _sha256(raw_feedback.get("manifest_sha256"), "manifest_sha256")

    source = _record(raw_feedback.get("source"), "source")
    _require_fields(source, ("audio_sha256", "source_audio_sha256", "duration_s", "song_title"), "source")
    _sha256(source["audio_sha256"], "source.audio_sha256")
    _sha256(source["source_audio_sha256"], "source.source_audio_sha256")
    duration = _finite_number(source["duration_s"], "source.duration_s")
    if duration <= 0:
        _fail("source.duration_s must be greater than zero")
    _string(source["song_title"], "source.song_title", nonempty=True)

    annotations = _record(raw_feedback.get("annotations"), "annotations")
    _require_fields(annotations, ("layers", "active_layer_id", "selected_segment_id", "global_notes"), "annotations")
    layers = annotations["layers"]
    if not isinstance(layers, list) or not layers:
        _fail("annotations.layers must be a non-empty array")
    active_layer_id = _string(annotations["active_layer_id"], "annotations.active_layer_id")
    selected_segment_id = _string(annotations["selected_segment_id"], "annotations.selected_segment_id")
    global_notes = _string(annotations["global_notes"], "annotations.global_notes")

    seen_ids: set[str] = set()
    raw_layers: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    active_segments: set[str] | None = None
    for layer_index, raw_layer in enumerate(layers):
        path = f"annotations.layers[{layer_index}]"
        layer = _record(raw_layer, path)
        _require_fields(layer, ("id", "name", "segments"), path)
        layer_id = _string(layer["id"], f"{path}.id", nonempty=True)
        if layer_id in seen_ids:
            _fail(f"duplicate id: {layer_id}")
        seen_ids.add(layer_id)
        _string(layer["name"], f"{path}.name")
        segments = layer["segments"]
        if not isinstance(segments, list) or not segments:
            _fail(f"{path}.segments must be a non-empty array")
        expected_start: float | int = 0
        copied_segments: list[dict[str, Any]] = []
        for segment_index, raw_segment in enumerate(segments):
            segment_path = f"{path}.segments[{segment_index}]"
            segment = _record(raw_segment, segment_path)
            _require_fields(
                segment,
                ("id", "start_s", "end_s", "label", "motif", "notes", "certainty"),
                segment_path,
            )
            segment_id = _string(segment["id"], f"{segment_path}.id", nonempty=True)
            if segment_id in seen_ids:
                _fail(f"duplicate id: {segment_id}")
            seen_ids.add(segment_id)
            start = _finite_number(segment["start_s"], f"{segment_path}.start_s")
            end = _finite_number(segment["end_s"], f"{segment_path}.end_s")
            if abs(start - expected_start) > _PARTITION_TOLERANCE_S:
                _fail(f"{segment_path} is unsorted, gapped, or overlapping")
            if start < -_PARTITION_TOLERANCE_S or end > duration + _PARTITION_TOLERANCE_S or end <= start:
                _fail(f"{segment_path} has invalid bounds")
            label = _string(segment["label"], f"{segment_path}.label")
            motif = _string(segment["motif"], f"{segment_path}.motif")
            notes = _string(segment["notes"], f"{segment_path}.notes")
            certainty = _string(segment["certainty"], f"{segment_path}.certainty")
            if certainty not in _CERTAINTIES:
                _fail(f"{segment_path}.certainty is invalid")
            copied_segments.append(
                {
                    "id": segment_id,
                    "start_s": start,
                    "end_s": end,
                    "label": label,
                    "motif": motif,
                    "notes": notes,
                    "certainty": certainty,
                }
            )
            expected_start = end
        if abs(expected_start - duration) > _PARTITION_TOLERANCE_S:
            _fail(f"{path} does not end at source.duration_s")
        if layer_id == active_layer_id:
            active_segments = {segment["id"] for segment in copied_segments}
        raw_layers.append((layer, copied_segments))
    if active_segments is None:
        _fail("annotations.active_layer_id does not identify a layer")
    if selected_segment_id not in active_segments:
        _fail("annotations.selected_segment_id must belong to the active layer")

    span_ids = {segment["id"] for _, segments in raw_layers for segment in segments}
    mappings = _validate_interpretations(interpretations, supplied_digest, span_ids)

    normalized_layers: list[dict[str, Any]] = []
    for layer_index, (raw_layer, segments) in enumerate(raw_layers):
        motif_groups: dict[str, list[str]] = {}
        for segment in segments:
            # Empty motif means no identity claim.  Names are never trimmed,
            # normalized, or compared across layers.
            if segment["motif"]:
                motif_groups.setdefault(segment["motif"], []).append(segment["id"])
        identity_by_span: dict[str, str] = {}
        identity_groups: list[dict[str, Any]] = []
        for group_index, (motif, group_span_ids) in enumerate(motif_groups.items()):
            identity_id = f"layer-{layer_index}-identity-{group_index}"
            for span_id in group_span_ids:
                identity_by_span[span_id] = identity_id
            identity_groups.append({"id": identity_id, "name": motif, "span_ids": list(group_span_ids)})

        spans: list[dict[str, Any]] = []
        for segment in segments:
            interpretation = mappings.get(segment["id"], {})
            spans.append(
                {
                    **segment,
                    "identity_id": identity_by_span.get(segment["id"]),
                    "variation": interpretation.get("variation"),
                    "transition": interpretation.get("transition"),
                    "interpretation_rationale": interpretation.get("rationale"),
                }
            )
        normalized_layers.append(
            {
                "id": raw_layer["id"],
                "name": raw_layer["name"],
                "spans": spans,
                "identity_groups": identity_groups,
            }
        )

    return {
        "schema_version": 1,
        "kind": "songviz-structural-development-reference",
        "source": deepcopy(source),
        "provenance": {
            "feedback_sha256": supplied_digest,
            "manifest_sha256": manifest_sha256,
            "status": "development_feedback",
            "interpretations": "explicit_analyst_mapping" if interpretations is not None else "none",
        },
        "layers": normalized_layers,
        "global_notes": global_notes,
        "limitations": [
            "Raw labels are opaque and do not imply a universal section taxonomy.",
            "Motif identities are positive assertions only and are scoped to their layer.",
            "Variation and transition values are present only when explicitly mapped by an analyst.",
            "This development feedback does not establish holdout quality or calibrated novelty.",
        ],
    }
