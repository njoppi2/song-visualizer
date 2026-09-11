"""Descriptive local-change diagnostics for structural-development data.

This module deliberately compares timestamps and intervals without turning the
comparison into a detector score.  In particular, reference identity remains a
positive, layer-scoped annotation and recurrence values remain the persisted
offline phrase diagnostics that produced them.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


_CERTAINTIES = {"unspecified", "clear", "uncertain"}


def _fail(message: str) -> None:
    raise ValueError(f"Invalid local structure evaluation input: {message}")


def _record(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{path} must be an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{path} must be an array")
    return value


def _finite(value: Any, path: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{path} must be a finite number")
    return value


def _event_id(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{path} must be a non-empty string")
    return value


def _source_duration(reference: dict) -> float | int:
    source = _record(reference.get("source"), "reference.source")
    duration = _finite(source.get("duration_s"), "reference.source.duration_s")
    if duration <= 0:
        _fail("reference.source.duration_s must be greater than zero")
    return duration


def _in_bounds_time(value: Any, path: str, duration: float | int) -> float | int:
    value = _finite(value, path)
    if value < 0 or value > duration:
        _fail(f"{path} must be within source duration")
    return value


def _validate_predictions(predictions: dict, duration: float | int) -> tuple[list[dict], list[dict]]:
    predictions = _record(predictions, "predictions")
    changes = _list(predictions.get("changes"), "predictions.changes")
    transitions = _list(predictions.get("transitions"), "predictions.transitions")
    event_ids: set[str] = set()

    def claim_id(event: dict, path: str) -> str:
        event_id = _event_id(event.get("id"), f"{path}.id")
        if event_id in event_ids:
            _fail(f"duplicate prediction id: {event_id}")
        event_ids.add(event_id)
        return event_id

    normalized_changes: list[dict] = []
    for index, value in enumerate(changes):
        path = f"predictions.changes[{index}]"
        event = _record(value, path)
        normalized_changes.append({"id": claim_id(event, path),
                                   "time_s": _in_bounds_time(event.get("time_s"), f"{path}.time_s", duration)})

    normalized_transitions: list[dict] = []
    for index, value in enumerate(transitions):
        path = f"predictions.transitions[{index}]"
        event = _record(value, path)
        event_id = claim_id(event, path)
        start = _in_bounds_time(event.get("start_s"), f"{path}.start_s", duration)
        end = _in_bounds_time(event.get("end_s"), f"{path}.end_s", duration)
        if end <= start:
            _fail(f"{path} must have end_s greater than start_s")
        normalized_transitions.append({"id": event_id, "start_s": start, "end_s": end})
    return normalized_changes, normalized_transitions


def _reference_layers(reference: dict, duration: float | int) -> list[dict]:
    layers = _list(reference.get("layers"), "reference.layers")
    normalized: list[dict] = []
    layer_ids: set[str] = set()
    for layer_index, raw_layer in enumerate(layers):
        path = f"reference.layers[{layer_index}]"
        layer = _record(raw_layer, path)
        layer_id = _event_id(layer.get("id"), f"{path}.id")
        if layer_id in layer_ids:
            _fail(f"duplicate reference layer id: {layer_id}")
        layer_ids.add(layer_id)
        name = layer.get("name")
        if not isinstance(name, str):
            _fail(f"{path}.name must be a string")
        spans = _list(layer.get("spans"), f"{path}.spans")
        copied_spans: list[dict] = []
        span_ids: set[str] = set()
        previous_end: float | int | None = None
        for span_index, raw_span in enumerate(spans):
            span_path = f"{path}.spans[{span_index}]"
            span = _record(raw_span, span_path)
            span_id = _event_id(span.get("id"), f"{span_path}.id")
            if span_id in span_ids:
                _fail(f"duplicate reference span id in layer {layer_id}: {span_id}")
            span_ids.add(span_id)
            start = _in_bounds_time(span.get("start_s"), f"{span_path}.start_s", duration)
            end = _in_bounds_time(span.get("end_s"), f"{span_path}.end_s", duration)
            if end <= start:
                _fail(f"{span_path} must have end_s greater than start_s")
            if previous_end is not None and start < previous_end:
                _fail(f"{span_path} is unsorted or overlaps the preceding reference span")
            label = span.get("label")
            certainty = span.get("certainty")
            if not isinstance(label, str) or not isinstance(certainty, str):
                _fail(f"{span_path} label and certainty must be strings")
            if certainty not in _CERTAINTIES:
                _fail(f"{span_path}.certainty is invalid")
            identity = span.get("identity_id")
            if identity is not None and not isinstance(identity, str):
                _fail(f"{span_path}.identity_id must be a string or null")
            variation = span.get("variation")
            if variation is not None and not isinstance(variation, str):
                _fail(f"{span_path}.variation must be a string or null")
            transition = span.get("transition")
            if transition is not None and type(transition) is not bool:
                _fail(f"{span_path}.transition must be a bool or null")
            copied_spans.append({"id": span_id, "start_s": start, "end_s": end,
                                 "label": label, "certainty": certainty,
                                 "identity_id": identity, "variation": variation,
                                 "transition": transition})
            previous_end = end
        normalized.append({"id": layer_id, "name": name, "spans": copied_spans})
    return normalized


def _legacy_boundary_times(legacy_sections: list[dict], duration: float | int) -> list[float | int]:
    times: list[float | int] = []
    for index, raw_section in enumerate(_list(legacy_sections, "legacy_sections")):
        section = _record(raw_section, f"legacy_sections[{index}]")
        start = _in_bounds_time(section.get("start_s"), f"legacy_sections[{index}].start_s", duration)
        # A legacy interval is useful only as a supplied boundary source; it
        # need not partition the track or be promoted to a reference layer.
        if "end_s" in section:
            end = _in_bounds_time(section["end_s"], f"legacy_sections[{index}].end_s", duration)
            if end <= start:
                _fail(f"legacy_sections[{index}] must have end_s greater than start_s")
        if 0 < start < duration:
            times.append(start)
    return times


def _nearest(time_s: float | int, events: list[dict], *, time_key: str = "time_s") -> dict | None:
    if not events:
        return None
    event = min(events, key=lambda item: (abs(item[time_key] - time_s), item[time_key], item.get("id", "")))
    result = {"time_s": event[time_key], "delta_s": event[time_key] - time_s}
    if "id" in event:
        return {"id": event["id"], **result}
    return result


def _best_overlap(reference_span: dict, predictions: list[dict]) -> dict | None:
    candidates: list[tuple[float, dict, float]] = []
    for prediction in predictions:
        intersection = max(0, min(reference_span["end_s"], prediction["end_s"])
                           - max(reference_span["start_s"], prediction["start_s"]))
        if intersection == 0:
            continue
        union = ((reference_span["end_s"] - reference_span["start_s"])
                 + (prediction["end_s"] - prediction["start_s"]) - intersection)
        candidates.append((intersection / union, prediction, intersection))
    if not candidates:
        return None
    iou, prediction, intersection = max(candidates, key=lambda item: (item[0], item[1]["id"]))
    return {"id": prediction["id"], "start_s": prediction["start_s"], "end_s": prediction["end_s"],
            "iou": iou, "intersection_s": intersection,
            "start_error_s": prediction["start_s"] - reference_span["start_s"],
            "end_error_s": prediction["end_s"] - reference_span["end_s"]}


def _best_reference_overlap(prediction: dict, references: list[dict]) -> dict | None:
    candidate = _best_overlap(prediction, references)
    if candidate is None:
        return None
    reference = next(item for item in references if item["id"] == candidate["id"])
    return {"span_id": reference["id"], "label": reference["label"],
            "start_s": reference["start_s"], "end_s": reference["end_s"],
            "certainty": reference["certainty"], "iou": candidate["iou"],
            "intersection_s": candidate["intersection_s"],
            "start_error_s": prediction["start_s"] - reference["start_s"],
            "end_error_s": prediction["end_s"] - reference["end_s"]}


def _bounded_score_or_none(value: Any, path: str) -> float | int | None:
    if value is None:
        return None
    value = _finite(value, path)
    if not 0 <= value <= 1:
        _fail(f"{path} must be between zero and one or null")
    return value


def _recurrence_context(recurrence_results: list[dict], changes: list[dict], duration: float | int) -> list[dict]:
    rows: list[dict] = []
    for result_index, raw_result in enumerate(_list(recurrence_results, "recurrence_results")):
        result = _record(raw_result, f"recurrence_results[{result_index}]")
        scale_beats = result.get("scale_beats")
        stride_beats = result.get("stride_beats")
        if isinstance(scale_beats, bool) or not isinstance(scale_beats, int) or scale_beats <= 0:
            _fail(f"recurrence_results[{result_index}].scale_beats must be a positive integer")
        if isinstance(stride_beats, bool) or not isinstance(stride_beats, int) or stride_beats <= 0:
            _fail(f"recurrence_results[{result_index}].stride_beats must be a positive integer")
        windows: list[dict] = []
        for window_index, raw_window in enumerate(_list(result.get("spans"), f"recurrence_results[{result_index}].spans")):
            window = _record(raw_window, f"recurrence_results[{result_index}].spans[{window_index}]")
            start = _in_bounds_time(window.get("start_s"), f"recurrence_results[{result_index}].spans[{window_index}].start_s", duration)
            end = _in_bounds_time(window.get("end_s"), f"recurrence_results[{result_index}].spans[{window_index}].end_s", duration)
            if end <= start:
                _fail(f"recurrence_results[{result_index}].spans[{window_index}] must have positive duration")
            windows.append({"start_s": start, "end_s": end})
        contexts: dict[int, dict] = {}
        for context_index, raw_context in enumerate(_list(result.get("context"), f"recurrence_results[{result_index}].context")):
            context = _record(raw_context, f"recurrence_results[{result_index}].context[{context_index}]")
            span_index = context.get("span")
            if isinstance(span_index, bool) or not isinstance(span_index, int) or not 0 <= span_index < len(windows):
                _fail(f"recurrence_results[{result_index}].context[{context_index}].span is invalid")
            if span_index in contexts:
                _fail(f"duplicate recurrence context span: {span_index}")
            contexts[span_index] = context

        for change in changes:
            eligible = [(window["start_s"], index, window) for index, window in enumerate(windows)
                        if window["start_s"] >= change["time_s"]]
            target = min(eligible, key=lambda item: (item[0], item[1])) if eligible else None
            base = {"event_id": change["id"], "change_time_s": change["time_s"],
                    "scale_beats": scale_beats, "stride_beats": stride_beats}
            if target is None:
                rows.append({**base, "window": None, "offset_from_change_s": None, "available_at_s": None,
                             "best_prior_span": None, "historical_pattern_novelty": None,
                             "local_pattern_change": None, "local_arrangement_change": None,
                             "comparable_prior_count": None})
                continue
            _, target_index, window = target
            context = contexts.get(target_index)
            if context is None:
                raw_best_prior = None
                scores = {"historical_pattern_novelty": None, "local_pattern_change": None,
                          "local_arrangement_change": None, "comparable_prior_count": None}
            else:
                raw_best_prior = context.get("best_prior_span")
                scores = {key: _bounded_score_or_none(context.get(key),
                                                       f"recurrence_results[{result_index}].context[{target_index}].{key}")
                          for key in ("historical_pattern_novelty", "local_pattern_change", "local_arrangement_change")}
                comparable = context.get("comparable_prior_count")
                if isinstance(comparable, bool) or not isinstance(comparable, int) or comparable < 0:
                    _fail(f"recurrence_results[{result_index}].context[{target_index}].comparable_prior_count is invalid")
                scores["comparable_prior_count"] = comparable
            if raw_best_prior is None:
                best_prior = None
            else:
                if isinstance(raw_best_prior, bool) or not isinstance(raw_best_prior, int) or not 0 <= raw_best_prior < len(windows):
                    _fail(f"recurrence_results[{result_index}].context[{target_index}].best_prior_span is invalid")
                if raw_best_prior >= target_index:
                    _fail(f"recurrence_results[{result_index}].context[{target_index}].best_prior_span must precede its target")
                prior = windows[raw_best_prior]
                if prior["end_s"] > window["start_s"]:
                    _fail(f"recurrence_results[{result_index}].context[{target_index}].best_prior_span overlaps its target")
                best_prior = {"span_index": raw_best_prior, "start_s": prior["start_s"], "end_s": prior["end_s"]}
            rows.append({**base, "window": deepcopy(window),
                         "offset_from_change_s": window["start_s"] - change["time_s"],
                         # Availability is the real persisted target-window end,
                         # rather than a recomputed onset timestamp.
                         "available_at_s": window["end_s"], "best_prior_span": best_prior, **scores})
    return rows


def evaluate_local_structure(
    reference: dict,
    predictions: dict,
    legacy_sections: list[dict],
    recurrence_results: list[dict],
) -> dict:
    """Return layer-separated, descriptive local-structure diagnostics.

    No timing tolerance, one-to-one assignment, calibrated metric, hierarchy,
    or semantic identity prediction is introduced by this comparison.
    """
    reference = _record(reference, "reference")
    duration = _source_duration(reference)
    changes, predicted_transitions = _validate_predictions(predictions, duration)
    layers = _reference_layers(reference, duration)
    legacy_times = _legacy_boundary_times(legacy_sections, duration)

    evaluated_layers: list[dict] = []
    for layer in layers:
        spans = layer["spans"]
        boundary_times = [right["start_s"] for _, right in zip(spans, spans[1:])]
        boundary_rows = []
        for left, right in zip(spans, spans[1:]):
            known_identity = left["identity_id"] is not None and right["identity_id"] is not None
            same_identity = known_identity and left["identity_id"] == right["identity_id"]
            time_s = right["start_s"]
            boundary_rows.append({
                "time_s": time_s, "left_span_id": left["id"], "right_span_id": right["id"],
                "identity_relation": ("same_explicit_group" if same_identity else
                                      "different_named_groups" if known_identity else "unknown"),
                "variation_change": (left["variation"] != right["variation"]
                                     if same_identity and left["variation"] is not None and right["variation"] is not None
                                     else None),
                "transition_start": right["transition"] is True,
                "transition_end": left["transition"] is True,
                "nearest_change": _nearest(time_s, changes),
                "nearest_legacy": _nearest(time_s, [{"time_s": value} for value in legacy_times]),
            })
        explicit_transitions = [span for span in spans if span["transition"] is True]
        transition_rows = [{"span_id": span["id"], "label": span["label"],
                            "start_s": span["start_s"], "end_s": span["end_s"],
                            "certainty": span["certainty"],
                            "predicted_best_overlap": _best_overlap(span, predicted_transitions)}
                           for span in explicit_transitions]
        predicted_transition_rows = [
            {"id": transition["id"], "start_s": transition["start_s"], "end_s": transition["end_s"],
             "reference_best_overlap": _best_reference_overlap(transition, explicit_transitions)}
            for transition in predicted_transitions
        ]
        predicted_change_rows = [
            {"id": change["id"], "time_s": change["time_s"],
             "nearest_annotated_boundary": _nearest(change["time_s"], [{"time_s": value} for value in boundary_times])}
            for change in changes
        ]
        evaluated_layers.append({"id": layer["id"], "name": layer["name"],
                                 "boundary_rows": boundary_rows, "transition_rows": transition_rows,
                                 "predicted_change_rows": predicted_change_rows,
                                 "predicted_transition_rows": predicted_transition_rows})

    minutes = duration / 60
    return {
        "schema_version": 1,
        "kind": "songviz-local-structure-development-evaluation",
        "source": deepcopy(reference["source"]),
        "counts": {"changes": {"count": len(changes), "density_per_minute": len(changes) / minutes},
                   "transitions": {"count": len(predicted_transitions),
                                   "density_per_minute": len(predicted_transitions) / minutes}},
        "layers": evaluated_layers,
        "recurrence_context": _recurrence_context(recurrence_results, changes, duration),
        "limitations": [
            "Development diagnostics only; nearest events and best IoU are descriptive, not accuracy, recall, precision, or calibrated tolerance claims.",
            "Best interval overlaps are independent many-to-one comparisons; zero-overlap candidates remain null.",
            "Different explicitly named identity groups are unlabeled contrasts, not negatives; unnamed identity is unknown.",
            "Reference layers remain independent annotations and are not forced into partitions or a hierarchy.",
            "Recurrence values are persisted offline phrase diagnostics, not recomputed acoustic evidence or semantic identity assignments.",
            "Recurrence context is prior to its target window, but an alignment gap can place that prior phrase after the detected change; it is window relation evidence, not a causal onset account.",
            "Target phrase evidence becomes available at target-window end and may be unavailable near track end; unsupported scores are null rather than zero.",
        ],
    }
