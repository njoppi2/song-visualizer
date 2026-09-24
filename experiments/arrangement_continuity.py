"""Fixed numerical arrangement/continuity comparison.

This module deliberately accepts only already-saved beat-domain features.  It has
no knowledge of song identity, excerpts, labels, or audio files.
"""

from __future__ import annotations

from typing import Any

import numpy as np


SCALES = (4, 8)
AUDIBILITY_TRACK_FRACTION = 0.02
AUDIBILITY_MINIMUM = 1e-8
LEVEL_RELATIVE_CHANGE = 0.5
LEVEL_SIDE_FRACTION = 0.75
UNCHANGED_RELATIVE_CHANGE = 0.25
UNCHANGED_AUDIBLE_FRACTION = 0.75
UNCHANGED_COSINE = 0.90


def _numeric_array(value: object, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must have a real numeric dtype")
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must not have boolean dtype")
    array = array.astype(float, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    # The fixed comparison reverses saved log1p CQTs.  A finite log value that
    # overflows on expm1 cannot provide a finite cosine and is invalid input.
    if ndim == 2 and np.any(array > np.log(np.finfo(float).max)):
        raise ValueError(f"{name} must remain finite after expm1")
    return array


def _validate_inputs(
    features: dict[str, Any], energy: dict[str, Any], beat_times: Any
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, int]:
    if not isinstance(features, dict) or not isinstance(energy, dict) or not features:
        raise ValueError("features and energy must be non-empty dictionaries")
    if set(features) != set(energy):
        raise ValueError("features and energy must have identical stem names")
    if any(not isinstance(stem, str) or not stem for stem in features):
        raise ValueError("stem names must be non-empty strings")

    beats = _numeric_array(beat_times, "beat_times", 1)
    if len(beats) < 2:
        raise ValueError("beat_times must contain at least two boundaries")
    if np.any(np.diff(beats) <= 0):
        raise ValueError("beat_times must be strictly increasing")

    checked_features: dict[str, np.ndarray] = {}
    checked_energy: dict[str, np.ndarray] = {}
    expected_beats = len(beats) - 1
    for stem in features:
        cqt = _numeric_array(features[stem], f"features[{stem!r}]", 2)
        rms = _numeric_array(energy[stem], f"energy[{stem!r}]", 1)
        if cqt.shape[0] == 0:
            raise ValueError(f"features[{stem!r}] must contain at least one frequency bin")
        if cqt.shape[1] != expected_beats or rms.shape[0] != expected_beats:
            raise ValueError(
                f"features[{stem!r}] and energy[{stem!r}] must each have "
                "one value per beat interval"
            )
        checked_features[stem] = cqt
        checked_energy[stem] = rms
    return checked_features, checked_energy, beats, expected_beats


def _relative_difference(left: float, right: float) -> float | None:
    largest = max(left, right)
    return None if largest == 0 else 1.0 - min(left, right) / largest


def _cosine(left: np.ndarray, right: np.ndarray) -> tuple[float | None, str | None]:
    left_raw = np.expm1(left).reshape(-1)
    right_raw = np.expm1(right).reshape(-1)
    left_norm = float(np.linalg.norm(left_raw))
    right_norm = float(np.linalg.norm(right_raw))
    if left_norm <= 0 or right_norm <= 0:
        return None, "nonpositive_norm"
    cosine = float(np.dot(left_raw, right_raw) / (left_norm * right_norm))
    if not np.isfinite(cosine):
        return None, "nonfinite_cosine"
    return float(np.clip(cosine, 0.0, 1.0)), None


def _stem_record(
    stem: str, cqt: np.ndarray, rms: np.ndarray, start: int, anchor: int, end: int
) -> dict[str, Any]:
    left = rms[start:anchor]
    right = rms[anchor:end]
    median_left = float(np.median(left))
    median_right = float(np.median(right))
    floor = float(max(AUDIBILITY_TRACK_FRACTION * float(np.max(rms)), AUDIBILITY_MINIMUM))
    relative = _relative_difference(median_left, median_right)
    left_above = float(np.mean(left > floor))
    right_above = float(np.mean(right > floor))
    midpoint = (median_left + median_right) / 2.0
    if median_right > median_left:
        direction = "increase"
        left_side = float(np.mean(left < midpoint))
        right_side = float(np.mean(right > midpoint))
    elif median_right < median_left:
        direction = "decrease"
        left_side = float(np.mean(left > midpoint))
        right_side = float(np.mean(right < midpoint))
    else:
        direction = None
        left_side = right_side = 0.0

    level_reasons: list[str] = []
    if not (median_left > floor or median_right > floor):
        level_reasons.append("below_audibility_floor")
    if relative is None or relative < LEVEL_RELATIVE_CHANGE:
        level_reasons.append("relative_change_below_threshold")
    if left_side < LEVEL_SIDE_FRACTION:
        level_reasons.append("left_midpoint_fraction_below_threshold")
    if right_side < LEVEL_SIDE_FRACTION:
        level_reasons.append("right_midpoint_fraction_below_threshold")
    level_change = not level_reasons

    cosine, cosine_reason = _cosine(cqt[:, start:anchor], cqt[:, anchor:end])
    unchanged_reasons: list[str] = []
    if left_above < UNCHANGED_AUDIBLE_FRACTION:
        unchanged_reasons.append("left_audible_fraction_below_threshold")
    if right_above < UNCHANGED_AUDIBLE_FRACTION:
        unchanged_reasons.append("right_audible_fraction_below_threshold")
    if relative is None or relative > UNCHANGED_RELATIVE_CHANGE:
        unchanged_reasons.append("median_relative_change_above_threshold")
    if cosine_reason:
        unchanged_reasons.append(cosine_reason)
    elif cosine is None or cosine < UNCHANGED_COSINE:
        unchanged_reasons.append("cosine_below_threshold")
    if level_change:
        unchanged_reasons.append("qualifying_changed_layer")

    return {
        "stem": stem,
        "floor": floor,
        "median_left": median_left,
        "median_right": median_right,
        "relative_change": relative,
        "left_above_floor_fraction": left_above,
        "right_above_floor_fraction": right_above,
        "left_midpoint_fraction": left_side,
        "right_midpoint_fraction": right_side,
        "crosses_floor": bool((median_left <= floor < median_right) or (median_right <= floor < median_left)),
        "direction": direction if level_change else None,
        "level_change": level_change,
        "level_change_reasons": level_reasons,
        "cosine_similarity": cosine,
        "cosine_reason": cosine_reason,
        "unchanged": not unchanged_reasons,
        "unchanged_reasons": unchanged_reasons,
    }


def _sample(
    scale: int, anchor: int, features: dict[str, np.ndarray], energy: dict[str, np.ndarray], beats: np.ndarray
) -> dict[str, Any] | None:
    n_beats = len(beats) - 1
    start, end = anchor - scale, anchor + scale
    if start < 0 or end > n_beats:
        return None
    records = [_stem_record(stem, features[stem], energy[stem], start, anchor, end) for stem in features]
    changed = [{"stem": r["stem"], "direction": r["direction"]} for r in records if r["level_change"]]
    unchanged = [r["stem"] for r in records if r["unchanged"]]
    has_role = any(stem in {"bass", "other"} for stem in unchanged)
    continuity = len(unchanged) >= 2 and has_role
    continuity_reasons: list[str] = []
    if len(unchanged) < 2:
        continuity_reasons.append("fewer_than_two_unchanged_stems")
    if not has_role:
        continuity_reasons.append("missing_bass_or_other_unchanged_stem")
    return {
        "scale_beats": scale,
        "anchor_index": anchor,
        "anchor_s": float(beats[anchor]),
        "support_start_s": float(beats[start]),
        "support_end_s": float(beats[end]),
        "available_at_s": float(beats[end]),
        "per_stem": records,
        "changed_layers": changed,
        "unchanged_layers": unchanged,
        "continuity_support": continuity,
        "continuity_reasons": continuity_reasons,
    }


def _common_changes(samples: list[dict[str, Any]]) -> list[dict[str, str]]:
    common = {(item["stem"], item["direction"]) for item in samples[0]["changed_layers"]}
    for sample in samples[1:]:
        common &= {(item["stem"], item["direction"]) for item in sample["changed_layers"]}
    return [{"stem": stem, "direction": direction} for stem, direction in sorted(common)]


def compute_arrangement_continuity(
    features: dict[str, Any], energy: dict[str, Any], beat_times: Any
) -> dict[str, Any]:
    """Compute the frozen two-arm arrangement comparison on beat-domain arrays.

    ``features[stem]`` is a finite, nonnegative saved log1p-CQT array of shape
    ``(frequency_bins, n_beats)``; ``energy[stem]`` is a matching RMS vector;
    ``beat_times`` has ``n_beats + 1`` strictly increasing boundaries.  Unsupported
    edge anchors are represented by ``None`` so they cannot be read as negatives.
    """
    checked_features, checked_energy, beats, n_beats = _validate_inputs(features, energy, beat_times)
    scale_samples = {
        scale: [_sample(scale, anchor, checked_features, checked_energy, beats) for anchor in range(n_beats + 1)]
        for scale in SCALES
    }
    anchors: list[dict[str, Any] | None] = []
    for anchor in range(n_beats + 1):
        if anchor + 1 > n_beats:
            anchors.append(None)
            continue
        comparisons = [scale_samples[scale][index] for index in (anchor, anchor + 1) for scale in SCALES]
        if any(sample is None for sample in comparisons):
            anchors.append(None)
            continue
        usable = [sample for sample in comparisons if sample is not None]
        changes = _common_changes(usable)
        level_only = bool(changes)
        continuity = all(sample["continuity_support"] for sample in usable)
        anchor_samples = {str(scale): scale_samples[scale][anchor] for scale in SCALES}
        next_samples = {str(scale): scale_samples[scale][anchor + 1] for scale in SCALES}
        anchors.append({
            "anchor_index": anchor,
            "anchor_s": float(beats[anchor]),
            "support_union_start_s": min(sample["support_start_s"] for sample in usable),
            "support_union_end_s": max(sample["support_end_s"] for sample in usable),
            "available_at_s": max(sample["available_at_s"] for sample in usable),
            "samples": {"anchor": anchor_samples, "next_anchor": next_samples},
            "changed_layers": changes,
            "level_change": level_only,
            "continuity_support": continuity,
            "arrangement_candidate": bool(level_only and continuity),
        })
    return {
        "config": {
            "scales_beats": list(SCALES),
            "audibility_floor": "max(0.02 * track_max_rms, 1e-8)",
            "level_relative_change_minimum": LEVEL_RELATIVE_CHANGE,
            "level_midpoint_fraction_minimum": LEVEL_SIDE_FRACTION,
            "unchanged_audible_fraction_minimum": UNCHANGED_AUDIBLE_FRACTION,
            "unchanged_relative_change_maximum": UNCHANGED_RELATIVE_CHANGE,
            "unchanged_cosine_minimum": UNCHANGED_COSINE,
            "unchanged_stems_minimum": 2,
            "required_unchanged_roles": ["bass", "other"],
            "persistence": "same changed stem and direction at both scales at anchor and next adjacent anchor",
        },
        "scales": [{"scale_beats": scale, "samples": scale_samples[scale]} for scale in SCALES],
        "anchors": anchors,
    }
