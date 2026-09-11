"""Bounded local-structure detector variants for comparison with the frozen control.

This module deliberately leaves :mod:`songviz.local_structure` untouched.  The
``control`` variant is a byte-for-byte semantic passthrough to that detector;
the other variants retain its contrast curves and dip proposals as a fixed
comparison point while adding narrowly specified proposal policies.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from numbers import Real

import numpy as np

from songviz.local_structure import (
    LocalStructureConfig,
    _integer,
    _validate_config,
    _validated_inputs,
    detect_local_structure,
)


@dataclass(frozen=True)
class LocalStructureVariantConfig:
    """Fixed parameters for the bounded channel/activity comparison.

    ``base_detector_config`` is serialized with every non-control result so a
    variant package always states exactly which unchanged detector supplied its
    curves and energy-dip proposals.
    """

    pattern_threshold_floor: float = .20
    arrangement_threshold_floor: float = .20
    mad_multiplier: float = 2.5
    activity_scales: tuple[int, ...] = (4, 8)
    activity_min_change: float = .5
    activity_support_fraction: float = .75
    peak_spacing_beats: int = 2
    base_detector_config: LocalStructureConfig = field(default_factory=LocalStructureConfig)


_VALID_VARIANTS = {"control", "separate_channels", "sustained_activity", "combined"}


def _finite_number(value: object) -> bool:
    return (not isinstance(value, (bool, np.bool_)) and isinstance(value, (Real, np.integer, np.floating))
            and bool(np.isfinite(value)))


def _validate_variant_config(config: LocalStructureVariantConfig) -> None:
    if not isinstance(config, LocalStructureVariantConfig):
        raise ValueError("config must be a LocalStructureVariantConfig")
    _validate_config(config.base_detector_config)
    for value in (config.pattern_threshold_floor, config.arrangement_threshold_floor,
                  config.mad_multiplier, config.activity_min_change,
                  config.activity_support_fraction):
        if not _finite_number(value):
            raise ValueError("variant threshold parameters must be finite numbers")
    if not 0 <= float(config.pattern_threshold_floor) <= 1:
        raise ValueError("pattern_threshold_floor must be in [0, 1]")
    if not 0 <= float(config.arrangement_threshold_floor) <= 1:
        raise ValueError("arrangement_threshold_floor must be in [0, 1]")
    if float(config.mad_multiplier) < 0:
        raise ValueError("mad_multiplier must be nonnegative")
    if not 0 < float(config.activity_min_change) <= 1:
        raise ValueError("activity_min_change must be in (0, 1]")
    if not 0 < float(config.activity_support_fraction) <= 1:
        raise ValueError("activity_support_fraction must be in (0, 1]")
    scales = config.activity_scales
    if (not isinstance(scales, tuple) or not scales
            or any(not _integer(scale) or scale < 1 for scale in scales)
            or len(set(scales)) != len(scales)):
        raise ValueError("activity_scales must be a nonempty tuple of distinct positive integers")
    if not _integer(config.peak_spacing_beats) or config.peak_spacing_beats < 1:
        raise ValueError("peak_spacing_beats must be a positive integer")


def _channel_threshold(values: list[float], floor: float, multiplier: float) -> float | None:
    if not values:
        return None
    vector = np.asarray(values, dtype=float)
    median = float(np.median(vector))
    mad = float(np.median(np.abs(vector - median)))
    return float(min(1.0, max(floor, median + multiplier * 1.4826 * mad)))


def _strict_peaks(curve: list[float | None], threshold: float | None, spacing: int) -> list[int]:
    """Return leftmost local-max plateaus whose score is strictly above threshold."""
    if threshold is None:
        return []
    peaks: list[int] = []
    for index, value in enumerate(curve):
        if value is None or value <= threshold:
            continue
        nearby = [item for item in curve[max(0, index - spacing):index + spacing + 1]
                  if item is not None]
        if nearby and value == max(nearby):
            before = curve[max(0, index - spacing):index]
            if not any(item == value for item in before):
                peaks.append(index)
    return peaks


def _activity_peaks(curve: list[float | None], minimum: float, spacing: int) -> list[int]:
    """As ``_strict_peaks``, except an exactly-qualified activity change counts."""
    peaks: list[int] = []
    for index, value in enumerate(curve):
        if value is None or value < minimum:
            continue
        nearby = [item for item in curve[max(0, index - spacing):index + spacing + 1]
                  if item is not None]
        if nearby and value == max(nearby):
            before = curve[max(0, index - spacing):index]
            if not any(item == value for item in before):
                peaks.append(index)
    return peaks


def _normalized_exceedance(score: float, threshold: float) -> float:
    """Threshold-relative ranking measure, not a probability or calibrated score."""
    return float((score - threshold) / max(1.0 - threshold, 1e-12))


def _separate_channel_candidates(
    features: dict[str, np.ndarray], energy: dict[str, np.ndarray], bt: np.ndarray, n: int,
    config: LocalStructureVariantConfig,
) -> tuple[list[dict], list[dict]]:
    names = sorted(features)
    floors = {name: max(float(energy[name].max()) * .02, 1e-8) for name in names}
    records: list[dict] = []
    raw: list[dict] = []
    base_scales = config.base_detector_config.contrast_scales
    for h in base_scales:
        pattern: list[float | None] = [None] * (n + 1)
        arrangement: list[float | None] = [None] * (n + 1)
        evidence_at: dict[int, dict[str, dict[str, float | None]]] = {}
        for k in range(h, n - h + 1):
            pattern_values: list[float] = []
            arrangement_values: list[float] = []
            evidence: dict[str, dict[str, float | None]] = {}
            for name in names:
                left_energy = float(energy[name][k - h:k].mean())
                right_energy = float(energy[name][k:k + h].mean())
                left_active, right_active = left_energy > floors[name], right_energy > floors[name]
                if left_active and right_active:
                    left = features[name][:, k - h:k].mean(axis=1)
                    right = features[name][:, k:k + h].mean(axis=1)
                    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
                    pattern_value = (float(1 - np.clip(np.dot(left, right) / denominator, 0, 1))
                                     if denominator > 1e-12 else None)
                    arrangement_value = float(1 - min(left_energy, right_energy) / max(left_energy, right_energy))
                elif left_active or right_active:
                    pattern_value, arrangement_value = None, 1.0
                else:
                    pattern_value, arrangement_value = None, None
                evidence[name] = {"pattern_change": pattern_value, "arrangement_change": arrangement_value}
                if pattern_value is not None:
                    pattern_values.append(pattern_value)
                if arrangement_value is not None:
                    arrangement_values.append(arrangement_value)
            pattern[k] = float(np.mean(sorted(pattern_values, reverse=True)[:2])) if pattern_values else None
            arrangement[k] = float(np.mean(sorted(arrangement_values, reverse=True)[:2])) if arrangement_values else None
            evidence_at[k] = evidence
        thresholds = {
            "pattern": _channel_threshold([v for v in pattern if v is not None],
                                          float(config.pattern_threshold_floor), float(config.mad_multiplier)),
            "arrangement": _channel_threshold([v for v in arrangement if v is not None],
                                                float(config.arrangement_threshold_floor), float(config.mad_multiplier)),
        }
        records.append({"scale_beats": int(h), "pattern_change": pattern,
                        "arrangement_change": arrangement, "thresholds": thresholds,
                        "stem_evidence": [evidence_at.get(k) for k in range(n + 1)]})
        for channel, curve in (("pattern", pattern), ("arrangement", arrangement)):
            threshold = thresholds[channel]
            for k in _strict_peaks(curve, threshold, config.peak_spacing_beats):
                score = float(curve[k])
                raw.append({
                    "beat_index": k, "scale_beats": int(h), "channel": channel,
                    "strength": score, "normalized_exceedance": _normalized_exceedance(score, float(threshold)),
                    "threshold": float(threshold), "pattern_change": pattern[k],
                    "arrangement_change": arrangement[k], "evidence": evidence_at[k],
                })
    # Channel scores retain their raw contrast units.  This ordering uses only
    # excess above each channel's own threshold, normalized by its remaining
    # [threshold, 1] headroom; it is deliberately not a confidence probability.
    raw.sort(key=lambda item: (-item["normalized_exceedance"], -item["strength"],
                               item["scale_beats"], item["channel"], item["beat_index"]))
    selected: list[dict] = []
    for proposal in raw:
        blockers = [winner for winner in selected
                    if abs(winner["beat_index"] - proposal["beat_index"]) < config.peak_spacing_beats]
        if blockers:
            blockers[0]["support"].append(proposal)
        else:
            proposal["support"] = [dict(proposal)]
            selected.append(proposal)
    return selected, records


def _activity_candidates(
    energy: dict[str, np.ndarray], bt: np.ndarray, n: int, config: LocalStructureVariantConfig,
) -> tuple[list[dict], list[dict]]:
    names = sorted(energy)
    floors = {name: max(float(energy[name].max()) * .02, 1e-8) for name in names}
    records: list[dict] = []
    raw: list[dict] = []
    for h in config.activity_scales:
        curve: list[float | None] = [None] * (n + 1)
        evidence_series: dict[str, list[dict | None]] = {name: [None] * (n + 1) for name in names}
        aggregate_evidence: list[dict | None] = [None] * (n + 1)
        for k in range(h, n - h + 1):
            supported: list[dict] = []
            qualified: list[dict] = []
            for name in names:
                left, right = energy[name][k - h:k], energy[name][k:k + h]
                left_median, right_median = float(np.median(left)), float(np.median(right))
                left_audible, right_audible = left_median > floors[name], right_median > floors[name]
                audible = left_audible or right_audible
                if not audible:
                    item = {"supported": False, "floor": floors[name], "left_median": left_median,
                            "right_median": right_median, "direction": None, "audibility_state": "silent",
                            "relative_change": None,
                            "left_support_fraction": None, "right_support_fraction": None,
                            "score": None, "qualified": False}
                else:
                    midpoint = (left_median + right_median) / 2
                    direction = "increase" if right_median > left_median else "decrease" if left_median > right_median else "none"
                    audibility_state = ("entrance" if not left_audible and right_audible else
                                        "exit" if left_audible and not right_audible else "continuous")
                    if direction == "increase":
                        left_fraction = float(np.mean(left < midpoint))
                        right_fraction = float(np.mean(right > midpoint))
                    elif direction == "decrease":
                        left_fraction = float(np.mean(left > midpoint))
                        right_fraction = float(np.mean(right < midpoint))
                    else:
                        left_fraction = right_fraction = 0.0
                    relative = float(1 - min(left_median, right_median) / max(left_median, right_median))
                    qualifies = (relative >= config.activity_min_change
                                 and left_fraction >= config.activity_support_fraction
                                 and right_fraction >= config.activity_support_fraction)
                    item = {"supported": True, "floor": floors[name], "left_median": left_median,
                            "right_median": right_median, "direction": direction,
                            "audibility_state": audibility_state,
                            "relative_change": relative, "left_support_fraction": left_fraction,
                            "right_support_fraction": right_fraction,
                            "score": relative if qualifies else 0.0, "qualified": bool(qualifies)}
                    supported.append(item)
                    if qualifies:
                        qualified.append(item)
                evidence_series[name][k] = item
            if supported:
                curve[k] = max((item["score"] for item in qualified), default=0.0)
                # Stable primary evidence chooses score, then the lexical stem name.
                primary_name = next((name for name in names if evidence_series[name][k] in qualified
                                     and evidence_series[name][k]["score"] == curve[k]), None)
                aggregate_evidence[k] = {"supported": True, "primary_stem": primary_name,
                                         "stem_evidence": {name: evidence_series[name][k] for name in names}}
            else:
                aggregate_evidence[k] = {"supported": False, "primary_stem": None,
                                         "stem_evidence": {name: evidence_series[name][k] for name in names}}
        records.append({"scale_beats": int(h), "activity_change": curve,
                        "threshold": float(config.activity_min_change),
                        "support_fraction_required": float(config.activity_support_fraction),
                        "stem_evidence": evidence_series, "evidence": aggregate_evidence})
        for k in _activity_peaks(curve, float(config.activity_min_change), config.peak_spacing_beats):
            evidence = aggregate_evidence[k]
            primary = evidence["primary_stem"]
            raw.append({"beat_index": k, "scale_beats": int(h), "channel": "sustained_activity",
                        "strength": float(curve[k]), "threshold": float(config.activity_min_change),
                        "pattern_change": None, "arrangement_change": None,
                        "primary_stem": primary, "evidence": evidence["stem_evidence"]})
    raw.sort(key=lambda item: (-item["strength"], item["scale_beats"], item["primary_stem"], item["beat_index"]))
    selected: list[dict] = []
    for proposal in raw:
        blockers = [winner for winner in selected
                    if abs(winner["beat_index"] - proposal["beat_index"]) < config.peak_spacing_beats]
        if blockers:
            blockers[0]["support"].append(proposal)
        else:
            proposal["support"] = [dict(proposal)]
            selected.append(proposal)
    return selected, records


def _proposal_support(proposal: dict, bt: np.ndarray, source: str) -> list[dict]:
    support = []
    for item in proposal["support"]:
        k, h = item["beat_index"], item["scale_beats"]
        support.append({"source": source, "channel": item["channel"], "scale_beats": h,
                        "time_s": float(bt[k]), "start_s": float(bt[k - h]), "end_s": float(bt[k + h]),
                        "available_at_s": float(bt[k + h]), "strength": item["strength"],
                        "threshold": item["threshold"],
                        "normalized_exceedance": item.get("normalized_exceedance"),
                        "primary_stem": item.get("primary_stem"), "stem_evidence": item["evidence"]})
    return sorted(support, key=lambda item: (item["scale_beats"], item["channel"], item["time_s"]))


def _new_change(proposal: dict, bt: np.ndarray, source: str, number: int) -> dict:
    support = _proposal_support(proposal, bt, source)
    k = proposal["beat_index"]
    return {
        "id": f"{source}-change-{number}", "time_s": float(bt[k]), "beat_index": k,
        "strength": proposal["strength"], "pattern_change": proposal["pattern_change"],
        "arrangement_change": proposal["arrangement_change"], "primary_scale_beats": proposal["scale_beats"],
        "supporting_scales_beats": sorted({item["scale_beats"] for item in support}),
        "scale_support": [{key: item[key] for key in ("scale_beats", "time_s", "start_s", "end_s", "strength")}
                          for item in support],
        "stem_evidence": proposal["evidence"], "primary_stem": proposal.get("primary_stem"),
        "source": source, "channel": proposal["channel"],
        "threshold": proposal["threshold"], "normalized_exceedance": proposal.get("normalized_exceedance"),
        "variant_support": support, "support_start_s": min(item["start_s"] for item in support),
        "support_end_s": max(item["end_s"] for item in support),
        "available_at_s": max(item["available_at_s"] for item in support), "kind": "local_change",
    }


def _base_variant_support(change: dict) -> list[dict]:
    return [{"source": "control_contrast", "channel": "combined", "scale_beats": item["scale_beats"],
             "time_s": item["time_s"], "start_s": item["start_s"], "end_s": item["end_s"],
             "available_at_s": item["end_s"], "strength": item["strength"], "threshold": None,
             "normalized_exceedance": None, "primary_stem": None, "stem_evidence": None}
            for item in change["scale_support"]]


def _merge_changes(base_changes: list[dict], additions: list[dict], spacing: int) -> list[dict]:
    """Merge additions into base contrast candidates before their own NMS winners.

    A nearby base candidate remains the timestamp/primary record.  This makes a
    sustained-activity augmentation retain the frozen-control timestamp, and in
    ``combined`` makes an activity augmentation retain the separate-channel
    contrast timestamp.  Every contributing support window remains in
    ``variant_support`` and expands availability to the union.
    """
    selected = [deepcopy(change) for change in base_changes]
    additions = sorted(additions, key=lambda item: (-item["strength"], item["primary_scale_beats"],
                                                     item["channel"], item["beat_index"]))
    for addition in additions:
        blockers = [change for change in selected if abs(change["beat_index"] - addition["beat_index"]) < spacing]
        if not blockers:
            selected.append(addition)
            continue
        # Existing base/contrast timestamps are deliberately preferred.  For a
        # tie among existing candidates use time then id, independent of input order.
        target = min(blockers, key=lambda item: (item["time_s"], item["id"]))
        prior = target.get("variant_support", _base_variant_support(target))
        combined_support = prior + addition["variant_support"]
        target["variant_support"] = sorted(combined_support,
                                           key=lambda item: (item["time_s"], item["source"], item["channel"], item["scale_beats"]))
        target["supporting_scales_beats"] = sorted(set(target["supporting_scales_beats"])
                                                   | {item["scale_beats"] for item in addition["variant_support"]})
        target["support_start_s"] = min(item["start_s"] for item in target["variant_support"])
        target["support_end_s"] = max(item["end_s"] for item in target["variant_support"])
        target["available_at_s"] = max(item["available_at_s"] for item in target["variant_support"])
    return sorted(selected, key=lambda item: (item["beat_index"], item["time_s"], item["id"]))


def detect_local_structure_variant(
    features: dict[str, np.ndarray], energy: dict[str, np.ndarray], beat_times: np.ndarray,
    *, variant: str, config: LocalStructureVariantConfig | None = None,
) -> dict:
    """Run one fixed local-structure variant without using labels or recurrence.

    Separate-channel ranking is ``(raw_score - threshold) / (1 - threshold)``.
    It merely compares how far each raw contrast clears its own threshold; it is
    not a probability.  Activity requires sustained, directional per-beat
    support, so isolated spikes and alternating high/low windows are retained as
    zero-valued supported evidence rather than proposals.  A median-window step
    can produce a plateau whose leftmost selected boundary is up to one quarter
    of its window before the underlying step; ``variant_support`` retains the
    actual raw window/timestamp rather than hiding that timing bias.
    """
    if variant not in _VALID_VARIANTS:
        raise ValueError("variant must be one of control, separate_channels, sustained_activity, combined")
    config = LocalStructureVariantConfig() if config is None else config
    _validate_variant_config(config)
    # This call is intentionally first: all variants share exactly the frozen
    # contrast curves and bounded dip/recovery proposals from the same config.
    control = detect_local_structure(features, energy, beat_times, config=config.base_detector_config)
    if variant == "control":
        return control
    checked_features, checked_energy, bt, n = _validated_inputs(features, energy, beat_times)
    result = deepcopy(control)
    result["method"] = "local_structure_variant_separate_channels_and_sustained_activity_v1"
    result["config"] = asdict(config)
    result["variant"] = variant
    result["limitations"] = (
        "Offline evidence uses future right context and whole-track audibility floors. "
        "Channel exceedance is threshold-relative ranking, not a probability; activity evidence can miss ramps, "
        "short changes, and changes below the audibility floor. Median activity plateaus use leftmost peak selection "
        "and can precede a step by up to one quarter-window; raw support windows preserve that bias. Candidates are "
        "not sections, identity, novelty, or human-label predictions."
    )

    separate, channel_curves = _separate_channel_candidates(checked_features, checked_energy, bt, n, config)
    activity, activity_curves = _activity_candidates(checked_energy, bt, n, config)
    result["channel_curves"] = channel_curves
    result["activity_curves"] = activity_curves
    separate_changes = [_new_change(item, bt, "separate_channels", number)
                        for number, item in enumerate(sorted(separate, key=lambda item: item["beat_index"]), 1)]
    activity_changes = [_new_change(item, bt, "sustained_activity", number)
                        for number, item in enumerate(sorted(activity, key=lambda item: item["beat_index"]), 1)]
    if variant == "separate_channels":
        result["changes"] = separate_changes
    elif variant == "sustained_activity":
        result["changes"] = _merge_changes(result["changes"], activity_changes, config.peak_spacing_beats)
    else:
        contrast_and_activity = _merge_changes(separate_changes, activity_changes, config.peak_spacing_beats)
        result["changes"] = contrast_and_activity
    return result
