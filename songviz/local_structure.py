"""Short-scale, label-free local structural candidate extraction.

The output is deliberately evidence rather than a segmentation or identity model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class LocalStructureConfig:
    contrast_scales: tuple[int, ...] = (2, 4, 8)
    threshold_floor: float = 0.20
    mad_multiplier: float = 2.5
    peak_spacing_beats: int = 2
    dip_min_beats: int = 2
    dip_max_beats: int = 12
    dip_flank_beats: int = 4
    dip_ratio: float = 0.65


def _integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _validate_config(config: LocalStructureConfig) -> None:
    if not isinstance(config, LocalStructureConfig):
        raise ValueError("config must be a LocalStructureConfig")
    scales = config.contrast_scales
    if not isinstance(scales, tuple) or not scales or any(not _integer(h) or h < 1 for h in scales):
        raise ValueError("contrast_scales must be a nonempty tuple of positive integers")
    if len(set(scales)) != len(scales):
        raise ValueError("contrast_scales must not contain duplicates")
    for value in (config.peak_spacing_beats, config.dip_min_beats,
                  config.dip_max_beats, config.dip_flank_beats):
        if not _integer(value) or value < 1:
            raise ValueError("beat counts must be positive integers")
    if config.dip_min_beats > config.dip_max_beats:
        raise ValueError("dip_min_beats must not exceed dip_max_beats")
    for value in (config.threshold_floor, config.mad_multiplier, config.dip_ratio):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
            raise ValueError("threshold parameters must be finite numbers")
    if not 0 <= float(config.threshold_floor) <= 1 or float(config.mad_multiplier) < 0 or not 0 <= float(config.dip_ratio) <= 1:
        raise ValueError("threshold parameters are outside their valid range")


def _validated_inputs(
    features: dict[str, np.ndarray], energy: dict[str, np.ndarray], beat_times: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, int]:
    try:
        bt = np.asarray(beat_times, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("beat_times must be finite increasing nonnegative endpoints") from exc
    if (bt.ndim != 1 or bt.size < 2 or not np.isfinite(bt).all() or np.any(bt < 0)
            or np.any(np.diff(bt) <= 0)):
        raise ValueError("beat_times must be finite increasing nonnegative endpoints")
    if not isinstance(features, dict) or not features or not isinstance(energy, dict) or set(features) != set(energy):
        raise ValueError("matching nonempty feature and energy stems are required")
    n = bt.size - 1
    checked_features: dict[str, np.ndarray] = {}
    checked_energy: dict[str, np.ndarray] = {}
    for name in sorted(features):
        try:
            f = np.asarray(features[name], dtype=float)
            e = np.asarray(energy[name], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("stems require finite nonnegative features and RMS") from exc
        if (f.ndim != 2 or f.shape[0] == 0 or f.shape[1] != n or e.shape != (n,) or not np.isfinite(f).all()
                or not np.isfinite(e).all() or np.any(f < 0) or np.any(e < 0)):
            raise ValueError("stems require finite nonnegative features and RMS")
        checked_features[name] = f
        checked_energy[name] = e
    return checked_features, checked_energy, bt, n


def _top_two(values: list[float]) -> float | None:
    return float(np.mean(sorted(values, reverse=True)[:2])) if values else None


def _threshold(values: list[float], config: LocalStructureConfig) -> float | None:
    if not values:
        return None
    vector = np.asarray(values, dtype=float)
    median = float(np.median(vector))
    mad = float(np.median(np.abs(vector - median)))
    return float(min(1.0, max(config.threshold_floor, median + config.mad_multiplier * 1.4826 * mad)))


def _local_peaks(curve: list[float | None], threshold: float | None, spacing: int) -> list[int]:
    """Return leftmost members of same-height local-max plateaus."""
    if threshold is None:
        return []
    peaks = []
    for k, value in enumerate(curve):
        if value is None or value <= threshold:
            continue
        nearby = [v for v in curve[max(0, k - spacing):k + spacing + 1] if v is not None]
        if nearby and value == max(nearby):
            # Equal peaks select their leftmost occurrence in this neighbourhood.
            before = curve[max(0, k - spacing):k]
            if not any(v == value for v in before):
                peaks.append(k)
    return peaks


def detect_local_structure(
    features: dict[str, np.ndarray], energy: dict[str, np.ndarray], beat_times: np.ndarray,
    *, config: LocalStructureConfig | None = None,
) -> dict:
    """Propose bounded local changes and energy dip/recovery intervals.

    All calculations use complete beat intervals.  ``None`` records unavailable
    evidence rather than interpreting unsupported/silent regions as no change.
    """
    config = LocalStructureConfig() if config is None else config
    _validate_config(config)
    features, energy, bt, n = _validated_inputs(features, energy, beat_times)
    names = sorted(features)
    floors = {name: max(float(energy[name].max()) * .02, 1e-8) for name in names}
    scale_records: list[dict] = []
    raw_peaks: list[dict] = []

    for h in config.contrast_scales:
        pattern: list[float | None] = [None] * (n + 1)
        arrangement: list[float | None] = [None] * (n + 1)
        combined: list[float | None] = [None] * (n + 1)
        evidence_at: dict[int, dict[str, dict[str, float | None]]] = {}
        for k in range(h, n - h + 1):
            stem_evidence: dict[str, dict[str, float | None]] = {}
            pattern_values: list[float] = []
            arrangement_values: list[float] = []
            for name in names:
                left_energy = float(energy[name][k - h:k].mean())
                right_energy = float(energy[name][k:k + h].mean())
                left_active = left_energy > floors[name]
                right_active = right_energy > floors[name]
                if left_active and right_active:
                    left = features[name][:, k - h:k].mean(axis=1)
                    right = features[name][:, k:k + h].mean(axis=1)
                    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
                    pattern_value = (float(1 - np.clip(np.dot(left, right) / denominator, 0, 1))
                                     if denominator > 1e-12 else None)
                    arrangement_value = float(1 - min(left_energy, right_energy) / max(left_energy, right_energy))
                elif left_active or right_active:
                    pattern_value = None
                    arrangement_value = 1.0
                else:
                    pattern_value = None
                    arrangement_value = None
                if pattern_value is not None:
                    pattern_values.append(pattern_value)
                if arrangement_value is not None:
                    arrangement_values.append(arrangement_value)
                stem_evidence[name] = {"pattern_change": pattern_value,
                                       "arrangement_change": arrangement_value}
            pattern[k] = _top_two(pattern_values)
            arrangement[k] = _top_two(arrangement_values)
            known = [value for value in (pattern[k], arrangement[k]) if value is not None]
            combined[k] = float(max(known)) if known else None
            evidence_at[k] = stem_evidence
        threshold = _threshold([value for value in combined if value is not None], config)
        record = {"scale_beats": int(h), "pattern_change": pattern,
                  "arrangement_change": arrangement, "combined_change": combined,
                  "threshold": threshold}
        scale_records.append(record)
        for k in _local_peaks(combined, threshold, config.peak_spacing_beats):
            raw_peaks.append({"beat_index": k, "scale_beats": int(h),
                              "strength": float(combined[k]), "pattern_change": pattern[k],
                              "arrangement_change": arrangement[k], "evidence": evidence_at[k]})

    # Rank first so cross-scale ties have a stable preference for smaller scales,
    # then earlier beats.  Suppressed peaks remain support for the retained one.
    raw_peaks.sort(key=lambda item: (-item["strength"], item["scale_beats"], item["beat_index"]))
    selected: list[dict] = []
    for peak in raw_peaks:
        blockers = [winner for winner in selected
                    if abs(winner["beat_index"] - peak["beat_index"]) < config.peak_spacing_beats]
        if blockers:
            blockers[0]["supporting_scales"].add(peak["scale_beats"])
            blockers[0]["scale_support"].append(peak)
            continue
        peak["supporting_scales"] = {peak["scale_beats"]}
        peak["scale_support"] = [dict(peak)]
        selected.append(peak)
    changes = []
    for number, peak in enumerate(sorted(selected, key=lambda item: item["beat_index"]), 1):
        k, h = peak["beat_index"], peak["scale_beats"]
        support = [{"scale_beats": p['scale_beats'], "time_s": float(bt[p['beat_index']]),
                    "start_s": float(bt[p['beat_index'] - p['scale_beats']]),
                    "end_s": float(bt[p['beat_index'] + p['scale_beats']]),
                    "strength": p['strength']} for p in peak['scale_support']]
        support.sort(key=lambda p: (p['scale_beats'], p['time_s']))
        support_start = min(p['start_s'] for p in support)
        support_end = max(p['end_s'] for p in support)
        changes.append({
            "id": f"change-{number}", "time_s": float(bt[k]), "beat_index": k,
            "strength": peak["strength"], "pattern_change": peak["pattern_change"],
            "arrangement_change": peak["arrangement_change"],
            "supporting_scales_beats": sorted(peak["supporting_scales"]),
            "primary_scale_beats": h, "scale_support": support,
            "stem_evidence": peak["evidence"], "support_start_s": support_start,
            "support_end_s": support_end, "available_at_s": support_end,
            "kind": "local_change",
        })

    aggregate = np.sqrt(sum(np.square(energy[name]) for name in names))
    energy_floor = max(float(aggregate.max()) * .02, 1e-8)
    proposals: list[dict] = []
    flank = config.dip_flank_beats
    for a in range(flank, n):
        for duration in range(config.dip_min_beats, config.dip_max_beats + 1):
            b = a + duration
            if b + flank > n:
                continue
            left = aggregate[a - flank:a]
            right = aggregate[b:b + flank]
            inside = aggregate[a:b]
            left_level, right_level = float(np.median(left)), float(np.median(right))
            min_flank = min(left_level, right_level)
            if min_flank <= energy_floor:
                continue
            cutoff = config.dip_ratio * min_flank
            if (not np.all(inside < cutoff) or aggregate[a - 1] < cutoff or aggregate[b] < cutoff
                    or np.count_nonzero(left > cutoff) < 3 * flank / 4
                    or np.count_nonzero(right > cutoff) < 3 * flank / 4):
                continue
            inside_level = float(np.mean(inside))
            affected = [name for name in names
                        if float(np.mean(energy[name][a:b])) < config.dip_ratio * min(
                            float(np.mean(energy[name][a - flank:a])),
                            float(np.mean(energy[name][b:b + flank])))]
            proposals.append({"start_beat": a, "end_beat": b,
                              "strength": float(np.clip(1 - inside_level / min_flank, 0, 1)),
                              "left_level": left_level, "right_level": right_level,
                              "inside_level": inside_level, "affected_stems": affected})
    proposals.sort(key=lambda item: (-item["strength"], -(item["end_beat"] - item["start_beat"]),
                                    item["start_beat"], item["end_beat"]))
    accepted: list[dict] = []
    for proposal in proposals:
        if any(proposal["start_beat"] < other["end_beat"] and other["start_beat"] < proposal["end_beat"]
               for other in accepted):
            continue
        accepted.append(proposal)
    transitions = []
    for number, proposal in enumerate(sorted(accepted, key=lambda item: item["start_beat"]), 1):
        a, b = proposal["start_beat"], proposal["end_beat"]
        transitions.append({
            "id": f"transition-{number}", "time_s": float(bt[a]), "start_s": float(bt[a]),
            "end_s": float(bt[b]), "start_beat": a, "end_beat": b,
            "strength": proposal["strength"], "kind": "energy_dip_recovery",
            "evidence": {key: proposal[key] for key in ("left_level", "right_level", "inside_level", "affected_stems")},
            "support_start_s": float(bt[a - flank]), "support_end_s": float(bt[b + flank]),
            "available_at_s": float(bt[b + flank]),
        })
    return {
        "schema_version": 1, "kind": "songviz-local-structure-candidates", "method": "multiscale_stem_contrast_and_bounded_rms_dip_v1",
        "config": asdict(config), "times_s": bt.tolist(), "curves": scale_records,
        "changes": changes, "transitions": transitions,
        "limitations": "Offline evidence uses future right context and a whole-track audibility floor; beat/frame resolution limits timing. Candidate events are not sections or novelty/identity probabilities. The dip detector misses fills, ramps, and non-energy dips.",
    }
