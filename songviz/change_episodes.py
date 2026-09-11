"""Multiscale, per-stem local-change episodes.

This is evidence for a later visual-direction experiment, not a segmentation,
transition-duration estimator, or perceptual-importance model.  It deliberately
keeps every qualifying stem/scale/channel response rather than applying NMS.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from songviz.local_structure import LocalStructureConfig, _integer, _validated_inputs
from songviz.local_structure_variants import (
    LocalStructureVariantConfig,
    _separate_channel_candidates,
)


@dataclass(frozen=True)
class ChangeEpisodeConfig:
    """Frozen parameters for per-channel response episodes."""

    scales: tuple[int, ...] = (2, 4, 8)
    threshold_floor: float = .20
    mad_multiplier: float = 2.5
    release_ratio: float = .5


def _finite_number(value: object) -> bool:
    return (not isinstance(value, (bool, np.bool_))
            and isinstance(value, (Real, np.integer, np.floating))
            and bool(np.isfinite(value)))


def _validate_config(config: ChangeEpisodeConfig) -> None:
    if not isinstance(config, ChangeEpisodeConfig):
        raise ValueError("config must be a ChangeEpisodeConfig")
    if (not isinstance(config.scales, tuple) or not config.scales
            or any(not _integer(scale) or scale < 1 for scale in config.scales)
            or len(set(config.scales)) != len(config.scales)):
        raise ValueError("scales must be a nonempty tuple of distinct positive integers")
    for value in (config.threshold_floor, config.mad_multiplier, config.release_ratio):
        if not _finite_number(value):
            raise ValueError("threshold parameters must be finite numbers")
    if not 0 <= float(config.threshold_floor) <= 1:
        raise ValueError("threshold_floor must be in [0, 1]")
    if float(config.mad_multiplier) < 0:
        raise ValueError("mad_multiplier must be nonnegative")
    if not 0 <= float(config.release_ratio) <= 1:
        raise ValueError("release_ratio must be in [0, 1]")


def _threshold(values: list[float], config: ChangeEpisodeConfig) -> float | None:
    if not values:
        return None
    vector = np.asarray(values, dtype=float)
    median = float(np.median(vector))
    mad = float(np.median(np.abs(vector - median)))
    return float(min(1., max(float(config.threshold_floor),
                              median + float(config.mad_multiplier) * 1.4826 * mad)))


def _episodes(values: list[float | None], high: float | None, release_ratio: float) -> list[dict]:
    """Return maximal above-release runs which contain a strict high crossing."""
    if high is None:
        return []
    low = float(release_ratio * high)
    found: list[dict] = []
    index = 0
    while index < len(values):
        value = values[index]
        if value is None or value <= low:
            index += 1
            continue
        first = index
        while index < len(values) and values[index] is not None and values[index] > low:
            index += 1
        last = index - 1
        run = [float(item) for item in values[first:last + 1] if item is not None]
        peak_value = max(run)
        if peak_value > high:
            peak = next(k for k in range(first, last + 1) if values[k] == peak_value)
            left_reason = "array_boundary" if first == 0 else (
                "missing_curve_value" if values[first - 1] is None else None)
            right_reason = "array_boundary" if last == len(values) - 1 else (
                "missing_curve_value" if values[last + 1] is None else None)
            found.append({
                "first": first,
                "last": last,
                "peak": peak,
                "peak_value": float(peak_value),
                "left_censored": left_reason is not None,
                "right_censored": right_reason is not None,
                "left_censor_reason": left_reason,
                "right_censor_reason": right_reason,
            })
    return found


def _raw_rms_directions(energy: np.ndarray, h: int, n: int) -> list[dict | None]:
    """Window-side RMS summaries; unlike contrast they retain signed level direction."""
    directions: list[dict | None] = [None] * (n + 1)
    for k in range(h, n - h + 1):
        left = float(np.mean(energy[k - h:k]))
        right = float(np.mean(energy[k:k + h]))
        directions[k] = {
            "left_mean_rms": left,
            "right_mean_rms": right,
            "signed_rms_difference": float(right - left),
        }
    return directions


def _spectral_concentration(window: np.ndarray) -> float | None:
    mean = np.mean(window, axis=1)
    total = float(np.sum(mean))
    if total <= 0:
        return None
    if mean.size == 1:
        return 1.0
    probabilities = mean / total
    entropy = -float(np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])))
    return float(np.clip(1. - entropy / np.log(mean.size), 0., 1.))


def _adjacent_spectral_change(features: np.ndarray, energy: np.ndarray, floor: float) -> float | None:
    values: list[float] = []
    for index in range(features.shape[1] - 1):
        if energy[index] <= floor or energy[index + 1] <= floor:
            continue
        left, right = features[:, index], features[:, index + 1]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator > 1e-12:
            values.append(float(1. - np.clip(np.dot(left, right) / denominator, 0., 1.)))
    return float(np.mean(values)) if values else None


def _side_context(
    features: np.ndarray, energy: np.ndarray, floor: float, denominator: float,
) -> dict:
    power = float(np.sum(np.square(energy)))
    audible = bool(np.any(energy > floor))
    return {
        "mean_rms": float(np.mean(energy)),
        "active_fraction": float(np.mean(energy > floor)),
        "rms_power_share": float(power / denominator) if denominator > 0 else None,
        "spectral_concentration": _spectral_concentration(features) if audible else None,
        "adjacent_spectral_change": _adjacent_spectral_change(features, energy, floor),
    }


def _peak_context(
    features: dict[str, np.ndarray], energy: dict[str, np.ndarray], peak: int, h: int,
) -> dict:
    names = sorted(features)
    left_total = float(sum(np.sum(np.square(energy[name][peak - h:peak])) for name in names))
    right_total = float(sum(np.sum(np.square(energy[name][peak:peak + h])) for name in names))
    stems: dict[str, dict] = {}
    for name in names:
        floor = max(float(np.max(energy[name])) * .02, 1e-8)
        left_energy, right_energy = energy[name][peak - h:peak], energy[name][peak:peak + h]
        left_features, right_features = features[name][:, peak - h:peak], features[name][:, peak:peak + h]
        left = _side_context(left_features, left_energy, floor, left_total)
        right = _side_context(right_features, right_energy, floor, right_total)
        stems[name] = {
            "left": left,
            "right": right,
            "changes": {"signed_rms_difference": float(right["mean_rms"] - left["mean_rms"] )},
        }
    return {"scale_beats": int(h), "stems": stems}


def detect_change_episodes(
    features: dict[str, np.ndarray], energy: dict[str, np.ndarray], beat_times: np.ndarray,
    *, config: ChangeEpisodeConfig | None = None,
) -> dict:
    """Extract independent hysteresis response episodes from frozen contrast evidence.

    The response extent is a run in a windowed contrast curve.  It is not a
    certified physical onset/settling interval or a judgment of importance.
    """
    config = ChangeEpisodeConfig() if config is None else config
    _validate_config(config)
    checked_features, checked_energy, bt, n = _validated_inputs(features, energy, beat_times)
    contrast_config = LocalStructureVariantConfig(
        pattern_threshold_floor=float(config.threshold_floor),
        arrangement_threshold_floor=float(config.threshold_floor),
        mad_multiplier=float(config.mad_multiplier),
        base_detector_config=LocalStructureConfig(contrast_scales=config.scales),
    )
    _, contrast_records = _separate_channel_candidates(
        checked_features, checked_energy, bt, n, contrast_config,
    )

    curves: list[dict] = []
    raw_episodes: list[dict] = []
    for source in contrast_records:
        h = int(source["scale_beats"])
        for stem in sorted(checked_features):
            directions = _raw_rms_directions(checked_energy[stem], h, n)
            for channel in ("pattern", "arrangement"):
                evidence_key = f"{channel}_change"
                values = [None if evidence is None else evidence[stem][evidence_key]
                          for evidence in source["stem_evidence"]]
                high = _threshold([float(value) for value in values if value is not None], config)
                low = None if high is None else float(high * config.release_ratio)
                curves.append({
                    "stem": stem,
                    "scale_beats": h,
                    "channel": channel,
                    "values": values,
                    "high_threshold": high,
                    "low_threshold": low,
                    "raw_rms_directions": directions,
                })
                for response in _episodes(values, high, float(config.release_ratio)):
                    raw_episodes.append({
                        "stem": stem, "scale_beats": h, "channel": channel,
                        "values": values, "high_threshold": high, "low_threshold": low,
                        **response,
                    })

    raw_episodes.sort(key=lambda item: (item["stem"], item["scale_beats"], item["channel"], item["first"]))
    episodes: list[dict] = []
    for number, episode in enumerate(raw_episodes, 1):
        first, last, peak, h = episode["first"], episode["last"], episode["peak"], episode["scale_beats"]
        # A known below-release neighbour certifies that the run has ended, but
        # computing that neighbour needs its own h-beat context.  Null/boundary
        # censoring is intentionally not treated as evidence beyond the run.
        response_support_start, response_support_end = first - h, last + h
        support_start = response_support_start - (1 if episode["left_censor_reason"] is None else 0)
        support_end = response_support_end + (1 if episode["right_censor_reason"] is None else 0)
        episodes.append({
            "id": f"episode-{number:04d}",
            "stem": episode["stem"],
            "scale_beats": h,
            "channel": episode["channel"],
            "raw_indices": list(range(first, last + 1)),
            "first_above_low_index": first,
            "last_above_low_index": last,
            "peak_index": peak,
            "peak_value": episode["peak_value"],
            "high_threshold": episode["high_threshold"],
            "low_threshold": episode["low_threshold"],
            "start_s": float(bt[first]),
            "end_s": float(bt[last + 1]),
            "peak_s": float(bt[peak]),
            "response_support_start_s": float(bt[response_support_start]),
            "response_support_end_s": float(bt[response_support_end]),
            "support_start_s": float(bt[support_start]),
            "support_end_s": float(bt[support_end]),
            "available_at_s": float(bt[support_end]),
            "left_censored": episode["left_censored"],
            "right_censored": episode["right_censored"],
            "left_censor_reason": episode["left_censor_reason"],
            "right_censor_reason": episode["right_censor_reason"],
            "contributing_stems": [episode["stem"]],
            "peak_context": _peak_context(checked_features, checked_energy, peak, h),
            "physical_onset_s": None,
            "physical_settled_s": None,
            "perceived_importance": None,
            "vocal_function": None,
        })
    return {
        "schema_version": 1,
        "kind": "songviz-change-episodes",
        "method": "per_stem_multiscale_windowed_contrast_hysteresis_v1",
        "config": {
            "scales": list(config.scales),
            "threshold_floor": float(config.threshold_floor),
            "mad_multiplier": float(config.mad_multiplier),
            "release_ratio": float(config.release_ratio),
        },
        "times_s": bt.tolist(),
        "curves": curves,
        "episodes": episodes,
        "limitations": (
            "Episodes are independent per-stem, per-scale, and per-channel windowed-contrast responses; "
            "they are not sections, a cross-scale hierarchy, or a visual-cut policy. Thresholds use whole-track "
            "statistics and each value needs future right-window context, so availability is offline. Response "
            "extent is not certified physical transition duration: physical_onset_s and physical_settled_s are "
            "unknown. Window support records only feature-comparison support and does not measure preparation or "
            "recovery. Acoustic descriptors are proxies, not mix-energy, importance, speech, laughter, melody, "
            "or vocal-function assignments; quiet pattern evidence remains unknown."
        ),
    }
