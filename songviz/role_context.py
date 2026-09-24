"""Continuous, multiscale before/after acoustic context for every beat boundary.

This module deliberately describes every supported boundary instead of selecting
candidate events.  Its output is acoustic evidence only: it does not infer a
musical role, vocal function, physical transition timing, or salience.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from songviz.change_episodes import _peak_context
from songviz.local_structure import _integer, _validated_inputs


@dataclass(frozen=True)
class RoleContextConfig:
    """The symmetric half-window sizes, in complete beat intervals."""

    scales: tuple[int, ...] = (2, 4, 8)


def _validate_config(config: RoleContextConfig) -> None:
    if not isinstance(config, RoleContextConfig):
        raise ValueError("config must be a RoleContextConfig")
    if (not isinstance(config.scales, tuple) or not config.scales
            or any(not _integer(scale) or scale < 1 for scale in config.scales)
            or len(set(config.scales)) != len(config.scales)):
        raise ValueError("scales must be a nonempty tuple of distinct positive integers")


def _signed_difference(right: float | None, left: float | None) -> float | None:
    return None if right is None or left is None else float(right - left)


def _context_sample(
    features: dict[str, np.ndarray], energy: dict[str, np.ndarray], bt: np.ndarray, k: int, h: int,
) -> dict:
    """Build one supported sample while preserving episode descriptor semantics."""
    peak = _peak_context(features, energy, k, h)
    stems: dict[str, dict] = {}
    for name in sorted(peak["stems"]):
        source = peak["stems"][name]
        left, right = source["left"], source["right"]
        stems[name] = {
            "left": left,
            "right": right,
            "changes": {
                "signed_rms_difference": _signed_difference(right["mean_rms"], left["mean_rms"]),
                "signed_active_fraction_difference": _signed_difference(
                    right["active_fraction"], left["active_fraction"]),
                "signed_rms_power_share_difference": _signed_difference(
                    right["rms_power_share"], left["rms_power_share"]),
                "signed_spectral_concentration_difference": _signed_difference(
                    right["spectral_concentration"], left["spectral_concentration"]),
                "signed_adjacent_spectral_change_difference": _signed_difference(
                    right["adjacent_spectral_change"], left["adjacent_spectral_change"]),
            },
            "musical_role": None,
            "vocal_function": None,
            "perceived_importance": None,
        }
    return {
        "anchor_index": int(k),
        "anchor_s": float(bt[k]),
        "scale_beats": int(h),
        "support_start_s": float(bt[k - h]),
        "support_end_s": float(bt[k + h]),
        "available_at_s": float(bt[k + h]),
        "left_support": {"start_s": float(bt[k - h]), "end_s": float(bt[k])},
        "right_support": {"start_s": float(bt[k]), "end_s": float(bt[k + h])},
        "stems": stems,
        "physical_onset_s": None,
        "physical_settled_s": None,
    }


def compute_role_context(
    features: dict[str, np.ndarray], energy: dict[str, np.ndarray], beat_times: np.ndarray,
    *, config: RoleContextConfig | None = None,
) -> dict:
    """Return all-stem before/after descriptors at every beat boundary and scale.

    A sample is ``None`` when its symmetric h-beat windows cannot both fit in
    the beat grid.  Spectral descriptors can separately be ``None`` for a
    silent/inaudible side; that is unknown evidence, not zero spectral change.
    """
    config = RoleContextConfig() if config is None else config
    _validate_config(config)
    checked_features, checked_energy, bt, n = _validated_inputs(features, energy, beat_times)
    names = sorted(checked_features)
    floors = {name: max(float(np.max(checked_energy[name])) * .02, 1e-8) for name in names}

    curves: list[dict] = []
    for h in config.scales:
        samples: list[dict | None] = [None] * (n + 1)
        for k in range(h, n - h + 1):
            samples[k] = _context_sample(checked_features, checked_energy, bt, k, h)
        curves.append({"scale_beats": int(h), "samples": samples})

    return {
        "schema_version": 1,
        "kind": "songviz-role-context",
        "method": "all_stem_multiscale_symmetric_before_after_context_v1",
        "config": {"scales": list(config.scales)},
        "times_s": bt.tolist(),
        "stem_names": names,
        "audibility_floors": floors,
        "curves": curves,
        "limitations": [
            "Each curve sample is null when its complete symmetric h-beat left/right feature support does not fit the beat grid.",
            "Audibility floors are calibrated offline from each stem's whole-track maximum (max(2% of maximum RMS, 1e-8)); available_at_s records only right-side feature support, not streaming availability.",
            "A null spectral descriptor or signed spectral difference means one or both required sides are inaudible or lack usable adjacent spectral evidence, not no spectral change.",
            "RMS-power share is null when a side has zero summed stem RMS power; spectral concentration is null when its averaged spectrum has zero total mass.",
            "Side descriptors use unweighted complete-beat summaries, including on an irregular beat grid; they are not time-integrated energy estimates.",
            "Descriptors are acoustic proxies only; musical_role, vocal_function, perceived_importance, physical_onset_s, and physical_settled_s are intentionally unknown.",
            "This API evaluates every supported boundary and does not select episodes, apply thresholds, or infer a visual-cut policy.",
        ],
    }
