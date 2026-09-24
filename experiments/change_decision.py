"""Fixed local level and distribution-contrast decision screen."""
from __future__ import annotations

import numpy as np

from songviz.local_structure import _validated_inputs
from songviz.local_structure_variants import detect_local_structure_variant


_SCALES = (4, 8)
_THRESHOLD = .20


def _unit_vectors(saved_log: np.ndarray, rms: np.ndarray) -> list[np.ndarray | None]:
    """Reverse log1p CQT per beat without overflowing before L2 normalization."""
    floor = max(.02 * float(rms.max()), 1e-8)
    vectors: list[np.ndarray | None] = []
    for index in range(rms.size):
        column = saved_log[:, index]
        if rms[index] <= floor:
            vectors.append(None)
            continue
        # expm1(column) / ||expm1(column)||, with exp(max(column)) cancelled.
        maximum = float(column.max())
        scaled = np.exp(column - maximum) * (-np.expm1(-column))
        norm = float(np.linalg.norm(scaled))
        vectors.append(scaled / norm if np.isfinite(norm) and norm > 0 else None)
    return vectors


def _score(vectors: list[np.ndarray | None], k: int, h: int) -> dict:
    left = [item for item in vectors[k - h:k] if item is not None]
    right = [item for item in vectors[k:k + h] if item is not None]
    base = {"usable_left": len(left), "usable_right": len(right)}
    if len(left) < 2 or len(right) < 2 or len(left) < .75 * h or len(right) < .75 * h:
        return base | {"score": None, "within_left": None, "within_right": None, "cross": None,
                       "pattern_qualified": None}
    def mean_pairs(items: list[np.ndarray]) -> float:
        return float(np.mean([np.dot(items[i], items[j]) for i in range(len(items)) for j in range(i)]))
    within_left, within_right = mean_pairs(left), mean_pairs(right)
    cross = float(np.mean([np.dot(a, b) for a in left for b in right]))
    score = float((within_left + within_right) / 2 - cross)
    return base | {"score": score, "within_left": within_left, "within_right": within_right,
                   "cross": cross, "pattern_qualified": bool(score >= _THRESHOLD)}


def compute_change_decision(features: dict[str, np.ndarray], energy: dict[str, np.ndarray], beat_times: np.ndarray) -> dict:
    """Return fixed, offline acoustic evidence; semantic fields intentionally remain null."""
    checked_features, checked_energy, bt, n = _validated_inputs(features, energy, beat_times)
    activity = detect_local_structure_variant(checked_features, checked_energy, bt, variant="sustained_activity")
    levels = {record["scale_beats"]: record["stem_evidence"] for record in activity["activity_curves"]}
    vectors = {stem: _unit_vectors(checked_features[stem], checked_energy[stem]) for stem in sorted(checked_features)}
    scale_rows: dict[int, list[dict | None]] = {}
    for h in _SCALES:
        rows: list[dict | None] = []
        for k in range(n + 1):
            if k < h or k + h > n:
                rows.append(None)
                continue
            per_stem = {}
            for stem in vectors:
                per_stem[stem] = _score(vectors[stem], k, h) | {"level": levels[h][stem][k]}
            rows.append({"anchor_index": k, "anchor_s": float(bt[k]), "support_start_s": float(bt[k - h]),
                         "support_end_s": float(bt[k + h]), "per_stem": per_stem})
        scale_rows[h] = rows
    anchors: list[dict | None] = []
    for k in range(n + 1):
        if k < 8 or k > n - 9:
            anchors.append(None)
            continue
        level_changes, pattern_changes, unknown = [], [], []
        for stem in vectors:
            comparisons = [scale_rows[h][j]["per_stem"][stem] for h in _SCALES for j in (k, k + 1)]
            qualifications = [item["pattern_qualified"] for item in comparisons]
            if any(value is None for value in qualifications):
                unknown.append(stem)
            elif all(qualifications):
                pattern_changes.append(stem)
            level_records = [item["level"] for item in comparisons]
            directions = [item["direction"] for item in level_records]
            if all(item["qualified"] for item in level_records) and len(set(directions)) == 1:
                level_changes.append({"stem": stem, "direction": directions[0]})
        pattern_signal = True if pattern_changes else (None if unknown else False)
        if level_changes and pattern_signal is True:
            classification = "both"
        elif level_changes:
            classification = "level_change"
        elif pattern_signal is True:
            classification = "pattern_shift"
        elif pattern_signal is None:
            classification = "insufficient_pattern_evidence"
        else:
            classification = "no_supported_change"
        anchors.append({"anchor_index": k, "anchor_s": float(bt[k]), "support_start_s": float(bt[k - 8]),
                        "support_end_s": float(bt[k + 9]), "available_at_s": float(bt[k + 9]),
                        "level_changes": level_changes, "pattern_changes": pattern_changes,
                        "pattern_unknown_stems": unknown, "pattern_signal": pattern_signal,
                        "classification": classification, "musical_continuity": None, "section_identity": None,
                        "vocal_behavior": None, "importance": None})
    return {"config": {"pattern_threshold": _THRESHOLD, "scales_beats": list(_SCALES),
                       "usable_fraction": .75, "rms_floor_fraction": .02},
            "beat_times_s": [float(value) for value in bt],
            "scales": [{"scale_beats": h, "samples": scale_rows[h]} for h in _SCALES], "anchors": anchors}
