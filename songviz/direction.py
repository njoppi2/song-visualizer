"""Experimental evidence-driven direction plans, independent of frame rendering.

Rules describe changes in measured activity, not certified musical sections or
surprise. A plan is reproducible and inspectable; its reasons are not evidence of
artistic quality. No model service is called here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np

LAYERS = ("pulse", "kick", "snare", "hh", "bass", "vocals", "other")
TREATMENTS = {"ring", "ribbon", "ticks", "rails"}


def content_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_signals(signals: dict) -> None:
    def ordered(values):
        return isinstance(values, list) and all(_number(v) and v >= 0 for v in values) and all(a < b for a, b in zip(values, values[1:]))
    if not ordered(signals.get("beat_times_s")):
        raise ValueError("Invalid beat times")
    hits = signals.get("hits")
    if not isinstance(hits, list):
        raise ValueError("Missing hits")
    for h in hits:
        if not isinstance(h, dict) or h.get("component") not in {"kick", "snare", "hh"} or not _number(h.get("t")) or h["t"] < 0 or not _number(h.get("velocity")) or not 0 <= h["velocity"] <= 1:
            raise ValueError("Invalid component hit")
    energy = signals.get("energy", {})
    times = energy.get("times_s")
    if not ordered(times) or not times:
        raise ValueError("Invalid energy times")
    for name in ("bass", "vocals", "other"):
        values = energy.get(name)
        if not isinstance(values, list) or len(values) != len(times) or not all(_number(v) and 0 <= v <= 1 for v in values):
            raise ValueError(f"Invalid {name} energy")


def validate_plan(plan: dict, signals: dict) -> None:
    """Reject malformed, incompatible, or unsupported plans before any render."""
    validate_signals(signals)
    start, end = plan.get("start_s"), plan.get("end_s")
    if plan.get("schema_version") != 1 or not _number(start) or not _number(end) or not 0 <= start < end:
        raise ValueError("Invalid plan version or interval")
    times = signals["energy"]["times_s"]
    if start < times[0] or end > times[-1]:
        raise ValueError("Plan outside available signal coverage")
    if plan.get("signals_sha256") != content_hash(signals):
        raise ValueError("Plan does not match signal evidence")
    evidence = plan.get("evidence", [])
    if not isinstance(evidence, list) or any(not isinstance(e, dict) or not isinstance(e.get("id"), str) for e in evidence):
        raise ValueError("Invalid evidence records")
    ids = {e["id"] for e in evidence}
    if len(ids) != len(evidence):
        raise ValueError("Duplicate evidence IDs")
    segments = plan.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Empty plan")
    cursor = start
    for s in segments:
        if not isinstance(s, dict) or not _number(s.get("start_s")) or not _number(s.get("end_s")) or abs(s["start_s"] - cursor) > 1e-8 or not cursor < s["end_s"] <= end:
            raise ValueError("Segments must cover the interval without gaps or overlaps")
        if s.get("palette") not in {"warm", "cool"} or not isinstance(s.get("motif"), str) or not s["motif"]:
            raise ValueError("Unsupported palette or motif")
        transition = s.get("transition_s")
        if not _number(transition) or not 0 <= transition <= min(2, s["end_s"] - s["start_s"]):
            raise ValueError("Invalid transition")
        layers = s.get("layers", {})
        if set(layers) != set(LAYERS):
            raise ValueError("Plan must declare every supported layer")
        for layer in layers.values():
            if not isinstance(layer, dict) or type(layer.get("visible")) is not bool or layer.get("treatment") not in TREATMENTS or not _number(layer.get("gain")) or not 0 <= layer["gain"] <= 1:
                raise ValueError("Unsupported layer treatment or parameters")
        focus = s.get("focus")
        if focus not in layers or not layers[focus]["visible"] or layers[focus]["gain"] <= 0:
            raise ValueError("Focus must be visible")
        refs = s.get("evidence_ids")
        if not isinstance(refs, list) or not refs or any(not isinstance(r, str) or r not in ids for r in refs) or not isinstance(s.get("reason"), str) or not s["reason"]:
            raise ValueError("Missing reason or unknown evidence")
        cursor = s["end_s"]
    if abs(cursor - end) > 1e-8:
        raise ValueError("Segments do not cover the end")


def _layers() -> dict:
    treatments = dict(zip(LAYERS, ("ring", "rails", "ring", "ticks", "ribbon", "ring", "ribbon")))
    return {name: {"visible": False, "treatment": treatments[name], "gain": 0.0} for name in LAYERS}


def make_plan(signals: dict, start: float, end: float, *, min_gap_s: float = 4.0, tail_s: float = .35) -> dict:
    """Select focus from long main-percussion gaps and measured stem energy.

    Thresholds are explicit development choices, not inferred musical laws.
    Gaps are detected across the available song, not hand-entered clip boundaries.
    """
    validate_signals(signals)
    if not _number(start) or not _number(end) or not 0 <= start < end or not _number(min_gap_s) or not _number(tail_s) or not 0 < tail_s < min_gap_s:
        raise ValueError("Invalid planning interval or gap parameters")
    times = np.asarray(signals["energy"]["times_s"])
    if start < times[0] or end > times[-1]:
        raise ValueError("Plan outside available signal coverage")
    main = sorted({h["t"] for h in signals["hits"] if h["component"] in {"kick", "snare"} and h["velocity"] > 0})
    # Include leading/trailing sparse regions, without inventing anchor hits.
    edges = [times[0], *main, times[-1]]
    gaps = [(a + (tail_s if i > 0 else 0), b) for i, (a, b) in enumerate(zip(edges, edges[1:])) if b - a >= min_gap_s]
    boundaries = sorted({start, end, *(float(t) for g in gaps for t in g if start < t < end)})
    plan = {"schema_version": 1, "start_s": start, "end_s": end,
            "signals_sha256": content_hash(signals), "planner": {"kind": "rules", "version": "activity_direction_v1", "min_gap_s": min_gap_s, "tail_s": tail_s, "vocal_focus_threshold": .65, "active_energy_threshold": .08},
            "uncertainty": "Separated-stem activity, not certified sections, instrument truth or surprise. Artistic choices require review.",
            "context": {"available_song_interval_s": [float(times[0]), float(times[-1])], "main_percussion_gaps_s": gaps, "motif_basis": "recurring main-percussion activity; not a verified harmonic or melodic recurrence"},
            "evidence": [], "segments": []}
    for i, (lo, hi) in enumerate(zip(boundaries, boundaries[1:])):
        mask = (times >= lo) & (times < hi)
        means = {name: float(np.asarray(signals["energy"][name])[mask].mean()) if mask.any() else float(np.interp(lo, times, signals["energy"][name])) for name in ("bass", "vocals", "other")}
        counts = {name: sum(lo <= h["t"] < hi and h["component"] == name and h["velocity"] > 0 for h in signals["hits"]) for name in ("kick", "snare", "hh")}
        sparse = any(a <= (lo + hi) / 2 < b for a, b in gaps)
        eid = f"activity-{i}"
        plan["evidence"].append({"id": eid, "start_s": lo, "end_s": hi, "prominent_hit_counts": counts, "mean_normalized_stem_energy": means, "main_percussion_sparse": sparse})
        layers = _layers()
        if sparse:
            focus = max(means, key=means.get) if max(means.values()) >= .08 else "pulse"
            reason = f"Long gap in prominent kick/snare evidence; foreground {focus} using remaining stem energy. Hide main percussion rather than imply continued hits."
            palette, motif = "cool", "sparse-texture"
            layers[focus] = {"visible": True, "treatment": "ribbon" if focus != "pulse" else "ring", "gain": .95}
            if focus != "pulse":
                layers["pulse"].update(visible=True, gain=.08)
        else:
            focus = "vocals" if means["vocals"] >= .65 else next((n for n in ("snare", "kick", "hh") if counts[n]), max(means, key=means.get))
            reason = f"Prominent percussion is present; foreground {focus}. Reuse warm groove identity and keep other layers subordinate or hidden."
            palette, motif = "warm", "percussion-groove"
            for name in ("kick", "snare", "hh"):
                if counts[name]:
                    layers[name].update(visible=True, gain=.65 if name != "hh" else .22)
            layers["pulse"].update(visible=True, gain=.12)
            if means["bass"] >= .08:
                layers["bass"].update(visible=True, gain=.25)
            layers[focus].update(visible=True, gain=1., treatment="ring")
        plan["segments"].append({"start_s": lo, "end_s": hi, "focus": focus, "motif": motif, "palette": palette,
                                 "transition_s": 0. if i == 0 else min(.6, (hi - lo) / 2), "layers": layers, "reason": reason, "evidence_ids": [eid]})
    validate_plan(plan, signals)
    return plan


def fixed_plan(plan: dict, signals: dict) -> dict:
    """Always-on comparator with the same rendering vocabulary and evidence."""
    result = copy.deepcopy(plan)
    layers = _layers()
    for name in LAYERS:
        layers[name].update(visible=True, gain=1. if name == "snare" else .5)
    layers["pulse"]["gain"] = .12
    result["planner"] = {"kind": "fixed_comparison", "version": "v1"}
    result["segments"] = [{"start_s": plan["start_s"], "end_s": plan["end_s"], "focus": "snare", "motif": "percussion-groove", "palette": "warm", "transition_s": 0., "layers": layers,
                           "reason": "Constant focus and visibility comparison; underlying audio and signals unchanged.", "evidence_ids": [e["id"] for e in result["evidence"]]}]
    validate_plan(result, signals)
    return result
