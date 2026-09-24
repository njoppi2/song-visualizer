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
# The first four are the frozen prototype vocabulary.  The authored vocabulary
# below is opt-in through ``make_visual_plan``; retaining this distinction keeps
# saved v1/v2 controls pixel-compatible in the renderer.
TREATMENTS = {"ring", "ribbon", "ticks", "rails", "filament", "shards", "impact", "contour"}


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
    version = plan.get("schema_version")
    if version not in {1, 2} or not _number(start) or not _number(end) or not 0 <= start < end:
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
    evidence_by_id = {e["id"]: e for e in evidence}
    planner = plan.get("planner")
    generated_v2 = (version == 2 and isinstance(planner, dict)
                    and planner.get("kind") == "rules"
                    and planner.get("version") == "gradual_activity_direction_v2")
    if version == 2 and not generated_v2:
        # A v2 curve without the known planner must say that it is authored.
        # In particular, deleting the gradual planner cannot turn a generated
        # record into an unprovenanced authored one.  ``visual_policy`` is
        # deliberately not envelope provenance: it changes only presentation.
        provenance = plan.get("envelope_provenance")
        if (not isinstance(provenance, dict) or provenance.get("kind") != "authored"
                or not isinstance(provenance.get("description"), str)
                or not provenance["description"]):
            raise ValueError("Non-generated version 2 envelopes require authored provenance")
    segments = plan.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Empty plan")
    cursor = start
    previous_layers = None
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
        for name, layer in layers.items():
            if not isinstance(layer, dict) or type(layer.get("visible")) is not bool or layer.get("treatment") not in TREATMENTS or not _number(layer.get("gain")) or not 0 <= layer["gain"] <= 1:
                raise ValueError("Unsupported layer treatment or parameters")
            anchor = layer.get("anchor")
            if anchor is not None and (not isinstance(anchor, list) or len(anchor) != 2
                                       or not all(_number(value) and 0 <= value <= 1 for value in anchor)):
                raise ValueError("Layer anchor must be a normalized [x, y] pair")
            envelope = layer.get("envelope")
            if version == 2:
                if not isinstance(envelope, list) or len(envelope) < 2:
                    raise ValueError("Version 2 layers require a gain/emphasis envelope")
                if any(not isinstance(k, dict) or not _number(k.get("t_s")) or not _number(k.get("gain")) or not _number(k.get("emphasis")) or not 0 <= k["gain"] <= 1 or not 0 <= k["emphasis"] <= 1 for k in envelope):
                    raise ValueError("Invalid envelope keyframe")
                if abs(envelope[0]["t_s"] - s["start_s"]) > 1e-8 or abs(envelope[-1]["t_s"] - s["end_s"]) > 1e-8 or any(a["t_s"] >= b["t_s"] for a, b in zip(envelope, envelope[1:])):
                    raise ValueError("Envelope must be ordered and cover its segment")
                if not layer["visible"] and any(k["gain"] != 0 for k in envelope):
                    raise ValueError("Hidden layers must have zero envelope gain")
                # ``gain`` is retained as a bounded legacy compatibility field
                # for frozen v2 plans, but envelopes are authoritative for v2
                # rendering and focus.  Do not require it to equal any knot:
                # directed-review-02 intentionally contains boundary-adjusted
                # scalar/curve differences and remains replayable unchanged.
                # A layer may enter/leave at a scene boundary: the segment
                # crossfade handles that without a fake instantaneous curve.
                # When it remains visible, its own curve must meet smoothly.
                if previous_layers is not None and previous_layers[name]["visible"] and layer["visible"]:
                    prior = previous_layers[name].get("envelope", [])[-1]
                    current = envelope[0]
                    if not prior or abs(prior["gain"] - current["gain"]) > 1e-8 or abs(prior["emphasis"] - current["emphasis"]) > 1e-8:
                        raise ValueError("Adjacent envelopes must be continuous")
            elif envelope is not None:
                raise ValueError("Version 1 plans do not support envelopes")
        focus = s.get("focus")
        focus_has_effect = (focus in layers and layers[focus]["visible"] and
                            (layers[focus]["gain"] > 0 if version == 1 else
                             any(key["gain"] > 0 for key in layers[focus]["envelope"])))
        if not focus_has_effect:
            raise ValueError("Focus must be visible")
        refs = s.get("evidence_ids")
        if not isinstance(refs, list) or not refs or any(not isinstance(r, str) or r not in ids for r in refs) or not isinstance(s.get("reason"), str) or not s["reason"]:
            raise ValueError("Missing reason or unknown evidence")
        if generated_v2:
            for ref in refs:
                record = evidence_by_id[ref]
                if record.get('start_s') != s['start_s'] or record.get('end_s') != s['end_s']:
                    raise ValueError('Generated envelope support must describe its segment')
                _validate_generated_envelope_support(evidence_by_id[ref], times)
        cursor = s["end_s"]
        previous_layers = layers
    if abs(cursor - end) > 1e-8:
        raise ValueError("Segments do not cover the end")


def _validate_generated_envelope_support(record: dict, times: list[float]) -> None:
    """Validate the exact offline support contract emitted by gradual v2."""
    start, end = record.get("start_s"), record.get("end_s")
    support = record.get("envelope_support")
    if (not _number(start) or not _number(end) or not times[0] <= start < end <= times[-1]
            or not isinstance(support, dict)):
        raise ValueError("Generated version 2 evidence requires envelope support")
    interval = support.get("envelope_interval_s")
    if (support.get("kind") != "finite_1.5s_trailing_per_stem_normalized_activity"
            or support.get("window_s") != 1.5
            or not isinstance(interval, list) or len(interval) != 2
            or not all(_number(value) for value in interval)
            or abs(interval[0] - start) > 1e-8 or abs(interval[1] - end) > 1e-8
            or not _number(support.get("support_start_s"))
            or not _number(support.get("available_through_s"))
            or any(not isinstance(support.get(field), str) or not support[field]
                   for field in ("sample_clock", "render_interpolation", "normalization"))):
        raise ValueError("Malformed generated envelope support")
    # The gradual planner uses a completed sample at/preceding each endpoint.
    # Its support-start convention includes the immediately preceding block
    # before the 1.5s trailing window; preserve that frozen convention exactly.
    clock = np.asarray(times, dtype=float)
    end_index = int(np.searchsorted(clock, end, side="right")) - 1
    anchor_index = int(np.searchsorted(clock, start, side="right")) - 1
    if end_index < 0 or anchor_index < 0:
        raise ValueError("Generated envelope support is outside the signal clock")
    first = int(np.searchsorted(clock, clock[anchor_index] - 1.5, side="left"))
    expected_start = float(clock[max(0, first - 1)])
    expected_end = float(clock[end_index])
    if (abs(support["support_start_s"] - expected_start) > 1e-8
            or abs(support["available_through_s"] - expected_end) > 1e-8):
        raise ValueError("Generated envelope support does not match the signal clock")


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
    # The fixed comparator deliberately stays the established constant v1
    # contract, even when its directed source is a v2 envelope plan.
    result["schema_version"] = 1
    layers = _layers()
    for name in LAYERS:
        layers[name].update(visible=True, gain=1. if name == "snare" else .5)
    layers["pulse"]["gain"] = .12
    result["planner"] = {"kind": "fixed_comparison", "version": "v1"}
    result["segments"] = [{"start_s": plan["start_s"], "end_s": plan["end_s"], "focus": "snare", "motif": "percussion-groove", "palette": "warm", "transition_s": 0., "layers": layers,
                           "reason": "Constant focus and visibility comparison; underlying audio and signals unchanged.", "evidence_ids": [e["id"] for e in result["evidence"]]}]
    validate_plan(result, signals)
    return result


def _trailing_smooth(times: np.ndarray, values: np.ndarray, window_s: float) -> np.ndarray:
    """Finite causal trailing mean with no cross-stem comparison."""
    result = np.empty_like(values, dtype=float)
    for i, time in enumerate(times):
        first = int(np.searchsorted(times, float(time) - window_s, side="left"))
        result[i] = float(values[first:i + 1].mean())
    return result


def make_gradual_plan(signals: dict, baseline_plan: dict) -> dict:
    """Make a bounded v2 comparison with continuous sparse-span emphasis.

    This deliberately retains the baseline's gap boundaries, palettes and visual
    motif labels.  It only turns each stem's own trailing normalized activity
    into an inspectable visual envelope; it does not infer loudness between
    stems, vocal leadership, salience, or musical recurrence.
    """
    validate_plan(baseline_plan, signals)
    if baseline_plan["schema_version"] != 1:
        raise ValueError("Gradual planning requires a version 1 baseline plan")
    result = copy.deepcopy(baseline_plan)
    result["schema_version"] = 2
    result["planner"] = {
        "kind": "rules", "version": "gradual_activity_direction_v2",
        "baseline_policy": baseline_plan.get("planner", {}).get("version"),
        "trailing_smoothing_s": 1.5,
        "sparse_other_gain": [0.38, 0.82],
        "sparse_vocals_gain": [0.04, 0.94],
        "sparse_vocals_emphasis": [0.08, 0.92],
        "policy": "Per-stem normalized activity only; no top-k allocation or cross-stem loudness comparison.",
    }
    result["uncertainty"] = ("Visual envelopes use trailing smoothed per-stem normalized activity. "
                             "They do not establish vocal leadership, salience, instrument identity, "
                             "or verified musical recurrence; artistic effect requires review.")
    times = np.asarray(signals["energy"]["times_s"], dtype=float)
    smooth = {name: _trailing_smooth(times, np.asarray(signals["energy"][name], dtype=float), 1.5)
              for name in ("bass", "vocals", "other")}

    def activity(name: str, time: float) -> float:
        """Sample-and-hold a completed activity block; never interpolate ahead."""
        if name not in smooth:
            return 0.0
        position = int(np.searchsorted(times, time, side="right")) - 1
        return float(smooth[name][position]) if position >= 0 else 0.0

    evidence_by_id = {record["id"]: record for record in result["evidence"]}

    for segment in result["segments"]:
        lo, hi = float(segment["start_s"]), float(segment["end_s"])
        references = [evidence_by_id[ref] for ref in segment["evidence_ids"]]
        sparse = any(bool(record.get("main_percussion_sparse")) for record in references)
        knots = [lo, *(float(t) for t in times if lo < t < hi), hi]
        for name, layer in segment["layers"].items():
            original_gain = float(layer["gain"])
            visible = bool(layer["visible"])
            curve = []
            for time in knots:
                value = activity(name, time)
                if not visible:
                    gain, emphasis = 0.0, 0.0
                elif sparse and name == "vocals":
                    # Keep a quiet support trace, then let this stem's own
                    # recent activity move it inward before percussion returns.
                    gain, emphasis = .04 + .90 * value, .08 + .84 * value
                elif sparse and name == "other":
                    gain, emphasis = .38 + .44 * value, 1.0 if name == segment["focus"] else .25 + .45 * value
                else:
                    gain, emphasis = original_gain, 1.0 if name == segment["focus"] else .25
                curve.append({"t_s": time, "gain": gain, "emphasis": emphasis})
            layer["envelope"] = curve
        # Stable allocation in sparse spans: percussion/bass remain omitted;
        # other and vocals do not compete for a single flickering slot.
        if sparse and segment["focus"] != "pulse":
            for name in ("vocals", "other"):
                layer = segment["layers"][name]
                layer.update(visible=True, treatment="ribbon", gain=layer["envelope"][0]["gain"])
                # Rebuild after visibility changes above.
                layer["envelope"] = [{"t_s": time,
                                      "gain": (.04 + .90 * activity(name, time)) if name == "vocals" else (.38 + .44 * activity(name, time)),
                                      "emphasis": (.08 + .84 * activity(name, time)) if name == "vocals" else (1.0 if name == segment["focus"] else .25 + .45 * activity(name, time))}
                                     for time in knots]
                layer["gain"] = layer["envelope"][0]["gain"]
        support_end_index = int(np.searchsorted(times, hi, side="right")) - 1
        first_anchor = int(np.searchsorted(times, lo, side="right")) - 1
        first_block = int(np.searchsorted(times, times[first_anchor] - 1.5, side="left"))
        support = {"kind": "finite_1.5s_trailing_per_stem_normalized_activity",
                   "window_s": 1.5, "envelope_interval_s": [lo, hi],
                   "support_start_s": float(times[max(0, first_block - 1)]),
                   "available_through_s": float(times[support_end_index]),
                   "sample_clock": "Each knot samples the last completed trailing activity estimate; support includes the contributing blocks' starts.",
                   "render_interpolation": "Offline linear interpolation between saved knots uses the next knot; local lookahead is at most one input block. This is not a streaming plan.",
                   "normalization": "frozen whole-song per-stem p95 from input signals; offline planning"}
        for record in references:
            record["envelope_support"] = support
            record["semantic_interpretation"] = {"vocal_leadership": "unknown", "salience": "unknown", "musical_recurrence": "unknown"}
        segment["baseline_reason"] = segment["reason"]
        segment["reason"] = (
            "Keep stable other/vocals traces in this sparse-percussion span; their own trailing activity controls gradual visual emphasis."
            if sparse and segment["focus"] != "pulse" else
            "Retain the earlier layer selection and visual motif; use the saved emphasis geometry and scene crossfade."
        ) + " Musical leadership and salience remain unknown."
    for previous, current in zip(result["segments"], result["segments"][1:]):
        for name in LAYERS:
            old, new = previous["layers"][name], current["layers"][name]
            if old["visible"] and new["visible"]:
                # Preserve position and intensity at a shared knot.  The next
                # keyframe then evolves smoothly under the new local policy.
                new["envelope"][0]["gain"] = old["envelope"][-1]["gain"]
                new["envelope"][0]["emphasis"] = old["envelope"][-1]["emphasis"]
    validate_plan(result, signals)
    return result


def make_visual_plan(signals: dict, baseline_plan: dict) -> dict:
    """Apply an authored, source-stable visual vocabulary to a gradual plan.

    This is intentionally *not* another direction policy.  It deep-copies the
    saved schema-v2 plan and changes only treatment identity, stable placement,
    and declared visual authorship.  Segment clocks, crossfades, visibility,
    focus, gains, emphasis curves, motifs, and evidence therefore remain the
    exact inspected response plan.
    """
    validate_plan(baseline_plan, signals)
    if baseline_plan.get("schema_version") != 2:
        raise ValueError("Visual planning requires a schema version 2 gradual plan")
    result = copy.deepcopy(baseline_plan)
    vocabulary = {
        "pulse": ("ring", [0.16, 0.48]),
        "kick": ("impact", [0.50, 0.79]),
        "snare": ("shards", [0.80, 0.47]),
        "hh": ("ticks", [0.18, 0.20]),
        "bass": ("ribbon", [0.50, 0.67]),
        "vocals": ("filament", [0.50, 0.42]),
        # A deliberately broad, lower atmospheric field.  It is behind the
        # vocal anchor in composition, not a second central voice trace.
        "other": ("contour", [0.50, 0.66]),
    }
    for segment in result["segments"]:
        for name, layer in segment["layers"].items():
            layer["treatment"], layer["anchor"] = vocabulary[name]
    result["visual_policy"] = {
        "kind": "authored_source_vocabulary_v1",
        "scope": "Visual vocabulary and stable layout only; no timing, signal, evidence, visibility, focus, gain, or emphasis change.",
        "treatments": {
            "vocals": "filament: continuous vertical flowing veil; decorative response, not pitch transcription",
            "snare": "shards: crisp angular broken strokes",
            "kick": "impact: compact grounded oval with a short halo expansion",
            "other": "contour: broad layered atmospheric curves behind and below the vocal field",
            "bass": "ribbon; hh: ticks; pulse: quiet ring",
        },
        "placement": "Stable anchors preserve legibility across focus changes; emphasis changes geometry scale rather than selecting a new position.",
        "stage": "Clean vignette without construction guides; brighter treatment strokes support readability at small review sizes.",
        "motion": "Existing event and stem-energy clocks only; no semantic or pitch-derived motion.",
    }
    validate_plan(result, signals)
    return result


_VOCAL_EMPHASIS_START = 119.0
_VOCAL_EMPHASIS_END = 132.0
_VOCAL_EMPHASIS_KNOTS = (119.0, 123.0, 125.5, 132.0)
# These deliberately modest support values are fixed experimental staging, not
# an allocation inferred from stem level or from the listening response.
_VOCAL_EMPHASIS_ACCOMPANIMENT = {
    "pulse": (.10, .25), "kick": (.38, .35), "snare": (.42, .45),
    "hh": (.18, .22), "bass": (.26, .30), "other": (.32, .35),
}


def _feedback_sha256(feedback_record: dict) -> str | None:
    file_record = feedback_record.get("file")
    if isinstance(file_record, dict) and isinstance(file_record.get("sha256"), str):
        return file_record["sha256"]
    value = feedback_record.get("sha256")
    return value if isinstance(value, str) else None


def _validate_vocal_emphasis_feedback(feedback_record: dict) -> None:
    """Require the hash-bound, unabridged verse-ending listening response."""
    if (not isinstance(feedback_record, dict) or not _feedback_sha256(feedback_record)
            or not isinstance(feedback_record.get("file"), (dict, str))):
        raise ValueError("Vocal-emphasis feedback requires its source file and SHA-256")
    answer = feedback_record.get("answer")
    if (not isinstance(answer, dict) or answer.get("example_id") != "verse-ending"
            or answer.get("perceived_change") != "subtle"
            or not isinstance(answer.get("notes"), str) or not answer["notes"]):
        raise ValueError("Vocal-emphasis feedback requires the full subtle verse-ending answer")


def _vocal_emphasis_envelope(values: tuple[tuple[float, float], ...]) -> list[dict]:
    return [{"t_s": time, "gain": gain, "emphasis": emphasis}
            for time, (gain, emphasis) in zip(_VOCAL_EMPHASIS_KNOTS, values)]


def _validate_vocal_emphasis_plan(plan: dict, signals: dict) -> None:
    """Validate fixed authored-study boundaries in addition to normal v2 rules."""
    validate_plan(plan, signals)
    if (plan.get("schema_version") != 2 or plan.get("start_s") != _VOCAL_EMPHASIS_START
            or plan.get("end_s") != _VOCAL_EMPHASIS_END
            or plan.get("planner") != {"kind": "authored", "version": "vocal_emphasis_comparison_v2"}):
        raise ValueError("Vocal-emphasis plan must use the fixed authored experiment boundary")
    provenance = plan.get("envelope_provenance")
    if (not isinstance(provenance, dict) or provenance.get("kind") != "authored"
            or not isinstance(provenance.get("description"), str)
            or not provenance["description"]):
        raise ValueError("Vocal-emphasis plan requires authored envelope provenance")
    segments = plan.get("segments")
    if not isinstance(segments, list) or len(segments) != 1:
        raise ValueError("Vocal-emphasis experiment has exactly one scene")
    segment = segments[0]
    if (segment.get("start_s") != _VOCAL_EMPHASIS_START
            or segment.get("end_s") != _VOCAL_EMPHASIS_END
            or segment.get("focus") != "vocals"):
        raise ValueError("Vocal-emphasis experiment keeps fixed vocal focus")
    vocal = segment["layers"]["vocals"]
    if vocal.get("gain") != 1.0 or not vocal.get("visible"):
        raise ValueError("Vocal scalar gain and visibility are fixed in this experiment")
    envelope = vocal.get("envelope", [])
    if [key["t_s"] for key in envelope] != list(_VOCAL_EMPHASIS_KNOTS):
        raise ValueError("Vocal-emphasis experiment requires its four fixed knots")
    if any(name not in _VOCAL_EMPHASIS_ACCOMPANIMENT for name in LAYERS if name != "vocals"):
        raise AssertionError("Incomplete fixed accompaniment declaration")
    for name, (gain, emphasis) in _VOCAL_EMPHASIS_ACCOMPANIMENT.items():
        layer = segment["layers"][name]
        if (not layer.get("visible") or layer.get("gain") != gain
                or layer.get("envelope") != _vocal_emphasis_envelope(((gain, emphasis),) * 4)):
            raise ValueError("Vocal-emphasis accompaniment must remain fixed and constant")
    evidence = plan.get("evidence")
    if (not isinstance(evidence, list) or len(evidence) != 1
            or evidence[0].get("id") != "verse-ending-feedback"
            or set(evidence[0]) != {"id", "feedback_record"}):
        raise ValueError("Vocal-emphasis experiment requires its one feedback record")
    _validate_vocal_emphasis_feedback(evidence[0]["feedback_record"])


def make_vocal_emphasis_plans(signals: dict, visual_parent: dict,
                              feedback_record: dict) -> tuple[dict, dict]:
    """Build reduced/steady authored plans for the fixed 119–132s study.

    ``visual_parent`` supplies only the already-reviewed visual treatments and
    anchors.  The returned plans deliberately do not derive their envelope from
    its activity or evidence. ``feedback_record`` is stored unchanged under the
    sole evidence record; it must bind a source-file SHA-256 and the complete
    raw ``verse-ending`` answer.
    """
    validate_plan(visual_parent, signals)
    if visual_parent.get("schema_version") != 2:
        raise ValueError("Vocal-emphasis planning requires a visual schema-v2 parent")
    _validate_vocal_emphasis_feedback(feedback_record)
    parent_layers = visual_parent["segments"][0].get("layers", {})
    if set(parent_layers) != set(LAYERS) or any(
            "anchor" not in parent_layers[name] for name in LAYERS):
        raise ValueError("Vocal-emphasis planning requires refreshed parent anchors")

    layers = copy.deepcopy(parent_layers)
    for name, (gain, emphasis) in _VOCAL_EMPHASIS_ACCOMPANIMENT.items():
        layers[name].update(visible=True, gain=gain)
        layers[name]["envelope"] = _vocal_emphasis_envelope(((gain, emphasis),) * 4)
    layers["vocals"].update(visible=True, gain=1.0)
    common = {
        "schema_version": 2, "start_s": _VOCAL_EMPHASIS_START,
        "end_s": _VOCAL_EMPHASIS_END, "signals_sha256": content_hash(signals),
        "planner": {"kind": "authored", "version": "vocal_emphasis_comparison_v2"},
        "uncertainty": ("The vocal envelope is an authored comparison choice informed by a "
                        "qualified listening note; it does not infer vocal function, importance, "
                        "laughter, leadership, or a detected boundary."),
        "context": {"accompaniment_envelopes": "Fixed modest constants; source-driven rendering remains unchanged."},
        "evidence": [{"id": "verse-ending-feedback", "feedback_record": copy.deepcopy(feedback_record)}],
        "visual_policy": copy.deepcopy(visual_parent.get("visual_policy")),
    }

    def plan(description: str, vocal_values: tuple[tuple[float, float], ...]) -> dict:
        result = copy.deepcopy(common)
        result["envelope_provenance"] = {"kind": "authored", "description": description}
        scene_layers = copy.deepcopy(layers)
        scene_layers["vocals"]["envelope"] = _vocal_emphasis_envelope(vocal_values)
        result["segments"] = [{
            "start_s": _VOCAL_EMPHASIS_START, "end_s": _VOCAL_EMPHASIS_END,
            "focus": "vocals", "motif": visual_parent["segments"][0]["motif"],
            "palette": visual_parent["segments"][0]["palette"], "transition_s": 0.0,
            "layers": scene_layers,
            "reason": "Fixed authored vocal-emphasis comparison; all non-vocal envelope values remain constant.",
            "evidence_ids": ["verse-ending-feedback"],
        }]
        _validate_vocal_emphasis_plan(result, signals)
        return result

    reduced = plan(
        "Authored reduced vocal emphasis: hold through 123s, linearly reduce by 125.5s, then hold.",
        ((1.0, 1.0), (1.0, 1.0), (.45, .30), (.45, .30)),
    )
    steady = plan(
        "Authored steady vocal emphasis control: retain full visual emphasis throughout the fixed scene.",
        ((1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
    )
    validate_vocal_emphasis_comparison(reduced, steady, signals)
    return reduced, steady


def validate_vocal_emphasis_comparison(plan: dict, baseline: dict, signals: dict) -> None:
    """Require the paired study to differ only in its authorized vocal curves."""
    _validate_vocal_emphasis_plan(plan, signals)
    _validate_vocal_emphasis_plan(baseline, signals)
    allowed_plan = copy.deepcopy(plan)
    allowed_baseline = copy.deepcopy(baseline)
    for item in (allowed_plan, allowed_baseline):
        item["envelope_provenance"].pop("description")
        for key in item["segments"][0]["layers"]["vocals"]["envelope"]:
            key.pop("gain")
            key.pop("emphasis")
    if allowed_plan != allowed_baseline:
        raise ValueError("Vocal-emphasis comparison changed a shared field")
    reduced = plan["segments"][0]["layers"]["vocals"]["envelope"]
    steady = baseline["segments"][0]["layers"]["vocals"]["envelope"]
    if ([key["gain"] for key in reduced], [key["emphasis"] for key in reduced]) != ([1., 1., .45, .45], [1., 1., .30, .30]):
        raise ValueError("Reduced vocal plan does not use the fixed authored curve")
    if any(key["gain"] != 1.0 or key["emphasis"] != 1.0 for key in steady):
        raise ValueError("Steady vocal plan does not use the fixed authored curve")
