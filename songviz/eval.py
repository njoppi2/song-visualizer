"""Evaluation framework for reduced-representation quality.

Compares extracted drum hits, vocal notes, and bass notes against
human-curated reference annotations.  Supports three evidence levels:

- **gold**: trusted datasets or synthetic ground truth
- **silver**: tabs, MIDI, score-like references, verified by listening
- **weak**: rough annotations, unverified

Usage::

    songviz eval path/to/song.flac
    songviz eval path/to/song.flac --reference-dir benchmark/references/feel-good-inc
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# ── GM drum map (MIDI note → component name) ──

_GM_NOTE_TO_COMPONENT: dict[int, str] = {
    35: "kick", 36: "kick",
    38: "snare", 40: "snare",
    42: "hh", 44: "hh", 46: "hh",
    47: "toms", 48: "toms", 45: "toms", 43: "toms", 41: "toms",
    51: "ride", 59: "ride",
    49: "crash", 57: "crash",
}

# ── Benchmark directory layout ──

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmark"


def _load_song_index() -> dict[str, str]:
    """Load ``benchmark/songs.json`` → ``{song_id: reference_subdir}``."""
    index_path = _BENCHMARK_DIR / "songs.json"
    if not index_path.exists():
        return {}
    return json.loads(index_path.read_text(encoding="utf-8"))


def references_dir_for_song(song_id: str) -> Path | None:
    """Look up the reference directory for a *song_id*, or ``None``."""
    index = _load_song_index()
    subdir = index.get(song_id)
    if subdir is None:
        return None
    ref_dir = _BENCHMARK_DIR / "references" / subdir
    return ref_dir if ref_dir.is_dir() else None


def load_reference(path: Path) -> dict[str, Any]:
    """Load a JSON reference annotation file."""
    return json.loads(path.read_text(encoding="utf-8"))


# ── Event filtering ──


def _events_in_range(
    events: list[dict[str, Any]],
    start_s: float,
    end_s: float,
    time_key: str = "onset_s",
) -> list[dict[str, Any]]:
    """Return events whose onset falls within [start_s, end_s)."""
    return [e for e in events if start_s <= e[time_key] < end_s]


# ── Activity evaluation ──


def evaluate_activity(
    events: list[dict[str, Any]],
    activity: list[dict[str, Any]],
    *,
    time_key: str = "onset_s",
    min_events_for_active: int = 3,
) -> dict[str, Any]:
    """Compare extracted events against section-level activity annotations.

    Each entry in *activity* must have ``start_s``, ``end_s``, ``active`` (bool).

    Returns section-level breakdown, precision/recall/F1 for active detection,
    and false-positive counts in inactive regions.
    """
    tp = fp = fn = tn = 0
    silent_fp_count = 0
    section_results: list[dict[str, Any]] = []

    for sec in activity:
        in_range = _events_in_range(events, sec["start_s"], sec["end_s"], time_key)
        n_events = len(in_range)
        detected_active = n_events >= min_events_for_active
        ref_active = sec["active"]

        if ref_active and detected_active:
            tp += 1
        elif ref_active and not detected_active:
            fn += 1
        elif not ref_active and detected_active:
            fp += 1
        else:
            tn += 1

        if not ref_active:
            silent_fp_count += n_events

        section_results.append({
            "start_s": sec["start_s"],
            "end_s": sec["end_s"],
            "label": sec.get("label", ""),
            "ref_active": ref_active,
            "detected_active": detected_active,
            "event_count": n_events,
            "match": ref_active == detected_active,
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    total_events = len(events)
    return {
        "sections": section_results,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "silent_fp_count": silent_fp_count,
        "silent_fp_rate": round(silent_fp_count / total_events, 4) if total_events > 0 else 0.0,
    }


# ── Pitch range evaluation ──


def evaluate_pitch_range(
    notes: list[dict[str, Any]],
    expected_range: tuple[float, float],
    *,
    activity: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fraction of notes with MIDI pitch within *expected_range* ``[lo, hi]``.

    If *activity* is provided, only notes in active sections are checked.
    """
    if activity:
        active_notes: list[dict[str, Any]] = []
        for sec in activity:
            if sec["active"]:
                active_notes.extend(_events_in_range(notes, sec["start_s"], sec["end_s"]))
        check = active_notes
    else:
        check = notes

    if not check:
        return {"in_range_pct": 0.0, "checked": 0}

    lo, hi = expected_range
    midis = [n["midi"] for n in check]
    in_range = sum(1 for m in midis if lo <= m <= hi)

    return {
        "checked": len(check),
        "in_range_count": in_range,
        "in_range_pct": round(100.0 * in_range / len(check), 1),
        "midi_median": round(float(np.median(midis)), 1),
        "midi_mean": round(float(np.mean(midis)), 1),
        "below_range_pct": round(100.0 * sum(1 for m in midis if m < lo) / len(check), 1),
        "above_range_pct": round(100.0 * sum(1 for m in midis if m > hi) / len(check), 1),
    }


# ── Onset matching ──


def evaluate_onsets(
    extracted_times: list[float],
    reference_times: list[float],
    tolerance_s: float = 0.05,
) -> dict[str, Any]:
    """Match extracted onsets to reference onsets (greedy nearest-neighbour).

    Returns precision, recall, F1.
    """
    if not reference_times:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "matched": 0, "fp": len(extracted_times), "fn": 0,
            "tolerance_s": tolerance_s,
        }
    if not extracted_times:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "matched": 0, "fp": 0, "fn": len(reference_times),
            "tolerance_s": tolerance_s,
        }

    ext = sorted(extracted_times)
    ref = sorted(reference_times)
    matched_ext: set[int] = set()
    matched_count = 0

    for rt in ref:
        best_dist = float("inf")
        best_ei = -1
        for ei, et in enumerate(ext):
            if ei in matched_ext:
                continue
            dist = abs(et - rt)
            if dist < best_dist:
                best_dist = dist
                best_ei = ei
            elif et > rt + tolerance_s:
                break
        if best_ei >= 0 and best_dist <= tolerance_s:
            matched_ext.add(best_ei)
            matched_count += 1

    fp = len(ext) - matched_count
    fn = len(ref) - matched_count
    precision = matched_count / (matched_count + fp) if (matched_count + fp) > 0 else 0.0
    recall = matched_count / (matched_count + fn) if (matched_count + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched": matched_count,
        "fp": fp,
        "fn": fn,
        "tolerance_s": tolerance_s,
    }


# ── Note-level transcription evaluation ──


def _load_midi_reference_notes(midi_path: Path, layer: str) -> list[dict[str, Any]]:
    """Load notes from a MIDI reference file.

    Returns a list of dicts with ``onset_s``, ``midi``, and ``velocity``.
    For drums, also includes ``component`` (kick/snare/hh/etc.) and ``t``.

    Returns an empty list if ``pretty_midi`` is not installed.
    """
    try:
        import pretty_midi
    except ImportError:
        return []

    pm = pretty_midi.PrettyMIDI(str(midi_path))

    if layer == "drums":
        hits: list[dict[str, Any]] = []
        for inst in pm.instruments:
            if inst.is_drum:
                for note in inst.notes:
                    comp = _GM_NOTE_TO_COMPONENT.get(note.pitch, "kick")
                    hits.append({
                        "t": note.start,
                        "onset_s": note.start,
                        "component": comp,
                        "midi": float(note.pitch),
                        "velocity": note.velocity / 127.0,
                    })
        return sorted(hits, key=lambda h: h["t"])
    else:
        notes: list[dict[str, Any]] = []
        for inst in pm.instruments:
            if not inst.is_drum:
                for note in inst.notes:
                    notes.append({
                        "onset_s": note.start,
                        "offset_s": note.end,
                        "midi": float(note.pitch),
                        "velocity": note.velocity / 127.0,
                    })
        return sorted(notes, key=lambda n: n["onset_s"])


def evaluate_note_transcription(
    detected_notes: list[dict[str, Any]],
    ref_notes: list[dict[str, Any]],
    *,
    onset_tol_s: float = 0.05,
    pitch_tol_st: float = 1.0,
) -> dict[str, Any]:
    """Note-level precision / recall / F1 against a MIDI reference.

    Computes:

    - **onset_f1**: onset-only matching (right time, any pitch)
    - **note_f1**: onset + absolute pitch matching
    - **note_f1_octave_invariant**: onset + pitch-class matching (wraps octaves)
    - **pitch_accuracy**: % of onset-matched pairs with correct pitch
    - **mean_onset_error_ms** / **mean_pitch_error_st**: average errors

    Both *detected_notes* and *ref_notes* must have ``onset_s`` and ``midi``
    fields.  Drums hits should also have them (``onset_s`` = ``t``,
    ``midi`` = GM note number — for drums only onset matching is meaningful).
    """
    if not ref_notes and not detected_notes:
        return {"ref_note_count": 0, "det_note_count": 0, "onset_f1": 0.0, "note_f1": 0.0}

    det = sorted(detected_notes, key=lambda n: n["onset_s"])
    ref = sorted(ref_notes, key=lambda n: n["onset_s"])

    def _greedy_match(
        ref_list: list[dict[str, Any]],
        det_list: list[dict[str, Any]],
        require_pitch: bool,
        octave_invariant: bool = False,
    ) -> tuple[int, list[float], list[float]]:
        """Return (match_count, onset_errors_ms, pitch_errors_st).

        pitch_errors_st is always collected for matched pairs (used for
        pitch_accuracy computation even in the onset-only pass).
        """
        matched_det: set[int] = set()
        match_count = 0
        onset_errors: list[float] = []
        pitch_errors: list[float] = []

        for rn in ref_list:
            best_dist = float("inf")
            best_di = -1
            for di, dn in enumerate(det_list):
                if di in matched_det:
                    continue
                odist = abs(dn["onset_s"] - rn["onset_s"])
                if dn["onset_s"] > rn["onset_s"] + onset_tol_s:
                    break
                if odist > onset_tol_s:
                    continue
                if require_pitch:
                    if octave_invariant:
                        rpc = int(round(rn["midi"])) % 12
                        dpc = int(round(dn["midi"])) % 12
                        pdist = min(abs(rpc - dpc), 12 - abs(rpc - dpc))
                    else:
                        pdist = abs(dn["midi"] - rn["midi"])
                    if pdist > pitch_tol_st:
                        continue
                if odist < best_dist:
                    best_dist = odist
                    best_di = di

            if best_di >= 0:
                matched_det.add(best_di)
                match_count += 1
                onset_errors.append(best_dist * 1000.0)
                # Always record pitch error for matched pairs
                pitch_errors.append(abs(det_list[best_di]["midi"] - rn["midi"]))

        return match_count, onset_errors, pitch_errors

    def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return round(prec, 4), round(rec, 4), round(f1, 4)

    # Onset-only
    o_tp, o_err_ms, o_pitch_err = _greedy_match(ref, det, require_pitch=False)
    o_fp = len(det) - o_tp
    o_fn = len(ref) - o_tp
    o_prec, o_rec, o_f1 = _f1(o_tp, o_fp, o_fn)

    # Pitch accuracy from onset-matched pairs (separate pass without pitch filter)
    pitch_accuracy = (
        sum(1 for e in o_pitch_err if e <= pitch_tol_st) / len(o_pitch_err)
        if o_pitch_err else 0.0
    )

    # Note matching (onset + absolute pitch)
    n_tp, _, _ = _greedy_match(ref, det, require_pitch=True, octave_invariant=False)
    n_fp = len(det) - n_tp
    n_fn = len(ref) - n_tp
    n_prec, n_rec, n_f1 = _f1(n_tp, n_fp, n_fn)

    # Octave-invariant note matching (onset + pitch-class)
    pc_tp, _, _ = _greedy_match(ref, det, require_pitch=True, octave_invariant=True)
    pc_fp = len(det) - pc_tp
    pc_fn = len(ref) - pc_tp
    _, _, pc_f1 = _f1(pc_tp, pc_fp, pc_fn)

    return {
        "ref_note_count": len(ref),
        "det_note_count": len(det),
        "fragmentation_ratio": round(len(det) / len(ref), 3) if len(ref) > 0 else 0.0,
        "onset_tp": o_tp, "onset_fp": o_fp, "onset_fn": o_fn,
        "onset_precision": o_prec, "onset_recall": o_rec, "onset_f1": o_f1,
        "note_tp": n_tp, "note_fp": n_fp, "note_fn": n_fn,
        "note_precision": n_prec, "note_recall": n_rec, "note_f1": n_f1,
        "note_f1_octave_invariant": pc_f1,
        "pitch_accuracy": round(pitch_accuracy, 4),
        "mean_onset_error_ms": round(float(np.mean(o_err_ms)), 1) if o_err_ms else 0.0,
        "mean_pitch_error_st": round(float(np.mean(o_pitch_err)), 2) if o_pitch_err else 0.0,
        "onset_tol_s": onset_tol_s,
        "pitch_tol_st": pitch_tol_st,
    }


# ── Octave-invariant pitched-layer metrics ──

_PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _pitch_class_histogram(notes: list[dict[str, Any]]) -> np.ndarray:
    """12-element pitch-class histogram, normalized to sum=1."""
    h = np.zeros(12, dtype=np.float64)
    for n in notes:
        h[int(round(n["midi"])) % 12] += 1
    total = h.sum()
    if total > 0:
        h /= total
    return h


def _contour_direction(notes: list[dict[str, Any]]) -> list[int]:
    """Direction between successive notes: +1=up, -1=down, 0=same."""
    dirs: list[int] = []
    for i in range(len(notes) - 1):
        diff = notes[i + 1]["midi"] - notes[i]["midi"]
        dirs.append(1 if diff > 0.5 else (-1 if diff < -0.5 else 0))
    return dirs


def _contour_proportions(dirs: list[int]) -> dict[str, float]:
    """Fraction of up / down / same in a contour direction vector."""
    n = len(dirs) if dirs else 1
    return {
        "up": round(dirs.count(1) / n, 4),
        "down": round(dirs.count(-1) / n, 4),
        "same": round(dirs.count(0) / n, 4),
    }


def _ioi_vector(notes: list[dict[str, Any]]) -> list[float]:
    """Inter-onset intervals (seconds) between successive notes."""
    return [notes[i + 1]["onset_s"] - notes[i]["onset_s"] for i in range(len(notes) - 1)]


def _histogram_intersection(a: np.ndarray, b: np.ndarray) -> float:
    """Histogram intersection (overlap) in [0, 1]."""
    return float(np.minimum(a, b).sum())


def evaluate_pitch_class(
    notes: list[dict[str, Any]],
    reference_pitch: dict[str, Any],
    *,
    activity: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Octave-invariant pitch-class evaluation.

    Compares the extracted pitch-class histogram against expected scale PCs
    from the reference.  Also reports the detected dominant PC and how it
    relates to the expected root.
    """
    if activity:
        check: list[dict[str, Any]] = []
        for sec in activity:
            if sec["active"]:
                check.extend(_events_in_range(notes, sec["start_s"], sec["end_s"]))
    else:
        check = notes
    if not check:
        return {"checked": 0}

    hist = _pitch_class_histogram(check)
    dominant_pc = int(np.argmax(hist))

    result: dict[str, Any] = {
        "checked": len(check),
        "dominant_pc": dominant_pc,
        "dominant_pc_name": _PC_NAMES[dominant_pc],
        "dominant_pc_pct": round(float(hist[dominant_pc]) * 100, 1),
    }

    # Compare against expected root and scale
    root_pc = reference_pitch.get("root_pc")
    if root_pc is None and "root_midi" in reference_pitch:
        root_pc = int(reference_pitch["root_midi"]) % 12
    if root_pc is not None:
        result["expected_root_pc"] = root_pc
        result["expected_root_name"] = _PC_NAMES[root_pc]
        result["root_pc_match"] = dominant_pc == root_pc
        result["root_pc_pct"] = round(float(hist[root_pc]) * 100, 1)
        # Check if the root mass leaked to neighbours
        left = (root_pc - 1) % 12
        right = (root_pc + 1) % 12
        result["root_neighbor_pct"] = round(float(hist[left] + hist[right]) * 100, 1)

    scale_pcs = reference_pitch.get("scale_pcs")
    if scale_pcs is not None:
        in_scale = sum(float(hist[pc]) for pc in scale_pcs)
        result["in_scale_pct"] = round(in_scale * 100, 1)
        # Build reference histogram for comparison
        ref_hist = np.zeros(12, dtype=np.float64)
        for pc in scale_pcs:
            ref_hist[pc] = 1.0
        ref_hist /= ref_hist.sum()
        result["scale_overlap"] = round(_histogram_intersection(hist, ref_hist), 4)

        # Per-class report: expected vs got (including ±1 neighbours)
        per_class: list[dict[str, Any]] = []
        for pc in scale_pcs:
            left = (pc - 1) % 12
            right = (pc + 1) % 12
            per_class.append({
                "expected": _PC_NAMES[pc],
                "pc": pc,
                "pct": round(float(hist[pc]) * 100, 1),
                "neighbor_left_pct": round(float(hist[left]) * 100, 1),
                "neighbor_right_pct": round(float(hist[right]) * 100, 1),
            })
        result["per_class"] = per_class

    return result


def evaluate_cross_section_consistency(
    notes: list[dict[str, Any]],
    activity: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare pitch-class, contour, and rhythm across active sections.

    High consistency means the extraction is internally stable even if the
    absolute register is uncertain.
    """
    active_sections = [s for s in activity if s["active"]]
    if len(active_sections) < 2:
        return {"section_count": len(active_sections), "pairs": []}

    # Gather per-section features
    sec_data: list[dict[str, Any]] = []
    for sec in active_sections:
        sec_notes = sorted(
            _events_in_range(notes, sec["start_s"], sec["end_s"]),
            key=lambda n: n["onset_s"],
        )
        if len(sec_notes) < 3:
            continue
        pc_hist = _pitch_class_histogram(sec_notes)
        contour = _contour_direction(sec_notes)
        iois = _ioi_vector(sec_notes)
        midis = [n["midi"] for n in sec_notes]
        sec_data.append({
            "label": sec.get("label", ""),
            "start_s": sec["start_s"],
            "end_s": sec["end_s"],
            "n_notes": len(sec_notes),
            "density": round(len(sec_notes) / (sec["end_s"] - sec["start_s"]), 2),
            "pc_hist": pc_hist,
            "contour_props": _contour_proportions(contour),
            "ioi_median": round(float(np.median(iois)), 4) if iois else 0.0,
            "midi_median": round(float(np.median(midis)), 1),
            "midi_std": round(float(np.std(midis)), 2),
        })

    # Pairwise comparisons
    pairs: list[dict[str, Any]] = []
    for i in range(len(sec_data)):
        for j in range(i + 1, len(sec_data)):
            a, b = sec_data[i], sec_data[j]
            pc_overlap = _histogram_intersection(a["pc_hist"], b["pc_hist"])

            # Contour similarity: absolute difference in proportions
            cp_a, cp_b = a["contour_props"], b["contour_props"]
            contour_sim = 1.0 - (
                abs(cp_a["up"] - cp_b["up"])
                + abs(cp_a["down"] - cp_b["down"])
                + abs(cp_a["same"] - cp_b["same"])
            ) / 2.0

            # IOI ratio
            ioi_ratio = (
                min(a["ioi_median"], b["ioi_median"]) / max(a["ioi_median"], b["ioi_median"])
                if a["ioi_median"] > 0 and b["ioi_median"] > 0
                else 0.0
            )

            # Density ratio
            density_ratio = (
                min(a["density"], b["density"]) / max(a["density"], b["density"])
                if a["density"] > 0 and b["density"] > 0
                else 0.0
            )

            pairs.append({
                "sections": f"{a['label']} vs {b['label']}",
                "pc_overlap": round(pc_overlap, 4),
                "contour_similarity": round(contour_sim, 4),
                "ioi_ratio": round(ioi_ratio, 4),
                "density_ratio": round(density_ratio, 4),
            })

    # Averages
    avg_pc = float(np.mean([p["pc_overlap"] for p in pairs])) if pairs else 0.0
    avg_contour = float(np.mean([p["contour_similarity"] for p in pairs])) if pairs else 0.0
    avg_ioi = float(np.mean([p["ioi_ratio"] for p in pairs])) if pairs else 0.0

    return {
        "section_count": len(sec_data),
        "pairs": pairs,
        "avg_pc_overlap": round(avg_pc, 4),
        "avg_contour_similarity": round(avg_contour, 4),
        "avg_ioi_ratio": round(avg_ioi, 4),
        "sections": [
            {k: v for k, v in sd.items() if k != "pc_hist"}
            for sd in sec_data
        ],
    }


def evaluate_register_stability(
    notes: list[dict[str, Any]],
    *,
    activity: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Octave-jump rate, register drift, and pitch variance.

    All metrics are reference-free — they measure internal consistency
    of the extraction.
    """
    if activity:
        check: list[dict[str, Any]] = []
        for sec in activity:
            if sec["active"]:
                check.extend(_events_in_range(notes, sec["start_s"], sec["end_s"]))
    else:
        check = notes
    if len(check) < 2:
        return {"checked": len(check)}

    midis = [n["midi"] for n in check]
    intervals = [midis[i + 1] - midis[i] for i in range(len(midis) - 1)]
    abs_intervals = [abs(iv) for iv in intervals]

    octave_jumps = sum(1 for iv in abs_intervals if iv >= 10)
    large_jumps = sum(1 for iv in abs_intervals if iv >= 7)

    return {
        "checked": len(check),
        "midi_std": round(float(np.std(midis)), 2),
        "midi_range": round(float(max(midis) - min(midis)), 1),
        "octave_jump_count": octave_jumps,
        "octave_jump_pct": round(100.0 * octave_jumps / len(intervals), 1),
        "large_jump_count": large_jumps,
        "large_jump_pct": round(100.0 * large_jumps / len(intervals), 1),
        "median_abs_interval": round(float(np.median(abs_intervals)), 2),
        "mean_abs_interval": round(float(np.mean(abs_intervals)), 2),
    }


# ── Layer-level evaluation ──


def evaluate_layer(
    layer_data: dict[str, Any],
    reference: dict[str, Any],
    *,
    ref_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate a single extracted layer against its reference.

    If *ref_dir* is provided and a ``<layer>_notes.mid`` file exists there,
    note-level transcription metrics are also computed.
    """
    layer = reference["layer"]

    if layer == "drums":
        events = layer_data.get("hits", [])
        time_key = "t"
    else:
        events = layer_data.get("notes", [])
        time_key = "onset_s"

    result: dict[str, Any] = {
        "layer": layer,
        "confidence": reference.get("confidence", "unknown"),
        "ref_source": reference.get("source", "unknown"),
        "extraction_source": layer_data.get("source", "unknown"),
        "event_count": len(events),
    }

    activity = reference.get("activity")
    if activity:
        result["activity"] = evaluate_activity(events, activity, time_key=time_key)

    pitch = reference.get("pitch")

    # ── Octave-invariant metrics (high trust) ──
    if pitch and layer in ("bass", "vocals") and events:
        result["octave_invariant"] = {}
        result["octave_invariant"]["pitch_class"] = evaluate_pitch_class(
            events, pitch, activity=activity,
        )
        if activity:
            result["octave_invariant"]["cross_section"] = evaluate_cross_section_consistency(
                events, activity,
            )
        result["octave_invariant"]["register_stability"] = evaluate_register_stability(
            events, activity=activity,
        )

    # ── Octave-sensitive metrics (provisional) ──
    if pitch and layer in ("bass", "vocals") and events:
        result["octave_sensitive"] = {}
        result["octave_sensitive"]["pitch_range"] = evaluate_pitch_range(
            events,
            (pitch["range"][0], pitch["range"][1]),
            activity=activity,
        )

    ref_onsets = reference.get("onsets")
    if ref_onsets is not None:
        ext_times = [e[time_key] for e in events]
        ref_times = [o["t"] for o in ref_onsets]
        result["onsets"] = evaluate_onsets(ext_times, ref_times)

    # ── Note-level transcription (requires <layer>_notes.mid) ──
    if ref_dir is not None:
        midi_path = ref_dir / f"{layer}_notes.mid"
        if midi_path.exists() and events:
            ref_notes = _load_midi_reference_notes(midi_path, layer)
            if ref_notes:
                # For drums: onset-only matching (pitch = GM note, not musical pitch)
                if layer == "drums":
                    det_hits = [{"onset_s": e["t"], "midi": 0.0} for e in events]
                    ref_hits = [{"onset_s": n["onset_s"], "midi": 0.0} for n in ref_notes]
                    nt = evaluate_note_transcription(
                        det_hits, ref_hits, onset_tol_s=0.05, pitch_tol_st=999.0,
                    )
                else:
                    nt = evaluate_note_transcription(events, ref_notes)
                result["note_transcription"] = nt

    return result


# ── Top-level evaluation ──


def evaluate_reduced(
    reduced: dict[str, Any],
    ref_dir: Path,
) -> dict[str, Any]:
    """Evaluate all layers of a reduced dict against available references."""
    results: dict[str, Any] = {"layers": {}}

    for layer in ("drums", "vocals", "bass"):
        ref_path = ref_dir / f"{layer}.json"
        if not ref_path.exists():
            continue
        layer_data = reduced.get(layer)
        if not layer_data:
            results["layers"][layer] = {"error": "no extraction data", "layer": layer}
            continue
        reference = load_reference(ref_path)
        results["layers"][layer] = evaluate_layer(layer_data, reference, ref_dir=ref_dir)

    return results


# ── Report formatting ──


def format_report(results: dict[str, Any]) -> str:
    """Format evaluation results as a human-readable report."""
    lines: list[str] = []

    for layer_name in ("drums", "vocals", "bass"):
        lr = results.get("layers", {}).get(layer_name)
        if lr is None:
            continue

        if "error" in lr:
            lines.append(f"\n--- {layer_name.capitalize()} ---")
            lines.append(f"  {lr['error']}")
            continue

        conf = lr.get("confidence", "?")
        ref_src = lr.get("ref_source", "?")
        ext_src = lr.get("extraction_source", "?")
        n = lr.get("event_count", 0)

        lines.append(f"\n--- {layer_name.capitalize()} ({conf}: {ref_src}) ---")
        lines.append(f"  Source: {ext_src} | {n} events")

        act = lr.get("activity")
        if act:
            lines.append(
                f"  Activity: F1={act['f1']:.2f} "
                f"(P={act['precision']:.2f} R={act['recall']:.2f})"
            )
            for sr in act.get("sections", []):
                status = "OK" if sr["match"] else "MISS"
                ref_str = "active" if sr["ref_active"] else "silent"
                det_str = "active" if sr["detected_active"] else "silent"
                lines.append(
                    f"    [{status}] {sr['start_s']:6.1f}-{sr['end_s']:6.1f}s "
                    f"({sr.get('label', ''):20s}): ref={ref_str:6s} det={det_str:6s} "
                    f"({sr['event_count']} events)"
                )
            lines.append(
                f"  Silent FP: {act['silent_fp_count']} events "
                f"({act['silent_fp_rate']:.1%})"
            )

        # ── Octave-invariant findings (high trust) ──
        oi = lr.get("octave_invariant")
        if oi:
            lines.append("  [Octave-invariant — high trust]")

            pc = oi.get("pitch_class")
            if pc and pc.get("checked", 0) > 0:
                lines.append(
                    f"    Dominant PC: {pc['dominant_pc_name']} "
                    f"({pc['dominant_pc_pct']:.1f}%)"
                )
                if "expected_root_name" in pc:
                    match_str = "YES" if pc.get("root_pc_match") else "NO"
                    lines.append(
                        f"    Root match: {match_str} "
                        f"(expected {pc['expected_root_name']}, "
                        f"got {pc['root_pc_pct']:.1f}% exact, "
                        f"{pc['root_neighbor_pct']:.1f}% in neighbors)"
                    )
                if "in_scale_pct" in pc:
                    lines.append(f"    In expected scale: {pc['in_scale_pct']:.1f}%")

                per_class = pc.get("per_class", [])
                if per_class:
                    for pcc in per_class:
                        lines.append(
                            f"      {pcc['expected']:2s}: "
                            f"{pcc['pct']:5.1f}% exact, "
                            f"+{pcc['neighbor_left_pct']:.1f}%/{pcc['neighbor_right_pct']:.1f}% "
                            f"in +-1 neighbors"
                        )

            cs = oi.get("cross_section")
            if cs and cs.get("section_count", 0) >= 2:
                lines.append(
                    f"    Cross-section consistency ({cs['section_count']} sections):"
                )
                lines.append(
                    f"      PC overlap: {cs['avg_pc_overlap']:.2f}  "
                    f"Contour sim: {cs['avg_contour_similarity']:.2f}  "
                    f"IOI ratio: {cs['avg_ioi_ratio']:.2f}"
                )
                secs = cs.get("sections", [])
                for sd in secs:
                    lines.append(
                        f"      {sd['label']:20s}: "
                        f"{sd['n_notes']:3d} notes, "
                        f"density={sd['density']:.1f}/s, "
                        f"IOI={sd['ioi_median']:.3f}s, "
                        f"MIDI std={sd['midi_std']:.1f}"
                    )
                for pair in cs.get("pairs", []):
                    lines.append(
                        f"      {pair['sections']}: "
                        f"PC={pair['pc_overlap']:.2f} "
                        f"contour={pair['contour_similarity']:.2f} "
                        f"IOI={pair['ioi_ratio']:.2f} "
                        f"density={pair['density_ratio']:.2f}"
                    )

            rs = oi.get("register_stability")
            if rs and rs.get("checked", 0) >= 2:
                lines.append(
                    f"    Register: MIDI std={rs['midi_std']:.1f}, "
                    f"range={rs['midi_range']:.0f}st"
                )
                lines.append(
                    f"    Octave jumps (>=10st): {rs['octave_jump_count']} "
                    f"({rs['octave_jump_pct']:.1f}%)"
                )
                lines.append(
                    f"    Large jumps (>=7st): {rs['large_jump_count']} "
                    f"({rs['large_jump_pct']:.1f}%)"
                )
                lines.append(
                    f"    Median interval: {rs['median_abs_interval']:.1f}st"
                )

        # ── Octave-sensitive findings (provisional) ──
        os_ = lr.get("octave_sensitive")
        if os_:
            lines.append("  [Octave-sensitive — provisional, requires verified register]")
            pr = os_.get("pitch_range")
            if pr:
                lines.append(
                    f"    Pitch range: {pr['in_range_pct']:.1f}% in range "
                    f"| median MIDI={pr['midi_median']:.1f}"
                )
                if pr["below_range_pct"] > 0:
                    lines.append(f"      Below range: {pr['below_range_pct']:.1f}%")
                if pr["above_range_pct"] > 0:
                    lines.append(f"      Above range: {pr['above_range_pct']:.1f}%")

        ons = lr.get("onsets")
        if ons:
            lines.append(
                f"  Onsets: F1={ons['f1']:.2f} "
                f"(P={ons['precision']:.2f} R={ons['recall']:.2f}) "
                f"@ {ons['tolerance_s'] * 1000:.0f}ms tolerance"
            )

        nt = lr.get("note_transcription")
        if nt:
            lines.append("  [Note-level transcription — vs MIDI reference]")
            lines.append(
                f"    Onset F1:  {nt['onset_f1']:.3f} "
                f"(P={nt['onset_precision']:.3f} R={nt['onset_recall']:.3f}) "
                f"| timing error: {nt['mean_onset_error_ms']:.1f}ms avg"
            )
            lines.append(
                f"    Note  F1:  {nt['note_f1']:.3f} "
                f"(P={nt['note_precision']:.3f} R={nt['note_recall']:.3f}) "
                f"| pitch_tol=±{nt['pitch_tol_st']:.0f}st"
            )
            lines.append(
                f"    OctInv F1: {nt['note_f1_octave_invariant']:.3f} "
                f"| pitch accuracy: {nt['pitch_accuracy']:.1%} "
                f"| avg pitch error: {nt['mean_pitch_error_st']:.2f}st"
            )
            lines.append(
                f"    Counts: ref={nt['ref_note_count']} det={nt['det_note_count']} "
                f"fragmentation={nt['fragmentation_ratio']:.2f}x"
            )

    return "\n".join(lines)
