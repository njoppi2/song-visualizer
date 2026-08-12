"""Compare bass extractors: basic-pitch vs pYIN on Feel Good Inc.

Four extraction variants (raw vs final x basic-pitch vs pYIN) evaluated
against the silver reference to isolate pitch-class smearing by source.

Usage:
    .songviz/venv/bin/python experiments/compare_bass_extractors.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from songviz.features import bass_note_events_basic_pitch, bass_pitch_hz
from songviz.reduction import extract_bass_notes, _notes_from_pitch_track
from songviz.eval import (
    evaluate_layer,
    evaluate_pitch_class,
    evaluate_cross_section_consistency,
    evaluate_register_stability,
    evaluate_activity,
    _events_in_range,
    _pitch_class_histogram,
    load_reference,
    _PC_NAMES,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SONG = "Gorillaz - Feel Good Inc (featuring De La Soul)"
RESULTS_DIR = ROOT / "experiments" / "results" / "feel_good_inc"
BASS_STEM = ROOT / "outputs" / SONG / "stems" / "bass.wav"
REFERENCE_PATH = ROOT / "benchmark" / "references" / "feel-good-inc" / "bass.json"

SR = 22050


def _load_mono(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y


# ---------------------------------------------------------------------------
# Extract all four variants
# ---------------------------------------------------------------------------

def extract_all_four(bass_stem_path: Path) -> dict[str, dict]:
    """Produce bp_raw, bp_final, pyin_raw, pyin_final note dicts."""
    y = _load_mono(bass_stem_path)

    # --- basic-pitch ---
    print("  Running basic-pitch...")
    raw_events = bass_note_events_basic_pitch(str(bass_stem_path))

    bp_raw = {
        "source": "basic_pitch_raw",
        "notes": [
            {
                "onset_s": round(float(ev["start_s"]), 4),
                "offset_s": round(float(ev["end_s"]), 4),
                "midi": round(float(ev["midi"]), 2),
                "velocity": round(float(ev["velocity"]), 4),
            }
            for ev in raw_events
        ],
    }

    bp_final = extract_bass_notes(raw_events, None, y, SR)

    # --- pYIN ---
    print("  Running pYIN...")
    pitch_hz_track = bass_pitch_hz(y, SR)

    pyin_raw = _notes_from_pitch_track(
        pitch_hz_track, y, SR,
        source="pyin_raw",
        hop_length=512,
        max_gap_frames=3,
    )

    pyin_final = extract_bass_notes(None, pitch_hz_track, y, SR)

    return {
        "bp_raw": bp_raw,
        "bp_final": bp_final,
        "pyin_raw": pyin_raw,
        "pyin_final": pyin_final,
        "_audio": y,
        "_pitch_hz": pitch_hz_track,
    }


# ---------------------------------------------------------------------------
# Evaluate a single variant
# ---------------------------------------------------------------------------

def evaluate_variant(
    notes_dict: dict,
    reference: dict,
) -> dict:
    """Run evaluate_layer on a variant, returning the full result dict."""
    return evaluate_layer(notes_dict, reference)


# ---------------------------------------------------------------------------
# Per-section pitch-class breakdown
# ---------------------------------------------------------------------------

def per_section_pitch_class(
    notes: list[dict],
    activity: list[dict],
) -> list[dict]:
    """PC histogram for each active section."""
    results = []
    for sec in activity:
        if not sec["active"]:
            continue
        sec_notes = _events_in_range(notes, sec["start_s"], sec["end_s"])
        if len(sec_notes) < 3:
            results.append({"label": sec.get("label", ""), "n_notes": len(sec_notes)})
            continue
        hist = _pitch_class_histogram(sec_notes)
        dominant_pc = int(np.argmax(hist))
        results.append({
            "label": sec.get("label", ""),
            "n_notes": len(sec_notes),
            "dominant_pc": _PC_NAMES[dominant_pc],
            "dominant_pct": round(float(hist[dominant_pc]) * 100, 1),
            "G_pct": round(float(hist[7]) * 100, 1),
            "F#_pct": round(float(hist[6]) * 100, 1),
            "G#_pct": round(float(hist[8]) * 100, 1),
            "Bb_pct": round(float(hist[10]) * 100, 1),
            "D_pct": round(float(hist[2]) * 100, 1),
            "Eb_pct": round(float(hist[3]) * 100, 1),
            "in_scale_pct": round(sum(float(hist[pc]) for pc in [7, 10, 2, 3]) * 100, 1),
        })
    return results


# ---------------------------------------------------------------------------
# pYIN pitch track stats
# ---------------------------------------------------------------------------

def pyin_pitch_track_stats(
    pitch_hz: np.ndarray,
    activity: list[dict],
) -> dict:
    """Raw pitch track voiced ratio and median Hz per active section."""
    hop = 512
    times = librosa.frames_to_time(np.arange(len(pitch_hz)), sr=SR, hop_length=hop)
    overall_voiced = float(np.isfinite(pitch_hz).mean())

    sections = []
    for sec in activity:
        if not sec["active"]:
            continue
        mask = (times >= sec["start_s"]) & (times < sec["end_s"])
        seg = pitch_hz[mask]
        voiced = seg[np.isfinite(seg)]
        sections.append({
            "label": sec.get("label", ""),
            "voiced_ratio": round(float(np.isfinite(seg).mean()), 4) if seg.size > 0 else 0.0,
            "median_hz": round(float(np.median(voiced)), 1) if voiced.size > 0 else None,
            "voiced_frames": int(voiced.size),
            "total_frames": int(seg.size),
        })

    return {"overall_voiced_ratio": round(overall_voiced, 4), "sections": sections}


# ---------------------------------------------------------------------------
# Print comparison
# ---------------------------------------------------------------------------

def _get_metric(result: dict, *keys: str, default=None):
    """Navigate nested dicts safely."""
    d = result
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is default:
            return default
    return d


def print_comparison(
    all_results: dict,
    variants: dict[str, dict],
    pyin_stats: dict,
    activity: list[dict],
) -> None:
    """Print the 4-column comparison table."""
    names = ["bp_raw", "bp_final", "pyin_raw", "pyin_final"]

    # Precompute full PC histograms from notes (active sections only)
    pc_hists: dict[str, np.ndarray] = {}
    for name in names:
        notes = variants[name].get("notes", [])
        active_notes: list[dict] = []
        for sec in activity:
            if sec["active"]:
                active_notes.extend(_events_in_range(notes, sec["start_s"], sec["end_s"]))
        pc_hists[name] = _pitch_class_histogram(active_notes)
    W = 12  # column width

    def _row(label: str, values: list) -> str:
        vals = "".join(f"{str(v):>{W}}" for v in values)
        return f"  {label:<30}{vals}"

    def _val(name: str, *keys: str, fmt: str = ".1f", default="—"):
        v = _get_metric(all_results[name], *keys)
        if v is None:
            return default
        if isinstance(v, str):
            return v
        return f"{v:{fmt}}"

    print("=" * 78)
    print("BASS EXTRACTOR COMPARISON: Feel Good Inc")
    print("=" * 78)

    # Header
    header = "".join(f"{n:>{W}}" for n in names)
    print(f"  {'':30}{header}")
    print("  " + "-" * (30 + W * 4))

    # --- Coverage ---
    print("\n--- Coverage ---")
    print(_row("Note count", [
        _get_metric(all_results[n], "event_count", default="—") for n in names
    ]))
    for n in names:
        r = all_results[n]
        act = r.get("activity")
        if act:
            # compute active density
            total_active_s = 0
            for sec in act.get("sections", []):
                if sec.get("ref_active"):
                    total_active_s += sec["end_s"] - sec["start_s"]
            ec = r.get("event_count", 0)
            all_results[n]["_active_density"] = round(ec / total_active_s, 2) if total_active_s > 0 else 0.0
    print(_row("Active density (/s)", [
        _get_metric(all_results[n], "_active_density", default="—") for n in names
    ]))

    # --- Activity ---
    print("\n--- Activity ---")
    print(_row("F1", [_val(n, "activity", "f1", fmt=".2f") for n in names]))
    print(_row("Precision", [_val(n, "activity", "precision", fmt=".2f") for n in names]))
    print(_row("Recall", [_val(n, "activity", "recall", fmt=".2f") for n in names]))
    print(_row("Silent FP count", [_val(n, "activity", "silent_fp_count", fmt="d") for n in names]))
    print(_row("Silent FP rate", [
        f"{_get_metric(all_results[n], 'activity', 'silent_fp_rate', default=0)*100:.1f}%"
        for n in names
    ]))

    # --- Pitch-Class Precision ---
    print("\n--- Pitch-Class Precision (octave-invariant) ---")
    for label, *keys in [
        ("Root G pct", "octave_invariant", "pitch_class", "root_pc_pct"),
        ("In-scale pct", "octave_invariant", "pitch_class", "in_scale_pct"),
        ("Dominant PC", "octave_invariant", "pitch_class", "dominant_pc_name"),
        ("Dominant PC pct", "octave_invariant", "pitch_class", "dominant_pc_pct"),
    ]:
        vals = []
        for n in names:
            v = _get_metric(all_results[n], *keys)
            if v is None:
                vals.append("—")
            elif isinstance(v, str):
                vals.append(v)
            else:
                vals.append(f"{v:.1f}%")
        print(_row(label, vals))

    # Neighbour detail for root G
    print(_row("F# neighbor", [
        f"{pc_hists[n][6]*100:.1f}%" for n in names
    ]))
    print(_row("G# neighbor", [
        f"{pc_hists[n][8]*100:.1f}%" for n in names
    ]))

    # Per expected scale note
    for pc, pc_name in [(7, "G"), (10, "Bb"), (2, "D"), (3, "Eb")]:
        print(_row(f"  {pc_name} (pc={pc})", [
            f"{pc_hists[n][pc]*100:.1f}%" for n in names
        ]))

    # --- Contour & Cross-Section ---
    print("\n--- Contour & Cross-Section Consistency ---")
    for label, key in [
        ("Avg PC overlap", "avg_pc_overlap"),
        ("Contour sim", "avg_contour_similarity"),
        ("IOI ratio", "avg_ioi_ratio"),
    ]:
        print(_row(label, [
            _val(n, "octave_invariant", "cross_section", key, fmt=".2f") for n in names
        ]))

    # --- Register Stability ---
    print("\n--- Register Stability ---")
    for label, key, fmt in [
        ("MIDI std", "midi_std", ".1f"),
        ("Octave jump %", "octave_jump_pct", ".1f"),
        ("Median interval", "median_abs_interval", ".1f"),
        ("MIDI range", "midi_range", ".0f"),
    ]:
        vals = [_val(n, "octave_invariant", "register_stability", key, fmt=fmt) for n in names]
        if key == "median_abs_interval":
            vals = [f"{v}st" if v != "—" else v for v in vals]
        elif key == "octave_jump_pct":
            vals = [f"{v}%" if v != "—" else v for v in vals]
        elif key == "midi_range":
            vals = [f"{v}st" if v != "—" else v for v in vals]
        print(_row(label, vals))

    # --- Octave-sensitive ---
    print("\n--- Octave-Sensitive (provisional) ---")
    for label, key, fmt in [
        ("In-range pct", "in_range_pct", ".1f"),
        ("MIDI median", "midi_median", ".1f"),
        ("Below range %", "below_range_pct", ".1f"),
    ]:
        print(_row(label, [
            _val(n, "octave_sensitive", "pitch_range", key, fmt=fmt) for n in names
        ]))

    # --- pYIN pitch track stats ---
    print("\n--- pYIN Pitch Track Stats ---")
    print(f"  Overall voiced ratio: {pyin_stats['overall_voiced_ratio']:.4f}")
    for sec in pyin_stats["sections"]:
        print(
            f"    {sec['label']:20s}: voiced={sec['voiced_ratio']:.2f} "
            f"({sec['voiced_frames']}/{sec['total_frames']} frames)"
            + (f", median={sec['median_hz']:.1f} Hz" if sec["median_hz"] else "")
        )


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def print_verdict(all_results: dict) -> dict:
    """Print and return a verdict on which extractor is better."""

    def _metric(name: str, *keys: str):
        return _get_metric(all_results[name], *keys)

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    verdicts = {}

    # 1. Pitch-class precision
    bp_in_scale = _metric("bp_final", "octave_invariant", "pitch_class", "in_scale_pct") or 0
    pyin_in_scale = _metric("pyin_final", "octave_invariant", "pitch_class", "in_scale_pct") or 0
    bp_root = _metric("bp_final", "octave_invariant", "pitch_class", "root_pc_pct") or 0
    pyin_root = _metric("pyin_final", "octave_invariant", "pitch_class", "root_pc_pct") or 0

    if pyin_in_scale > bp_in_scale + 5 and pyin_root > bp_root + 3:
        pc_winner = "pyin"
    elif bp_in_scale > pyin_in_scale + 5 and bp_root > pyin_root + 3:
        pc_winner = "basic_pitch"
    else:
        pc_winner = "tie"
    verdicts["pitch_class_winner"] = pc_winner
    print(f"\n  1. Pitch-class precision: {pc_winner}")
    print(f"     bp_final  in-scale={bp_in_scale:.1f}%, root G={bp_root:.1f}%")
    print(f"     pyin_final in-scale={pyin_in_scale:.1f}%, root G={pyin_root:.1f}%")

    # 2. Contour / cross-section consistency
    bp_cs = _metric("bp_final", "octave_invariant", "cross_section", "avg_pc_overlap") or 0
    pyin_cs = _metric("pyin_final", "octave_invariant", "cross_section", "avg_pc_overlap") or 0
    bp_contour = _metric("bp_final", "octave_invariant", "cross_section", "avg_contour_similarity") or 0
    pyin_contour = _metric("pyin_final", "octave_invariant", "cross_section", "avg_contour_similarity") or 0

    cs_score_bp = (bp_cs + bp_contour) / 2
    cs_score_pyin = (pyin_cs + pyin_contour) / 2
    if cs_score_pyin > cs_score_bp + 0.03:
        cs_winner = "pyin"
    elif cs_score_bp > cs_score_pyin + 0.03:
        cs_winner = "basic_pitch"
    else:
        cs_winner = "tie"
    verdicts["consistency_winner"] = cs_winner
    print(f"\n  2. Cross-section consistency: {cs_winner}")
    print(f"     bp_final  PC overlap={bp_cs:.2f}, contour sim={bp_contour:.2f}")
    print(f"     pyin_final PC overlap={pyin_cs:.2f}, contour sim={pyin_contour:.2f}")

    # 3. Silence behavior / false positives
    bp_fp = _metric("bp_final", "activity", "silent_fp_count") or 0
    pyin_fp = _metric("pyin_final", "activity", "silent_fp_count") or 0
    if pyin_fp < bp_fp * 0.7:
        fp_winner = "pyin"
    elif bp_fp < pyin_fp * 0.7:
        fp_winner = "basic_pitch"
    else:
        fp_winner = "tie"
    verdicts["silence_winner"] = fp_winner
    print(f"\n  3. Silence behavior: {fp_winner}")
    print(f"     bp_final  silent FP={bp_fp}")
    print(f"     pyin_final silent FP={pyin_fp}")

    # 4. Overall
    bp_f1 = _metric("bp_final", "activity", "f1") or 0
    pyin_f1 = _metric("pyin_final", "activity", "f1") or 0
    bp_notes = _metric("bp_final", "event_count") or 0
    pyin_notes = _metric("pyin_final", "event_count") or 0

    # Decision logic
    both_smearing = (bp_in_scale < 60 and pyin_in_scale < 60)
    pyin_clearly_better = (
        pc_winner == "pyin"
        and pyin_notes > 30  # acceptable coverage
        and pyin_f1 >= bp_f1 * 0.8  # acceptable activity
    )

    if pyin_clearly_better:
        overall = "SWITCH to pYIN as default"
    elif both_smearing:
        overall = "KEEP basic-pitch as default — pitch-class precision UNRESOLVED (stem quality issue)"
    elif pc_winner == "basic_pitch":
        overall = "KEEP basic-pitch as default"
    else:
        overall = "KEEP basic-pitch as default — pitch-class precision unresolved"

    verdicts["overall"] = overall
    print(f"\n  4. OVERALL: {overall}")
    print(f"     bp_final:   {bp_notes} notes, F1={bp_f1:.2f}, in-scale={bp_in_scale:.1f}%")
    print(f"     pyin_final: {pyin_notes} notes, F1={pyin_f1:.2f}, in-scale={pyin_in_scale:.1f}%")

    return verdicts


# ---------------------------------------------------------------------------
# Per-section pitch-class tables
# ---------------------------------------------------------------------------

def print_per_section(all_variants: dict, activity: list[dict]) -> dict:
    """Print per-section PC tables for all 4 variants."""
    names = ["bp_raw", "bp_final", "pyin_raw", "pyin_final"]

    print("\n" + "=" * 78)
    print("PER-SECTION PITCH-CLASS BREAKDOWN")
    print("=" * 78)

    per_section_data = {}
    for name in names:
        notes = all_variants[name].get("notes", [])
        per_section_data[name] = per_section_pitch_class(notes, activity)

    active_labels = [s.get("label", "") for s in activity if s["active"]]

    for i, label in enumerate(active_labels):
        print(f"\n  Section: {label}")
        W = 12
        header = "".join(f"{n:>{W}}" for n in names)
        print(f"    {'':20}{header}")
        print(f"    " + "-" * (20 + W * 4))

        for metric in ["n_notes", "dominant_pc", "G_pct", "F#_pct", "G#_pct",
                        "Bb_pct", "D_pct", "Eb_pct", "in_scale_pct"]:
            vals = []
            for name in names:
                sec_data = per_section_data[name]
                if i < len(sec_data):
                    v = sec_data[i].get(metric, "—")
                    if isinstance(v, float):
                        vals.append(f"{v:.1f}%")
                    else:
                        vals.append(str(v))
                else:
                    vals.append("—")
            row_vals = "".join(f"{v:>{W}}" for v in vals)
            print(f"    {metric:20}{row_vals}")

    return per_section_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading bass stem and reference...")
    reference = load_reference(REFERENCE_PATH)
    activity = reference.get("activity", [])

    variants = extract_all_four(BASS_STEM)
    audio = variants.pop("_audio")
    pitch_hz = variants.pop("_pitch_hz")

    # Evaluate all 4 variants
    print("\n  Evaluating variants...")
    all_results = {}
    for name, notes_dict in variants.items():
        all_results[name] = evaluate_variant(notes_dict, reference)

    # pYIN pitch track stats
    pyin_stats = pyin_pitch_track_stats(pitch_hz, activity)

    # Print tables
    print_comparison(all_results, variants, pyin_stats, activity)
    print_per_section(variants, activity)
    verdicts = print_verdict(all_results)

    # Save JSON
    output = {
        "variants": {},
        "pyin_pitch_track_stats": pyin_stats,
        "verdicts": verdicts,
    }
    for name in ["bp_raw", "bp_final", "pyin_raw", "pyin_final"]:
        output["variants"][name] = {
            "eval": all_results[name],
            "per_section_pc": per_section_pitch_class(
                variants[name].get("notes", []), activity,
            ),
        }

    out_path = RESULTS_DIR / "bass_extractor_comparison.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str) + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
