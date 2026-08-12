"""Bass pitch diagnostic pass 3: Mix vs Stem CQT comparison.

Resolves whether the G2 spectral valley found in pass 2 is:
  (a) A Demucs separation artifact — G2 valley in stem but peak in mix
  (b/c) Inherent to the recording — G2 valley in both mix and stem

Caveat: the mix CQT includes kick drum and all instruments, not just bass.
We are comparing *relative spectral shape* (valley vs peak), not absolute
magnitude.

Usage:
    .songviz/venv/bin/python experiments/diagnose_bass_pitch_v3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SONG = "Gorillaz - Feel Good Inc (featuring De La Soul)"
MIX_PATH = ROOT / "songs" / f"{SONG}.flac"
BASS_STEM = ROOT / "outputs" / SONG / "stems" / "bass.wav"
RESULTS_DIR = ROOT / "experiments" / "results" / "feel_good_inc"

SR = 22050
HOP = 512

SECTIONS = {
    "verse1+chorus1": (12.5, 63.0),
    "chorus2": (95.0, 138.0),
}

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_REF_NOTES = {
    "D2": librosa.note_to_hz("D2"),
    "Eb2": librosa.note_to_hz("Eb2"),
    "F#2": librosa.note_to_hz("F#2"),
    "G2": librosa.note_to_hz("G2"),
    "G#2": librosa.note_to_hz("G#2"),
    "Bb2": librosa.note_to_hz("Bb2"),
    "B2": librosa.note_to_hz("B2"),
}

G2_HZ = float(librosa.note_to_hz("G2"))  # ~98.0


# ---------------------------------------------------------------------------
# Reused helpers from v2
# ---------------------------------------------------------------------------
def _load_section(y: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    return y[int(start_s * SR):int(end_s * SR)]


def _hz_to_midi(hz: float | np.ndarray) -> float | np.ndarray:
    return 69.0 + 12.0 * np.log2(np.maximum(hz, 1e-8) / 440.0)


def _midi_to_note(midi: float) -> str:
    m = int(round(midi))
    return f"{PC_NAMES[m % 12]}{m // 12 - 1}"


def _nearest_ref(hz: float) -> tuple[str, float]:
    best_name, best_cents = "", 9999.0
    for name, ref_hz in _REF_NOTES.items():
        cents = abs(1200 * np.log2(hz / ref_hz))
        if cents < best_cents:
            best_name, best_cents = name, cents
    return best_name, round(best_cents, 1)


def dense_cqt(y_section: np.ndarray) -> dict:
    """Report every CQT bin in [80–130 Hz] with 1/3-semitone resolution."""
    bpo = 36
    fmin = librosa.note_to_hz("C1")
    n_bins = 180

    C = np.abs(librosa.cqt(
        y=y_section.astype(np.float32), sr=SR,
        fmin=fmin, n_bins=n_bins, bins_per_octave=bpo, hop_length=HOP,
    ))
    mean_mag = C.mean(axis=1)
    freqs = fmin * 2.0 ** (np.arange(n_bins) / bpo)

    mask = (freqs >= 80.0) & (freqs <= 130.0)
    indices = np.where(mask)[0]

    bins_data = []
    for idx in indices:
        hz = float(freqs[idx])
        midi = float(_hz_to_midi(hz))
        mag = float(mean_mag[idx])
        nearest_note, cents_off = _nearest_ref(hz)
        bins_data.append({
            "bin_idx": int(idx),
            "hz": round(hz, 2),
            "midi": round(midi, 2),
            "note_approx": _midi_to_note(midi),
            "magnitude": round(mag, 6),
            "nearest_ref": nearest_note,
            "cents_from_ref": cents_off,
        })

    mags = np.array([b["magnitude"] for b in bins_data])
    peak_idx = int(np.argmax(mags))
    peak = bins_data[peak_idx]

    local_peaks = []
    for i in range(1, len(bins_data) - 1):
        if (bins_data[i]["magnitude"] > bins_data[i - 1]["magnitude"]
                and bins_data[i]["magnitude"] > bins_data[i + 1]["magnitude"]):
            local_peaks.append(bins_data[i])

    return {
        "range_hz": [80, 130],
        "bins_per_octave": bpo,
        "cents_per_bin": round(1200 / bpo, 1),
        "all_bins": bins_data,
        "global_peak": peak,
        "local_peaks": local_peaks,
    }


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------
def _is_local_minimum(bins_data: list[dict], target_hz: float) -> bool:
    """Check if the bin nearest to target_hz is a local minimum."""
    if len(bins_data) < 3:
        return False
    distances = [abs(b["hz"] - target_hz) for b in bins_data]
    nearest_idx = int(np.argmin(distances))
    if nearest_idx == 0 or nearest_idx == len(bins_data) - 1:
        return False
    mag = bins_data[nearest_idx]["magnitude"]
    return (mag < bins_data[nearest_idx - 1]["magnitude"]
            and mag < bins_data[nearest_idx + 1]["magnitude"])


def compare_cqt_pair(
    mix_cqt: dict, stem_cqt: dict,
) -> dict:
    """Compare mix and stem CQT for G2 valley analysis."""
    mix_bins = mix_cqt["all_bins"]
    stem_bins = stem_cqt["all_bins"]

    # Side-by-side table: match bins by Hz (same CQT config → same bins)
    side_by_side = []
    for mb, sb in zip(mix_bins, stem_bins):
        ratio = sb["magnitude"] / max(mb["magnitude"], 1e-10)
        side_by_side.append({
            "hz": mb["hz"],
            "note_approx": mb["note_approx"],
            "mix_mag": mb["magnitude"],
            "stem_mag": sb["magnitude"],
            "stem_mix_ratio": round(ratio, 4),
        })

    g2_valley_in_mix = _is_local_minimum(mix_bins, G2_HZ)
    g2_valley_in_stem = _is_local_minimum(stem_bins, G2_HZ)

    if g2_valley_in_stem and not g2_valley_in_mix:
        verdict = "G2_valley_separation_artifact"
        explanation = ("G2 is a valley in the Demucs stem but NOT in the mix. "
                       "Demucs is removing G2 energy during separation.")
    elif g2_valley_in_stem and g2_valley_in_mix:
        verdict = "G2_valley_inherent"
        explanation = ("G2 is a valley in BOTH mix and stem. The spectral dip "
                       "is inherent to the recording, not a Demucs artifact.")
    elif not g2_valley_in_stem:
        verdict = "G2_not_valley_in_stem"
        explanation = ("G2 is NOT a valley in the stem (contradicts v2 finding). "
                       "May depend on section or aggregation.")
    else:
        verdict = "inconclusive"
        explanation = "Unexpected combination of valley/peak states."

    return {
        "g2_valley_in_mix": g2_valley_in_mix,
        "g2_valley_in_stem": g2_valley_in_stem,
        "verdict": verdict,
        "explanation": explanation,
        "side_by_side": side_by_side,
    }


def _print_side_by_side(comparison: dict) -> None:
    """Print the side-by-side table."""
    print(f"\n  {'Hz':>8} {'note':>5} {'mix_mag':>10} {'stem_mag':>10} {'ratio':>8}")
    print(f"  {'—'*8} {'—'*5} {'—'*10} {'—'*10} {'—'*8}")
    for row in comparison["side_by_side"]:
        marker = " ◄G2" if abs(row["hz"] - G2_HZ) < 2.0 else ""
        print(f"  {row['hz']:>8.2f} {row['note_approx']:>5} "
              f"{row['mix_mag']:>10.6f} {row['stem_mag']:>10.6f} "
              f"{row['stem_mix_ratio']:>8.4f}{marker}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    for path, label in [(MIX_PATH, "Mix"), (BASS_STEM, "Bass stem")]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    print("Loading mix and bass stem...")
    y_mix, _ = librosa.load(str(MIX_PATH), sr=SR, mono=True)
    y_stem, _ = librosa.load(str(BASS_STEM), sr=SR, mono=True)
    print(f"  Mix:  {len(y_mix)/SR:.1f}s ({len(y_mix)} samples)")
    print(f"  Stem: {len(y_stem)/SR:.1f}s ({len(y_stem)} samples)")

    all_results = {}

    for sec_name, (start_s, end_s) in SECTIONS.items():
        print(f"\n{'='*70}")
        print(f"SECTION: {sec_name} ({start_s}–{end_s}s)")
        print(f"{'='*70}")

        mix_sec = _load_section(y_mix, start_s, end_s)
        stem_sec = _load_section(y_stem, start_s, end_s)

        print("\n  Computing dense CQT for mix...")
        mix_cqt = dense_cqt(mix_sec)
        print("  Computing dense CQT for stem...")
        stem_cqt = dense_cqt(stem_sec)

        comparison = compare_cqt_pair(mix_cqt, stem_cqt)

        _print_side_by_side(comparison)

        print(f"\n  G2 valley in mix?  {comparison['g2_valley_in_mix']}")
        print(f"  G2 valley in stem? {comparison['g2_valley_in_stem']}")
        print(f"  VERDICT: {comparison['verdict']}")
        print(f"  → {comparison['explanation']}")

        # Also print peaks for reference
        print(f"\n  Mix peaks:")
        for lp in mix_cqt["local_peaks"]:
            print(f"    {lp['hz']:.2f} Hz ({lp['note_approx']}) "
                  f"mag={lp['magnitude']:.6f}")
        print(f"  Stem peaks:")
        for lp in stem_cqt["local_peaks"]:
            print(f"    {lp['hz']:.2f} Hz ({lp['note_approx']}) "
                  f"mag={lp['magnitude']:.6f}")

        all_results[sec_name] = {
            "mix_cqt": mix_cqt,
            "stem_cqt": stem_cqt,
            "comparison": comparison,
        }

    # --- Overall verdict ---
    verdicts = [r["comparison"]["verdict"] for r in all_results.values()]
    if all(v == "G2_valley_inherent" for v in verdicts):
        overall = "G2_valley_inherent"
    elif all(v == "G2_valley_separation_artifact" for v in verdicts):
        overall = "G2_valley_separation_artifact"
    else:
        overall = f"mixed ({', '.join(verdicts)})"

    print(f"\n{'='*70}")
    print(f"OVERALL VERDICT: {overall}")
    print(f"{'='*70}")
    print(f"Caveat: mix CQT includes all instruments (kick, etc.), not just bass.")
    print(f"We compare spectral *shape* (valley vs peak), not absolute magnitude.")

    # --- Save JSON ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "diagnostic": "bass_pitch_v3_mix_vs_stem_cqt",
        "song": SONG,
        "sections": dict(SECTIONS),
        "caveat": ("Mix CQT includes all instruments. Comparison is relative "
                    "spectral shape (valley vs peak), not absolute magnitude."),
        "overall_verdict": overall,
        "per_section": {
            sec_name: {
                "mix_cqt": data["mix_cqt"],
                "stem_cqt": data["stem_cqt"],
                "g2_valley_in_mix": data["comparison"]["g2_valley_in_mix"],
                "g2_valley_in_stem": data["comparison"]["g2_valley_in_stem"],
                "verdict": data["comparison"]["verdict"],
                "explanation": data["comparison"]["explanation"],
                "side_by_side": data["comparison"]["side_by_side"],
            }
            for sec_name, data in all_results.items()
        },
    }
    out_path = RESULTS_DIR / "bass_pitch_diagnostic_v3.json"
    out_path.write_text(json.dumps(artifact, indent=2, default=str) + "\n")
    print(f"\nJSON artifact saved to {out_path}")


if __name__ == "__main__":
    main()
