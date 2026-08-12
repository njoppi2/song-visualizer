"""Bass pitch diagnostic pass 2: dense CQT + pYIN ablation.

Pass 1 had a critical gap: CQT was sampled only at semitone centers despite
having 1/3-semitone resolution.  And the pYIN failure (5% voiced) was observed
but its cause was not isolated.

This pass does exactly two things:
  D1 — Dense CQT: report ALL bins in [80–130 Hz] to see actual spectral shape
  D2 — pYIN ablation: with vs without harmonic preprocessing

Usage:
    .songviz/venv/bin/python experiments/diagnose_bass_pitch_v2.py
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
BASS_STEM = ROOT / "outputs" / SONG / "stems" / "bass.wav"
RESULTS_DIR = ROOT / "experiments" / "results" / "feel_good_inc"

SR = 22050
HOP = 512

SECTIONS = {
    "verse1+chorus1": (12.5, 63.0),
    "chorus2": (95.0, 138.0),
}

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Reference note frequencies for annotation
_REF_NOTES = {
    "D2": librosa.note_to_hz("D2"),    # 73.4
    "Eb2": librosa.note_to_hz("Eb2"),  # 77.8
    "F#2": librosa.note_to_hz("F#2"),  # 92.5
    "G2": librosa.note_to_hz("G2"),    # 98.0
    "G#2": librosa.note_to_hz("G#2"),  # 103.8
    "Bb2": librosa.note_to_hz("Bb2"),  # 116.5
    "B2": librosa.note_to_hz("B2"),    # 123.5
}


def _load_section(y: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    return y[int(start_s * SR):int(end_s * SR)]


def _hz_to_midi(hz: float | np.ndarray) -> float | np.ndarray:
    return 69.0 + 12.0 * np.log2(np.maximum(hz, 1e-8) / 440.0)


def _midi_to_note(midi: float) -> str:
    m = int(round(midi))
    return f"{PC_NAMES[m % 12]}{m // 12 - 1}"


def _nearest_ref(hz: float) -> tuple[str, float]:
    """Return (note_name, distance_in_cents) to nearest reference note."""
    best_name, best_cents = "", 9999.0
    for name, ref_hz in _REF_NOTES.items():
        cents = abs(1200 * np.log2(hz / ref_hz))
        if cents < best_cents:
            best_name, best_cents = name, cents
    return best_name, round(best_cents, 1)


# ---------------------------------------------------------------------------
# D1: Dense CQT inspection [80–130 Hz]
# ---------------------------------------------------------------------------
def dense_cqt(y_section: np.ndarray) -> dict:
    """Report every CQT bin in [80–130 Hz] with 1/3-semitone resolution."""
    bpo = 36  # 33.3 cents per bin
    fmin = librosa.note_to_hz("C1")
    n_bins = 180

    C = np.abs(librosa.cqt(
        y=y_section.astype(np.float32), sr=SR,
        fmin=fmin, n_bins=n_bins, bins_per_octave=bpo, hop_length=HOP,
    ))
    mean_mag = C.mean(axis=1)
    freqs = fmin * 2.0 ** (np.arange(n_bins) / bpo)

    # Select bins in [80, 130] Hz
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

    # Find peak
    mags = np.array([b["magnitude"] for b in bins_data])
    peak_idx = int(np.argmax(mags))
    peak = bins_data[peak_idx]

    # Find local peaks (bins higher than both neighbors)
    local_peaks = []
    for i in range(1, len(bins_data) - 1):
        if bins_data[i]["magnitude"] > bins_data[i-1]["magnitude"] and \
           bins_data[i]["magnitude"] > bins_data[i+1]["magnitude"]:
            local_peaks.append(bins_data[i])

    # Check for consistent tuning drift: for each reference note that has a
    # nearby local peak, compute the offset
    tuning_offsets = []
    for lp in local_peaks:
        if lp["cents_from_ref"] < 50 and lp["magnitude"] > 0.1:
            # This local peak is reasonably close to a reference note
            # Sign: negative = flat, positive = sharp
            ref_hz = _REF_NOTES[lp["nearest_ref"]]
            offset_cents = 1200 * np.log2(lp["hz"] / ref_hz)
            tuning_offsets.append({
                "ref_note": lp["nearest_ref"],
                "peak_hz": lp["hz"],
                "ref_hz": round(ref_hz, 2),
                "offset_cents": round(offset_cents, 1),
            })

    return {
        "range_hz": [80, 130],
        "bins_per_octave": bpo,
        "cents_per_bin": round(1200 / bpo, 1),
        "all_bins": bins_data,
        "global_peak": peak,
        "local_peaks": local_peaks,
        "tuning_offsets": tuning_offsets,
    }


# ---------------------------------------------------------------------------
# D2: pYIN ablation — with vs without harmonic preprocessing
# ---------------------------------------------------------------------------
def _run_pyin(y_section: np.ndarray, *, label: str) -> dict:
    """Run pYIN with production gate, return detailed stats."""
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y=y_section, fmin=30.0, fmax=400.0, sr=SR,
        frame_length=2048, hop_length=HOP,
    )
    f0 = np.asarray(f0, dtype=np.float32)
    voiced_flag = np.asarray(voiced_flag, dtype=bool)
    voiced_prob = np.asarray(voiced_prob, dtype=np.float32)

    rms = librosa.feature.rms(
        y=y_section, frame_length=2048, hop_length=HOP, center=True,
    )[0].astype(np.float32, copy=False)

    n = int(min(f0.size, rms.size))
    f0, rms = f0[:n], rms[:n]
    voiced_flag, voiced_prob = voiced_flag[:n], voiced_prob[:n]

    # Production gate (features.py:99-101)
    thr = float(np.percentile(rms, 25)) * 0.8
    thr = max(thr, float(rms.max()) * 0.08, 1e-6)
    voiced = (rms >= thr) & voiced_flag & (voiced_prob >= 0.75)

    # Also report intermediate gate stages
    rms_pass = int(np.sum(rms >= thr))
    flag_pass = int(np.sum(voiced_flag))
    prob_pass = int(np.sum(voiced_prob >= 0.75))
    all_pass = int(np.sum(voiced))

    f0v = f0[voiced]
    valid = f0v[np.isfinite(f0v) & (f0v > 0)]

    result = {
        "label": label,
        "total_frames": int(n),
        "rms_threshold": round(float(thr), 6),
        "rms_max": round(float(rms.max()), 6),
        "rms_p25": round(float(np.percentile(rms, 25)), 6),
        "gate_stages": {
            "rms_pass": rms_pass,
            "voiced_flag_pass": flag_pass,
            "voiced_prob_pass": prob_pass,
            "all_gates_pass": all_pass,
        },
    }

    if valid.size == 0:
        result["voiced_frames"] = 0
        result["voiced_ratio"] = 0.0
        result["error"] = "no voiced frames after gating"
        return result

    midi = _hz_to_midi(valid)
    result["voiced_frames"] = int(valid.size)
    result["voiced_ratio"] = round(float(valid.size) / max(n, 1), 4)
    result["median_hz"] = round(float(np.median(valid)), 2)
    result["median_midi"] = round(float(np.median(midi)), 2)
    result["nearest_semitone"] = _midi_to_note(float(np.median(midi)))

    # Distribution: 1-semitone bins from MIDI 30 to 55
    hist_counts, hist_edges = np.histogram(midi, bins=np.arange(29.5, 55.6, 1.0))
    midi_dist = {}
    for i, count in enumerate(hist_counts):
        if count > 0:
            m = 30 + i
            midi_dist[str(m)] = {
                "note": _midi_to_note(m),
                "count": int(count),
                "pct": round(100.0 * count / valid.size, 1),
            }
    result["midi_distribution"] = midi_dist

    return result


def pyin_ablation(y_section: np.ndarray) -> dict:
    """Compare pYIN with and without harmonic preprocessing."""
    # Condition A: with harmonic preprocessing (production config)
    yh = librosa.effects.harmonic(y=y_section, margin=6.0).astype(np.float32, copy=False)
    with_harmonic = _run_pyin(yh, label="with_harmonic(margin=6.0)")

    # Condition B: without harmonic preprocessing (raw stem)
    without_harmonic = _run_pyin(
        y_section.astype(np.float32, copy=False), label="without_harmonic",
    )

    return {
        "with_harmonic": with_harmonic,
        "without_harmonic": without_harmonic,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not BASS_STEM.exists():
        print(f"ERROR: Bass stem not found: {BASS_STEM}")
        sys.exit(1)

    print("Loading bass stem...")
    y, _ = librosa.load(str(BASS_STEM), sr=SR, mono=True)
    print(f"  Duration: {len(y)/SR:.1f}s, samples: {len(y)}")

    all_cqt = {}
    all_ablation = {}

    for sec_name, (start_s, end_s) in SECTIONS.items():
        print(f"\n{'='*70}")
        print(f"SECTION: {sec_name} ({start_s}–{end_s}s)")
        print(f"{'='*70}")

        y_sec = _load_section(y, start_s, end_s)

        # --- D1: Dense CQT ---
        print(f"\n  [D1] Dense CQT — all bins in [80–130 Hz]")
        d1 = dense_cqt(y_sec)
        all_cqt[sec_name] = d1

        print(f"  {'bin':>4} {'Hz':>8} {'MIDI':>6} {'note':>5} "
              f"{'mag':>9} {'nearest':>6} {'¢ off':>6}  bar")
        print(f"  {'—'*4} {'—'*8} {'—'*6} {'—'*5} {'—'*9} {'—'*6} {'—'*6}  {'—'*30}")
        max_mag = max(b["magnitude"] for b in d1["all_bins"]) or 1.0
        for b in d1["all_bins"]:
            bar_len = int(30 * b["magnitude"] / max_mag)
            bar = "█" * bar_len
            is_peak = " ◄" if b == d1["global_peak"] else ""
            is_local = " *" if b in d1["local_peaks"] else ""
            print(f"  {b['bin_idx']:>4} {b['hz']:>8.2f} {b['midi']:>6.2f} "
                  f"{b['note_approx']:>5} {b['magnitude']:>9.6f} "
                  f"{b['nearest_ref']:>6} {b['cents_from_ref']:>5.1f}¢  "
                  f"{bar}{is_peak}{is_local}")

        print(f"\n  Global peak: {d1['global_peak']['hz']:.2f} Hz "
              f"(MIDI {d1['global_peak']['midi']:.2f}, "
              f"~{d1['global_peak']['note_approx']}) "
              f"mag={d1['global_peak']['magnitude']:.6f}")

        if d1["local_peaks"]:
            print(f"  Local peaks:")
            for lp in d1["local_peaks"]:
                print(f"    {lp['hz']:.2f} Hz ({lp['note_approx']}) "
                      f"mag={lp['magnitude']:.6f}  "
                      f"nearest={lp['nearest_ref']} {lp['cents_from_ref']:.0f}¢")

        if d1["tuning_offsets"]:
            print(f"  Tuning offsets from reference notes:")
            for to in d1["tuning_offsets"]:
                direction = "flat" if to["offset_cents"] < 0 else "sharp"
                print(f"    {to['ref_note']}: peak at {to['peak_hz']:.1f} Hz "
                      f"vs ref {to['ref_hz']:.1f} Hz → "
                      f"{abs(to['offset_cents']):.1f}¢ {direction}")

        # --- D2: pYIN ablation ---
        print(f"\n  [D2] pYIN ablation: with vs without harmonic preprocessing")
        d2 = pyin_ablation(y_sec)
        all_ablation[sec_name] = d2

        for condition in ["with_harmonic", "without_harmonic"]:
            r = d2[condition]
            print(f"\n    --- {r['label']} ---")
            gs = r["gate_stages"]
            print(f"    Gate stages ({r['total_frames']} total):")
            print(f"      RMS ≥ {r['rms_threshold']:.6f}: "
                  f"{gs['rms_pass']} ({100*gs['rms_pass']/r['total_frames']:.1f}%)")
            print(f"      voiced_flag=True:    "
                  f"{gs['voiced_flag_pass']} ({100*gs['voiced_flag_pass']/r['total_frames']:.1f}%)")
            print(f"      voiced_prob ≥ 0.75:  "
                  f"{gs['voiced_prob_pass']} ({100*gs['voiced_prob_pass']/r['total_frames']:.1f}%)")
            print(f"      ALL gates:           "
                  f"{gs['all_gates_pass']} ({100*gs['all_gates_pass']/r['total_frames']:.1f}%)")

            if "error" in r:
                print(f"    Result: {r['error']}")
            else:
                print(f"    Voiced: {r['voiced_frames']}/{r['total_frames']} "
                      f"({r['voiced_ratio']*100:.1f}%)")
                print(f"    Median: {r['median_hz']:.2f} Hz = "
                      f"MIDI {r['median_midi']:.2f} ({r['nearest_semitone']})")
                if r["midi_distribution"]:
                    print(f"    MIDI distribution:")
                    for midi_str, info in sorted(
                        r["midi_distribution"].items(),
                        key=lambda x: x[1]["count"], reverse=True,
                    ):
                        print(f"      MIDI {midi_str:>3} ({info['note']:>4}): "
                              f"{info['count']:>4} ({info['pct']:>5.1f}%)")

    # --- Save JSON ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "diagnostic": "bass_pitch_v2_dense_cqt_and_pyin_ablation",
        "song": SONG,
        "sections": dict(SECTIONS),
        "dense_cqt": all_cqt,
        "pyin_ablation": all_ablation,
    }
    out_path = RESULTS_DIR / "bass_pitch_diagnostic_v2.json"
    out_path.write_text(json.dumps(artifact, indent=2, default=str) + "\n")
    print(f"\nJSON artifact saved to {out_path}")


if __name__ == "__main__":
    main()
