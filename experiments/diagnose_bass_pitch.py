"""Diagnose why pYIN and basic-pitch both report 0% pitch-class G on Feel Good Inc bass.

Hypotheses:
  H1 — Tuning offset: recording is flat, G quantizes to F#
  H2 — Weak fundamental: Demucs lost ~98 Hz, 2nd harmonic dominates
  H3 — Shared extractor limitation: both methods fail at bass freq
  H4 — Reference error: root is not G

Usage:
    .songviz/venv/bin/python experiments/diagnose_bass_pitch.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from songviz.features import bass_note_events_basic_pitch

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SONG = "Gorillaz - Feel Good Inc (featuring De La Soul)"
BASS_STEM = ROOT / "outputs" / SONG / "stems" / "bass.wav"
RESULTS_DIR = ROOT / "experiments" / "results" / "feel_good_inc"

SR = 22050
HOP = 512

# Active sections from benchmark reference
SECTIONS = {
    "verse1+chorus1": (12.5, 63.0),
    "chorus2": (95.0, 138.0),
}

# Pitch-class names for display
PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _load_section(y: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    s, e = int(start_s * SR), int(end_s * SR)
    return y[s:e]


def _hz_to_midi(hz: np.ndarray) -> np.ndarray:
    return 69.0 + 12.0 * np.log2(np.maximum(hz, 1e-8) / 440.0)


def _midi_to_note(midi: float) -> str:
    m = int(round(midi))
    return f"{PC_NAMES[m % 12]}{m // 12 - 1}"


# ---------------------------------------------------------------------------
# Diagnostic 1: Continuous pYIN pitch distribution
# ---------------------------------------------------------------------------
def diagnostic_pyin(y_section: np.ndarray) -> dict:
    """Run pYIN with exact production parameters, report continuous pitch stats."""
    # Replicate features.py:70 — harmonic_margin=6.0 for bass
    yh = librosa.effects.harmonic(y=y_section, margin=6.0).astype(np.float32, copy=False)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y=yh, fmin=30.0, fmax=400.0, sr=SR,
        frame_length=2048, hop_length=HOP,
    )
    f0 = np.asarray(f0, dtype=np.float32)
    voiced_flag = np.asarray(voiced_flag, dtype=bool)
    voiced_prob = np.asarray(voiced_prob, dtype=np.float32)

    rms = librosa.feature.rms(
        y=yh, frame_length=2048, hop_length=HOP, center=True,
    )[0].astype(np.float32, copy=False)

    n = int(min(f0.size, rms.size))
    f0, rms = f0[:n], rms[:n]
    voiced_flag, voiced_prob = voiced_flag[:n], voiced_prob[:n]

    # Same gate as production (features.py:99-101)
    thr = float(np.percentile(rms, 25)) * 0.8
    thr = max(thr, float(rms.max()) * 0.08, 1e-6)
    voiced = (rms >= thr) & voiced_flag & (voiced_prob >= 0.75)

    f0v = f0[voiced]
    valid = f0v[np.isfinite(f0v) & (f0v > 0)]

    if valid.size == 0:
        return {"voiced_frames": 0, "total_frames": n, "error": "no voiced frames"}

    midi_continuous = _hz_to_midi(valid)

    # Fractional offset from nearest semitone
    offsets = midi_continuous - np.round(midi_continuous)
    median_hz = float(np.median(valid))
    median_midi = float(np.median(midi_continuous))
    median_offset = float(np.median(offsets))

    # Semitone bin distribution for F#2(42), G2(43), G#2(44)
    semitone_bins = {}
    for target_midi, name in [(42, "F#2"), (43, "G2"), (44, "G#2")]:
        in_bin = np.sum(np.abs(midi_continuous - target_midi) < 0.5)
        semitone_bins[name] = float(in_bin / len(midi_continuous))

    # Fine-grained histogram: 0.1-semitone bins across MIDI [40, 50]
    bins_edges = np.arange(40.0, 50.05, 0.1)
    hist_counts, _ = np.histogram(midi_continuous, bins=bins_edges)
    total_in_range = hist_counts.sum()
    fine_hist = {}
    for i, count in enumerate(hist_counts):
        if count > 0:
            center = bins_edges[i] + 0.05
            fine_hist[f"{center:.1f}"] = {
                "count": int(count),
                "pct": round(100.0 * count / max(total_in_range, 1), 1),
                "note": _midi_to_note(center),
            }

    return {
        "voiced_frames": int(np.sum(voiced)),
        "total_frames": int(n),
        "voiced_ratio": round(float(np.sum(voiced)) / max(n, 1), 3),
        "median_hz": round(median_hz, 2),
        "median_midi": round(median_midi, 2),
        "nearest_semitone": _midi_to_note(median_midi),
        "median_offset_st": round(median_offset, 3),
        "semitone_bin_pct": {k: round(v * 100, 1) for k, v in semitone_bins.items()},
        "fine_histogram_0.1st": fine_hist,
    }


# ---------------------------------------------------------------------------
# Diagnostic 2: CQT spectral energy
# ---------------------------------------------------------------------------
def diagnostic_cqt(y_section: np.ndarray) -> dict:
    """CQT with 1/3-semitone resolution to see between semitones."""
    bpo = 36  # bins per octave → 33 cents per bin
    fmin = librosa.note_to_hz("C1")
    n_bins = 180  # 5 octaves (C1 to C6)

    C = np.abs(librosa.cqt(
        y=y_section.astype(np.float32), sr=SR,
        fmin=fmin, n_bins=n_bins, bins_per_octave=bpo, hop_length=HOP,
    ))

    # Mean magnitude per bin across time
    mean_mag = C.mean(axis=1)

    # Map bins to frequencies
    freqs = fmin * 2.0 ** (np.arange(n_bins) / bpo)

    # Energy at specific notes (find closest bin)
    targets = {
        "F#2": librosa.note_to_hz("F#2"),
        "G2": librosa.note_to_hz("G2"),
        "G#2": librosa.note_to_hz("G#2"),
        "G3": librosa.note_to_hz("G3"),  # 2nd harmonic
        "D4": librosa.note_to_hz("D4"),  # 3rd harmonic
    }
    target_energy = {}
    for name, target_hz in targets.items():
        idx = int(np.argmin(np.abs(freqs - target_hz)))
        target_energy[name] = {
            "magnitude": round(float(mean_mag[idx]), 6),
            "bin_hz": round(float(freqs[idx]), 2),
        }

    # Fundamental-to-2nd-harmonic ratio
    g2_mag = target_energy["G2"]["magnitude"]
    g3_mag = target_energy["G3"]["magnitude"]
    f2h_ratio = round(g2_mag / max(g3_mag, 1e-9), 3) if g3_mag > 1e-9 else None

    # Strongest bin in [80, 120] Hz
    mask = (freqs >= 80) & (freqs <= 120)
    if mask.any():
        sub_mag = mean_mag[mask]
        sub_freqs = freqs[mask]
        peak_idx = int(np.argmax(sub_mag))
        peak_hz = float(sub_freqs[peak_idx])
        peak_midi = _hz_to_midi(np.array([peak_hz]))[0]
        strongest = {
            "hz": round(peak_hz, 2),
            "midi": round(float(peak_midi), 2),
            "note": _midi_to_note(float(peak_midi)),
            "magnitude": round(float(sub_mag[peak_idx]), 6),
        }
    else:
        strongest = None

    return {
        "bins_per_octave": bpo,
        "target_energy": target_energy,
        "fundamental_to_2nd_harmonic_ratio_G2_G3": f2h_ratio,
        "strongest_bin_80_120Hz": strongest,
    }


# ---------------------------------------------------------------------------
# Diagnostic 3: Basic-pitch MIDI value distribution
# ---------------------------------------------------------------------------
def diagnostic_basic_pitch(audio_path: str, start_s: float, end_s: float) -> dict:
    """Count basic-pitch notes per integer MIDI in the section."""
    events = bass_note_events_basic_pitch(audio_path)

    # Filter to section
    section_events = [
        e for e in events
        if e["start_s"] < end_s and e["end_s"] > start_s
    ]

    midi_counts: dict[int, int] = {}
    for e in section_events:
        m = int(round(e["midi"]))
        midi_counts[m] = midi_counts.get(m, 0) + 1

    # Build distribution for MIDI [30, 55]
    distribution = {}
    for m in range(30, 56):
        c = midi_counts.get(m, 0)
        if c > 0:
            distribution[str(m)] = {
                "note": _midi_to_note(m),
                "count": c,
            }

    total = sum(midi_counts.values())

    return {
        "total_notes_in_section": total,
        "midi_distribution": distribution,
    }


# ---------------------------------------------------------------------------
# Hypothesis evaluation
# ---------------------------------------------------------------------------
def evaluate_hypotheses(pyin_results: dict, cqt_results: dict) -> dict:
    """Automated hypothesis evaluation from diagnostic results."""
    verdicts = {}

    # Collect median offsets and CQT ratios across sections
    offsets = []
    f2h_ratios = []
    strongest_notes = []
    for sec_name, sec_data in pyin_results.items():
        if "error" in sec_data:
            continue
        offsets.append(sec_data["median_offset_st"])
    for sec_name, sec_data in cqt_results.items():
        r = sec_data.get("fundamental_to_2nd_harmonic_ratio_G2_G3")
        if r is not None:
            f2h_ratios.append(r)
        s = sec_data.get("strongest_bin_80_120Hz")
        if s:
            strongest_notes.append(s)

    # H1: Tuning offset
    if offsets:
        mean_offset = float(np.mean(offsets))
        if abs(mean_offset) > 0.3:
            h1 = "SUPPORTED"
            h1_evidence = f"Median offset = {mean_offset:+.3f} st (>{0.3} threshold)"
        elif abs(mean_offset) < 0.2:
            h1 = "WEAKENED"
            h1_evidence = f"Median offset = {mean_offset:+.3f} st (<0.2, within tolerance)"
        else:
            h1 = "INCONCLUSIVE"
            h1_evidence = f"Median offset = {mean_offset:+.3f} st (between 0.2 and 0.3)"
    else:
        h1 = "NO_DATA"
        h1_evidence = "No voiced frames"
    verdicts["H1_tuning_offset"] = {"verdict": h1, "evidence": h1_evidence}

    # H2: Weak fundamental
    if f2h_ratios:
        mean_ratio = float(np.mean(f2h_ratios))
        if mean_ratio < 0.5:
            h2 = "SUPPORTED"
            h2_evidence = f"G2/G3 energy ratio = {mean_ratio:.3f} (<0.5, fundamental weak)"
        elif mean_ratio > 1.0:
            h2 = "WEAKENED"
            h2_evidence = f"G2/G3 energy ratio = {mean_ratio:.3f} (>1.0, fundamental present)"
        else:
            h2 = "INCONCLUSIVE"
            h2_evidence = f"G2/G3 energy ratio = {mean_ratio:.3f} (between 0.5 and 1.0)"
    else:
        h2 = "NO_DATA"
        h2_evidence = "No CQT ratio data"
    verdicts["H2_weak_fundamental"] = {"verdict": h2, "evidence": h2_evidence}

    # H3: Shared extractor limitation
    if h1 in ("WEAKENED", "NO_DATA") and h2 in ("WEAKENED", "NO_DATA"):
        h3 = "SUPPORTED"
        h3_evidence = "Neither tuning nor weak fundamental explains the miss — extractor limitation likely"
    elif h1 == "SUPPORTED" or h2 == "SUPPORTED":
        h3 = "WEAKENED"
        h3_evidence = f"H1={h1}, H2={h2} — a simpler cause is supported"
    else:
        h3 = "INCONCLUSIVE"
        h3_evidence = f"H1={h1}, H2={h2}"
    verdicts["H3_extractor_limitation"] = {"verdict": h3, "evidence": h3_evidence}

    # H4: Reference error
    if strongest_notes:
        # Check if strongest bin near G2 (MIDI 43, ~98 Hz)
        avg_hz = float(np.mean([s["hz"] for s in strongest_notes]))
        g2_hz = librosa.note_to_hz("G2")
        dist_cents = abs(1200 * np.log2(avg_hz / g2_hz))
        if dist_cents > 100:
            h4 = "SUPPORTED"
            h4_evidence = (
                f"Strongest bin at {avg_hz:.1f} Hz ({dist_cents:.0f} cents from G2={g2_hz:.1f} Hz) — "
                f"root may not be G"
            )
        else:
            h4 = "WEAKENED"
            h4_evidence = (
                f"Strongest bin at {avg_hz:.1f} Hz ({dist_cents:.0f} cents from G2={g2_hz:.1f} Hz) — "
                f"consistent with G root"
            )
    else:
        h4 = "NO_DATA"
        h4_evidence = "No strongest-bin data"
    verdicts["H4_reference_error"] = {"verdict": h4, "evidence": h4_evidence}

    return verdicts


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

    all_pyin = {}
    all_cqt = {}
    all_bp = {}

    for sec_name, (start_s, end_s) in SECTIONS.items():
        print(f"\n{'='*60}")
        print(f"SECTION: {sec_name} ({start_s}–{end_s}s)")
        print(f"{'='*60}")

        y_sec = _load_section(y, start_s, end_s)

        # --- Diagnostic 1: pYIN ---
        print("\n  [D1] Continuous pYIN pitch distribution")
        d1 = diagnostic_pyin(y_sec)
        all_pyin[sec_name] = d1

        if "error" not in d1:
            print(f"    Voiced frames: {d1['voiced_frames']}/{d1['total_frames']} "
                  f"({d1['voiced_ratio']*100:.1f}%)")
            print(f"    Median Hz: {d1['median_hz']}")
            print(f"    Median MIDI: {d1['median_midi']} "
                  f"(nearest: {d1['nearest_semitone']})")
            print(f"    Offset from nearest semitone: {d1['median_offset_st']:+.3f} st")
            print(f"    Semitone bin %: ", end="")
            for note, pct in d1["semitone_bin_pct"].items():
                print(f"{note}={pct:.1f}%  ", end="")
            print()

            # Top 5 fine-grained bins
            hist = d1["fine_histogram_0.1st"]
            sorted_bins = sorted(hist.items(), key=lambda x: x[1]["count"], reverse=True)
            print(f"    Top fine-grained bins (0.1st):")
            for midi_str, info in sorted_bins[:8]:
                print(f"      MIDI {midi_str} ({info['note']}): "
                      f"{info['count']} frames ({info['pct']:.1f}%)")
        else:
            print(f"    ERROR: {d1['error']}")

        # --- Diagnostic 2: CQT ---
        print(f"\n  [D2] CQT spectral energy (1/3-semitone resolution)")
        d2 = diagnostic_cqt(y_sec)
        all_cqt[sec_name] = d2

        print(f"    Target note energy:")
        for note, info in d2["target_energy"].items():
            print(f"      {note}: {info['magnitude']:.6f} (bin @ {info['bin_hz']:.1f} Hz)")
        print(f"    G2/G3 ratio: {d2['fundamental_to_2nd_harmonic_ratio_G2_G3']}")
        s = d2["strongest_bin_80_120Hz"]
        if s:
            print(f"    Strongest in [80-120 Hz]: {s['hz']:.1f} Hz "
                  f"(MIDI {s['midi']:.2f}, {s['note']}) mag={s['magnitude']:.6f}")

        # --- Diagnostic 3: Basic-pitch ---
        print(f"\n  [D3] Basic-pitch MIDI distribution")
        d3 = diagnostic_basic_pitch(str(BASS_STEM), start_s, end_s)
        all_bp[sec_name] = d3

        print(f"    Total notes in section: {d3['total_notes_in_section']}")
        if d3["midi_distribution"]:
            for midi_str, info in sorted(d3["midi_distribution"].items(),
                                         key=lambda x: x[1]["count"], reverse=True):
                print(f"      MIDI {midi_str} ({info['note']}): {info['count']}")

    # --- Hypothesis evaluation ---
    print(f"\n{'='*60}")
    print("HYPOTHESIS EVALUATION")
    print(f"{'='*60}")

    verdicts = evaluate_hypotheses(all_pyin, all_cqt)
    for hyp_id, v in verdicts.items():
        status = v["verdict"]
        marker = {"SUPPORTED": "+", "WEAKENED": "-", "INCONCLUSIVE": "?", "NO_DATA": "!"}
        print(f"\n  [{marker.get(status, '?')}] {hyp_id}: {status}")
        print(f"      {v['evidence']}")

    # --- Save JSON artifact ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "diagnostic": "bass_pitch_g_detection",
        "song": SONG,
        "sections": dict(SECTIONS),
        "pyin": all_pyin,
        "cqt": all_cqt,
        "basic_pitch": all_bp,
        "hypotheses": verdicts,
    }
    out_path = RESULTS_DIR / "bass_pitch_diagnostic.json"
    out_path.write_text(json.dumps(artifact, indent=2, default=str) + "\n")
    print(f"\nJSON artifact saved to {out_path}")


if __name__ == "__main__":
    main()
