"""Bass harmonic-salience contour: harmonics-first pitch estimation.

Both pYIN and basic-pitch are fundamental-first estimators that fail when
Demucs removes the G2 fundamental energy (98 Hz).  The remaining harmonics
(196, 294, 392 Hz) are still present — human listeners can hear the bass line
— but neither extractor uses harmonic evidence to infer the missing
fundamental.

This prototype uses librosa.salience on a high-res CQT: for each candidate
f0, it sums energy at harmonics 2f, 3f, 4f, 5f (where the energy actually
lives), then picks the f0 with strongest total harmonic support.

Usage:
    .songviz/venv/bin/python experiments/bass_harmonic_contour.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SONG = "Gorillaz - Feel Good Inc (featuring De La Soul)"
BASS_STEM = ROOT / "outputs" / SONG / "stems" / "bass.wav"
REDUCED_JSON = ROOT / "outputs" / SONG / "analysis" / "reduced.json"
PREV_CONTOUR_JSON = (
    ROOT / "experiments" / "results" / "feel_good_inc" / "bass_contour_prototype.json"
)
RESULTS_DIR = ROOT / "experiments" / "results" / "feel_good_inc"

SR = 22050
HOP = 512

# CQT parameters
BPO = 36          # bins per octave (1/3-semitone resolution)
N_BINS = 220      # C1 (~33 Hz) to ~2200 Hz — covers 5th harmonic of 400 Hz
FMIN = float(librosa.note_to_hz("C1"))  # ~32.7 Hz

# Harmonic salience
HARMONICS = [1, 2, 3, 4, 5]
WEIGHTS = np.array([0.2, 1.0, 0.8, 0.6, 0.4])

# Bass range for argmax
BASS_RANGE = (30.0, 400.0)

# Temporal smoothing
SMOOTH_WIN = 11
JUMP_THR_ST = 5.0

# Sonification
GAIN_BASS = 0.4
NORM_PEAK = 0.9

SECTIONS = {
    "verse1+chorus1": (12.5, 63.0),
    "chorus2": (95.0, 138.0),
}

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ---------------------------------------------------------------------------
# Helpers (copied from bass_contour_prototype.py / features.py)
# ---------------------------------------------------------------------------
def _nanmedian_smooth(x: np.ndarray, *, win: int) -> np.ndarray:
    """NaN-aware running median smoothing."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    if win <= 1:
        return x.astype(np.float32, copy=True)
    win = int(win)
    if win % 2 == 0:
        win += 1
    r = win // 2
    out = np.empty_like(x, dtype=np.float32)
    for i in range(x.size):
        lo = max(0, i - r)
        hi = min(x.size, i + r + 1)
        sl = x[lo:hi]
        if not np.isfinite(sl).any():
            out[i] = np.nan
            continue
        v = np.nanmedian(sl)
        out[i] = (0.0 if not np.isfinite(v) else float(v))
    return out


def _hz_to_midi(hz: float | np.ndarray) -> float | np.ndarray:
    return 69.0 + 12.0 * np.log2(np.maximum(hz, 1e-8) / 440.0)


def _midi_to_hz(midi: float | np.ndarray) -> float | np.ndarray:
    return 440.0 * 2.0 ** ((np.asarray(midi) - 69.0) / 12.0)


def _midi_to_note(midi: float) -> str:
    m = int(round(midi))
    return f"{PC_NAMES[m % 12]}{m // 12 - 1}"


def _pitch_class_distribution(midi_values: np.ndarray) -> dict:
    """Compute pitch-class distribution from MIDI values."""
    valid = midi_values[np.isfinite(midi_values)]
    if valid.size == 0:
        return {}
    pcs = np.round(valid).astype(int) % 12
    dist = {}
    for pc in range(12):
        count = int(np.sum(pcs == pc))
        if count > 0:
            dist[PC_NAMES[pc]] = {
                "pc": pc,
                "count": count,
                "pct": round(100.0 * count / valid.size, 1),
            }
    return dict(sorted(dist.items(), key=lambda x: x[1]["count"], reverse=True))


# ---------------------------------------------------------------------------
# Jump rejection
# ---------------------------------------------------------------------------
def _reject_jumps(midi: np.ndarray, *, max_st: float) -> np.ndarray:
    """For each frame: if |midi[i] - median(neighbors)| > max_st → NaN.

    Neighbors = up to 5 frames each side (excluding NaN).
    """
    out = midi.copy()
    n = out.size
    for i in range(n):
        if not np.isfinite(out[i]):
            continue
        # Gather neighbor values (5 each side, excluding NaN)
        lo = max(0, i - 5)
        hi = min(n, i + 6)
        neighbors = out[lo:hi]
        valid = neighbors[np.isfinite(neighbors)]
        if valid.size < 2:
            continue
        med = float(np.median(valid))
        if abs(out[i] - med) > max_st:
            out[i] = np.nan
    return out


# ---------------------------------------------------------------------------
# Harmonic salience contour extraction
# ---------------------------------------------------------------------------
def extract_harmonic_contour(y: np.ndarray) -> dict:
    """Extract bass pitch contour using harmonic salience on CQT."""

    # 1. CQT — high resolution, raw stem (no harmonic preprocessing)
    C = np.abs(librosa.cqt(
        y=y.astype(np.float32),
        sr=SR,
        fmin=FMIN,
        n_bins=N_BINS,
        bins_per_octave=BPO,
        hop_length=HOP,
    ))
    freqs = FMIN * 2.0 ** (np.arange(N_BINS) / BPO)

    # 2. Harmonic salience
    sal = librosa.salience(
        S=C,
        freqs=freqs,
        harmonics=HARMONICS,
        weights=WEIGHTS,
        filter_peaks=False,
        fill_value=0.0,
    )
    # sal shape: (N_BINS, n_frames)

    # 3. Restrict to bass range, argmax per frame
    bass_mask = (freqs >= BASS_RANGE[0]) & (freqs <= BASS_RANGE[1])
    bass_indices = np.where(bass_mask)[0]
    bass_sal = sal[bass_indices, :]  # (n_bass_bins, n_frames)

    # Per-frame argmax within bass range
    best_local = np.argmax(bass_sal, axis=0)  # index into bass_indices
    f0_hz = freqs[bass_indices[best_local]].astype(np.float32)

    # Per-frame confidence: peak salience value
    n_frames = sal.shape[1]
    confidence = np.array(
        [float(bass_sal[best_local[i], i]) for i in range(n_frames)],
        dtype=np.float32,
    )

    # 4. RMS activity gating (same formula as production)
    rms = librosa.feature.rms(
        y=y, frame_length=2048, hop_length=HOP, center=True,
    )[0].astype(np.float32, copy=False)

    # Align lengths
    n = int(min(n_frames, rms.size))
    f0_hz = f0_hz[:n]
    confidence = confidence[:n]
    rms = rms[:n]

    thr = float(np.percentile(rms, 25)) * 0.8
    thr = max(thr, float(rms.max()) * 0.08, 1e-6)
    activity = rms >= thr

    # Gate inactive frames
    f0_hz[~activity] = np.nan

    # 5. Confidence gating: reject frames where salience is low
    #    (below 25th percentile of active-frame salience)
    active_confidence = confidence[activity]
    if active_confidence.size > 0:
        conf_thr = float(np.percentile(active_confidence, 25))
        low_conf = confidence < conf_thr
        f0_hz[low_conf] = np.nan

    # 6. Temporal smoothing in MIDI domain
    midi = np.where(
        np.isfinite(f0_hz),
        _hz_to_midi(f0_hz),
        np.nan,
    ).astype(np.float32)

    midi = _nanmedian_smooth(midi, win=SMOOTH_WIN)
    midi = _reject_jumps(midi, max_st=JUMP_THR_ST)

    # Convert back to Hz
    pitch_hz = np.where(
        np.isfinite(midi),
        _midi_to_hz(midi).astype(np.float32),
        np.nan,
    ).astype(np.float32)

    # Normalized energy
    rms_max = float(rms.max()) if rms.max() > 0 else 1.0
    energy = (rms / rms_max).astype(np.float32)

    times_s = librosa.frames_to_time(
        np.arange(n), sr=SR, hop_length=HOP,
    ).astype(np.float32)

    return {
        "times_s": times_s,
        "pitch_hz": pitch_hz,
        "energy": energy,
        "activity": activity,
        "confidence": confidence,
        "n_frames": n,
        "rms_threshold": thr,
    }


# ---------------------------------------------------------------------------
# Sonification (phase-accumulating triangle wave)
# ---------------------------------------------------------------------------
def sonify_contour(contour: dict, out_path: Path) -> None:
    """Sonify the contour as a phase-accumulating triangle wave."""
    n_frames = contour["n_frames"]
    pitch_hz = contour["pitch_hz"]
    energy = contour["energy"]

    total_samples = n_frames * HOP
    buf = np.zeros(total_samples, dtype=np.float32)
    phase = 0.0

    for i in range(n_frames):
        hz = float(pitch_hz[i])
        amp = float(energy[i]) * GAIN_BASS
        start = i * HOP
        end = min(start + HOP, total_samples)
        n_samp = end - start

        if not np.isfinite(hz) or hz < 1.0 or amp < 1e-6:
            continue

        t_local = np.arange(n_samp, dtype=np.float64) / SR
        omega = 2.0 * np.pi * hz
        sig = (
            np.sin(omega * t_local + phase)
            - (1.0 / 9.0) * np.sin(3.0 * omega * t_local + 3.0 * phase)
            + (1.0 / 25.0) * np.sin(5.0 * omega * t_local + 5.0 * phase)
        ).astype(np.float32)

        buf[start:end] += sig * amp
        phase = (phase + 2.0 * np.pi * hz * n_samp / SR) % (2.0 * np.pi)

    # Peak normalize
    peak = np.max(np.abs(buf))
    if peak > 1e-8:
        buf *= NORM_PEAK / peak

    sf.write(str(out_path), buf, SR)
    print(f"  Sonified contour → {out_path} ({len(buf)/SR:.1f}s)")


# ---------------------------------------------------------------------------
# Three-way comparison
# ---------------------------------------------------------------------------
def compare_three_ways(
    contour: dict,
    reduced_path: Path,
    prev_contour_path: Path | None,
) -> dict:
    """Compare: note events vs pYIN contour vs harmonic-salience contour."""

    # --- 1. Load note events from reduced.json ---
    with open(reduced_path) as f:
        reduced = json.load(f)
    bass_layer = reduced.get("bass", {})
    bass_notes = bass_layer.get("notes", []) if isinstance(bass_layer, dict) else bass_layer
    note_midis = np.array([n["midi"] for n in bass_notes], dtype=np.float32)
    note_total_dur = sum(n["offset_s"] - n["onset_s"] for n in bass_notes)
    song_dur = float(contour["times_s"][-1]) if contour["n_frames"] > 0 else 1.0

    note_stats = {
        "method": "note_events (basic-pitch)",
        "active_pct": round(100.0 * note_total_dur / max(song_dur, 1.0), 1),
        "median_midi": round(float(np.median(note_midis)), 2) if note_midis.size else None,
        "median_note": _midi_to_note(float(np.median(note_midis))) if note_midis.size else None,
        "pc_distribution": _pitch_class_distribution(note_midis),
    }

    # --- 2. Load previous pYIN contour ---
    pyin_stats = None
    if prev_contour_path and prev_contour_path.exists():
        with open(prev_contour_path) as f:
            prev = json.load(f)
        prev_pitch = prev.get("contour", {}).get("pitch_hz", [])
        prev_midi = np.array([
            _hz_to_midi(h) if h is not None and np.isfinite(h) else np.nan
            for h in prev_pitch
        ], dtype=np.float32)
        valid_prev = prev_midi[np.isfinite(prev_midi)]
        n_voiced_prev = int(np.sum(np.isfinite(prev_midi)))
        pyin_stats = {
            "method": "pYIN contour (voiced_prob≥0.3)",
            "voiced_pct": round(100.0 * n_voiced_prev / max(len(prev_midi), 1), 1),
            "median_midi": round(float(np.median(valid_prev)), 2) if valid_prev.size else None,
            "median_note": _midi_to_note(float(np.median(valid_prev))) if valid_prev.size else None,
            "pc_distribution": _pitch_class_distribution(valid_prev),
        }

    # --- 3. Harmonic salience contour stats ---
    pitch_hz = contour["pitch_hz"]
    midi = np.where(
        np.isfinite(pitch_hz), _hz_to_midi(pitch_hz), np.nan,
    ).astype(np.float32)
    valid_midi = midi[np.isfinite(midi)]
    n_voiced = int(np.sum(np.isfinite(pitch_hz)))

    harm_stats = {
        "method": "harmonic salience contour",
        "voiced_pct": round(100.0 * n_voiced / max(contour["n_frames"], 1), 1),
        "median_midi": round(float(np.median(valid_midi)), 2) if valid_midi.size else None,
        "median_note": _midi_to_note(float(np.median(valid_midi))) if valid_midi.size else None,
        "pc_distribution": _pitch_class_distribution(valid_midi),
    }

    # --- Per-section breakdown ---
    per_section = {}
    for sec_name, (start_s, end_s) in SECTIONS.items():
        times = contour["times_s"]
        sec_mask = (times >= start_s) & (times < end_s)
        sec_midi = midi[sec_mask]
        sec_valid = sec_midi[np.isfinite(sec_midi)]
        sec_voiced = int(np.sum(np.isfinite(sec_midi)))
        sec_total = int(np.sum(sec_mask))

        # Note events in section
        sec_notes = [n for n in bass_notes if start_s <= n["onset_s"] < end_s]
        sec_note_midis = np.array([n["midi"] for n in sec_notes], dtype=np.float32)

        per_section[sec_name] = {
            "harm_voiced_pct": round(100.0 * sec_voiced / max(sec_total, 1), 1),
            "harm_median_midi": round(float(np.median(sec_valid)), 2) if sec_valid.size else None,
            "harm_pc_dist": _pitch_class_distribution(sec_valid),
            "notes_count": len(sec_notes),
            "notes_median_midi": round(float(np.median(sec_note_midis)), 2) if sec_note_midis.size else None,
            "notes_pc_dist": _pitch_class_distribution(sec_note_midis),
        }

    # --- G pitch-class comparison ---
    g_note = note_stats["pc_distribution"].get("G", {}).get("pct", 0.0)
    g_pyin = pyin_stats["pc_distribution"].get("G", {}).get("pct", 0.0) if pyin_stats else None
    g_harm = harm_stats["pc_distribution"].get("G", {}).get("pct", 0.0)

    return {
        "note_event_stats": note_stats,
        "pyin_contour_stats": pyin_stats,
        "harmonic_contour_stats": harm_stats,
        "per_section": per_section,
        "g_pitch_class": {
            "note_events_pct": g_note,
            "pyin_contour_pct": g_pyin,
            "harmonic_contour_pct": g_harm,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    for path, label in [(BASS_STEM, "Bass stem"), (REDUCED_JSON, "reduced.json")]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    print("Loading bass stem...")
    y, _ = librosa.load(str(BASS_STEM), sr=SR, mono=True)
    print(f"  Duration: {len(y)/SR:.1f}s, samples: {len(y)}")

    print("\nExtracting harmonic-salience contour...")
    print(f"  CQT: BPO={BPO}, N_BINS={N_BINS}, fmin={FMIN:.1f} Hz")
    print(f"  Harmonics: {HARMONICS}, weights: {WEIGHTS.tolist()}")
    print(f"  Bass range: {BASS_RANGE[0]}–{BASS_RANGE[1]} Hz")
    contour = extract_harmonic_contour(y)

    # --- Print contour summary ---
    pitch_hz = contour["pitch_hz"]
    valid_hz = pitch_hz[np.isfinite(pitch_hz)]
    valid_midi = _hz_to_midi(valid_hz) if valid_hz.size else np.array([])
    n_voiced = int(np.sum(np.isfinite(pitch_hz)))
    voiced_pct = 100.0 * n_voiced / max(contour["n_frames"], 1)

    print(f"\n{'='*60}")
    print("HARMONIC SALIENCE CONTOUR SUMMARY")
    print(f"{'='*60}")
    print(f"  Frames: {contour['n_frames']}")
    print(f"  Voiced: {n_voiced} ({voiced_pct:.1f}%)")
    if valid_hz.size:
        print(f"  Pitch range: {valid_hz.min():.1f}–{valid_hz.max():.1f} Hz "
              f"(MIDI {valid_midi.min():.1f}–{valid_midi.max():.1f})")
        print(f"  Median: {np.median(valid_hz):.1f} Hz "
              f"(MIDI {np.median(valid_midi):.1f}, "
              f"~{_midi_to_note(float(np.median(valid_midi)))})")

    pc_dist = _pitch_class_distribution(valid_midi)
    if pc_dist:
        print("  Pitch-class distribution:")
        for name, info in pc_dist.items():
            marker = " ◄ TARGET" if name == "G" else ""
            print(f"    {name:>3} (pc={info['pc']:>2}): "
                  f"{info['count']:>5} ({info['pct']:>5.1f}%){marker}")

    # --- Three-way comparison ---
    print(f"\n{'='*60}")
    print("THREE-WAY COMPARISON")
    print(f"{'='*60}")
    prev_path = PREV_CONTOUR_JSON if PREV_CONTOUR_JSON.exists() else None
    comparison = compare_three_ways(contour, REDUCED_JSON, prev_path)

    ns = comparison["note_event_stats"]
    ps = comparison["pyin_contour_stats"]
    hs = comparison["harmonic_contour_stats"]

    print(f"\n  {'Metric':<25} {'Note Events':>15} ", end="")
    if ps:
        print(f"{'pYIN Contour':>15} ", end="")
    print(f"{'Harmonic Sal.':>15}")

    print(f"  {'—'*25} {'—'*15} ", end="")
    if ps:
        print(f"{'—'*15} ", end="")
    print(f"{'—'*15}")

    print(f"  {'Active/Voiced %':<25} {ns['active_pct']:>14.1f}% ", end="")
    if ps:
        print(f"{ps['voiced_pct']:>14.1f}% ", end="")
    print(f"{hs['voiced_pct']:>14.1f}%")

    print(f"  {'Median MIDI':<25} {str(ns['median_midi']):>15} ", end="")
    if ps:
        print(f"{str(ps['median_midi']):>15} ", end="")
    print(f"{str(hs['median_midi']):>15}")

    print(f"  {'Median note':<25} {str(ns['median_note']):>15} ", end="")
    if ps:
        print(f"{str(ps['median_note']):>15} ", end="")
    print(f"{str(hs['median_note']):>15}")

    g = comparison["g_pitch_class"]
    print(f"  {'G (pc=7) %':<25} {g['note_events_pct']:>14.1f}% ", end="")
    if g["pyin_contour_pct"] is not None:
        print(f"{g['pyin_contour_pct']:>14.1f}% ", end="")
    print(f"{g['harmonic_contour_pct']:>14.1f}%")

    # Per-section
    for sec_name, sec_data in comparison["per_section"].items():
        print(f"\n  Section: {sec_name}")
        print(f"    Harmonic contour voiced: {sec_data['harm_voiced_pct']:.1f}%, "
              f"median MIDI: {sec_data['harm_median_midi']}")
        print(f"    Notes count: {sec_data['notes_count']}, "
              f"median MIDI: {sec_data['notes_median_midi']}")
        if sec_data["harm_pc_dist"]:
            top3 = list(sec_data["harm_pc_dist"].items())[:3]
            print(f"    Harmonic top PCs: "
                  + ", ".join(f"{n} {d['pct']:.1f}%" for n, d in top3))
        if sec_data["notes_pc_dist"]:
            top3 = list(sec_data["notes_pc_dist"].items())[:3]
            print(f"    Notes top PCs:    "
                  + ", ".join(f"{n} {d['pct']:.1f}%" for n, d in top3))

    print(f"\n  KEY QUESTION: Does harmonic salience detect G?")
    print(f"    Note events:       G = {g['note_events_pct']:.1f}%")
    if g["pyin_contour_pct"] is not None:
        print(f"    pYIN contour:      G = {g['pyin_contour_pct']:.1f}%")
    print(f"    Harmonic salience: G = {g['harmonic_contour_pct']:.1f}%")
    target_met = g["harmonic_contour_pct"] > 5.0
    print(f"    Target (>5%): {'YES ✓' if target_met else 'NO ✗'}")

    # --- Sonify ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = RESULTS_DIR / "bass_harmonic_contour.wav"
    sonify_contour(contour, wav_path)

    # --- Save JSON artifact ---
    contour_serializable = {
        "times_s": contour["times_s"].tolist(),
        "pitch_hz": [
            float(x) if np.isfinite(x) else None for x in contour["pitch_hz"]
        ],
        "energy": contour["energy"].tolist(),
        "activity": contour["activity"].tolist(),
        "confidence": contour["confidence"].tolist(),
        "n_frames": contour["n_frames"],
        "rms_threshold": contour["rms_threshold"],
    }

    artifact = {
        "diagnostic": "bass_harmonic_contour",
        "song": SONG,
        "params": {
            "sr": SR,
            "hop": HOP,
            "bpo": BPO,
            "n_bins": N_BINS,
            "fmin": FMIN,
            "harmonics": HARMONICS,
            "weights": WEIGHTS.tolist(),
            "bass_range": list(BASS_RANGE),
            "smooth_win": SMOOTH_WIN,
            "jump_thr_st": JUMP_THR_ST,
        },
        "contour": contour_serializable,
        "comparison": comparison,
    }
    out_path = RESULTS_DIR / "bass_harmonic_contour.json"
    out_path.write_text(json.dumps(artifact, indent=2, default=str) + "\n")
    print(f"\nJSON artifact saved to {out_path}")


if __name__ == "__main__":
    main()
