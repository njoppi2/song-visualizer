"""Bass contour prototype: continuous pitch+energy representation.

Tests whether a continuous pitch contour (lower voiced_prob threshold, no
semitone quantization) captures bass identity better than the failing
note-event extraction.

Key differences from production pYIN (features.py):
  - voiced_prob threshold: 0.3 (vs 0.75)
  - No semitone quantization (raw Hz kept)
  - Median smooth window: 5 (vs 7)

Usage:
    .songviz/venv/bin/python experiments/bass_contour_prototype.py
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
RESULTS_DIR = ROOT / "experiments" / "results" / "feel_good_inc"

SR = 22050
HOP = 512

SECTIONS = {
    "verse1+chorus1": (12.5, 63.0),
    "chorus2": (95.0, 138.0),
}

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Contour-specific parameters
VOICED_PROB_THR = 0.3  # lower than production 0.75
MEDIAN_WIN = 5         # slightly smaller than production 7
GAIN_BASS = 0.4
NORM_PEAK = 0.9


# ---------------------------------------------------------------------------
# _nanmedian_smooth — copied from songviz/features.py:16-41
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _hz_to_midi(hz: float | np.ndarray) -> float | np.ndarray:
    return 69.0 + 12.0 * np.log2(np.maximum(hz, 1e-8) / 440.0)


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
# Contour extraction
# ---------------------------------------------------------------------------
def extract_contour(y: np.ndarray) -> dict:
    """Extract continuous bass pitch contour with relaxed voicing threshold."""
    yh = librosa.effects.harmonic(y=y, margin=6.0).astype(np.float32, copy=False)

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

    # Activity mask (same formula as production)
    thr = float(np.percentile(rms, 25)) * 0.8
    thr = max(thr, float(rms.max()) * 0.08, 1e-6)
    activity = rms >= thr

    # Voicing gate with LOWER threshold
    voiced = activity & voiced_flag & (voiced_prob >= VOICED_PROB_THR)

    # Pitch: keep raw Hz (no quantization)
    pitch_hz = np.where(voiced, f0, np.nan).astype(np.float32)

    # Light median smoothing on MIDI domain
    midi = np.where(np.isfinite(pitch_hz), _hz_to_midi(pitch_hz), np.nan).astype(np.float32)
    midi = _nanmedian_smooth(midi, win=MEDIAN_WIN)
    # Convert back to Hz (still continuous, no rounding)
    pitch_hz = np.where(
        np.isfinite(midi),
        (440.0 * 2.0 ** ((midi - 69.0) / 12.0)).astype(np.float32),
        np.nan,
    )

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
        "voiced": voiced,
        "n_frames": n,
        "rms_threshold": thr,
    }


# ---------------------------------------------------------------------------
# Sonification
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
            # Advance without sound (keep phase reset for next voiced frame)
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
# Comparison with note events
# ---------------------------------------------------------------------------
def compare_with_note_events(contour: dict, reduced_path: Path) -> dict:
    """Compare contour stats with existing note-event extraction."""
    with open(reduced_path) as f:
        reduced = json.load(f)

    bass_layer = reduced.get("bass", {})
    bass_notes = bass_layer.get("notes", []) if isinstance(bass_layer, dict) else bass_layer

    # --- Note-event stats ---
    note_midis = np.array([n["midi"] for n in bass_notes], dtype=np.float32)
    note_total_dur = sum(n["offset_s"] - n["onset_s"] for n in bass_notes)
    song_dur = contour["times_s"][-1] if contour["n_frames"] > 0 else 1.0
    note_active_pct = 100.0 * note_total_dur / max(song_dur, 1.0)
    note_pc_dist = _pitch_class_distribution(note_midis)

    note_stats = {
        "n_notes": len(bass_notes),
        "total_dur_s": round(note_total_dur, 2),
        "active_pct": round(note_active_pct, 1),
        "median_midi": round(float(np.median(note_midis)), 2) if note_midis.size else None,
        "median_note": _midi_to_note(float(np.median(note_midis))) if note_midis.size else None,
        "pc_distribution": note_pc_dist,
    }

    # --- Contour stats ---
    pitch_hz = contour["pitch_hz"]
    midi = np.where(
        np.isfinite(pitch_hz),
        _hz_to_midi(pitch_hz),
        np.nan,
    ).astype(np.float32)
    valid_midi = midi[np.isfinite(midi)]
    n_voiced = int(np.sum(np.isfinite(pitch_hz)))
    voiced_pct = 100.0 * n_voiced / max(contour["n_frames"], 1)
    n_active = int(np.sum(contour["activity"]))
    active_pct = 100.0 * n_active / max(contour["n_frames"], 1)

    contour_pc_dist = _pitch_class_distribution(valid_midi)

    contour_stats = {
        "n_frames": contour["n_frames"],
        "n_voiced": n_voiced,
        "voiced_pct": round(voiced_pct, 1),
        "n_active": n_active,
        "active_pct": round(active_pct, 1),
        "median_midi": round(float(np.median(valid_midi)), 2) if valid_midi.size else None,
        "median_note": _midi_to_note(float(np.median(valid_midi))) if valid_midi.size else None,
        "median_hz": round(float(np.median(pitch_hz[np.isfinite(pitch_hz)])), 2) if valid_midi.size else None,
        "pc_distribution": contour_pc_dist,
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

        # Note events in this section
        sec_notes = [n for n in bass_notes if start_s <= n["onset_s"] < end_s]
        sec_note_midis = np.array([n["midi"] for n in sec_notes], dtype=np.float32)

        per_section[sec_name] = {
            "contour_voiced_pct": round(100.0 * sec_voiced / max(sec_total, 1), 1),
            "contour_median_midi": round(float(np.median(sec_valid)), 2) if sec_valid.size else None,
            "contour_pc_dist": _pitch_class_distribution(sec_valid),
            "notes_count": len(sec_notes),
            "notes_median_midi": round(float(np.median(sec_note_midis)), 2) if sec_note_midis.size else None,
            "notes_pc_dist": _pitch_class_distribution(sec_note_midis),
        }

    # --- Key question: does G (pc=7) appear? ---
    g_in_contour = contour_pc_dist.get("G", {}).get("pct", 0.0)
    g_in_notes = note_pc_dist.get("G", {}).get("pct", 0.0)

    return {
        "note_event_stats": note_stats,
        "contour_stats": contour_stats,
        "per_section": per_section,
        "g_pitch_class": {
            "contour_pct": g_in_contour,
            "note_event_pct": g_in_notes,
            "improvement": g_in_contour > g_in_notes,
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

    print("\nExtracting contour (voiced_prob ≥ 0.3, no quantization)...")
    contour = extract_contour(y)

    # --- Print contour summary ---
    pitch_hz = contour["pitch_hz"]
    valid_hz = pitch_hz[np.isfinite(pitch_hz)]
    valid_midi = _hz_to_midi(valid_hz) if valid_hz.size else np.array([])
    n_voiced = int(np.sum(np.isfinite(pitch_hz)))
    voiced_pct = 100.0 * n_voiced / max(contour["n_frames"], 1)

    print(f"\n{'='*60}")
    print(f"CONTOUR SUMMARY")
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
        print(f"  Pitch-class distribution:")
        for name, info in pc_dist.items():
            marker = " ◄" if name == "G" else ""
            print(f"    {name:>3} (pc={info['pc']:>2}): "
                  f"{info['count']:>5} ({info['pct']:>5.1f}%){marker}")

    # --- Compare with note events ---
    print(f"\n{'='*60}")
    print(f"COMPARISON: Contour vs Note Events")
    print(f"{'='*60}")
    comparison = compare_with_note_events(contour, REDUCED_JSON)

    ns = comparison["note_event_stats"]
    cs = comparison["contour_stats"]
    print(f"\n  {'Metric':<25} {'Note Events':>15} {'Contour':>15}")
    print(f"  {'—'*25} {'—'*15} {'—'*15}")
    print(f"  {'Active/Voiced %':<25} {ns['active_pct']:>14.1f}% {cs['voiced_pct']:>14.1f}%")
    print(f"  {'Median MIDI':<25} {str(ns['median_midi']):>15} {str(cs['median_midi']):>15}")
    print(f"  {'Median note':<25} {str(ns['median_note']):>15} {str(cs['median_note']):>15}")
    print(f"  {'G (pc=7) %':<25} "
          f"{comparison['g_pitch_class']['note_event_pct']:>14.1f}% "
          f"{comparison['g_pitch_class']['contour_pct']:>14.1f}%")

    # Per-section breakdown
    for sec_name, sec_data in comparison["per_section"].items():
        print(f"\n  Section: {sec_name}")
        print(f"    Contour voiced: {sec_data['contour_voiced_pct']:.1f}%, "
              f"median MIDI: {sec_data['contour_median_midi']}")
        print(f"    Notes count: {sec_data['notes_count']}, "
              f"median MIDI: {sec_data['notes_median_midi']}")
        if sec_data["contour_pc_dist"]:
            top3 = list(sec_data["contour_pc_dist"].items())[:3]
            print(f"    Contour top PCs: "
                  + ", ".join(f"{n} {d['pct']:.1f}%" for n, d in top3))
        if sec_data["notes_pc_dist"]:
            top3 = list(sec_data["notes_pc_dist"].items())[:3]
            print(f"    Notes top PCs:   "
                  + ", ".join(f"{n} {d['pct']:.1f}%" for n, d in top3))

    g = comparison["g_pitch_class"]
    print(f"\n  KEY QUESTION: Does lowered threshold help G detection?")
    print(f"    Note events: G = {g['note_event_pct']:.1f}%")
    print(f"    Contour:     G = {g['contour_pct']:.1f}%")
    print(f"    Improvement: {'YES' if g['improvement'] else 'NO'}")

    # --- Sonify ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = RESULTS_DIR / "bass_contour_prototype.wav"
    sonify_contour(contour, wav_path)

    # --- Save JSON ---
    # Convert numpy arrays to lists for JSON serialization
    contour_serializable = {
        "times_s": contour["times_s"].tolist(),
        "pitch_hz": [float(x) if np.isfinite(x) else None for x in contour["pitch_hz"]],
        "energy": contour["energy"].tolist(),
        "activity": contour["activity"].tolist(),
        "n_frames": contour["n_frames"],
        "rms_threshold": contour["rms_threshold"],
    }

    artifact = {
        "diagnostic": "bass_contour_prototype",
        "song": SONG,
        "params": {
            "voiced_prob_threshold": VOICED_PROB_THR,
            "median_smooth_window": MEDIAN_WIN,
            "sr": SR,
            "hop": HOP,
            "fmin": 30.0,
            "fmax": 400.0,
            "harmonic_margin": 6.0,
        },
        "contour": contour_serializable,
        "comparison": comparison,
    }
    out_path = RESULTS_DIR / "bass_contour_prototype.json"
    out_path.write_text(json.dumps(artifact, indent=2, default=str) + "\n")
    print(f"\nJSON artifact saved to {out_path}")


if __name__ == "__main__":
    main()
