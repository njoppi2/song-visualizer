"""Compare separation backends for Feel Good Inc.

Experiment A: Vocal front-end bake-off (4 models)
Experiment B: DrumSep representation vs frequency-band heuristic

Usage:
    .songviz/venv/bin/python experiments/compare_stems.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from songviz.features import vocals_pitch_hz, drums_band_energy_3

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SONG = "Gorillaz - Feel Good Inc (featuring De La Soul)"
RESULTS_DIR = ROOT / "experiments" / "results" / "feel_good_inc"

# Demucs stems (baseline)
DEMUCS_DIR = ROOT / "outputs" / SONG / "stems"

# Vocal models
VOCAL_MODELS = {
    "demucs_htdemucs": {
        "vocals": DEMUCS_DIR / "vocals.wav",
        "instrumental_parts": [DEMUCS_DIR / "drums.wav", DEMUCS_DIR / "bass.wav", DEMUCS_DIR / "other.wav"],
    },
    "melband_roformer": {
        "vocals": RESULTS_DIR / "melband_roformer" / f"{SONG}_(vocals)_vocals_mel_band_roformer.wav",
        "instrumental": RESULTS_DIR / "melband_roformer" / f"{SONG}_(other)_vocals_mel_band_roformer.wav",
    },
    "bs_roformer_viperx": {
        "vocals": RESULTS_DIR / "bs_roformer_viperx" / f"{SONG}_(Vocals)_model_bs_roformer_ep_368_sdr_12.wav",
        "instrumental": RESULTS_DIR / "bs_roformer_viperx" / f"{SONG}_(Instrumental)_model_bs_roformer_ep_368_sdr_12.wav",
    },
    "melband_bleedless": {
        "vocals": RESULTS_DIR / "melband_bleedless" / f"{SONG}_(vocals)_mel_band_roformer_kim_ft2_bleedless_unwa.wav",
        "instrumental": RESULTS_DIR / "melband_bleedless" / f"{SONG}_(other)_mel_band_roformer_kim_ft2_bleedless_unwa.wav",
    },
}

# DrumSep stems
DRUMSEP_DIR = RESULTS_DIR / "drumsep"
DRUMSEP_COMPONENTS = {
    "kick": DRUMSEP_DIR / "drums_(kick)_MDX23C-DrumSep-aufr33-jarredou.wav",
    "snare": DRUMSEP_DIR / "drums_(snare)_MDX23C-DrumSep-aufr33-jarredou.wav",
    "hh": DRUMSEP_DIR / "drums_(hh)_MDX23C-DrumSep-aufr33-jarredou.wav",
    "toms": DRUMSEP_DIR / "drums_(toms)_MDX23C-DrumSep-aufr33-jarredou.wav",
    "ride": DRUMSEP_DIR / "drums_(ride)_MDX23C-DrumSep-aufr33-jarredou.wav",
    "crash": DRUMSEP_DIR / "drums_(crash)_MDX23C-DrumSep-aufr33-jarredou.wav",
}

SR = 22050


def _load_mono(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y


# ---------------------------------------------------------------------------
# Proxy metrics
# ---------------------------------------------------------------------------

def voiced_frame_ratio(pitch_track: np.ndarray) -> float:
    if pitch_track.size == 0:
        return 0.0
    return float(np.isfinite(pitch_track).mean())


def cross_stem_rms_correlation(y_a: np.ndarray, y_b: np.ndarray) -> float:
    hop = 512
    rms_a = librosa.feature.rms(y=y_a, hop_length=hop)[0]
    rms_b = librosa.feature.rms(y=y_b, hop_length=hop)[0]
    n = min(len(rms_a), len(rms_b))
    if n < 2:
        return 0.0
    rms_a, rms_b = rms_a[:n], rms_b[:n]
    cc = np.corrcoef(rms_a, rms_b)[0, 1]
    return float(cc) if np.isfinite(cc) else 0.0


def pitch_jump_count(pitch_track: np.ndarray, *, semitone_thresh: float = 3.0) -> int:
    hz = pitch_track[np.isfinite(pitch_track)]
    if hz.size < 2:
        return 0
    midi = 69.0 + 12.0 * np.log2(np.maximum(hz, 1e-6) / 440.0)
    jumps = np.abs(np.diff(midi))
    return int((jumps > semitone_thresh).sum())


def onset_sharpness(y: np.ndarray) -> float:
    env = librosa.onset.onset_strength(y=y, sr=SR)
    if env.size == 0 or env.mean() < 1e-9:
        return 0.0
    return float(env.max() / env.mean())


def make_pitch_track_json(y_vocals: np.ndarray) -> list[list]:
    pitch = vocals_pitch_hz(y_vocals, SR)
    hop = 512
    times = librosa.frames_to_time(np.arange(len(pitch)), sr=SR, hop_length=hop)
    return [[round(float(t), 4), None if not np.isfinite(hz) else round(float(hz), 2)]
            for t, hz in zip(times, pitch)]


def make_onset_times_json(y: np.ndarray) -> list[float]:
    onsets = librosa.onset.onset_detect(y=y, sr=SR, units="time")
    return [round(float(t), 4) for t in onsets]


# ---------------------------------------------------------------------------
# Experiment A: Vocal front-end bake-off
# ---------------------------------------------------------------------------

def experiment_a() -> dict:
    print("=" * 60)
    print("EXPERIMENT A: Vocal Front-End Bake-Off")
    print("=" * 60)

    results = {}

    for name, paths in VOCAL_MODELS.items():
        print(f"\n  Loading {name}...")
        y_vocals = _load_mono(paths["vocals"])

        # Build instrumental
        if "instrumental" in paths:
            y_instr = _load_mono(paths["instrumental"])
        else:
            parts = [_load_mono(p) for p in paths["instrumental_parts"]]
            n = min(len(p) for p in parts)
            y_instr = sum(p[:n] for p in parts)

        print(f"  Computing pitch track...")
        pitch = vocals_pitch_hz(y_vocals, SR)
        pitch_json = make_pitch_track_json(y_vocals)

        voiced_count = sum(1 for _, hz in pitch_json if hz is not None)
        total_frames = len(pitch_json)

        m = {
            "voiced_frame_ratio": voiced_frame_ratio(pitch),
            "voiced_frames": voiced_count,
            "total_frames": total_frames,
            "pitch_jump_count_3st": pitch_jump_count(pitch, semitone_thresh=3.0),
            "pitch_jump_count_5st": pitch_jump_count(pitch, semitone_thresh=5.0),
            "vocal_instrumental_rms_corr": cross_stem_rms_correlation(y_vocals, y_instr),
        }
        results[name] = m

        # Save pitch track artifact
        out_dir = RESULTS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "pitch_track.json").write_text(json.dumps(pitch_json) + "\n")

    # Print comparison table
    print("\n" + "-" * 60)
    print("VOCAL MODEL COMPARISON")
    print("-" * 60)
    header = f"{'Metric':<35}" + "".join(f"{n:<20}" for n in results)
    print(header)
    print("-" * len(header))

    metric_keys = list(next(iter(results.values())).keys())
    for key in metric_keys:
        row = f"{key:<35}"
        for name in results:
            v = results[name][key]
            if isinstance(v, float):
                row += f"{v:<20.4f}"
            else:
                row += f"{v:<20}"
        print(row)

    return results


# ---------------------------------------------------------------------------
# Experiment B: DrumSep representation
# ---------------------------------------------------------------------------

def experiment_b() -> dict:
    print("\n" + "=" * 60)
    print("EXPERIMENT B: DrumSep Representation vs Frequency-Band Heuristic")
    print("=" * 60)

    # Load Demucs full drum stem
    print("\n  Loading Demucs drum stem...")
    demucs_drums = _load_mono(DEMUCS_DIR / "drums.wav")

    # Heuristic: drums_band_energy_3 on full drum stem
    print("  Computing frequency-band heuristic (drums_band_energy_3)...")
    heuristic_energy = drums_band_energy_3(demucs_drums, SR)

    # DrumSep: load individual components
    print("  Loading DrumSep components...")
    components = {}
    for comp_name, path in DRUMSEP_COMPONENTS.items():
        components[comp_name] = _load_mono(path)

    # Map DrumSep components to our 3-band scheme
    # kick → kick, snare+toms → snare, hh+ride+crash → hats
    hop = 512
    n_frames = heuristic_energy.shape[0]

    def _component_rms(y: np.ndarray) -> np.ndarray:
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        return rms[:n_frames] if len(rms) >= n_frames else np.pad(rms, (0, n_frames - len(rms)))

    kick_rms = _component_rms(components["kick"])
    snare_rms = _component_rms(components["snare"]) + _component_rms(components["toms"])
    hats_rms = _component_rms(components["hh"]) + _component_rms(components["ride"]) + _component_rms(components["crash"])

    # Normalize each to [0, 1] using 99th percentile cap (same as drums_band_energy_3)
    drumsep_energy = np.zeros((n_frames, 3), dtype=np.float32)
    for i, arr in enumerate([kick_rms, snare_rms, hats_rms]):
        cap = float(np.percentile(arr, 99)) if arr.size else 0.0
        cap = max(cap, float(arr.max()) if arr.size else 0.0, 1e-9)
        drumsep_energy[:, i] = np.clip(arr / cap, 0.0, 1.0)

    # Onset detection per component
    print("  Computing per-component onsets...")
    component_onsets = {}
    for comp_name, y in components.items():
        component_onsets[comp_name] = make_onset_times_json(y)

    # Full drum onset baseline
    full_onsets = make_onset_times_json(demucs_drums)

    # Correlation between heuristic and DrumSep energy curves
    print("\n" + "-" * 60)
    print("DRUMSEP vs FREQUENCY-BAND HEURISTIC")
    print("-" * 60)

    band_names = ["kick", "snare", "hats"]
    results = {"per_band_correlation": {}, "per_component_onsets": {}}
    for i, band in enumerate(band_names):
        cc = np.corrcoef(heuristic_energy[:, i], drumsep_energy[:, i])[0, 1]
        cc = float(cc) if np.isfinite(cc) else 0.0
        results["per_band_correlation"][band] = cc
        print(f"  {band} correlation (heuristic vs drumsep): {cc:.4f}")

    print(f"\n  Full drum onset count: {len(full_onsets)}")
    for comp_name, onsets in component_onsets.items():
        results["per_component_onsets"][comp_name] = len(onsets)
        print(f"  {comp_name} onsets: {len(onsets)}")

    # Onset sharpness per component vs full drum
    print(f"\n  Full drum onset sharpness: {onset_sharpness(demucs_drums):.2f}")
    results["component_onset_sharpness"] = {}
    for comp_name, y in components.items():
        sharp = onset_sharpness(y)
        results["component_onset_sharpness"][comp_name] = sharp
        print(f"  {comp_name} onset sharpness: {sharp:.2f}")

    # Save artifacts
    drumsep_out = RESULTS_DIR / "drumsep"
    np.save(str(drumsep_out / "drumsep_3band_energy.npy"), drumsep_energy)
    np.save(str(drumsep_out / "heuristic_3band_energy.npy"), heuristic_energy)
    for comp_name, onsets in component_onsets.items():
        (drumsep_out / f"onset_times_{comp_name}.json").write_text(json.dumps(onsets) + "\n")
    (drumsep_out / "onset_times_full.json").write_text(json.dumps(full_onsets) + "\n")

    # Qualitative check: do kick onsets align with heuristic kick peaks?
    kick_onset_arr = np.array(component_onsets["kick"])
    if kick_onset_arr.size > 0:
        kick_times = librosa.frames_to_time(np.arange(n_frames), sr=SR, hop_length=hop)
        # At each kick onset, what's the heuristic kick energy?
        heuristic_at_kick = []
        drumsep_at_kick = []
        for t in kick_onset_arr:
            idx = int(np.argmin(np.abs(kick_times - t)))
            if idx < n_frames:
                heuristic_at_kick.append(float(heuristic_energy[idx, 0]))
                drumsep_at_kick.append(float(drumsep_energy[idx, 0]))

        print(f"\n  At kick onsets ({len(kick_onset_arr)} events):")
        print(f"    Heuristic kick energy: mean={np.mean(heuristic_at_kick):.3f}, "
              f"median={np.median(heuristic_at_kick):.3f}")
        print(f"    DrumSep kick energy:   mean={np.mean(drumsep_at_kick):.3f}, "
              f"median={np.median(drumsep_at_kick):.3f}")
        results["kick_energy_at_onsets"] = {
            "heuristic_mean": float(np.mean(heuristic_at_kick)),
            "drumsep_mean": float(np.mean(drumsep_at_kick)),
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    vocal_results = experiment_a()
    drum_results = experiment_b()

    # Save combined results
    all_results = {
        "experiment_a_vocals": vocal_results,
        "experiment_b_drums": drum_results,
    }
    out_path = RESULTS_DIR / "comparison_v2.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str) + "\n")
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
