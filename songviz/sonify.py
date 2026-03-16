"""Sonify a reduced.json into a WAV file for debug/validation.

Reads drum hits, vocal notes, and bass notes from the reduced representation
and renders them as simple waveforms: noise/sine bursts for drums, sine for
vocals, triangle wave for bass.  Output is a mono WAV at ``SR`` Hz.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

# ── Tunable constants ──
# All of these are intentionally module-level so they're easy to tweak
# after listening to a few songs.

SR: int = 22050
"""Default sample rate (Hz)."""

# Per-layer pre-normalization gain.  After mixing, the combined signal is
# peak-normalized to 0.9, so these only control *relative* loudness.
GAIN_DRUMS: float = 0.5
GAIN_VOCALS: float = 0.7
GAIN_BASS: float = 0.4

# Drum sound durations (seconds).
DUR_KICK: float = 0.060
DUR_SNARE: float = 0.040
DUR_TOMS: float = 0.050
DUR_HH: float = 0.015
DUR_RIDE: float = 0.025
DUR_CRASH: float = 0.080

# Envelope ramps (seconds).
ATTACK_S: float = 0.005
RELEASE_S: float = 0.010

# Kick sine-sweep range (Hz).
KICK_FREQ_START: float = 150.0
KICK_FREQ_END: float = 50.0

# Toms sine frequency (Hz).  Lower than snare, higher than kick end.
TOMS_FREQ: float = 120.0

# Duration of the tail buffer appended after the last event (seconds).
TAIL_S: float = 0.1

# Peak level after normalization (headroom).
NORM_PEAK: float = 0.9

# ── Internal helpers ──


def _apply_envelope(
    buf: np.ndarray,
    sr: int,
    attack_s: float = ATTACK_S,
    release_s: float = RELEASE_S,
) -> np.ndarray:
    """Apply a linear attack/release envelope to *buf* (in-place safe)."""
    n = len(buf)
    attack_n = min(int(attack_s * sr), n)
    release_n = min(int(release_s * sr), max(n - attack_n, 0))
    env = np.ones(n, dtype=np.float32)
    if attack_n > 0:
        env[:attack_n] = np.linspace(0.0, 1.0, attack_n, dtype=np.float32)
    if release_n > 0:
        env[-release_n:] = np.linspace(1.0, 0.0, release_n, dtype=np.float32)
    return buf * env


# ── Drum template synthesis ──


def _synth_kick(sr: int) -> np.ndarray:
    """Low sine sweep (150 -> 50 Hz)."""
    n = int(DUR_KICK * sr)
    t = np.arange(n, dtype=np.float64) / sr
    freq = KICK_FREQ_START * (KICK_FREQ_END / KICK_FREQ_START) ** (t / DUR_KICK)
    phase = np.cumsum(freq / sr)
    sig = np.sin(2.0 * np.pi * phase).astype(np.float32)
    return _apply_envelope(sig, sr)


def _synth_snare(sr: int) -> np.ndarray:
    """Deterministic white-noise burst."""
    n = int(DUR_SNARE * sr)
    sig = np.random.default_rng(42).uniform(-1.0, 1.0, n).astype(np.float32)
    return _apply_envelope(sig, sr)


def _synth_toms(sr: int) -> np.ndarray:
    """Mid-pitched sine burst at 120 Hz — distinguishable from kick sweep."""
    n = int(DUR_TOMS * sr)
    t = np.arange(n, dtype=np.float64) / sr
    sig = np.sin(2.0 * np.pi * TOMS_FREQ * t).astype(np.float32)
    return _apply_envelope(sig, sr)


def _synth_hh(sr: int) -> np.ndarray:
    """Very short deterministic noise tick."""
    n = int(DUR_HH * sr)
    sig = np.random.default_rng(7).uniform(-1.0, 1.0, n).astype(np.float32)
    return _apply_envelope(sig, sr)


def _synth_ride(sr: int) -> np.ndarray:
    """Slightly longer noise ping — distinct from hh."""
    n = int(DUR_RIDE * sr)
    sig = np.random.default_rng(13).uniform(-1.0, 1.0, n).astype(np.float32)
    return _apply_envelope(sig, sr)


def _synth_crash(sr: int) -> np.ndarray:
    """Longer noise burst."""
    n = int(DUR_CRASH * sr)
    sig = np.random.default_rng(99).uniform(-1.0, 1.0, n).astype(np.float32)
    return _apply_envelope(sig, sr)


_DRUM_SYNTH: dict[str, Any] = {
    "kick": _synth_kick,
    "snare": _synth_snare,
    "toms": _synth_toms,
    "hh": _synth_hh,
    "ride": _synth_ride,
    "crash": _synth_crash,
}

_DRUM_DUR: dict[str, float] = {
    "kick": DUR_KICK,
    "snare": DUR_SNARE,
    "toms": DUR_TOMS,
    "hh": DUR_HH,
    "ride": DUR_RIDE,
    "crash": DUR_CRASH,
}


# ── Pitched-note synthesis ──


def _synth_sine(midi: float, duration_s: float, sr: int) -> np.ndarray:
    """Pure sine wave at the MIDI pitch — used for vocals."""
    freq = 440.0 * 2.0 ** ((midi - 69.0) / 12.0)
    n = max(1, int(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / sr
    sig = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    return _apply_envelope(sig, sr)


def _synth_triangle(midi: float, duration_s: float, sr: int) -> np.ndarray:
    """Triangle wave (3-harmonic additive) at the MIDI pitch — used for bass.

    Adds odd harmonics so low bass notes are audible on laptop speakers.
    """
    freq = 440.0 * 2.0 ** ((midi - 69.0) / 12.0)
    n = max(1, int(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / sr
    omega = 2.0 * np.pi * freq
    sig = (
        np.sin(omega * t)
        - (1.0 / 9.0) * np.sin(3.0 * omega * t)
        + (1.0 / 25.0) * np.sin(5.0 * omega * t)
    ).astype(np.float32)
    # Normalize raw waveform peak to ~1.0
    peak = np.max(np.abs(sig))
    if peak > 1e-8:
        sig /= peak
    return _apply_envelope(sig, sr)


# ── Mixing ──


def _mix_into(
    buf: np.ndarray, signal: np.ndarray, onset_s: float, gain: float, sr: int,
) -> None:
    """Add *signal* * *gain* into *buf* at the sample offset for *onset_s*."""
    start = int(onset_s * sr)
    end = start + len(signal)
    sig_start = max(0, -start)
    buf_start = max(0, start)
    buf_end = min(len(buf), end)
    sig_end = sig_start + (buf_end - buf_start)
    if buf_start < buf_end:
        buf[buf_start:buf_end] += signal[sig_start:sig_end] * gain


def _compute_duration_samples(reduced: dict[str, Any], sr: int) -> int:
    """Return total buffer length in samples from all events."""
    max_t = 0.5  # minimum duration
    for hit in reduced.get("drums", {}).get("hits", []):
        comp = hit.get("component", "kick")
        dur = _DRUM_DUR.get(comp, DUR_KICK)
        max_t = max(max_t, hit["t"] + dur)
    for layer in ("vocals", "bass"):
        for note in reduced.get(layer, {}).get("notes", []):
            max_t = max(max_t, note.get("offset_s", 0.0))
    return int((max_t + TAIL_S) * sr)


# ── Public API ──


def sonify_reduced(
    reduced: dict[str, Any],
    out_path: Path,
    sr: int = SR,
) -> None:
    """Render a reduced-representation dict into a mono WAV file.

    Parameters
    ----------
    reduced : parsed ``reduced.json`` content (keys: drums, vocals, bass)
    out_path : destination WAV path (parent dirs created automatically)
    sr : sample rate (default :data:`SR`)
    """
    n_samples = _compute_duration_samples(reduced, sr)
    buf = np.zeros(n_samples, dtype=np.float64)

    # Pre-compute drum templates (one per component, reused for every hit)
    drum_templates = {name: fn(sr) for name, fn in _DRUM_SYNTH.items()}

    # ── Drums ──
    for hit in reduced.get("drums", {}).get("hits", []):
        comp = hit.get("component", "kick")
        vel = hit.get("velocity", 0.5)
        template = drum_templates.get(comp, drum_templates["kick"])
        _mix_into(buf, template, hit["t"], vel * GAIN_DRUMS, sr)

    # ── Vocals ──
    for note in reduced.get("vocals", {}).get("notes", []):
        dur = note["offset_s"] - note["onset_s"]
        if dur <= 0:
            continue
        vel = note.get("velocity", 0.5)
        sig = _synth_sine(note["midi"], dur, sr)
        _mix_into(buf, sig, note["onset_s"], vel * GAIN_VOCALS, sr)

    # ── Bass ──
    for note in reduced.get("bass", {}).get("notes", []):
        dur = note["offset_s"] - note["onset_s"]
        if dur <= 0:
            continue
        vel = note.get("velocity", 0.5)
        sig = _synth_triangle(note["midi"], dur, sr)
        _mix_into(buf, sig, note["onset_s"], vel * GAIN_BASS, sr)

    # ── Normalize ──
    peak = float(np.max(np.abs(buf)))
    if peak > 1e-6:
        buf *= NORM_PEAK / peak

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), buf.astype(np.float32), sr)


# ── Per-layer rendering (for debug WAVs) ──


def _render_layer(
    reduced: dict[str, Any],
    layer: str,
    sr: int,
    n_samples: int,
) -> np.ndarray:
    """Render a single layer into a float64 buffer (no normalization)."""
    buf = np.zeros(n_samples, dtype=np.float64)
    if layer == "drums":
        templates = {name: fn(sr) for name, fn in _DRUM_SYNTH.items()}
        for hit in reduced.get("drums", {}).get("hits", []):
            comp = hit.get("component", "kick")
            vel = hit.get("velocity", 0.5)
            tmpl = templates.get(comp, templates["kick"])
            _mix_into(buf, tmpl, hit["t"], vel * GAIN_DRUMS, sr)
    elif layer in ("vocals", "bass"):
        synth_fn = _synth_sine if layer == "vocals" else _synth_triangle
        gain = GAIN_VOCALS if layer == "vocals" else GAIN_BASS
        for note in reduced.get(layer, {}).get("notes", []):
            dur = note["offset_s"] - note["onset_s"]
            if dur <= 0:
                continue
            vel = note.get("velocity", 0.5)
            sig = synth_fn(note["midi"], dur, sr)
            _mix_into(buf, sig, note["onset_s"], vel * gain, sr)
    return buf


def _normalize_and_write(buf: np.ndarray, path: Path, sr: int) -> None:
    peak = float(np.max(np.abs(buf)))
    if peak > 1e-6:
        buf = buf * (NORM_PEAK / peak)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), buf.astype(np.float32), sr)


def sonify_reduced_layers(
    reduced: dict[str, Any],
    out_dir: Path,
    sr: int = SR,
) -> dict[str, Path]:
    """Write per-layer debug WAVs and return a dict of name → path.

    Writes: reduced_drums_only.wav, reduced_vocals_only.wav,
    reduced_bass_only.wav, reduced_vocals_plus_bass.wav,
    reduced_bass_only_up1oct.wav.
    """
    import copy

    n_samples = _compute_duration_samples(reduced, sr)
    paths: dict[str, Path] = {}

    for layer in ("drums", "vocals", "bass"):
        buf = _render_layer(reduced, layer, sr, n_samples)
        p = out_dir / f"reduced_{layer}_only.wav"
        _normalize_and_write(buf, p, sr)
        paths[f"{layer}_only"] = p

    # vocals + bass combined
    vb = _render_layer(reduced, "vocals", sr, n_samples) + _render_layer(reduced, "bass", sr, n_samples)
    p = out_dir / "reduced_vocals_plus_bass.wav"
    _normalize_and_write(vb, p, sr)
    paths["vocals_plus_bass"] = p

    # bass transposed up one octave (audibility check)
    bass_up = copy.deepcopy(reduced)
    for note in bass_up.get("bass", {}).get("notes", []):
        note["midi"] += 12
    buf = _render_layer(bass_up, "bass", sr, n_samples)
    p = out_dir / "reduced_bass_only_up1oct.wav"
    _normalize_and_write(buf, p, sr)
    paths["bass_only_up1oct"] = p

    return paths


# ── Diagnostics ──


def diagnose_reduced(
    reduced: dict[str, Any],
    sr: int = SR,
) -> dict[str, Any]:
    """Compute diagnostic stats and warning heuristics for a reduced dict.

    Returns a dict with per-layer stats, energy fractions, and a ``warnings``
    list of short string codes describing detected problems.
    """
    n_samples = _compute_duration_samples(reduced, sr)
    song_dur_s = n_samples / sr

    # Render per-layer buffers (pre-normalization)
    drum_buf = _render_layer(reduced, "drums", sr, n_samples)
    vocal_buf = _render_layer(reduced, "vocals", sr, n_samples)
    bass_buf = _render_layer(reduced, "bass", sr, n_samples)

    drum_energy = float(np.sum(drum_buf ** 2))
    vocal_energy = float(np.sum(vocal_buf ** 2))
    bass_energy = float(np.sum(bass_buf ** 2))
    total_energy = drum_energy + vocal_energy + bass_energy

    def _rms(buf: np.ndarray) -> float:
        return float(np.sqrt(np.mean(buf ** 2)))

    def _energy_pct(e: float) -> float:
        return 100.0 * e / total_energy if total_energy > 1e-12 else 0.0

    # ── Drum stats ──
    hits = reduced.get("drums", {}).get("hits", [])
    comp_counts: dict[str, int] = {}
    for h in hits:
        c = h.get("component", "kick")
        comp_counts[c] = comp_counts.get(c, 0) + 1

    drums_stats: dict[str, Any] = {
        "event_count": len(hits),
        "density_hits_per_s": len(hits) / song_dur_s if song_dur_s > 0 else 0.0,
        "component_counts": comp_counts,
        "rms": _rms(drum_buf),
        "energy_pct": _energy_pct(drum_energy),
    }

    # ── Pitched layer stats ──
    def _pitched_stats(layer: str, buf: np.ndarray, energy: float) -> dict[str, Any]:
        notes = reduced.get(layer, {}).get("notes", [])
        source = reduced.get(layer, {}).get("source", "unknown")
        if not notes:
            return {
                "source": source,
                "event_count": 0,
                "total_active_s": 0.0,
                "coverage_pct": 0.0,
                "mean_duration_s": 0.0,
                "median_duration_s": 0.0,
                "midi_mean": 0.0,
                "midi_median": 0.0,
                "midi_min": 0.0,
                "midi_max": 0.0,
                "rms": _rms(buf),
                "energy_pct": _energy_pct(energy),
            }
        durs = [n["offset_s"] - n["onset_s"] for n in notes]
        midis = [n["midi"] for n in notes]
        total_active = sum(d for d in durs if d > 0)
        return {
            "source": source,
            "event_count": len(notes),
            "total_active_s": round(total_active, 2),
            "coverage_pct": round(100.0 * total_active / song_dur_s, 1) if song_dur_s > 0 else 0.0,
            "mean_duration_s": round(float(np.mean(durs)), 3),
            "median_duration_s": round(float(np.median(durs)), 3),
            "midi_mean": round(float(np.mean(midis)), 1),
            "midi_median": round(float(np.median(midis)), 1),
            "midi_min": float(np.min(midis)),
            "midi_max": float(np.max(midis)),
            "rms": _rms(buf),
            "energy_pct": _energy_pct(energy),
        }

    vocals_stats = _pitched_stats("vocals", vocal_buf, vocal_energy)
    bass_stats = _pitched_stats("bass", bass_buf, bass_energy)

    # ── Bass dynamics & isolation diagnostics ──
    bass_notes = reduced.get("bass", {}).get("notes", [])
    if bass_notes:
        velocities = [n["velocity"] for n in bass_notes]
        bass_stats["velocity_min"] = round(min(velocities), 4)
        bass_stats["velocity_max"] = round(max(velocities), 4)
        bass_stats["velocity_p10"] = round(float(np.percentile(velocities, 10)), 4)

        iso = 0
        for i, n in enumerate(bass_notes):
            gap_before = n["onset_s"] - bass_notes[i - 1]["offset_s"] if i > 0 else float("inf")
            gap_after = bass_notes[i + 1]["onset_s"] - n["offset_s"] if i < len(bass_notes) - 1 else float("inf")
            if gap_before > 4.0 and gap_after > 4.0:
                iso += 1
        bass_stats["isolated_note_count"] = iso

    # ── Warning heuristics ──
    warnings: list[str] = []

    if drums_stats["energy_pct"] > 60.0:
        warnings.append("drums_dominate")

    if vocals_stats["coverage_pct"] < 20.0:
        warnings.append("sparse_vocals")

    if bass_stats["coverage_pct"] < 15.0:
        warnings.append("sparse_bass")

    if vocals_stats["event_count"] > 0 and vocals_stats["midi_mean"] < 55.0:
        warnings.append("low_vocal_pitch")

    if (bass_stats["event_count"] > 50
            and bass_stats["median_duration_s"] < 0.2):
        warnings.append("fragmented_bass")

    if vocals_stats["source"] == "pyin":
        warnings.append("basic_pitch_unavailable_or_failed")

    if bass_stats.get("isolated_note_count", 0) > 5:
        warnings.append("many_isolated_bass_notes")
    if bass_stats.get("velocity_min", 1.0) > 0.15 and bass_stats["event_count"] > 20:
        warnings.append("bass_velocity_floor_high")

    return {
        "song_duration_s": round(song_dur_s, 1),
        "drums": drums_stats,
        "vocals": vocals_stats,
        "bass": bass_stats,
        "warnings": warnings,
    }
