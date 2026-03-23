"""Reduced representation: convert continuous per-stem features into discrete events.

Drum hits live under ``"drums"``, vocal notes under ``"vocals"``, bass notes
under ``"bass"`` in ``analysis/reduced.json``.
"""
from __future__ import annotations

from typing import Any, Sequence

import librosa
import numpy as np

# ── Reduced representation version ──
_REDUCED_SCHEMA_VERSION = 1

# ── Per-component peak-pick tuning ──
# At hop_length=512, sr=22050: 1 frame ≈ 23.2 ms.
#
# wait  = min frames between consecutive hits for this component
# delta = min onset-strength prominence to count as a peak
# pre_max / post_max = frames for local-max window
# pre_avg / post_avg = frames for local-mean comparison window
#
# Rationale:
#   hh/ride can fire as fast as 16th notes at ~160 BPM (~94 ms ≈ 4 frames)
#   kick/snare rarely faster than 8th notes (~188 ms ≈ 8 frames)
#   crash is sparse but has long decay, needs wider window

PEAK_PICK_PARAMS: dict[str, dict[str, int | float]] = {
    "kick":  {"pre_max": 3, "post_max": 3, "pre_avg": 7, "post_avg": 7, "delta": 0.07, "wait": 7},
    "snare": {"pre_max": 3, "post_max": 3, "pre_avg": 7, "post_avg": 7, "delta": 0.07, "wait": 7},
    "toms":  {"pre_max": 3, "post_max": 3, "pre_avg": 7, "post_avg": 7, "delta": 0.07, "wait": 7},
    "hh":    {"pre_max": 2, "post_max": 2, "pre_avg": 5, "post_avg": 5, "delta": 0.05, "wait": 3},
    "ride":  {"pre_max": 2, "post_max": 2, "pre_avg": 5, "post_avg": 5, "delta": 0.05, "wait": 3},
    "crash": {"pre_max": 3, "post_max": 3, "pre_avg": 7, "post_avg": 7, "delta": 0.06, "wait": 5},
}

# Frequency band boundaries for fallback classification (Hz).
FALLBACK_BANDS: dict[str, tuple[float, float | None]] = {
    "kick": (20.0, 150.0),
    "snare": (150.0, 2500.0),
    "hh": (2500.0, None),
}


def _compute_beat_alignment(hit_time: float, beat_times_s: np.ndarray) -> dict[str, Any]:
    """Return beat_idx and beat_phase for a single hit time."""
    idx = int(np.searchsorted(beat_times_s, hit_time))
    n = len(beat_times_s)

    # Find nearest beat
    if idx == 0:
        nearest = 0
    elif idx >= n:
        nearest = n - 1
    else:
        if abs(hit_time - beat_times_s[idx - 1]) <= abs(hit_time - beat_times_s[idx]):
            nearest = idx - 1
        else:
            nearest = idx

    beat_time = float(beat_times_s[nearest])

    # Compute phase within the surrounding beat interval
    if nearest < n - 1:
        interval = float(beat_times_s[nearest + 1] - beat_times_s[nearest])
    elif nearest > 0:
        interval = float(beat_times_s[nearest] - beat_times_s[nearest - 1])
    else:
        interval = 0.0

    if interval > 0:
        phase = (hit_time - beat_time) / interval
        # Wrap to [0.0, 1.0)
        phase = phase % 1.0
    else:
        phase = 0.0

    return {"beat_idx": nearest, "beat_phase": round(phase, 4)}


def extract_drum_hits(
    components: dict[str, np.ndarray],
    sr: int,
    *,
    hop_length: int = 512,
    beat_times_s: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Extract drum hits from DrumSep component waveforms.

    Parameters
    ----------
    components : dict mapping component name → mono waveform (numpy float32)
    sr : sample rate
    hop_length : hop length for onset/RMS computation
    beat_times_s : beat grid for alignment (optional)

    Returns
    -------
    dict with ``"source": "drumsep"`` and ``"hits": [...]``
    """
    beat_arr = np.asarray(beat_times_s, dtype=np.float64) if beat_times_s is not None and len(beat_times_s) > 0 else None
    all_hits: list[dict[str, Any]] = []

    for comp_name, y in components.items():
        if comp_name not in PEAK_PICK_PARAMS:
            continue

        # Skip silent components
        rms_check = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        if rms_check.max() < 1e-7:
            continue

        # Onset envelope (spectral flux — good for transient detection)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        # RMS for velocity
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        params = PEAK_PICK_PARAMS[comp_name]
        peaks = librosa.util.peak_pick(
            onset_env,
            pre_max=int(params["pre_max"]),
            post_max=int(params["post_max"]),
            pre_avg=int(params["pre_avg"]),
            post_avg=int(params["post_avg"]),
            delta=float(params["delta"]),
            wait=int(params["wait"]),
        )

        if len(peaks) == 0:
            continue

        # Gather RMS at peak frames (clamp to valid range)
        peak_rms_values = rms[np.clip(peaks, 0, len(rms) - 1)]

        # Normalisation cap: 99th percentile of detected-peak RMS
        cap = float(np.percentile(peak_rms_values, 99)) if len(peak_rms_values) > 0 else 1.0
        if cap < 1e-10:
            cap = 1.0

        for frame_idx, raw_vel in zip(peaks, peak_rms_values):
            t = float(librosa.frames_to_time(int(frame_idx), sr=sr, hop_length=hop_length))
            vel_norm = float(np.clip(raw_vel / cap, 0.0, 1.0))
            hit: dict[str, Any] = {
                "t": round(t, 4),
                "component": comp_name,
                "velocity": round(vel_norm, 4),
                "velocity_raw": round(float(raw_vel), 6),
            }
            if beat_arr is not None:
                hit.update(_compute_beat_alignment(t, beat_arr))
            all_hits.append(hit)

    all_hits.sort(key=lambda h: h["t"])
    return {"source": "drumsep", "hits": all_hits}


def extract_drum_hits_fallback(
    y_drums: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
    n_fft: int = 2048,
    beat_times_s: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Heuristic drum hit extraction when DrumSep is unavailable.

    Detects onsets on the full drum stem, then classifies each by
    spectral band energy.
    """
    beat_arr = np.asarray(beat_times_s, dtype=np.float64) if beat_times_s is not None and len(beat_times_s) > 0 else None

    onset_frames = librosa.onset.onset_detect(
        y=y_drums, sr=sr, hop_length=hop_length, backtrack=True, units="frames",
    )

    if len(onset_frames) == 0:
        return {"source": "heuristic", "hits": []}

    # RMS for velocity
    rms = librosa.feature.rms(y=y_drums, hop_length=hop_length)[0]

    # Compute STFT once for classification
    S = np.abs(librosa.stft(y_drums, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    all_hits: list[dict[str, Any]] = []

    # RMS values at onset frames for normalisation
    onset_rms_values = rms[np.clip(onset_frames, 0, len(rms) - 1)]
    cap = float(np.percentile(onset_rms_values, 99)) if len(onset_rms_values) > 0 else 1.0
    if cap < 1e-10:
        cap = 1.0

    for frame_idx in onset_frames:
        t = float(librosa.frames_to_time(int(frame_idx), sr=sr, hop_length=hop_length))

        # Classify by spectral band energy
        col = S[:, min(int(frame_idx), S.shape[1] - 1)]
        band_energy: dict[str, float] = {}
        for band_name, (lo, hi) in FALLBACK_BANDS.items():
            mask = freqs >= lo
            if hi is not None:
                mask = mask & (freqs < hi)
            band_energy[band_name] = float(np.sum(col[mask] ** 2))

        component = max(band_energy, key=band_energy.get)  # type: ignore[arg-type]

        raw_vel = float(rms[min(int(frame_idx), len(rms) - 1)])
        vel_norm = float(np.clip(raw_vel / cap, 0.0, 1.0))

        hit: dict[str, Any] = {
            "t": round(t, 4),
            "component": component,
            "velocity": round(vel_norm, 4),
            "velocity_raw": round(float(raw_vel), 6),
        }
        if beat_arr is not None:
            hit.update(_compute_beat_alignment(t, beat_arr))
        all_hits.append(hit)

    all_hits.sort(key=lambda h: h["t"])
    return {"source": "heuristic", "hits": all_hits}


# ── Pitch-track → note events (shared by vocals and bass) ──


def _notes_from_pitch_track(
    pitch_hz: np.ndarray,
    y: np.ndarray,
    sr: int,
    *,
    source: str,
    hop_length: int = 512,
    min_note_frames: int = 3,
    max_gap_frames: int = 0,
    beat_times_s: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Convert a frame-level Hz pitch track into discrete note events.

    This is the stem-agnostic core shared by vocals and bass extraction.

    Parameters
    ----------
    pitch_hz : per-frame fundamental frequency (NaN = unvoiced)
    y : mono waveform (for RMS-based velocity)
    sr : sample rate
    source : label for the ``"source"`` field (e.g. ``"pyin"``)
    hop_length : analysis hop
    min_note_frames : minimum consecutive frames to keep a note (default 3 ≈ 70 ms)
    max_gap_frames : merge same-MIDI notes separated by ≤ this many NaN frames
        (default 0 = no merging). Use ~3 for bass to bridge brief pYIN dropouts.
    beat_times_s : beat grid for alignment (optional)

    Returns
    -------
    dict with ``"source"`` and ``"notes": [...]``
    """
    beat_arr = (
        np.asarray(beat_times_s, dtype=np.float64)
        if beat_times_s is not None and len(beat_times_s) > 0
        else None
    )

    # Hz → MIDI (NaN stays NaN)
    with np.errstate(divide="ignore", invalid="ignore"):
        midi_raw = 69.0 + 12.0 * np.log2(pitch_hz / 440.0)
    midi_rounded = np.round(midi_raw).astype(float)  # NaN rounds to NaN

    # Group consecutive frames with the same rounded MIDI
    notes_raw: list[tuple[int, int, float]] = []  # (start_frame, end_frame, midi)
    n_frames = len(midi_rounded)
    i = 0
    while i < n_frames:
        m = midi_rounded[i]
        if np.isnan(m):
            i += 1
            continue
        j = i + 1
        while j < n_frames and midi_rounded[j] == m:
            j += 1
        if j - i >= min_note_frames:
            notes_raw.append((i, j - 1, m))
        i = j

    # Merge same-MIDI notes separated by small NaN gaps
    if max_gap_frames > 0 and len(notes_raw) > 1:
        merged: list[tuple[int, int, float]] = [notes_raw[0]]
        for start_f, end_f, midi_val in notes_raw[1:]:
            prev_start, prev_end, prev_midi = merged[-1]
            gap = start_f - prev_end - 1
            if midi_val == prev_midi and gap <= max_gap_frames:
                merged[-1] = (prev_start, end_f, midi_val)
            else:
                merged.append((start_f, end_f, midi_val))
        notes_raw = merged

    if not notes_raw:
        return {"source": source, "notes": []}

    # Compute per-note RMS velocity
    rms_values: list[float] = []
    for start_f, end_f, _ in notes_raw:
        s0 = start_f * hop_length
        s1 = min((end_f + 1) * hop_length, len(y))
        segment = y[s0:s1]
        rms_values.append(float(np.sqrt(np.mean(segment.astype(np.float64) ** 2))) if len(segment) > 0 else 0.0)

    # 99th-percentile cap for normalisation (same pattern as drum hits)
    cap = float(np.percentile(rms_values, 99)) if rms_values else 1.0
    if cap < 1e-10:
        cap = 1.0

    notes: list[dict[str, Any]] = []
    for (start_f, end_f, midi_val), raw_vel in zip(notes_raw, rms_values):
        onset_s = float(start_f * hop_length / sr)
        offset_s = float((end_f + 1) * hop_length / sr)
        vel_norm = float(np.clip(raw_vel / cap, 0.0, 1.0))
        note: dict[str, Any] = {
            "onset_s": round(onset_s, 4),
            "offset_s": round(offset_s, 4),
            "midi": float(midi_val),
            "velocity": round(vel_norm, 4),
        }
        if beat_arr is not None:
            note.update(_compute_beat_alignment(onset_s, beat_arr))
        notes.append(note)

    return {"source": source, "notes": notes}


def extract_vocal_notes_from_pitch_track(
    pitch_hz: np.ndarray,
    y: np.ndarray,
    sr: int,
    **kw: Any,
) -> dict[str, Any]:
    """Convert a pYIN vocal pitch track into discrete note events.

    Thin wrapper around :func:`_notes_from_pitch_track` with ``source="pyin"``.
    """
    return _notes_from_pitch_track(pitch_hz, y, sr, source="pyin", **kw)


def extract_vocal_notes(
    note_events: list[dict[str, float]] | None,
    pitch_hz: np.ndarray | None,
    y: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
    beat_times_s: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Top-level vocal note dispatcher — picks best available source.

    Parameters
    ----------
    note_events : basic-pitch output (list of {start_s, end_s, midi, velocity}), or None
    pitch_hz : pYIN frame-level Hz pitch track, or None
    y : mono waveform (for pYIN velocity computation)
    sr : sample rate
    hop_length : analysis hop
    beat_times_s : beat grid for alignment (optional)

    Returns
    -------
    dict with ``"source"`` and ``"notes"`` keys
    """
    # Primary: basic-pitch note events
    if note_events is not None and len(note_events) > 0:
        beat_arr = (
            np.asarray(beat_times_s, dtype=np.float64)
            if beat_times_s is not None and len(beat_times_s) > 0
            else None
        )
        notes: list[dict[str, Any]] = []
        for ev in note_events:
            note: dict[str, Any] = {
                "onset_s": round(float(ev["start_s"]), 4),
                "offset_s": round(float(ev["end_s"]), 4),
                "midi": round(float(ev["midi"]), 2),
                "velocity": round(float(ev["velocity"]), 4),
            }
            if beat_arr is not None:
                note.update(_compute_beat_alignment(note["onset_s"], beat_arr))
            notes.append(note)

        # Clean up octave artifacts from basic-pitch (vocal range 36--96)
        notes = _correct_octave_by_context(notes, midi_lo=36, midi_hi=96)
        notes = _smooth_vocal_octave_jumps(notes)

        return {"source": "basic_pitch", "notes": notes}

    # Fallback: pYIN pitch track
    if pitch_hz is not None and np.any(np.isfinite(pitch_hz)):
        result = extract_vocal_notes_from_pitch_track(
            pitch_hz, y, sr,
            hop_length=hop_length,
            beat_times_s=beat_times_s,
        )
        # Clean up octave artifacts from pYIN (vocal range 36--96)
        result["notes"] = _correct_octave_by_context(
            result["notes"], midi_lo=36, midi_hi=96,
        )
        result["notes"] = _smooth_vocal_octave_jumps(result["notes"])
        return result

    # No data
    return {"source": "none", "notes": []}


# ── Bass note extraction ──

# Octave-correction parameters for basic-pitch bass output.
_OCTAVE_CORRECT_WINDOW = 5      # neighbours each side for local median
_OCTAVE_CORRECT_MIN_GAIN = 6    # min semitone improvement to accept a shift

# Bass energy gating & isolation pruning parameters.
_BASS_RMS_GATE_PERCENTILE: int = 10
_BASS_RMS_GATE_SCALE: float = 0.5
_BASS_ISOLATION_WINDOW_S: float = 4.0
_BASS_ISOLATION_MAX_VELOCITY: float = 0.35


def _dedup_octave_overlaps(
    notes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove octave-doubled notes produced by basic-pitch.

    When two notes overlap in time and share the same pitch class (mod 12)
    but differ by one or more octaves, keep only the one with higher velocity.
    """
    notes = sorted(notes, key=lambda n: n["onset_s"])
    keep = [True] * len(notes)
    for i in range(len(notes)):
        if not keep[i]:
            continue
        ni = notes[i]
        mi = int(round(ni["midi"]))
        for j in range(i + 1, len(notes)):
            if not keep[j]:
                continue
            nj = notes[j]
            if nj["onset_s"] >= ni["offset_s"]:
                break
            mj = int(round(nj["midi"]))
            if mi % 12 == mj % 12 and mi != mj:
                if ni["velocity"] >= nj["velocity"]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    return [n for n, k in zip(notes, keep) if k]


def _correct_octave_by_context(
    notes: list[dict[str, Any]],
    *,
    window: int = _OCTAVE_CORRECT_WINDOW,
    min_gain: int = _OCTAVE_CORRECT_MIN_GAIN,
    midi_lo: float = 20,
    midi_hi: float = 80,
) -> list[dict[str, Any]]:
    """Shift notes ±12 semitones when local context strongly suggests a
    different octave.

    For each note, if moving it up or down by 12 brings it at least
    *min_gain* semitones closer to the median MIDI of its *window*
    nearest neighbours, apply the shift.

    Parameters
    ----------
    midi_lo, midi_hi : allowable MIDI range for shifted candidates.
        Defaults (20--80) suit bass; use ~36--96 for vocals.
    """
    notes = [dict(n) for n in notes]
    midis = np.array([n["midi"] for n in notes])
    for i in range(len(notes)):
        lo = max(0, i - window)
        hi = min(len(notes), i + window + 1)
        neighbours = [midis[j] for j in range(lo, hi) if j != i]
        if len(neighbours) < 2:
            continue
        med = float(np.median(neighbours))
        m = midis[i]
        dist_orig = abs(m - med)
        best_m, best_dist = m, dist_orig
        for shift in (-12, 12):
            cand = m + shift
            if cand < midi_lo or cand > midi_hi:
                continue
            d = abs(cand - med)
            if d < best_dist:
                best_dist = d
                best_m = cand
        if best_m != m and (dist_orig - best_dist) >= min_gain:
            notes[i] = dict(notes[i], midi=round(float(best_m), 2))
            midis[i] = best_m
    return notes


# ── Vocal octave-jump smoother ──

_VOCAL_OCTAVE_JUMP_THRESHOLD = 10  # semitones — jumps >= this are suspicious


def _neighbour_cost(
    m: float,
    prev_m: float | None,
    next_m: float | None,
) -> float:
    """Sum of absolute intervals to immediate neighbours (lower = smoother)."""
    cost = 0.0
    if prev_m is not None:
        cost += abs(m - prev_m)
    if next_m is not None:
        cost += abs(m - next_m)
    return cost


def _smooth_vocal_octave_jumps(
    notes: list[dict[str, Any]],
    *,
    jump_threshold: int = _VOCAL_OCTAVE_JUMP_THRESHOLD,
    midi_lo: float = 36,
    midi_hi: float = 96,
) -> list[dict[str, Any]]:
    """Resolve octave jumps between consecutive vocal notes.

    For each note, if it has a large jump (>= *jump_threshold* semitones)
    to **every** available neighbour (both sides for interior notes, single
    side for edge notes), it is considered an outlier.  If shifting it
    +/-12 semitones reduces the total neighbour cost, the shift is applied.

    This ensures only true outliers are corrected -- a note that jumps
    from its left neighbour but agrees with its right neighbour is a
    legitimate pitch change, not an octave error.
    """
    if len(notes) < 2:
        return notes
    notes = [dict(n) for n in notes]
    midis = np.array([n["midi"] for n in notes], dtype=float)
    n = len(midis)

    for i in range(n):
        m = midis[i]
        prev_m = midis[i - 1] if i > 0 else None
        next_m = midis[i + 1] if i < n - 1 else None

        # An outlier must have large jumps to BOTH neighbours.
        # Edge notes (only one neighbour) are left to _correct_octave_by_context.
        if prev_m is None or next_m is None:
            continue
        if abs(m - prev_m) < jump_threshold or abs(m - next_m) < jump_threshold:
            continue

        # This interior note jumps from both neighbours.  Try +/-12 shifts.
        best_m = m
        best_cost = _neighbour_cost(m, prev_m, next_m)
        for shift in (-12, 12):
            cand = m + shift
            if cand < midi_lo or cand > midi_hi:
                continue
            cost = _neighbour_cost(cand, prev_m, next_m)
            if cost < best_cost:
                best_cost = cost
                best_m = cand

        if best_m != m:
            notes[i] = dict(notes[i], midi=round(float(best_m), 2))
            midis[i] = best_m

    return notes


def _stem_rms_for_notes(
    notes: list[dict[str, Any]], y: np.ndarray, sr: int,
) -> list[float]:
    """Compute per-note RMS from the stem waveform."""
    rms_vals: list[float] = []
    for n in notes:
        s0 = int(n["onset_s"] * sr)
        s1 = int(n["offset_s"] * sr)
        seg = y[s0:s1]
        rms_vals.append(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) if len(seg) > 0 else 0.0)
    return rms_vals


def _rescale_velocity_to_stem_energy(
    notes: list[dict[str, Any]], y: np.ndarray, sr: int,
) -> list[dict[str, Any]]:
    """Replace detector-confidence velocity with stem-RMS-based velocity."""
    if not notes:
        return notes
    rms_vals = _stem_rms_for_notes(notes, y, sr)
    cap = float(np.percentile(rms_vals, 99))
    if cap < 1e-8:
        return notes  # zero-audio edge case — trust detector output
    return [dict(n, velocity=round(float(np.clip(r / cap, 0.0, 1.0)), 4))
            for n, r in zip(notes, rms_vals)]


def _gate_and_prune_bass_notes(
    notes: list[dict[str, Any]], y: np.ndarray, sr: int,
) -> list[dict[str, Any]]:
    """Remove false-positive bass notes via RMS gating and isolation pruning."""
    if not notes:
        return notes

    # Step A — RMS energy gate
    rms_vals = _stem_rms_for_notes(notes, y, sr)
    nonzero = [r for r in rms_vals if r > 1e-8]
    if nonzero:
        threshold = _BASS_RMS_GATE_SCALE * float(np.percentile(nonzero, _BASS_RMS_GATE_PERCENTILE))
        notes = [n for n, r in zip(notes, rms_vals) if r >= threshold]

    # Step B — Isolated weak note pruning
    if len(notes) <= 1:
        return notes
    keep: list[dict[str, Any]] = []
    for i, n in enumerate(notes):
        gap_before = n["onset_s"] - notes[i - 1]["offset_s"] if i > 0 else float("inf")
        gap_after = notes[i + 1]["onset_s"] - n["offset_s"] if i < len(notes) - 1 else float("inf")
        if gap_before > _BASS_ISOLATION_WINDOW_S and gap_after > _BASS_ISOLATION_WINDOW_S and n["velocity"] < _BASS_ISOLATION_MAX_VELOCITY:
            continue
        keep.append(n)
    return keep


def estimate_key_scale(
    chroma: np.ndarray,
    *,
    mode: str = "both",
) -> list[int]:
    """Estimate the key from a chroma matrix and return scale pitch classes.

    Parameters
    ----------
    chroma : (N, 12) chroma feature matrix (e.g. from ``other_chroma_12``).
    mode : ``"major"``, ``"minor"``, or ``"both"`` (default).
        When ``"both"``, the template with the highest correlation wins.

    Returns
    -------
    list of pitch classes (0–11) forming the estimated key's diatonic scale.
    """
    chroma = np.asarray(chroma, dtype=np.float64)
    if chroma.ndim != 2 or chroma.shape[1] != 12 or chroma.shape[0] == 0:
        # Cannot estimate — return chromatic (all 12 PCs, i.e. no filtering)
        return list(range(12))

    # Energy-weighted chroma profile (sum across frames)
    profile = chroma.sum(axis=0)
    total = profile.sum()
    if total < 1e-12:
        return list(range(12))
    profile = profile / total

    # Scale templates (intervals from root, in semitones)
    major_intervals = [0, 2, 4, 5, 7, 9, 11]   # Ionian
    minor_intervals = [0, 2, 3, 5, 7, 8, 10]    # Aeolian (natural minor)

    templates: list[tuple[str, list[int]]] = []
    if mode in ("major", "both"):
        templates.append(("major", major_intervals))
    if mode in ("minor", "both"):
        templates.append(("minor", minor_intervals))

    best_corr = -np.inf
    best_pcs: list[int] = list(range(12))

    for _mode_name, intervals in templates:
        for root in range(12):
            # Build binary template
            tmpl = np.zeros(12, dtype=np.float64)
            pcs = [(root + iv) % 12 for iv in intervals]
            for pc in pcs:
                tmpl[pc] = 1.0
            tmpl /= tmpl.sum()
            # Pearson correlation
            corr = float(np.corrcoef(profile, tmpl)[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_pcs = pcs

    return best_pcs


def _snap_bass_to_scale(
    notes: list[dict[str, Any]],
    scale_pcs: list[int],
    *,
    max_shift: int = 1,
) -> list[dict[str, Any]]:
    """Snap each bass note's MIDI to the nearest scale degree.

    Only shifts notes by at most *max_shift* semitones (default 1).
    Notes already on a scale degree are left unchanged.

    Parameters
    ----------
    notes : list of note dicts with ``"midi"`` key.
    scale_pcs : list of pitch classes (0–11) forming the target scale.
    max_shift : maximum semitone correction (default 1).

    Returns
    -------
    New list of note dicts with adjusted ``"midi"`` values.
    """
    if not notes or not scale_pcs or len(scale_pcs) >= 12:
        return notes

    scale_set = set(int(pc) % 12 for pc in scale_pcs)

    out: list[dict[str, Any]] = []
    for n in notes:
        midi = n["midi"]
        midi_int = int(round(midi))
        pc = midi_int % 12

        if pc in scale_set:
            out.append(n)
            continue

        # Find nearest scale PC within max_shift
        best_shift = 0
        best_dist = max_shift + 1  # sentinel — larger than allowed
        for delta in range(-max_shift, max_shift + 1):
            if delta == 0:
                continue
            candidate_pc = (midi_int + delta) % 12
            if candidate_pc in scale_set and abs(delta) < best_dist:
                best_dist = abs(delta)
                best_shift = delta

        if best_dist <= max_shift:
            new_midi = round(float(midi_int + best_shift), 2)
            out.append(dict(n, midi=new_midi))
        else:
            # Beyond max_shift — keep original
            out.append(n)

    return out


def extract_bass_notes(
    note_events: list[dict[str, float]] | None,
    pitch_hz: np.ndarray | None,
    y: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
    beat_times_s: Sequence[float] | np.ndarray | None = None,
    scale_pcs: list[int] | None = None,
) -> dict[str, Any]:
    """Top-level bass note dispatcher — picks best available source.

    Parameters
    ----------
    note_events : basic-pitch output (list of {start_s, end_s, midi, velocity}), or None
    pitch_hz : pYIN per-frame fundamental frequency (NaN = unvoiced), or None
    y : mono waveform (for pYIN velocity computation)
    sr : sample rate
    hop_length : analysis hop
    beat_times_s : beat grid for alignment (optional)
    scale_pcs : estimated key's scale pitch classes (0–11), or None to skip
        quantization.  When provided, notes are snapped to the nearest
        scale degree (±1 semitone) after octave correction and before
        energy gating.

    Returns
    -------
    dict with ``"source"`` and ``"notes": [...]``
    """
    # Primary: basic-pitch note events
    if note_events is not None and len(note_events) > 0:
        beat_arr = (
            np.asarray(beat_times_s, dtype=np.float64)
            if beat_times_s is not None and len(beat_times_s) > 0
            else None
        )
        notes: list[dict[str, Any]] = []
        for ev in note_events:
            note: dict[str, Any] = {
                "onset_s": round(float(ev["start_s"]), 4),
                "offset_s": round(float(ev["end_s"]), 4),
                "midi": round(float(ev["midi"]), 2),
                "velocity": round(float(ev["velocity"]), 4),
            }
            if beat_arr is not None:
                note.update(_compute_beat_alignment(note["onset_s"], beat_arr))
            notes.append(note)

        # Clean up octave artifacts from basic-pitch
        notes = _dedup_octave_overlaps(notes)
        notes = _correct_octave_by_context(notes)

        # Snap to estimated key scale (±1 semitone correction)
        if scale_pcs is not None:
            notes = _snap_bass_to_scale(notes, scale_pcs)

        # Energy gating & velocity rescaling
        notes = _rescale_velocity_to_stem_energy(notes, y, sr)
        notes = _gate_and_prune_bass_notes(notes, y, sr)

        return {"source": "basic_pitch", "notes": notes}

    # Fallback: pYIN pitch track
    if pitch_hz is not None and np.any(np.isfinite(pitch_hz)):
        result = _notes_from_pitch_track(
            pitch_hz, y, sr,
            source="pyin",
            hop_length=hop_length,
            max_gap_frames=3,
            beat_times_s=beat_times_s,
        )
        # Snap to estimated key scale (±1 semitone correction)
        if scale_pcs is not None:
            result["notes"] = _snap_bass_to_scale(result["notes"], scale_pcs)
        result["notes"] = _rescale_velocity_to_stem_energy(result["notes"], y, sr)
        result["notes"] = _gate_and_prune_bass_notes(result["notes"], y, sr)
        return result

    # No data
    return {"source": "none", "notes": []}
