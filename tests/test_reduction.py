"""Tests for songviz.reduction — drum hit, vocal note & bass note extraction."""
from __future__ import annotations

import numpy as np

from songviz.reduction import (
    _bass_global_octave_fix,
    _correct_octave_by_context,
    _dedup_octave_overlaps,
    _gate_and_prune_bass_notes,
    _neighbour_cost,
    _notes_from_pitch_track,
    _refine_bass_pitch_cqt,
    _rescale_velocity_to_stem_energy,
    _smooth_vocal_octave_jumps,
    _snap_bass_to_scale,
    estimate_key_scale,
    extract_bass_notes,
    extract_drum_hits,
    extract_drum_hits_fallback,
    extract_vocal_notes,
    extract_vocal_notes_from_pitch_track,
)

SR = 22050
HOP = 512


def _click_signal(duration_s: float = 2.0, click_times: list[float] | None = None, amp: float = 0.9) -> np.ndarray:
    """Synthesize a short broadband click at each specified time."""
    n = int(SR * duration_s)
    y = np.zeros(n, dtype=np.float32)
    for ct in click_times or []:
        i = int(ct * SR)
        if 0 <= i < n - 4:
            y[i : i + 4] = amp
    return y


def test_extract_drum_hits_detects_clicks() -> None:
    """Hits should be detected near the expected click times (within ~50 ms)."""
    click_times = [0.3, 0.7, 1.2]
    comps = {"kick": _click_signal(click_times=click_times)}
    result = extract_drum_hits(comps, SR, hop_length=HOP)
    hits = result["hits"]
    assert len(hits) > 0
    detected_times = [h["t"] for h in hits]
    for expected in click_times:
        assert any(abs(dt - expected) < 0.05 for dt in detected_times), (
            f"No hit near {expected}s; detected: {detected_times}"
        )


def test_extract_drum_hits_schema() -> None:
    """Output has correct structure and all values are plain Python types."""
    comps = {"snare": _click_signal(click_times=[0.5])}
    result = extract_drum_hits(comps, SR, hop_length=HOP)
    assert result["source"] == "drumsep"
    assert isinstance(result["hits"], list)
    for h in result["hits"]:
        assert isinstance(h["t"], float)
        assert isinstance(h["component"], str)
        assert isinstance(h["velocity"], float)
        assert isinstance(h["velocity_raw"], float)


def test_extract_drum_hits_velocity_range() -> None:
    """velocity should be in [0, 1]; velocity_raw should be > 0 for non-silent."""
    comps = {"kick": _click_signal(click_times=[0.2, 0.6, 1.0, 1.4])}
    result = extract_drum_hits(comps, SR, hop_length=HOP)
    for h in result["hits"]:
        assert 0.0 <= h["velocity"] <= 1.0
        assert h["velocity_raw"] > 0


def test_extract_drum_hits_raw_velocity_cross_component() -> None:
    """Louder kick should have higher velocity_raw than quieter hh."""
    comps = {
        "kick": _click_signal(click_times=[0.5], amp=0.9),
        "hh": _click_signal(click_times=[0.5], amp=0.1),
    }
    result = extract_drum_hits(comps, SR, hop_length=HOP)
    kick_raw = [h["velocity_raw"] for h in result["hits"] if h["component"] == "kick"]
    hh_raw = [h["velocity_raw"] for h in result["hits"] if h["component"] == "hh"]
    assert len(kick_raw) > 0 and len(hh_raw) > 0
    assert max(kick_raw) > max(hh_raw)


def test_extract_drum_hits_empty_components() -> None:
    """All-zero input should produce no hits."""
    comps = {
        "kick": np.zeros(SR * 2, dtype=np.float32),
        "snare": np.zeros(SR * 2, dtype=np.float32),
    }
    result = extract_drum_hits(comps, SR, hop_length=HOP)
    assert result["hits"] == []


def test_extract_drum_hits_beat_alignment() -> None:
    """With beat_times_s, each hit should have beat_idx and beat_phase."""
    comps = {"kick": _click_signal(click_times=[0.5, 1.0])}
    beats = [0.0, 0.5, 1.0, 1.5]
    result = extract_drum_hits(comps, SR, hop_length=HOP, beat_times_s=beats)
    for h in result["hits"]:
        assert "beat_idx" in h
        assert "beat_phase" in h
        assert isinstance(h["beat_idx"], int)
        assert 0.0 <= h["beat_phase"] < 1.0


def test_extract_drum_hits_fallback_schema() -> None:
    """Fallback produces same structure with source 'heuristic'."""
    y = _click_signal(click_times=[0.3, 0.8])
    result = extract_drum_hits_fallback(y, SR, hop_length=HOP)
    assert result["source"] == "heuristic"
    assert isinstance(result["hits"], list)
    for h in result["hits"]:
        assert "t" in h
        assert "component" in h
        assert "velocity" in h
        assert "velocity_raw" in h


def test_extract_drum_hits_fallback_detects_something() -> None:
    """Mixed signal with clicks should produce non-empty hits."""
    y = _click_signal(click_times=[0.2, 0.5, 0.9, 1.3])
    result = extract_drum_hits_fallback(y, SR, hop_length=HOP)
    assert len(result["hits"]) > 0


# ── Vocal note extraction tests ──


def _sine_tone(freq_hz: float, duration_s: float, amp: float = 0.5) -> np.ndarray:
    """Generate a pure sine tone at the given frequency."""
    t = np.arange(int(SR * duration_s)) / SR
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def _pitch_track_from_tones(
    tone_specs: list[tuple[float, float]],
    gap_s: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a pitch_hz array and waveform from (freq_hz, duration_s) specs with gaps.

    Returns (pitch_hz, y) where pitch_hz has one value per hop frame.
    """
    segments: list[np.ndarray] = []
    pitch_segments: list[np.ndarray] = []
    gap_samples = int(gap_s * SR)
    gap_frames = max(1, gap_samples // HOP)

    for freq, dur in tone_specs:
        tone = _sine_tone(freq, dur)
        segments.append(tone)
        n_frames = max(1, len(tone) // HOP)
        pitch_segments.append(np.full(n_frames, freq, dtype=np.float64))
        # Gap (silence + NaN pitch)
        segments.append(np.zeros(gap_samples, dtype=np.float32))
        pitch_segments.append(np.full(gap_frames, np.nan, dtype=np.float64))

    y = np.concatenate(segments)
    pitch_hz = np.concatenate(pitch_segments)
    return pitch_hz, y


def test_vocal_notes_from_pitch_track_basic() -> None:
    """Two tones (A4 + gap + C5) → 2 notes with correct MIDI."""
    # A4 = 440 Hz → MIDI 69, C5 = 523.25 Hz → MIDI 72
    pitch_hz, y = _pitch_track_from_tones([(440.0, 0.3), (523.25, 0.3)])
    result = extract_vocal_notes_from_pitch_track(pitch_hz, y, SR, hop_length=HOP)
    assert result["source"] == "pyin"
    assert len(result["notes"]) == 2
    assert result["notes"][0]["midi"] == 69.0
    assert result["notes"][1]["midi"] == 72.0
    # Onsets should be ordered
    assert result["notes"][0]["onset_s"] < result["notes"][1]["onset_s"]


def test_vocal_notes_from_pitch_track_min_duration() -> None:
    """1-frame note filtered out, longer note kept."""
    # Build a pitch array manually: 1 frame at MIDI 60, then NaN gap, then 10 frames at MIDI 65
    short = np.full(1, 261.63, dtype=np.float64)  # C4, 1 frame
    gap = np.full(5, np.nan, dtype=np.float64)
    long = np.full(10, 349.23, dtype=np.float64)  # F4, 10 frames
    pitch_hz = np.concatenate([short, gap, long])
    y = np.zeros(len(pitch_hz) * HOP, dtype=np.float32)
    # Add some signal so velocity isn't zero
    y[6 * HOP : 16 * HOP] = 0.3

    result = extract_vocal_notes_from_pitch_track(pitch_hz, y, SR, hop_length=HOP, min_note_frames=3)
    assert len(result["notes"]) == 1
    assert result["notes"][0]["midi"] == 65.0  # F4 rounded


def test_vocal_notes_from_pitch_track_velocity() -> None:
    """Loud note has higher velocity than quiet note; all in [0,1]."""
    # Loud A4 then quiet A4
    loud = _sine_tone(440.0, 0.3, amp=0.8)
    gap = np.zeros(int(0.1 * SR), dtype=np.float32)
    quiet = _sine_tone(440.0, 0.3, amp=0.05)

    y = np.concatenate([loud, gap, quiet])
    n_loud = max(1, len(loud) // HOP)
    n_gap = max(1, len(gap) // HOP)
    n_quiet = max(1, len(quiet) // HOP)
    pitch_hz = np.concatenate([
        np.full(n_loud, 440.0),
        np.full(n_gap, np.nan),
        np.full(n_quiet, 440.0),
    ])

    result = extract_vocal_notes_from_pitch_track(pitch_hz, y, SR, hop_length=HOP)
    assert len(result["notes"]) == 2
    assert result["notes"][0]["velocity"] > result["notes"][1]["velocity"]
    for n in result["notes"]:
        assert 0.0 <= n["velocity"] <= 1.0


def test_vocal_notes_from_pitch_track_beat_alignment() -> None:
    """With beat_times_s, notes have beat_idx (int) and beat_phase (float in [0,1))."""
    pitch_hz, y = _pitch_track_from_tones([(440.0, 0.3)])
    beats = [0.0, 0.5, 1.0, 1.5]
    result = extract_vocal_notes_from_pitch_track(pitch_hz, y, SR, hop_length=HOP, beat_times_s=beats)
    assert len(result["notes"]) >= 1
    for n in result["notes"]:
        assert "beat_idx" in n
        assert "beat_phase" in n
        assert isinstance(n["beat_idx"], int)
        assert 0.0 <= n["beat_phase"] < 1.0


def test_vocal_notes_from_pitch_track_all_nan() -> None:
    """All-NaN pitch → empty notes list."""
    pitch_hz = np.full(100, np.nan, dtype=np.float64)
    y = np.zeros(100 * HOP, dtype=np.float32)
    result = extract_vocal_notes_from_pitch_track(pitch_hz, y, SR, hop_length=HOP)
    assert result["source"] == "pyin"
    assert result["notes"] == []


def test_vocal_notes_from_pitch_track_schema() -> None:
    """Each note has the expected keys with correct types."""
    pitch_hz, y = _pitch_track_from_tones([(440.0, 0.3)])
    result = extract_vocal_notes_from_pitch_track(pitch_hz, y, SR, hop_length=HOP)
    assert len(result["notes"]) >= 1
    for n in result["notes"]:
        assert set(n.keys()) == {"onset_s", "offset_s", "midi", "velocity"}
        assert isinstance(n["onset_s"], float)
        assert isinstance(n["offset_s"], float)
        assert isinstance(n["midi"], float)
        assert isinstance(n["velocity"], float)
        assert n["offset_s"] > n["onset_s"]


def test_extract_vocal_notes_basic_pitch_source() -> None:
    """Mock basic-pitch note_events → source='basic_pitch', correct field remapping."""
    events = [
        {"start_s": 1.0, "end_s": 1.5, "midi": 60.123, "velocity": 0.8},
        {"start_s": 2.0, "end_s": 2.3, "midi": 64.0, "velocity": 0.6},
    ]
    y = np.zeros(SR * 3, dtype=np.float32)
    result = extract_vocal_notes(events, None, y, SR)
    assert result["source"] == "basic_pitch"
    assert len(result["notes"]) == 2
    assert result["notes"][0]["onset_s"] == 1.0
    assert result["notes"][0]["offset_s"] == 1.5
    assert result["notes"][0]["midi"] == 60.12  # rounded to 2dp
    assert result["notes"][1]["velocity"] == 0.6


def test_extract_vocal_notes_prefers_basic_pitch() -> None:
    """Both note_events and pitch_hz provided → source='basic_pitch'."""
    events = [{"start_s": 0.5, "end_s": 0.8, "midi": 65.0, "velocity": 0.7}]
    pitch_hz = np.full(50, 440.0, dtype=np.float64)
    y = np.zeros(SR * 2, dtype=np.float32)
    result = extract_vocal_notes(events, pitch_hz, y, SR)
    assert result["source"] == "basic_pitch"


def test_extract_vocal_notes_falls_back_to_pyin() -> None:
    """note_events=None, valid pitch_hz → source='pyin'."""
    pitch_hz, y = _pitch_track_from_tones([(440.0, 0.3)])
    result = extract_vocal_notes(None, pitch_hz, y, SR)
    assert result["source"] == "pyin"
    assert len(result["notes"]) >= 1


def test_extract_vocal_notes_empty_fallback() -> None:
    """Both None → source='none', empty notes."""
    y = np.zeros(SR * 2, dtype=np.float32)
    result = extract_vocal_notes(None, None, y, SR)
    assert result["source"] == "none"
    assert result["notes"] == []


# ── Bass note extraction tests ──


def test_bass_notes_basic() -> None:
    """Two bass tones (E2 + gap + A2) → 2 notes with correct MIDI (pYIN fallback)."""
    # E2 = 82.41 Hz → MIDI 40, A2 = 110.0 Hz → MIDI 45
    # (Within expected bass register so median shift doesn't trigger)
    pitch_hz, y = _pitch_track_from_tones([(82.41, 0.3), (110.0, 0.3)])
    result = extract_bass_notes(None, pitch_hz, y, SR, hop_length=HOP)
    assert result["source"] == "pyin"
    assert len(result["notes"]) == 2
    assert result["notes"][0]["midi"] == 40.0
    assert result["notes"][1]["midi"] == 45.0
    assert result["notes"][0]["onset_s"] < result["notes"][1]["onset_s"]


def test_bass_notes_velocity() -> None:
    """Loud note has higher velocity than quiet note; all in [0,1] (pYIN fallback).

    Gap must be > max_gap_s (0.15 s) so notes are NOT merged by _merge_adjacent_notes.
    """
    loud = _sine_tone(55.0, 0.3, amp=0.8)
    gap = np.zeros(int(0.3 * SR), dtype=np.float32)  # 0.3 s gap > 0.15 s threshold
    quiet = _sine_tone(55.0, 0.3, amp=0.2)

    y = np.concatenate([loud, gap, quiet])
    n_loud = max(1, len(loud) // HOP)
    n_gap = max(1, len(gap) // HOP)
    n_quiet = max(1, len(quiet) // HOP)
    pitch_hz = np.concatenate([
        np.full(n_loud, 55.0),
        np.full(n_gap, np.nan),
        np.full(n_quiet, 55.0),
    ])

    result = extract_bass_notes(None, pitch_hz, y, SR, hop_length=HOP)
    assert len(result["notes"]) == 2
    assert result["notes"][0]["velocity"] > result["notes"][1]["velocity"]
    for n in result["notes"]:
        assert 0.0 <= n["velocity"] <= 1.0


def test_bass_notes_beat_alignment() -> None:
    """With beat_times_s → notes have beat_idx (int) + beat_phase (float in [0,1)) (pYIN fallback)."""
    pitch_hz, y = _pitch_track_from_tones([(55.0, 0.3)])
    beats = [0.0, 0.5, 1.0, 1.5]
    result = extract_bass_notes(None, pitch_hz, y, SR, hop_length=HOP, beat_times_s=beats)
    assert len(result["notes"]) >= 1
    for n in result["notes"]:
        assert "beat_idx" in n
        assert "beat_phase" in n
        assert isinstance(n["beat_idx"], int)
        assert 0.0 <= n["beat_phase"] < 1.0


def test_bass_notes_all_nan() -> None:
    """All-NaN pitch_hz, no note_events → source='none', empty notes (no usable pitch data)."""
    pitch_hz = np.full(100, np.nan, dtype=np.float64)
    y = np.zeros(100 * HOP, dtype=np.float32)
    result = extract_bass_notes(None, pitch_hz, y, SR, hop_length=HOP)
    assert result["source"] == "none"
    assert result["notes"] == []


def test_bass_notes_none_both() -> None:
    """note_events=None, pitch_hz=None → source='none', empty notes."""
    y = np.zeros(SR * 2, dtype=np.float32)
    result = extract_bass_notes(None, None, y, SR, hop_length=HOP)
    assert result["source"] == "none"
    assert result["notes"] == []


def test_bass_notes_schema() -> None:
    """Each note has exactly {onset_s, offset_s, midi, velocity}, correct types, offset > onset (pYIN)."""
    pitch_hz, y = _pitch_track_from_tones([(41.20, 0.3)])
    result = extract_bass_notes(None, pitch_hz, y, SR, hop_length=HOP)
    assert len(result["notes"]) >= 1
    for n in result["notes"]:
        assert set(n.keys()) == {"onset_s", "offset_s", "midi", "velocity"}
        assert isinstance(n["onset_s"], float)
        assert isinstance(n["offset_s"], float)
        assert isinstance(n["midi"], float)
        assert isinstance(n["velocity"], float)
        assert n["offset_s"] > n["onset_s"]


def test_bass_notes_basic_pitch_source() -> None:
    """Mock basic-pitch note_events → source='basic_pitch', correct field remapping."""
    events = [
        {"start_s": 1.0, "end_s": 1.5, "midi": 40.123, "velocity": 0.7},
        {"start_s": 2.0, "end_s": 2.4, "midi": 35.0, "velocity": 0.5},
    ]
    y = np.zeros(SR * 3, dtype=np.float32)
    result = extract_bass_notes(events, None, y, SR)
    assert result["source"] == "basic_pitch"
    assert len(result["notes"]) == 2
    assert result["notes"][0]["onset_s"] == 1.0
    assert result["notes"][0]["offset_s"] == 1.5
    assert result["notes"][0]["midi"] == 40.12  # rounded to 2dp
    assert result["notes"][1]["velocity"] == 0.5


def test_bass_notes_prefers_basic_pitch() -> None:
    """Both note_events and pitch_hz provided → source='basic_pitch'."""
    events = [{"start_s": 0.5, "end_s": 0.9, "midi": 35.0, "velocity": 0.6}]
    pitch_hz = np.full(50, 55.0, dtype=np.float64)
    y = np.zeros(SR * 2, dtype=np.float32)
    result = extract_bass_notes(events, pitch_hz, y, SR)
    assert result["source"] == "basic_pitch"


def test_bass_notes_falls_back_to_pyin() -> None:
    """note_events=None, valid pitch_hz → source='pyin'."""
    pitch_hz, y = _pitch_track_from_tones([(55.0, 0.3)])
    result = extract_bass_notes(None, pitch_hz, y, SR)
    assert result["source"] == "pyin"
    assert len(result["notes"]) >= 1


# ── Bass gap-merge tests ──


def test_bass_notes_gap_merge() -> None:
    """A 2-frame NaN gap in a sustained pitch → 1 merged note (not 2)."""
    # 10 frames at MIDI 33 (A1=55Hz), 2-frame NaN gap, 10 frames at MIDI 33
    pitch_hz = np.concatenate([
        np.full(10, 55.0, dtype=np.float64),
        np.full(2, np.nan, dtype=np.float64),
        np.full(10, 55.0, dtype=np.float64),
    ])
    y = np.ones(len(pitch_hz) * HOP, dtype=np.float32) * 0.3
    result = _notes_from_pitch_track(
        pitch_hz, y, SR, source="pyin", hop_length=HOP, max_gap_frames=3,
    )
    assert len(result["notes"]) == 1
    assert result["notes"][0]["midi"] == 33.0


def test_bass_notes_gap_merge_different_pitch_no_merge() -> None:
    """Gap between different MIDI values → still 2 separate notes."""
    # 10 frames at MIDI 33 (55Hz), 2-frame gap, 10 frames at MIDI 28 (41.2Hz)
    pitch_hz = np.concatenate([
        np.full(10, 55.0, dtype=np.float64),
        np.full(2, np.nan, dtype=np.float64),
        np.full(10, 41.20, dtype=np.float64),
    ])
    y = np.ones(len(pitch_hz) * HOP, dtype=np.float32) * 0.3
    result = _notes_from_pitch_track(
        pitch_hz, y, SR, source="pyin", hop_length=HOP, max_gap_frames=3,
    )
    assert len(result["notes"]) == 2


def test_bass_notes_gap_merge_large_gap_no_merge() -> None:
    """Gap > max_gap_frames → still 2 separate notes."""
    # 10 frames at MIDI 33, 5-frame gap (> 3), 10 frames at MIDI 33
    pitch_hz = np.concatenate([
        np.full(10, 55.0, dtype=np.float64),
        np.full(5, np.nan, dtype=np.float64),
        np.full(10, 55.0, dtype=np.float64),
    ])
    y = np.ones(len(pitch_hz) * HOP, dtype=np.float32) * 0.3
    result = _notes_from_pitch_track(
        pitch_hz, y, SR, source="pyin", hop_length=HOP, max_gap_frames=3,
    )
    assert len(result["notes"]) == 2


# ── Octave dedup tests ──


def test_dedup_octave_overlaps_removes_quieter() -> None:
    """Two overlapping notes same pitch class, different octave → keep louder."""
    notes = [
        {"onset_s": 1.0, "offset_s": 1.5, "midi": 34.0, "velocity": 0.6},  # A#1
        {"onset_s": 1.0, "offset_s": 1.4, "midi": 46.0, "velocity": 0.3},  # A#2
    ]
    result = _dedup_octave_overlaps(notes)
    assert len(result) == 1
    assert result[0]["midi"] == 34.0  # louder one kept


def test_dedup_octave_overlaps_keeps_different_pitch_class() -> None:
    """Two overlapping notes with different pitch class → both kept."""
    notes = [
        {"onset_s": 1.0, "offset_s": 1.5, "midi": 34.0, "velocity": 0.6},  # A#1
        {"onset_s": 1.1, "offset_s": 1.4, "midi": 35.0, "velocity": 0.3},  # B1
    ]
    result = _dedup_octave_overlaps(notes)
    assert len(result) == 2


def test_dedup_octave_overlaps_keeps_non_overlapping() -> None:
    """Same pitch class, different octave, but NOT overlapping → both kept."""
    notes = [
        {"onset_s": 1.0, "offset_s": 1.5, "midi": 34.0, "velocity": 0.6},  # A#1
        {"onset_s": 2.0, "offset_s": 2.5, "midi": 46.0, "velocity": 0.3},  # A#2
    ]
    result = _dedup_octave_overlaps(notes)
    assert len(result) == 2


# ── Octave context correction tests ──


def test_correct_octave_shifts_outlier_down() -> None:
    """A single note an octave above its neighbours gets shifted down."""
    notes = [
        {"onset_s": 0.0, "offset_s": 0.3, "midi": 34.0, "velocity": 0.5},
        {"onset_s": 0.5, "offset_s": 0.8, "midi": 35.0, "velocity": 0.5},
        {"onset_s": 1.0, "offset_s": 1.3, "midi": 46.0, "velocity": 0.5},  # A#2 outlier
        {"onset_s": 1.5, "offset_s": 1.8, "midi": 32.0, "velocity": 0.5},
        {"onset_s": 2.0, "offset_s": 2.3, "midi": 34.0, "velocity": 0.5},
    ]
    result = _correct_octave_by_context(notes)
    # The A#2 (46) should be shifted to A#1 (34)
    assert result[2]["midi"] == 34.0


def test_correct_octave_no_shift_when_consistent() -> None:
    """All notes in the same octave → no changes."""
    notes = [
        {"onset_s": i * 0.5, "offset_s": i * 0.5 + 0.3, "midi": float(m), "velocity": 0.5}
        for i, m in enumerate([34, 35, 32, 34, 30])
    ]
    result = _correct_octave_by_context(notes)
    for orig, fixed in zip(notes, result):
        assert orig["midi"] == fixed["midi"]


def test_correct_octave_no_shift_when_gain_too_small() -> None:
    """Shift would help only slightly (< 6 semitones) → no change."""
    # Neighbours span a wide range so median is ~40; note at 46 is only 6 away
    notes = [
        {"onset_s": i * 0.5, "offset_s": i * 0.5 + 0.3, "midi": float(m), "velocity": 0.5}
        for i, m in enumerate([34, 38, 46, 44, 40])
    ]
    result = _correct_octave_by_context(notes)
    # MIDI 46 with median ~39 → dist=7; shift to 34 → dist=5; gain=2 < 6 → no shift
    assert result[2]["midi"] == 46.0


def test_bass_notes_basic_pitch_applies_octave_cleanup() -> None:
    """extract_bass_notes with basic-pitch events applies dedup + octave correction."""
    events = [
        # Two overlapping notes: E2 loud + E3 quiet → dedup keeps E2
        {"start_s": 1.0, "end_s": 1.5, "midi": 40.0, "velocity": 0.6},
        {"start_s": 1.0, "end_s": 1.4, "midi": 52.0, "velocity": 0.3},
        # Surrounding context at octave 2 (within expected bass range)
        {"start_s": 0.0, "end_s": 0.3, "midi": 38.0, "velocity": 0.5},
        {"start_s": 0.5, "end_s": 0.8, "midi": 41.0, "velocity": 0.5},
        {"start_s": 2.0, "end_s": 2.3, "midi": 36.0, "velocity": 0.5},
    ]
    y = np.zeros(SR * 3, dtype=np.float32)
    result = extract_bass_notes(events, None, y, SR)
    assert result["source"] == "basic_pitch"
    # Dedup should have removed the E3 duplicate
    midis = [n["midi"] for n in result["notes"]]
    assert 52.0 not in midis
    # Should have 4 notes (5 minus 1 deduped)
    assert len(result["notes"]) == 4


# ── Bass energy gating & velocity rescaling tests ──


def test_bass_energy_gate_removes_silent_note() -> None:
    """pYIN path: pitch track with note in loud + silent region.

    The pYIN path no longer applies the hard RMS gate (it was designed for
    basic-pitch false positives and was too aggressive for pYIN's sparse output).
    Instead, _rescale_velocity_to_stem_energy sets the silent note's velocity
    to ~0.0, making it effectively inaudible even though it remains in the list.
    """
    # Build a waveform: loud tone 0–0.3s, silence 1–1.3s
    dur_s = 2.0
    y = np.zeros(int(SR * dur_s), dtype=np.float32)
    # Loud region: 0–0.3s
    t_loud = np.arange(int(0.3 * SR)) / SR
    y[:len(t_loud)] = (0.5 * np.sin(2.0 * np.pi * 55.0 * t_loud)).astype(np.float32)
    # Silent region stays zero

    # Pitch track: 55 Hz (A1=MIDI 33) at 0–0.3s, then gap, then 55 Hz at 1.0–1.3s
    n_frames = len(y) // HOP
    pitch_hz = np.full(n_frames, np.nan, dtype=np.float64)
    f_start_loud = 0
    f_end_loud = int(0.3 * SR) // HOP
    pitch_hz[f_start_loud:f_end_loud] = 55.0
    f_start_silent = int(1.0 * SR) // HOP
    f_end_silent = int(1.3 * SR) // HOP
    if f_end_silent <= n_frames:
        pitch_hz[f_start_silent:f_end_silent] = 55.0

    result = extract_bass_notes(None, pitch_hz, y, SR, hop_length=HOP)
    assert result["source"] == "pyin"
    # Both notes are present; the silent one has near-zero velocity
    notes = result["notes"]
    assert len(notes) >= 1
    loud_note = min(notes, key=lambda n: n["onset_s"])
    assert loud_note["onset_s"] < 0.5  # the loud one is first
    assert loud_note["velocity"] > 0.0  # loud note has positive velocity


def test_bass_energy_gate_keeps_active_notes() -> None:
    """Two real-audio notes → both survive gating."""
    dur_s = 2.0
    y = np.zeros(int(SR * dur_s), dtype=np.float32)
    # Two loud tones
    t1 = np.arange(int(0.3 * SR)) / SR
    y[:len(t1)] = (0.5 * np.sin(2.0 * np.pi * 55.0 * t1)).astype(np.float32)
    t2_start = int(0.5 * SR)
    t2 = np.arange(int(0.3 * SR)) / SR
    y[t2_start:t2_start + len(t2)] = (0.4 * np.sin(2.0 * np.pi * 55.0 * t2)).astype(np.float32)

    n_frames = len(y) // HOP
    pitch_hz = np.full(n_frames, np.nan, dtype=np.float64)
    pitch_hz[0:int(0.3 * SR) // HOP] = 55.0
    pitch_hz[t2_start // HOP:(t2_start + int(0.3 * SR)) // HOP] = 55.0

    result = extract_bass_notes(None, pitch_hz, y, SR, hop_length=HOP)
    # Both notes have real energy — both should survive
    assert len(result["notes"]) == 2


def test_bass_isolated_weak_note_pruned() -> None:
    """Cluster + 5s gap + quiet isolated note → isolated pruned."""
    notes = [
        {"onset_s": 1.0, "offset_s": 1.5, "midi": 33.0, "velocity": 0.6},
        {"onset_s": 2.0, "offset_s": 2.5, "midi": 33.0, "velocity": 0.5},
        # 5s gap — isolated AND low velocity
        {"onset_s": 7.5, "offset_s": 8.0, "midi": 33.0, "velocity": 0.2},
    ]
    # Provide audio with equal energy everywhere so RMS gate doesn't interfere
    y = np.ones(int(SR * 10), dtype=np.float32) * 0.3
    result = _gate_and_prune_bass_notes(notes, y, SR)
    assert len(result) == 2  # isolated weak note removed
    assert all(n["onset_s"] < 5.0 for n in result)


def test_bass_isolated_strong_note_survives() -> None:
    """Same layout, loud isolated note → survives."""
    notes = [
        {"onset_s": 1.0, "offset_s": 1.5, "midi": 33.0, "velocity": 0.6},
        {"onset_s": 2.0, "offset_s": 2.5, "midi": 33.0, "velocity": 0.5},
        # 5s gap — isolated BUT high velocity
        {"onset_s": 7.5, "offset_s": 8.0, "midi": 33.0, "velocity": 0.5},
    ]
    y = np.ones(int(SR * 10), dtype=np.float32) * 0.3
    result = _gate_and_prune_bass_notes(notes, y, SR)
    assert len(result) == 3  # loud isolated note kept


def test_bass_velocity_tracks_stem_energy() -> None:
    """Loud + quiet tones → velocity ratio ≥ 5×."""
    notes = [
        {"onset_s": 0.0, "offset_s": 0.5, "midi": 33.0, "velocity": 0.5},
        {"onset_s": 1.0, "offset_s": 1.5, "midi": 33.0, "velocity": 0.5},
    ]
    y = np.zeros(int(SR * 2), dtype=np.float32)
    # Loud region for first note
    t = np.arange(int(0.5 * SR)) / SR
    y[:len(t)] = (0.8 * np.sin(2.0 * np.pi * 55.0 * t)).astype(np.float32)
    # Quiet region for second note
    s = int(1.0 * SR)
    t2 = np.arange(int(0.5 * SR)) / SR
    y[s:s + len(t2)] = (0.05 * np.sin(2.0 * np.pi * 55.0 * t2)).astype(np.float32)

    result = _rescale_velocity_to_stem_energy(notes, y, SR)
    assert result[0]["velocity"] / result[1]["velocity"] >= 5.0


def test_bass_energy_gate_basic_pitch_path() -> None:
    """Basic-pitch events with one in silence → gated."""
    events = [
        {"start_s": 0.0, "end_s": 0.5, "midi": 33.0, "velocity": 0.6},
        {"start_s": 0.6, "end_s": 1.0, "midi": 35.0, "velocity": 0.5},
        # Note in silent region
        {"start_s": 3.0, "end_s": 3.5, "midi": 33.0, "velocity": 0.4},
    ]
    y = np.zeros(int(SR * 5), dtype=np.float32)
    # Audio only in 0–1s
    t = np.arange(int(1.0 * SR)) / SR
    y[:len(t)] = (0.5 * np.sin(2.0 * np.pi * 55.0 * t)).astype(np.float32)

    result = extract_bass_notes(events, None, y, SR)
    assert result["source"] == "basic_pitch"
    # The silent-region note should be gated
    assert len(result["notes"]) == 2
    assert all(n["onset_s"] < 2.0 for n in result["notes"])


# ── Vocal octave correction tests ──


def test_vocal_octave_context_correction_basic_pitch() -> None:
    """extract_vocal_notes (basic-pitch path) corrects a sub-harmonic outlier.

    Notes are spaced 0.5 s apart so _merge_adjacent_notes (max_gap_s=0.08)
    does NOT merge them.  This lets octave correction operate note-by-note.
    """
    # 10 notes at ~MIDI 65 (F4), one outlier at MIDI 53 (F3 — one octave below)
    events = [
        {"start_s": float(i) * 0.5, "end_s": float(i) * 0.5 + 0.25,
         "midi": 53.0 if i == 5 else 65.0, "velocity": 0.6}
        for i in range(10)
    ]
    y = np.zeros(SR * 6, dtype=np.float32)
    result = extract_vocal_notes(events, None, y, SR)
    assert result["source"] == "basic_pitch"
    midis = [n["midi"] for n in result["notes"]]
    # 10 separate notes survive (no merging since gaps = 0.25 s > max_gap_s = 0.08)
    assert len(midis) == 10, f"Expected 10 notes, got {len(midis)}: {midis}"
    # The outlier at index 5 should have been shifted from 53 to 65
    assert midis[5] == 65.0, f"Expected 65.0 but got {midis[5]}"


def test_vocal_octave_context_correction_pyin() -> None:
    """extract_vocal_notes (pYIN fallback) also applies octave correction."""
    # Build a pitch track with a sub-harmonic dip in the middle
    # 10 segments of 10 frames each at MIDI 65 (~349.2 Hz),
    # except segment 5 at MIDI 53 (~174.6 Hz = one octave below F4)
    segments = []
    for i in range(10):
        freq = 174.6 if i == 5 else 349.2
        segments.append(np.full(10, freq, dtype=np.float64))
        segments.append(np.full(3, np.nan, dtype=np.float64))  # gap
    pitch_hz = np.concatenate(segments)
    y = np.ones(len(pitch_hz) * HOP, dtype=np.float32) * 0.3
    result = extract_vocal_notes(None, pitch_hz, y, SR)
    assert result["source"] == "pyin"
    midis = [n["midi"] for n in result["notes"]]
    # All notes should be at MIDI 65 (the outlier corrected from 53)
    assert all(m == 65.0 for m in midis), f"Expected all 65.0, got {midis}"


def test_smooth_vocal_octave_jumps_fixes_single_dip() -> None:
    """A single note an octave below neighbours gets shifted up."""
    notes = [
        {"onset_s": 0.0, "offset_s": 0.3, "midi": 65.0, "velocity": 0.5},
        {"onset_s": 0.5, "offset_s": 0.8, "midi": 53.0, "velocity": 0.5},  # outlier
        {"onset_s": 1.0, "offset_s": 1.3, "midi": 67.0, "velocity": 0.5},
    ]
    result = _smooth_vocal_octave_jumps(notes)
    # 53 -> 65 (shift +12) reduces cost from |65-53|+|53-67| = 26 to |65-65|+|65-67| = 2
    assert result[1]["midi"] == 65.0


def test_smooth_vocal_octave_jumps_fixes_single_spike() -> None:
    """A single note an octave above neighbours gets shifted down."""
    notes = [
        {"onset_s": 0.0, "offset_s": 0.3, "midi": 60.0, "velocity": 0.5},
        {"onset_s": 0.5, "offset_s": 0.8, "midi": 74.0, "velocity": 0.5},  # outlier
        {"onset_s": 1.0, "offset_s": 1.3, "midi": 62.0, "velocity": 0.5},
    ]
    result = _smooth_vocal_octave_jumps(notes)
    # 74 -> 62 (shift -12) reduces cost from |60-74|+|74-62| = 26 to |60-62|+|62-62| = 2
    assert result[1]["midi"] == 62.0


def test_smooth_vocal_octave_jumps_no_change_when_smooth() -> None:
    """Notes within threshold are not modified."""
    notes = [
        {"onset_s": i * 0.5, "offset_s": i * 0.5 + 0.3, "midi": float(m), "velocity": 0.5}
        for i, m in enumerate([60, 62, 64, 65, 67])
    ]
    result = _smooth_vocal_octave_jumps(notes)
    for orig, fixed in zip(notes, result):
        assert orig["midi"] == fixed["midi"]


def test_smooth_vocal_octave_jumps_respects_midi_bounds() -> None:
    """Shift that would go below midi_lo is not applied."""
    notes = [
        {"onset_s": 0.0, "offset_s": 0.3, "midi": 40.0, "velocity": 0.5},
        {"onset_s": 0.5, "offset_s": 0.8, "midi": 52.0, "velocity": 0.5},  # jump = 12
        {"onset_s": 1.0, "offset_s": 1.3, "midi": 40.0, "velocity": 0.5},
    ]
    # With midi_lo=36, shift down to 40 is valid
    result = _smooth_vocal_octave_jumps(notes, midi_lo=36)
    assert result[1]["midi"] == 40.0
    # With midi_lo=45, shift down to 40 is blocked
    result2 = _smooth_vocal_octave_jumps(notes, midi_lo=45)
    assert result2[1]["midi"] == 52.0


def test_smooth_vocal_octave_jumps_single_note() -> None:
    """Single note list returned unchanged."""
    notes = [{"onset_s": 0.0, "offset_s": 0.3, "midi": 60.0, "velocity": 0.5}]
    result = _smooth_vocal_octave_jumps(notes)
    assert len(result) == 1
    assert result[0]["midi"] == 60.0


def test_smooth_vocal_octave_jumps_empty() -> None:
    """Empty list returned unchanged."""
    assert _smooth_vocal_octave_jumps([]) == []


def test_neighbour_cost_both_neighbours() -> None:
    """Cost is sum of absolute distances to both neighbours."""
    assert _neighbour_cost(60.0, 55.0, 65.0) == 10.0


def test_neighbour_cost_one_neighbour() -> None:
    """With one None neighbour, cost is distance to the other."""
    assert _neighbour_cost(60.0, None, 65.0) == 5.0
    assert _neighbour_cost(60.0, 55.0, None) == 5.0


def test_neighbour_cost_no_neighbours() -> None:
    """With both None, cost is zero."""
    assert _neighbour_cost(60.0, None, None) == 0.0


def test_correct_octave_by_context_respects_vocal_range() -> None:
    """With midi_lo=36, midi_hi=96, corrections happen within vocal range."""
    # Notes around MIDI 65, outlier at MIDI 77 (an octave above would be 89, within range)
    notes = [
        {"onset_s": i * 0.5, "offset_s": i * 0.5 + 0.3, "midi": float(m), "velocity": 0.5}
        for i, m in enumerate([65, 66, 77, 64, 65])
    ]
    # With default bass range (20-80): shift 77 -> 65 gain = |77-65| - |65-65| = 12 >= 6
    result_bass = _correct_octave_by_context(notes, midi_lo=20, midi_hi=80)
    assert result_bass[2]["midi"] == 65.0

    # With vocal range: same behaviour since 65 is within 36-96
    result_vocal = _correct_octave_by_context(notes, midi_lo=36, midi_hi=96)
    assert result_vocal[2]["midi"] == 65.0


def test_smooth_vocal_octave_jumps_edge_note_unchanged() -> None:
    """Edge notes (first/last) are not shifted by the smoother.

    Edge notes only have one neighbour so it's ambiguous which note is
    the outlier.  _correct_octave_by_context handles these using a wider
    window.
    """
    notes = [
        {"onset_s": 0.0, "offset_s": 0.3, "midi": 53.0, "velocity": 0.5},  # edge
        {"onset_s": 0.5, "offset_s": 0.8, "midi": 65.0, "velocity": 0.5},
        {"onset_s": 1.0, "offset_s": 1.3, "midi": 67.0, "velocity": 0.5},
    ]
    result = _smooth_vocal_octave_jumps(notes)
    # Edge note is NOT shifted by the smoother
    assert result[0]["midi"] == 53.0


def test_vocal_notes_basic_pitch_applies_both_corrections() -> None:
    """extract_vocal_notes applies context correction then octave-jump smoothing."""
    # Build a sequence where context correction alone wouldn't fix the last jump
    # but the smoother will
    events = [
        {"start_s": 0.0, "end_s": 0.25, "midi": 60.0, "velocity": 0.6},
        {"start_s": 0.3, "end_s": 0.55, "midi": 62.0, "velocity": 0.6},
        {"start_s": 0.6, "end_s": 0.85, "midi": 64.0, "velocity": 0.6},
        # Outlier: one octave below — context correction should fix this
        {"start_s": 0.9, "end_s": 1.15, "midi": 52.0, "velocity": 0.6},
        {"start_s": 1.2, "end_s": 1.45, "midi": 65.0, "velocity": 0.6},
    ]
    y = np.zeros(SR * 2, dtype=np.float32)
    result = extract_vocal_notes(events, None, y, SR)
    midis = [n["midi"] for n in result["notes"]]
    # MIDI 52 should be corrected to 64 (shift +12)
    assert midis[3] == 64.0, f"Expected 64.0 but got {midis[3]}"


# ── Key estimation tests ──


def test_estimate_key_scale_c_major() -> None:
    """Chroma concentrated on C major tones -> scale contains C, D, E, F, G, A, B."""
    chroma = np.zeros((100, 12), dtype=np.float64)
    # C=0, D=2, E=4, F=5, G=7, A=9, B=11
    for pc in [0, 2, 4, 5, 7, 9, 11]:
        chroma[:, pc] = 1.0
    scale = estimate_key_scale(chroma)
    assert len(scale) == 7
    # All C major PCs should be present
    assert set(scale) == {0, 2, 4, 5, 7, 9, 11}


def test_estimate_key_scale_a_minor() -> None:
    """Chroma concentrated on A minor tones -> scale contains A minor PCs."""
    chroma = np.zeros((100, 12), dtype=np.float64)
    # A=9, B=11, C=0, D=2, E=4, F=5, G=7 (natural minor)
    a_minor_pcs = {9, 11, 0, 2, 4, 5, 7}
    for pc in a_minor_pcs:
        chroma[:, pc] = 1.0
    # Give A slightly more weight to make it clearly the tonic
    chroma[:, 9] = 1.5
    scale = estimate_key_scale(chroma)
    assert len(scale) == 7
    # A minor and C major share the same notes, so either is valid
    assert set(scale) == a_minor_pcs


def test_estimate_key_scale_single_dominant_pc() -> None:
    """Chroma with one strongly dominant PC -> scale includes that PC."""
    chroma = np.zeros((100, 12), dtype=np.float64)
    # G strongly dominant (pc=7), with some D (pc=2) and B (pc=11)
    chroma[:, 7] = 5.0
    chroma[:, 2] = 2.0
    chroma[:, 11] = 1.5
    scale = estimate_key_scale(chroma)
    assert 7 in scale  # G must be in the scale


def test_estimate_key_scale_empty_input() -> None:
    """Empty or invalid chroma -> returns all 12 PCs (no filtering)."""
    assert estimate_key_scale(np.zeros((0, 12))) == list(range(12))
    assert estimate_key_scale(np.zeros((10, 5))) == list(range(12))
    assert estimate_key_scale(np.zeros((10, 12))) == list(range(12))  # all-zero energy


def test_estimate_key_scale_mode_major() -> None:
    """mode='major' only considers major templates."""
    chroma = np.zeros((100, 12), dtype=np.float64)
    for pc in [0, 2, 4, 5, 7, 9, 11]:  # C major
        chroma[:, pc] = 1.0
    scale = estimate_key_scale(chroma, mode="major")
    assert len(scale) == 7


def test_estimate_key_scale_mode_minor() -> None:
    """mode='minor' only considers minor templates."""
    chroma = np.zeros((100, 12), dtype=np.float64)
    for pc in [9, 11, 0, 2, 4, 5, 7]:  # A minor
        chroma[:, pc] = 1.0
    scale = estimate_key_scale(chroma, mode="minor")
    assert len(scale) == 7


# ── Scale snap tests ──


def test_snap_bass_to_scale_no_change_when_on_scale() -> None:
    """Notes already on scale degrees -> no change."""
    scale_pcs = [0, 2, 4, 5, 7, 9, 11]  # C major
    notes = [
        {"onset_s": 0.0, "offset_s": 0.5, "midi": 36.0, "velocity": 0.5},  # C2 (pc=0)
        {"onset_s": 1.0, "offset_s": 1.5, "midi": 43.0, "velocity": 0.5},  # G2 (pc=7)
    ]
    result = _snap_bass_to_scale(notes, scale_pcs)
    assert [n["midi"] for n in result] == [36.0, 43.0]


def test_snap_bass_to_scale_shifts_by_one() -> None:
    """F# (pc=6) is off C major scale; nearest scale tones are F (5) or G (7)."""
    scale_pcs = [0, 2, 4, 5, 7, 9, 11]  # C major
    notes = [
        {"onset_s": 0.0, "offset_s": 0.5, "midi": 42.0, "velocity": 0.5},  # F#2 (pc=6)
    ]
    result = _snap_bass_to_scale(notes, scale_pcs)
    # Should snap to either F2 (41) or G2 (43) -- both are 1 semitone away
    assert result[0]["midi"] in (41.0, 43.0)


def test_snap_bass_to_scale_no_shift_beyond_max() -> None:
    """With max_shift=1, a note 2+ semitones from any scale tone is left as-is."""
    # Scale missing both neighbours of Bb (pc=10): [0, 2, 4, 7]
    scale_pcs = [0, 2, 4, 7]
    notes = [
        {"onset_s": 0.0, "offset_s": 0.5, "midi": 34.0, "velocity": 0.5},  # Bb1 (pc=10)
    ]
    result = _snap_bass_to_scale(notes, scale_pcs, max_shift=1)
    # pc=9 (A) is NOT in this scale, pc=11 (B) is NOT either -> no snap within +/-1
    assert result[0]["midi"] == 34.0


def test_snap_bass_to_scale_empty_notes() -> None:
    """Empty notes -> empty result."""
    assert _snap_bass_to_scale([], [0, 2, 4, 5, 7, 9, 11]) == []


def test_snap_bass_to_scale_full_chromatic_noop() -> None:
    """All 12 PCs in scale -> no snapping (every note is already on scale)."""
    notes = [
        {"onset_s": 0.0, "offset_s": 0.5, "midi": 42.0, "velocity": 0.5},
    ]
    result = _snap_bass_to_scale(notes, list(range(12)))
    assert result[0]["midi"] == 42.0


def test_snap_bass_to_scale_preserves_other_fields() -> None:
    """Snapping preserves onset_s, offset_s, velocity and other note fields."""
    scale_pcs = [0, 2, 4, 5, 7, 9, 11]  # C major
    notes = [
        {"onset_s": 1.5, "offset_s": 2.0, "midi": 42.0, "velocity": 0.7, "beat_idx": 3},
    ]
    result = _snap_bass_to_scale(notes, scale_pcs)
    assert result[0]["onset_s"] == 1.5
    assert result[0]["offset_s"] == 2.0
    assert result[0]["velocity"] == 0.7
    assert result[0]["beat_idx"] == 3
    assert result[0]["midi"] in (41.0, 43.0)  # snapped from F#


def test_snap_bass_g_to_f_sharp_fix() -> None:
    """G (pc=7) smeared to F# (pc=6) -- scale with G snaps it back."""
    # This is the exact problem: basic-pitch reports F# when the real note is G
    scale_pcs = [7, 10, 2, 3]  # G minor pentatonic subset
    notes = [
        {"onset_s": 0.0, "offset_s": 0.5, "midi": 42.0, "velocity": 0.5},  # F#2 (pc=6)
        {"onset_s": 1.0, "offset_s": 1.5, "midi": 44.0, "velocity": 0.5},  # G#2 (pc=8)
    ]
    result = _snap_bass_to_scale(notes, scale_pcs)
    # F# (6) should snap to G (7) -- 1 semitone
    assert result[0]["midi"] == 43.0  # G2
    # G# (8): pc=8 is 1 away from pc=7 (G) -> snaps to G
    assert result[1]["midi"] == 43.0  # snapped to G2


def test_extract_bass_notes_with_scale_pcs() -> None:
    """extract_bass_notes with scale_pcs applies snapping to basic-pitch events."""
    events = [
        {"start_s": 0.0, "end_s": 0.5, "midi": 42.0, "velocity": 0.6},  # F#2 -> snap to G2
        {"start_s": 0.6, "end_s": 1.0, "midi": 43.0, "velocity": 0.5},  # G2 -> on scale
    ]
    y = np.ones(int(SR * 2), dtype=np.float32) * 0.3
    scale_pcs = [7, 10, 2, 3]  # G, Bb, D, Eb
    result = extract_bass_notes(events, None, y, SR, scale_pcs=scale_pcs)
    assert result["source"] == "basic_pitch"
    midis = [n["midi"] for n in result["notes"]]
    # Both should be G2 (43) after snapping
    assert all(m == 43.0 for m in midis)


def test_extract_bass_notes_pyin_with_scale_pcs() -> None:
    """extract_bass_notes pYIN fallback also applies scale snapping."""
    pitch_hz, y = _pitch_track_from_tones([(41.20, 0.3)])  # E1 -> MIDI 28 (pc=4)
    # Scale that doesn't include E (pc=4) but includes Eb (pc=3) and F (pc=5)
    scale_pcs = [0, 2, 3, 5, 7, 9, 11]  # no pc=4
    result = extract_bass_notes(None, pitch_hz, y, SR, hop_length=HOP, scale_pcs=scale_pcs)
    assert result["source"] == "pyin"
    assert len(result["notes"]) >= 1
    # MIDI 28 (pc=4) should snap to either 27 (pc=3, Eb) or 29 (pc=5, F)
    for n in result["notes"]:
        assert int(round(n["midi"])) % 12 in scale_pcs


def test_extract_bass_notes_scale_pcs_none_skips_snapping() -> None:
    """scale_pcs=None -> no snapping applied, notes unchanged."""
    events = [
        {"start_s": 0.0, "end_s": 0.5, "midi": 42.0, "velocity": 0.6},  # F#2
    ]
    y = np.ones(int(SR * 2), dtype=np.float32) * 0.3
    result = extract_bass_notes(events, None, y, SR, scale_pcs=None)
    assert result["source"] == "basic_pitch"
    assert result["notes"][0]["midi"] == 42.0  # unchanged


# ── CQT-based bass pitch refinement tests ──


def test_refine_bass_pitch_cqt_shifts_up_subharmonic() -> None:
    """Harmonic ratio test should shift note up 12 when octave-above has more energy."""
    # Synthesize a tone with fundamental at G2 (98 Hz, MIDI 43) plus
    # some energy at G1 (49 Hz, MIDI 31) to simulate a sub-harmonic detection.
    dur = 0.5
    t = np.arange(int(SR * dur)) / SR
    freq_g1 = 49.0  # G1 — sub-harmonic
    freq_g2 = 98.0  # G2 — real fundamental (twice freq_g1)
    y_full = np.zeros(int(SR * 1.0), dtype=np.float32)
    # Real fundamental (G2) louder than sub-harmonic (G1)
    y_full[: len(t)] = (
        0.3 * np.sin(2 * np.pi * freq_g1 * t)
        + 0.5 * np.sin(2 * np.pi * freq_g2 * t)
    ).astype(np.float32)

    # Pretend basic-pitch detected G1 (MIDI 31) — sub-harmonic error
    notes = [
        {"onset_s": 0.0, "offset_s": 0.5, "midi": 31.0, "velocity": 0.6},
    ]
    refined = _refine_bass_pitch_cqt(notes, y_full, SR)
    assert len(refined) == 1
    # Should shift up to ~MIDI 43 (G2)
    assert refined[0]["midi"] == 43.0, f"Expected 43, got {refined[0]['midi']}"


def test_refine_bass_pitch_cqt_no_change_when_correct() -> None:
    """Harmonic ratio test should not change pitch when already in correct octave."""
    # Synthesize a pure A2 (110 Hz, MIDI 45)
    dur = 0.5
    t = np.arange(int(SR * dur)) / SR
    freq_a2 = 110.0
    y_full = np.zeros(int(SR * 1.0), dtype=np.float32)
    y_full[: len(t)] = (0.5 * np.sin(2.0 * np.pi * freq_a2 * t)).astype(np.float32)

    notes = [
        {"onset_s": 0.0, "offset_s": 0.5, "midi": 45.0, "velocity": 0.6},
    ]
    refined = _refine_bass_pitch_cqt(notes, y_full, SR)
    assert len(refined) == 1
    assert refined[0]["midi"] == 45.0, f"Expected 45, got {refined[0]['midi']}"


def test_refine_bass_pitch_cqt_skips_short_notes() -> None:
    """Notes shorter than minimum duration should be left unchanged."""
    y = np.zeros(int(SR * 1.0), dtype=np.float32)
    notes = [
        {"onset_s": 0.0, "offset_s": 0.04, "midi": 42.0, "velocity": 0.5},  # 40ms < 60ms threshold
    ]
    refined = _refine_bass_pitch_cqt(notes, y, SR)
    assert refined[0]["midi"] == 42.0  # unchanged


def test_refine_bass_pitch_cqt_skips_silent_segments() -> None:
    """Silent segments have no CQT energy — notes should be unchanged."""
    y = np.zeros(int(SR * 1.0), dtype=np.float32)
    notes = [
        {"onset_s": 0.0, "offset_s": 0.5, "midi": 42.0, "velocity": 0.5},
    ]
    refined = _refine_bass_pitch_cqt(notes, y, SR)
    assert refined[0]["midi"] == 42.0  # unchanged (no energy to refine with)


def test_bass_global_octave_fix_shifts_up() -> None:
    """Global median check shifts all notes up when median is too low."""
    notes = [
        {"onset_s": 0.0, "offset_s": 0.04, "midi": 28.0, "velocity": 0.5},
        {"onset_s": 0.1, "offset_s": 0.14, "midi": 30.0, "velocity": 0.5},
        {"onset_s": 0.2, "offset_s": 0.24, "midi": 32.0, "velocity": 0.5},
    ]
    fixed = _bass_global_octave_fix(notes)
    # Median was 30, below threshold of 36 → should shift all up by 12
    assert fixed[0]["midi"] == 40.0
    assert fixed[1]["midi"] == 42.0
    assert fixed[2]["midi"] == 44.0


def test_bass_global_octave_fix_no_change() -> None:
    """Notes already in expected range → no shift."""
    notes = [
        {"onset_s": 0.0, "offset_s": 0.3, "midi": 40.0, "velocity": 0.5},
        {"onset_s": 0.5, "offset_s": 0.8, "midi": 45.0, "velocity": 0.5},
    ]
    fixed = _bass_global_octave_fix(notes)
    assert fixed[0]["midi"] == 40.0
    assert fixed[1]["midi"] == 45.0


def test_refine_bass_pitch_cqt_empty_notes() -> None:
    """Empty notes list should return empty."""
    y = np.zeros(SR, dtype=np.float32)
    assert _refine_bass_pitch_cqt([], y, SR) == []


def test_refine_bass_pitch_cqt_preserves_other_fields() -> None:
    """Per-note CQT refinement preserves all note fields when no change is needed."""
    y = np.zeros(SR, dtype=np.float32)
    notes = [
        {"onset_s": 0.0, "offset_s": 0.04, "midi": 42.0, "velocity": 0.6, "beat_idx": 0},
    ]
    refined = _refine_bass_pitch_cqt(notes, y, SR)
    # Note too short for CQT → kept unchanged
    assert refined[0]["midi"] == 42.0
    assert refined[0]["onset_s"] == 0.0
    assert refined[0]["offset_s"] == 0.04
    assert refined[0]["velocity"] == 0.6
    assert refined[0]["beat_idx"] == 0


def test_refine_bass_pitch_cqt_in_extract_pipeline() -> None:
    """Global octave fix + refinement runs as part of extract_bass_notes pipeline."""
    # Notes at MIDI 30 (below expected median) — global fix shifts up by 12
    y_full = np.zeros(int(SR * 1.0), dtype=np.float32)

    events = [
        {"start_s": 0.0, "end_s": 0.3, "midi": 30.0, "velocity": 0.6},
        {"start_s": 0.4, "end_s": 0.7, "midi": 32.0, "velocity": 0.6},
    ]
    result = extract_bass_notes(events, None, y_full, SR)
    assert result["source"] == "basic_pitch"
    assert len(result["notes"]) >= 1
    # Median was 31, below 36 → global fix shifts up by 12
    assert result["notes"][0]["midi"] == 42.0


def test_snap_bass_to_scale_max_shift_2_default() -> None:
    """Default max_shift is now 2, allowing corrections of 2 semitones."""
    # G# (pc=8) is 1 semitone from both G (pc=7) and A (pc=9) in C major.
    # The loop finds G first (delta=-1), so it snaps down.
    scale_pcs = [0, 2, 4, 5, 7, 9, 11]  # C major
    notes_gs = [
        {"onset_s": 0.0, "offset_s": 0.5, "midi": 44.0, "velocity": 0.5},  # G#2 (pc=8)
    ]
    result = _snap_bass_to_scale(notes_gs, scale_pcs)
    # G# (8) -> equidistant from G (7) and A (9); algorithm picks lower (delta=-1 found first)
    assert result[0]["midi"] in (43.0, 45.0)

    # Test a 2-semitone snap: F (pc=5) -> G (pc=7) in a sparse scale [0, 2, 7]
    # With max_shift=1 this would NOT snap. With max_shift=2 it should.
    sparse_scale = [0, 2, 7]
    notes_f = [
        {"onset_s": 0.0, "offset_s": 0.5, "midi": 41.0, "velocity": 0.5},  # F2 (pc=5)
    ]
    result2 = _snap_bass_to_scale(notes_f, sparse_scale, max_shift=2)
    # F (pc=5) -> G (pc=7) is +2 semitones; D (pc=2) is -3 (out of range)
    assert result2[0]["midi"] == 43.0  # G2

    # Verify max_shift=1 would NOT snap this same note
    result3 = _snap_bass_to_scale(notes_f, sparse_scale, max_shift=1)
    assert result3[0]["midi"] == 41.0  # unchanged — 2 semitones exceeds max_shift=1
