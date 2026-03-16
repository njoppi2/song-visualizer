"""Tests for songviz.reduction — drum hit, vocal note & bass note extraction."""
from __future__ import annotations

import numpy as np

from songviz.reduction import (
    _correct_octave_by_context,
    _dedup_octave_overlaps,
    _gate_and_prune_bass_notes,
    _notes_from_pitch_track,
    _rescale_velocity_to_stem_energy,
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
    """Two bass tones (E1 + gap + A1) → 2 notes with correct MIDI (pYIN fallback)."""
    # E1 = 41.20 Hz → MIDI 28, A1 = 55.0 Hz → MIDI 33
    pitch_hz, y = _pitch_track_from_tones([(41.20, 0.3), (55.0, 0.3)])
    result = extract_bass_notes(None, pitch_hz, y, SR, hop_length=HOP)
    assert result["source"] == "pyin"
    assert len(result["notes"]) == 2
    assert result["notes"][0]["midi"] == 28.0
    assert result["notes"][1]["midi"] == 33.0
    assert result["notes"][0]["onset_s"] < result["notes"][1]["onset_s"]


def test_bass_notes_velocity() -> None:
    """Loud note has higher velocity than quiet note; all in [0,1] (pYIN fallback)."""
    loud = _sine_tone(55.0, 0.3, amp=0.8)
    gap = np.zeros(int(0.1 * SR), dtype=np.float32)
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
        # Two overlapping notes: A#1 loud + A#2 quiet → dedup keeps A#1
        {"start_s": 1.0, "end_s": 1.5, "midi": 34.0, "velocity": 0.6},
        {"start_s": 1.0, "end_s": 1.4, "midi": 46.0, "velocity": 0.3},
        # Surrounding context at octave 1
        {"start_s": 0.0, "end_s": 0.3, "midi": 32.0, "velocity": 0.5},
        {"start_s": 0.5, "end_s": 0.8, "midi": 35.0, "velocity": 0.5},
        {"start_s": 2.0, "end_s": 2.3, "midi": 30.0, "velocity": 0.5},
    ]
    y = np.zeros(SR * 3, dtype=np.float32)
    result = extract_bass_notes(events, None, y, SR)
    assert result["source"] == "basic_pitch"
    # Dedup should have removed the A#2 duplicate
    midis = [n["midi"] for n in result["notes"]]
    assert 46.0 not in midis
    # Should have 4 notes (5 minus 1 deduped)
    assert len(result["notes"]) == 4


# ── Bass energy gating & velocity rescaling tests ──


def test_bass_energy_gate_removes_silent_note() -> None:
    """pYIN path: pitch track with note in loud + silent region → only loud note survives."""
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
    # The silent-region note should be gated out
    assert len(result["notes"]) == 1
    assert result["notes"][0]["onset_s"] < 0.5  # the loud one


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
