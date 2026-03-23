"""Tests for songviz.eval — evaluation framework for reduced representation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from songviz.eval import (
    _contour_direction,
    _events_in_range,
    _pitch_class_histogram,
    evaluate_activity,
    evaluate_cross_section_consistency,
    evaluate_layer,
    evaluate_onsets,
    evaluate_pitch_class,
    evaluate_pitch_range,
    evaluate_reduced,
    evaluate_register_stability,
    format_report,
    load_reference,
)


# ── Helpers ──


def _make_notes(
    times: list[tuple[float, float]],
    midi: float = 43.0,
    velocity: float = 0.5,
) -> list[dict]:
    """Create note dicts at given (onset, offset) times."""
    return [
        {"onset_s": t[0], "offset_s": t[1], "midi": midi, "velocity": velocity}
        for t in times
    ]


def _make_hits(times: list[float], component: str = "kick") -> list[dict]:
    """Create drum hit dicts at given times."""
    return [{"t": t, "component": component, "velocity": 0.5} for t in times]


# ── _events_in_range ──


def test_events_in_range_basic() -> None:
    notes = _make_notes([(1.0, 1.5), (3.0, 3.5), (5.0, 5.5)])
    result = _events_in_range(notes, 2.0, 4.0)
    assert len(result) == 1
    assert result[0]["onset_s"] == 3.0


def test_events_in_range_inclusive_start_exclusive_end() -> None:
    notes = _make_notes([(2.0, 2.5), (4.0, 4.5)])
    result = _events_in_range(notes, 2.0, 4.0)
    assert len(result) == 1  # 2.0 included, 4.0 excluded


def test_events_in_range_drum_time_key() -> None:
    hits = _make_hits([1.0, 3.0, 5.0])
    result = _events_in_range(hits, 2.0, 4.0, time_key="t")
    assert len(result) == 1
    assert result[0]["t"] == 3.0


# ── evaluate_activity ──


def test_activity_perfect_match() -> None:
    """Notes in active sections, none in silent → F1 = 1.0."""
    activity = [
        {"start_s": 0.0, "end_s": 10.0, "active": False, "label": "silent"},
        {"start_s": 10.0, "end_s": 30.0, "active": True, "label": "active"},
    ]
    notes = _make_notes([(12.0, 13.0), (15.0, 16.0), (20.0, 21.0)])
    result = evaluate_activity(notes, activity)
    assert result["f1"] == 1.0
    assert result["silent_fp_count"] == 0


def test_activity_false_positives_in_silent() -> None:
    """Notes detected in a silent section → activity FP."""
    activity = [
        {"start_s": 0.0, "end_s": 10.0, "active": False, "label": "silent"},
        {"start_s": 10.0, "end_s": 30.0, "active": True, "label": "active"},
    ]
    # 5 notes in silent region, 5 in active
    notes = _make_notes(
        [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0), (9.0, 9.5)]
        + [(12.0, 13.0), (15.0, 16.0), (20.0, 21.0), (22.0, 23.0), (25.0, 26.0)]
    )
    result = evaluate_activity(notes, activity)
    assert result["fp"] == 1  # silent section wrongly detected active
    assert result["silent_fp_count"] == 5
    assert result["precision"] < 1.0


def test_activity_missed_active_section() -> None:
    """No notes in an active section → FN."""
    activity = [
        {"start_s": 0.0, "end_s": 10.0, "active": True, "label": "should_be_active"},
        {"start_s": 10.0, "end_s": 20.0, "active": True, "label": "also_active"},
    ]
    # Notes only in second section
    notes = _make_notes([(12.0, 13.0), (15.0, 16.0), (18.0, 19.0)])
    result = evaluate_activity(notes, activity)
    assert result["fn"] == 1
    assert result["recall"] == 0.5


def test_activity_drum_hits() -> None:
    """Activity evaluation works with drum hits (time_key='t')."""
    activity = [
        {"start_s": 0.0, "end_s": 5.0, "active": False},
        {"start_s": 5.0, "end_s": 20.0, "active": True},
    ]
    hits = _make_hits([6.0, 8.0, 10.0, 12.0])
    result = evaluate_activity(hits, activity, time_key="t")
    assert result["f1"] == 1.0
    assert result["silent_fp_count"] == 0


def test_activity_min_events_threshold() -> None:
    """Fewer than min_events_for_active notes → not detected as active."""
    activity = [
        {"start_s": 0.0, "end_s": 10.0, "active": True},
    ]
    # Only 2 notes, threshold is 3
    notes = _make_notes([(3.0, 4.0), (7.0, 8.0)])
    result = evaluate_activity(notes, activity, min_events_for_active=3)
    assert result["fn"] == 1  # missed because too few events


# ── evaluate_pitch_range ──


def test_pitch_range_all_in_range() -> None:
    notes = _make_notes([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)], midi=43.0)
    result = evaluate_pitch_range(notes, (38, 50))
    assert result["in_range_pct"] == 100.0
    assert result["below_range_pct"] == 0.0
    assert result["above_range_pct"] == 0.0


def test_pitch_range_mixed() -> None:
    notes = [
        {"onset_s": 1.0, "offset_s": 2.0, "midi": 30.0, "velocity": 0.5},  # below
        {"onset_s": 3.0, "offset_s": 4.0, "midi": 43.0, "velocity": 0.5},  # in range
        {"onset_s": 5.0, "offset_s": 6.0, "midi": 60.0, "velocity": 0.5},  # above
    ]
    result = evaluate_pitch_range(notes, (38, 50))
    assert abs(result["in_range_pct"] - 33.3) < 0.2
    assert abs(result["below_range_pct"] - 33.3) < 0.2
    assert abs(result["above_range_pct"] - 33.3) < 0.2


def test_pitch_range_with_activity_filter() -> None:
    """Only checks notes in active sections when activity is provided."""
    activity = [
        {"start_s": 0.0, "end_s": 5.0, "active": False},
        {"start_s": 5.0, "end_s": 15.0, "active": True},
    ]
    notes = [
        {"onset_s": 2.0, "offset_s": 3.0, "midi": 20.0, "velocity": 0.5},  # in silent (excluded)
        {"onset_s": 7.0, "offset_s": 8.0, "midi": 43.0, "velocity": 0.5},  # in active, in range
        {"onset_s": 10.0, "offset_s": 11.0, "midi": 43.0, "velocity": 0.5},  # in active, in range
    ]
    result = evaluate_pitch_range(notes, (38, 50), activity=activity)
    assert result["checked"] == 2
    assert result["in_range_pct"] == 100.0


# ── evaluate_onsets ──


def test_onsets_perfect_match() -> None:
    ext = [1.0, 2.0, 3.0]
    ref = [1.01, 2.02, 2.99]
    result = evaluate_onsets(ext, ref, tolerance_s=0.05)
    assert result["matched"] == 3
    assert result["f1"] == 1.0


def test_onsets_with_extra_and_missing() -> None:
    ext = [1.0, 2.0, 3.0, 4.0]  # 4.0 is extra (FP)
    ref = [1.0, 2.0, 5.0]  # 5.0 has no match (FN)
    result = evaluate_onsets(ext, ref, tolerance_s=0.05)
    assert result["matched"] == 2
    assert result["fp"] == 2
    assert result["fn"] == 1


def test_onsets_empty_extracted() -> None:
    result = evaluate_onsets([], [1.0, 2.0])
    assert result["fn"] == 2
    assert result["matched"] == 0


def test_onsets_empty_reference() -> None:
    result = evaluate_onsets([1.0, 2.0], [])
    assert result["fp"] == 2
    assert result["matched"] == 0


# ── evaluate_layer ──


def test_evaluate_layer_bass() -> None:
    layer_data = {
        "source": "basic_pitch",
        "notes": _make_notes(
            [(15.0, 16.0), (20.0, 21.0), (25.0, 26.0)],
            midi=43.0,
        ),
    }
    reference = {
        "layer": "bass",
        "confidence": "silver",
        "source": "tabs",
        "activity": [
            {"start_s": 0.0, "end_s": 10.0, "active": False},
            {"start_s": 10.0, "end_s": 30.0, "active": True},
        ],
        "pitch": {"range": [38, 50]},
    }
    result = evaluate_layer(layer_data, reference)
    assert result["layer"] == "bass"
    assert result["activity"]["f1"] == 1.0
    assert result["octave_sensitive"]["pitch_range"]["in_range_pct"] == 100.0


def test_evaluate_layer_drums() -> None:
    layer_data = {
        "source": "drumsep",
        "hits": _make_hits([5.0, 8.0, 12.0, 15.0]),
    }
    reference = {
        "layer": "drums",
        "confidence": "silver",
        "source": "listening",
        "activity": [
            {"start_s": 0.0, "end_s": 3.0, "active": False},
            {"start_s": 3.0, "end_s": 20.0, "active": True},
        ],
    }
    result = evaluate_layer(layer_data, reference)
    assert result["activity"]["f1"] == 1.0
    assert "octave_invariant" not in result  # drums have no pitch


# ── evaluate_reduced (integration) ──


def test_evaluate_reduced_with_temp_references(tmp_path: Path) -> None:
    """Full integration: write references, evaluate, check structure."""
    # Write bass reference
    bass_ref = {
        "layer": "bass",
        "confidence": "silver",
        "source": "test",
        "activity": [
            {"start_s": 0.0, "end_s": 5.0, "active": False},
            {"start_s": 5.0, "end_s": 15.0, "active": True},
        ],
        "pitch": {"range": [38, 50]},
    }
    (tmp_path / "bass.json").write_text(json.dumps(bass_ref))

    reduced = {
        "bass": {
            "source": "basic_pitch",
            "notes": _make_notes([(6.0, 7.0), (9.0, 10.0), (12.0, 13.0)], midi=43.0),
        },
    }

    results = evaluate_reduced(reduced, tmp_path)
    assert "bass" in results["layers"]
    assert results["layers"]["bass"]["activity"]["f1"] == 1.0


def test_evaluate_reduced_missing_layer(tmp_path: Path) -> None:
    """Layer in reference but not in reduced → error entry."""
    bass_ref = {
        "layer": "bass",
        "confidence": "silver",
        "source": "test",
        "activity": [{"start_s": 0.0, "end_s": 10.0, "active": True}],
    }
    (tmp_path / "bass.json").write_text(json.dumps(bass_ref))

    results = evaluate_reduced({}, tmp_path)
    assert results["layers"]["bass"]["error"] == "no extraction data"


# ── format_report ──


def test_format_report_produces_output() -> None:
    results = {
        "layers": {
            "bass": {
                "layer": "bass",
                "confidence": "silver",
                "ref_source": "tabs",
                "extraction_source": "basic_pitch",
                "event_count": 100,
                "activity": {
                    "sections": [],
                    "f1": 1.0, "precision": 1.0, "recall": 1.0,
                    "tp": 2, "fp": 0, "fn": 0, "tn": 1,
                    "silent_fp_count": 0, "silent_fp_rate": 0.0,
                },
                "octave_sensitive": {
                    "pitch_range": {
                        "checked": 100, "in_range_count": 95,
                        "in_range_pct": 95.0, "midi_median": 43.0,
                        "midi_mean": 43.5,
                        "below_range_pct": 3.0, "above_range_pct": 2.0,
                    },
                },
            },
        },
    }
    report = format_report(results)
    assert "Bass" in report
    assert "F1=1.00" in report
    assert "95.0%" in report
    assert "provisional" in report


# ── Octave-invariant metrics ──


def test_pitch_class_histogram_basic() -> None:
    """All-G notes → 100% on pitch class 7."""
    notes = _make_notes([(i, i + 0.5) for i in range(5)], midi=43.0)  # G2
    hist = _pitch_class_histogram(notes)
    assert hist[7] == 1.0  # G = pc 7
    assert hist.sum() == pytest.approx(1.0)


def test_pitch_class_histogram_octave_invariant() -> None:
    """G2 and G3 both map to pitch class 7."""
    notes = [
        {"onset_s": 0.0, "offset_s": 0.5, "midi": 43.0},  # G2
        {"onset_s": 1.0, "offset_s": 1.5, "midi": 55.0},  # G3
    ]
    hist = _pitch_class_histogram(notes)
    assert hist[7] == 1.0  # both are G


def test_contour_direction() -> None:
    notes = [
        {"onset_s": 0.0, "offset_s": 0.5, "midi": 40.0},
        {"onset_s": 1.0, "offset_s": 1.5, "midi": 43.0},  # up
        {"onset_s": 2.0, "offset_s": 2.5, "midi": 43.0},  # same
        {"onset_s": 3.0, "offset_s": 3.5, "midi": 38.0},  # down
    ]
    dirs = _contour_direction(notes)
    assert dirs == [1, 0, -1]


def test_evaluate_pitch_class_root_match() -> None:
    """Notes all on the expected root → root_pc_match=True."""
    notes = _make_notes([(i, i + 0.5) for i in range(10)], midi=43.0)
    ref_pitch = {"root_midi": 43, "scale_pcs": [7, 10, 2, 3]}
    result = evaluate_pitch_class(notes, ref_pitch)
    assert result["root_pc_match"] is True
    assert result["dominant_pc_name"] == "G"
    assert result["in_scale_pct"] == pytest.approx(100.0)


def test_evaluate_pitch_class_octave_shifted() -> None:
    """Notes one octave low still match pitch class."""
    notes = _make_notes([(i, i + 0.5) for i in range(10)], midi=31.0)  # G1
    ref_pitch = {"root_midi": 43, "scale_pcs": [7, 10, 2, 3]}
    result = evaluate_pitch_class(notes, ref_pitch)
    assert result["root_pc_match"] is True  # G1 has same pitch class as G2
    assert result["in_scale_pct"] == pytest.approx(100.0)


def test_evaluate_pitch_class_semitone_error() -> None:
    """Notes shifted +1 semitone → root mismatch, leakage to neighbor."""
    notes = _make_notes([(i, i + 0.5) for i in range(10)], midi=44.0)  # G#
    ref_pitch = {"root_midi": 43, "scale_pcs": [7]}
    result = evaluate_pitch_class(notes, ref_pitch)
    assert result["root_pc_match"] is False
    assert result["root_pc_pct"] == 0.0
    assert result["root_neighbor_pct"] == 100.0  # all in G#+G#=neighbor


def test_cross_section_consistency_identical() -> None:
    """Same notes in two sections → perfect consistency."""
    activity = [
        {"start_s": 0.0, "end_s": 5.0, "active": True, "label": "A"},
        {"start_s": 10.0, "end_s": 15.0, "active": True, "label": "B"},
    ]
    # Same pattern in both sections
    notes = (
        _make_notes([(1.0, 1.5), (2.0, 2.5), (3.0, 3.5), (4.0, 4.5)], midi=43.0)
        + _make_notes([(11.0, 11.5), (12.0, 12.5), (13.0, 13.5), (14.0, 14.5)], midi=43.0)
    )
    result = evaluate_cross_section_consistency(notes, activity)
    assert result["avg_pc_overlap"] == pytest.approx(1.0)
    assert result["avg_contour_similarity"] == pytest.approx(1.0)
    assert result["avg_ioi_ratio"] == pytest.approx(1.0)


def test_cross_section_consistency_different_pitch() -> None:
    """Different pitch classes → low PC overlap, but contour can still match."""
    activity = [
        {"start_s": 0.0, "end_s": 5.0, "active": True, "label": "A"},
        {"start_s": 10.0, "end_s": 15.0, "active": True, "label": "B"},
    ]
    notes_a = [
        {"onset_s": 1.0, "offset_s": 1.5, "midi": 43.0},
        {"onset_s": 2.0, "offset_s": 2.5, "midi": 45.0},
        {"onset_s": 3.0, "offset_s": 3.5, "midi": 43.0},
        {"onset_s": 4.0, "offset_s": 4.5, "midi": 45.0},
    ]
    notes_b = [
        {"onset_s": 11.0, "offset_s": 11.5, "midi": 60.0},  # different PC
        {"onset_s": 12.0, "offset_s": 12.5, "midi": 62.0},
        {"onset_s": 13.0, "offset_s": 13.5, "midi": 60.0},
        {"onset_s": 14.0, "offset_s": 14.5, "midi": 62.0},
    ]
    result = evaluate_cross_section_consistency(notes_a + notes_b, activity)
    assert result["avg_pc_overlap"] < 0.5  # different pitch classes
    assert result["avg_contour_similarity"] == pytest.approx(1.0)  # same up/down/same pattern
    assert result["avg_ioi_ratio"] == pytest.approx(1.0)  # same rhythm


def test_register_stability_no_jumps() -> None:
    """Stepwise motion → no octave jumps."""
    notes = [
        {"onset_s": float(i), "offset_s": float(i) + 0.5, "midi": 43.0 + (i % 3)}
        for i in range(10)
    ]
    result = evaluate_register_stability(notes)
    assert result["octave_jump_count"] == 0
    assert result["large_jump_count"] == 0
    assert result["midi_std"] < 2.0


def test_register_stability_with_jumps() -> None:
    """Alternating octaves → high jump rate."""
    notes = [
        {"onset_s": float(i), "offset_s": float(i) + 0.5,
         "midi": 43.0 if i % 2 == 0 else 55.0}
        for i in range(10)
    ]
    result = evaluate_register_stability(notes)
    assert result["octave_jump_count"] == 9  # every transition is a 12-semitone jump
    assert result["octave_jump_pct"] == 100.0


def test_evaluate_layer_bass_has_octave_invariant() -> None:
    """evaluate_layer produces both octave_invariant and octave_sensitive for bass."""
    layer_data = {
        "source": "basic_pitch",
        "notes": _make_notes(
            [(15.0, 16.0), (20.0, 21.0), (25.0, 26.0)], midi=43.0,
        ),
    }
    reference = {
        "layer": "bass",
        "confidence": "silver",
        "source": "tabs",
        "activity": [
            {"start_s": 0.0, "end_s": 10.0, "active": False},
            {"start_s": 10.0, "end_s": 30.0, "active": True},
        ],
        "pitch": {"root_midi": 43, "range": [38, 50], "scale_pcs": [7, 10, 2, 3]},
    }
    result = evaluate_layer(layer_data, reference)
    assert "octave_invariant" in result
    assert "octave_sensitive" in result
    assert result["octave_invariant"]["pitch_class"]["root_pc_match"] is True
    assert result["octave_invariant"]["register_stability"]["octave_jump_count"] == 0
