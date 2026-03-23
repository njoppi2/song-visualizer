"""Tests for songviz.bench — benchmark runner and baseline comparison."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from songviz.bench import (
    _compute_aggregate,
    compare_to_baseline,
    find_benchmark_songs,
    format_bench_report,
    format_comparison_report,
    save_baseline,
)


# ── Fixtures ──


def _make_song_result(
    *,
    activity_f1: float = 1.0,
    silent_fp_rate: float = 0.0,
    in_scale_pct: float = 50.0,
    root_pc_pct: float = 20.0,
    octave_jump_pct: float = 10.0,
    in_range_pct: float = 70.0,
    below_range_pct: float = 20.0,
) -> dict:
    """Build a minimal song result dict for testing aggregation."""
    return {
        "song_id": "abc123",
        "audio_file": "test.flac",
        "results": {
            "layers": {
                "bass": {
                    "layer": "bass",
                    "event_count": 100,
                    "activity": {
                        "f1": activity_f1,
                        "precision": 1.0,
                        "recall": 1.0,
                        "silent_fp_count": 0,
                        "silent_fp_rate": silent_fp_rate,
                    },
                    "octave_invariant": {
                        "pitch_class": {
                            "checked": 100,
                            "dominant_pc": 7,
                            "dominant_pc_name": "G",
                            "dominant_pc_pct": root_pc_pct,
                            "in_scale_pct": in_scale_pct,
                            "root_pc_pct": root_pc_pct,
                        },
                        "register_stability": {
                            "checked": 100,
                            "midi_std": 3.5,
                            "midi_range": 12.0,
                            "octave_jump_count": 5,
                            "octave_jump_pct": octave_jump_pct,
                            "large_jump_count": 10,
                            "large_jump_pct": 15.0,
                            "median_abs_interval": 3.0,
                            "mean_abs_interval": 3.5,
                        },
                    },
                    "octave_sensitive": {
                        "pitch_range": {
                            "checked": 100,
                            "in_range_count": 70,
                            "in_range_pct": in_range_pct,
                            "midi_median": 43.0,
                            "midi_mean": 42.5,
                            "below_range_pct": below_range_pct,
                            "above_range_pct": 10.0,
                        },
                    },
                },
                "drums": {
                    "layer": "drums",
                    "event_count": 200,
                    "activity": {
                        "f1": 1.0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "silent_fp_count": 0,
                        "silent_fp_rate": 0.0,
                    },
                },
            },
        },
    }


# ── Tests: _compute_aggregate ──


def test_aggregate_single_song():
    results = {"song_a": _make_song_result(in_scale_pct=40.0, root_pc_pct=10.0)}
    agg = _compute_aggregate(results)
    assert "bass" in agg
    assert agg["bass"]["in_scale_pct"]["mean"] == 40.0
    assert agg["bass"]["root_pc_pct"]["mean"] == 10.0
    assert agg["bass"]["in_scale_pct"]["n"] == 1


def test_aggregate_multiple_songs():
    results = {
        "song_a": _make_song_result(in_scale_pct=40.0, activity_f1=0.9),
        "song_b": _make_song_result(in_scale_pct=60.0, activity_f1=1.0),
    }
    agg = _compute_aggregate(results)
    assert agg["bass"]["in_scale_pct"]["mean"] == 50.0
    assert agg["bass"]["in_scale_pct"]["min"] == 40.0
    assert agg["bass"]["in_scale_pct"]["max"] == 60.0
    assert agg["bass"]["activity_f1"]["mean"] == 0.95
    assert agg["bass"]["activity_f1"]["n"] == 2


def test_aggregate_skips_errors():
    results = {
        "song_a": _make_song_result(in_scale_pct=40.0),
        "song_b": {"song_id": "xyz", "audio_file": "bad.flac", "error": "no stems"},
    }
    agg = _compute_aggregate(results)
    assert agg["bass"]["in_scale_pct"]["n"] == 1
    assert agg["bass"]["in_scale_pct"]["mean"] == 40.0


def test_aggregate_drums_has_activity_only():
    results = {"song_a": _make_song_result()}
    agg = _compute_aggregate(results)
    assert "drums" in agg
    assert "activity_f1" in agg["drums"]
    # Drums don't have pitch metrics
    assert "in_scale_pct" not in agg["drums"]


# ── Tests: compare_to_baseline ──


def test_comparison_no_regression():
    baseline = {
        "aggregate": {
            "bass": {
                "activity_f1": {"mean": 0.95, "min": 0.9, "max": 1.0, "n": 2},
                "in_scale_pct": {"mean": 40.0, "min": 38.0, "max": 42.0, "n": 2},
            },
        },
    }
    current = {
        "aggregate": {
            "bass": {
                "activity_f1": {"mean": 0.96, "min": 0.92, "max": 1.0, "n": 2},
                "in_scale_pct": {"mean": 41.0, "min": 39.0, "max": 43.0, "n": 2},
            },
        },
    }
    comp = compare_to_baseline(current, baseline)
    assert not comp["has_regressions"]
    assert len(comp["regressions"]) == 0


def test_comparison_detects_regression():
    baseline = {
        "aggregate": {
            "bass": {
                "activity_f1": {"mean": 1.0, "min": 1.0, "max": 1.0, "n": 2},
            },
        },
    }
    current = {
        "aggregate": {
            "bass": {
                # F1 dropped by 0.1 — well beyond 0.02 threshold
                "activity_f1": {"mean": 0.9, "min": 0.85, "max": 0.95, "n": 2},
            },
        },
    }
    comp = compare_to_baseline(current, baseline)
    assert comp["has_regressions"]
    assert len(comp["regressions"]) == 1
    assert comp["regressions"][0]["metric"] == "activity_f1"
    assert comp["regressions"][0]["delta"] == pytest.approx(-0.1, abs=0.01)


def test_comparison_detects_improvement():
    baseline = {
        "aggregate": {
            "bass": {
                "in_scale_pct": {"mean": 40.0, "min": 38.0, "max": 42.0, "n": 2},
            },
        },
    }
    current = {
        "aggregate": {
            "bass": {
                "in_scale_pct": {"mean": 70.0, "min": 65.0, "max": 75.0, "n": 2},
            },
        },
    }
    comp = compare_to_baseline(current, baseline)
    assert not comp["has_regressions"]
    assert len(comp["improvements"]) == 1
    assert comp["improvements"][0]["metric"] == "in_scale_pct"


def test_comparison_lower_is_better():
    """For metrics like octave_jump_pct, lower is better — increase = regression."""
    baseline = {
        "aggregate": {
            "bass": {
                "octave_jump_pct": {"mean": 5.0, "min": 4.0, "max": 6.0, "n": 2},
            },
        },
    }
    current = {
        "aggregate": {
            "bass": {
                # Octave jumps increased — regression
                "octave_jump_pct": {"mean": 15.0, "min": 12.0, "max": 18.0, "n": 2},
            },
        },
    }
    comp = compare_to_baseline(current, baseline)
    assert comp["has_regressions"]
    assert comp["regressions"][0]["metric"] == "octave_jump_pct"


# ── Tests: save_baseline ──


def test_save_baseline(tmp_path):
    results = {"timestamp": "2026-03-23T10:00:00Z", "songs": {}, "aggregate": {}}
    path = save_baseline(results, baselines_dir=tmp_path)
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["timestamp"] == "2026-03-23T10:00:00Z"

    # Also check latest.json
    latest = tmp_path / "latest.json"
    assert latest.exists()


# ── Tests: format_bench_report ──


def test_format_bench_report_produces_output():
    results = {
        "timestamp": "2026-03-23T10:00:00Z",
        "song_count": 1,
        "success_count": 1,
        "errors": [],
        "songs": {"test-song": _make_song_result()},
        "aggregate": {"bass": {"activity_f1": {"mean": 1.0, "min": 1.0, "max": 1.0, "n": 1}}},
    }
    report = format_bench_report(results)
    assert "test-song" in report
    assert "AGGREGATE" in report


def test_format_comparison_report_with_regressions():
    comp = {
        "has_regressions": True,
        "regressions": [{"layer": "bass", "metric": "activity_f1", "baseline": 1.0, "current": 0.9, "delta": -0.1, "direction": "higher"}],
        "improvements": [],
        "unchanged_count": 5,
    }
    report = format_comparison_report(comp)
    assert "REGRESSIONS" in report
    assert "activity_f1" in report


def test_format_comparison_report_no_regressions():
    comp = {
        "has_regressions": False,
        "regressions": [],
        "improvements": [{"layer": "bass", "metric": "in_scale_pct", "baseline": 40.0, "current": 70.0, "delta": 30.0, "direction": "higher"}],
        "unchanged_count": 3,
    }
    report = format_comparison_report(comp)
    assert "No regressions" in report
    assert "in_scale_pct" in report


# ── Tests: find_benchmark_songs ──


def test_find_benchmark_songs_returns_list():
    """Smoke test — just checks that the function returns a list without crashing."""
    songs_dir = Path("songs")
    if not songs_dir.exists():
        pytest.skip("songs/ directory not available")
    found = find_benchmark_songs(songs_dir)
    assert isinstance(found, list)
    # Each entry should have required keys
    for entry in found:
        assert "song_id" in entry
        assert "audio_path" in entry
        assert "ref_dir" in entry
