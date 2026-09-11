"""Tests for songviz.bench — benchmark runner and baseline comparison."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from songviz.bench import (
    _compute_aggregate,
    compare_to_baseline,
    evaluate_all_songs,
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


def _section_result(*, confidence: str = "silver", source: str = "listening", f1: float = 0.8) -> dict:
    return {
        "n_detected": 3,
        "n_reference": 3,
        "over_seg_ratio": 1.0,
        "under_seg_rate": 0.0,
        "boundary_f1_3s": {"f1": f1, "precision": f1, "recall": f1},
        "boundary_f1_05s": {"f1": f1 - 0.1, "precision": f1 - 0.1, "recall": f1 - 0.1},
        "pairwise_f1": {"f1": f1 + 0.05, "precision": f1 + 0.05, "recall": f1 + 0.05},
        "ref_boundaries_s": [10.0],
        "det_boundaries_s": [10.0],
        "ref_confidence": confidence,
        "ref_source": source,
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


def test_aggregate_sections_keeps_all_confidence_groups_diagnostic():
    silver = _make_song_result()
    silver["results"]["sections"] = _section_result(confidence="silver", f1=0.8)
    bronze = _make_song_result()
    bronze["results"]["sections"] = _section_result(
        confidence="bronze", source="audio inference", f1=0.2,
    )

    sections = _compute_aggregate({"silver": silver, "bronze": bronze})["sections"]
    assert sections["silver_gold_diagnostic"]["song_count"] == 1
    assert sections["silver_gold_diagnostic"]["boundary_f1_3s"]["mean"] == 0.8
    assert sections["all_references_diagnostic"]["song_count"] == 2
    assert sections["all_references_diagnostic"]["boundary_f1_3s"]["mean"] == 0.5
    assert sections["by_reference_confidence"]["bronze"]["song_count"] == 1


# ── Tests: evaluate_all_songs section story integration ──


def _benchmark_song(tmp_path: Path) -> tuple[dict, Path, Path]:
    audio_path = tmp_path / "song.flac"
    audio_path.write_bytes(b"not decoded")
    ref_dir = tmp_path / "references"
    ref_dir.mkdir()
    (ref_dir / "sections.json").write_text(json.dumps({
        "confidence": "silver",
        "source": "human listening",
        "sections": [
            {"start_s": 0.0, "end_s": 10.0, "label": "intro"},
            {"start_s": 10.0, "end_s": 20.0, "label": "verse"},
        ],
    }))
    reduced_path = tmp_path / "reduced.json"
    reduced_path.write_text("{}")
    return {
        "song_id": "song-id",
        "audio_path": audio_path,
        "ref_dir": ref_dir,
        "ref_name": "song-ref",
    }, reduced_path, tmp_path / "output"


def test_evaluate_all_songs_loads_story_and_evaluates_sections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    song, reduced_path, out_dir = _benchmark_song(tmp_path)
    story_path = out_dir / "analysis" / "story.json"
    story_path.parent.mkdir(parents=True)
    story_path.write_text(json.dumps({"sections": [
        {"start_s": 0.0, "end_s": 10.0},
        {"start_s": 10.0, "end_s": 20.0},
    ]}))
    monkeypatch.setattr("songviz.bench.find_benchmark_songs", lambda _songs_dir: [song])
    monkeypatch.setattr("songviz.bench.ensure_reduced", lambda *_args, **_kwargs: reduced_path)
    monkeypatch.setattr("songviz.bench.output_dir_for_audio", lambda *_args, **_kwargs: out_dir)

    result = evaluate_all_songs(tmp_path, force_reduce=True)
    song_result = result["songs"]["song-ref"]
    assert song_result["results"]["sections"]["boundary_f1_3s"]["f1"] == pytest.approx(1.0)
    assert song_result["section_evaluation"]["status"] == "evaluated"
    assert song_result["section_evaluation"]["reference"] == {
        "available": True,
        "sha256": sha256((song["ref_dir"] / "sections.json").read_bytes()).hexdigest(),
        "source": "human listening",
        "confidence": "silver",
    }
    assert song_result["section_evaluation"]["story_freshness"] == "stale_possible_after_force_reduce"


@pytest.mark.parametrize("story_text, expected_status", [
    (None, "not_evaluated_story_missing"),
    ("{not json", "not_evaluated_story_malformed"),
])
def test_evaluate_all_songs_keeps_layers_when_story_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    story_text: str | None,
    expected_status: str,
):
    song, reduced_path, out_dir = _benchmark_song(tmp_path)
    if story_text is not None:
        story_path = out_dir / "analysis" / "story.json"
        story_path.parent.mkdir(parents=True)
        story_path.write_text(story_text)
    seen_stories: list[dict | None] = []

    def fake_evaluate(reduced: dict, ref_dir: Path, *, story: dict | None = None) -> dict:
        seen_stories.append(story)
        return {"layers": {"bass": {"activity": {"f1": 0.7, "silent_fp_rate": 0.1}}}}

    monkeypatch.setattr("songviz.bench.find_benchmark_songs", lambda _songs_dir: [song])
    monkeypatch.setattr("songviz.bench.ensure_reduced", lambda *_args, **_kwargs: reduced_path)
    monkeypatch.setattr("songviz.bench.output_dir_for_audio", lambda *_args, **_kwargs: out_dir)
    monkeypatch.setattr("songviz.bench.evaluate_reduced", fake_evaluate)

    result = evaluate_all_songs(tmp_path)
    song_result = result["songs"]["song-ref"]
    assert seen_stories == [None]
    assert song_result["results"]["layers"]["bass"]["activity"]["f1"] == 0.7
    assert song_result["section_evaluation"]["status"] == expected_status
    assert song_result["section_evaluation"]["warnings"]


@pytest.mark.parametrize("sections", [
    [{"start_s": float("nan"), "end_s": 10.0}],
    [{"start_s": -1.0, "end_s": 10.0}],
    [{"start_s": 10.0, "end_s": 10.0}],
])
def test_evaluate_all_songs_rejects_unsafe_story_intervals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sections: list[dict],
):
    song, reduced_path, out_dir = _benchmark_song(tmp_path)
    story_path = out_dir / "analysis" / "story.json"
    story_path.parent.mkdir(parents=True)
    story_path.write_text(json.dumps({"sections": sections}))
    seen_stories: list[dict | None] = []

    def fake_evaluate(reduced: dict, ref_dir: Path, *, story: dict | None = None) -> dict:
        seen_stories.append(story)
        return {"layers": {"bass": {"activity": {"f1": 0.7, "silent_fp_rate": 0.1}}}}

    monkeypatch.setattr("songviz.bench.find_benchmark_songs", lambda _songs_dir: [song])
    monkeypatch.setattr("songviz.bench.ensure_reduced", lambda *_args, **_kwargs: reduced_path)
    monkeypatch.setattr("songviz.bench.output_dir_for_audio", lambda *_args, **_kwargs: out_dir)
    monkeypatch.setattr("songviz.bench.evaluate_reduced", fake_evaluate)

    song_result = evaluate_all_songs(tmp_path)["songs"]["song-ref"]
    assert seen_stories == [None]
    assert song_result["results"]["layers"]["bass"]["activity"]["f1"] == 0.7
    assert song_result["section_evaluation"]["status"] == "not_evaluated_story_malformed"


def test_evaluate_all_songs_rejects_malformed_section_reference_without_losing_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    song, reduced_path, out_dir = _benchmark_song(tmp_path)
    story_path = out_dir / "analysis" / "story.json"
    story_path.parent.mkdir(parents=True)
    story_path.write_text(json.dumps({"sections": [
        {"start_s": 0.0, "end_s": 10.0}, {"start_s": 10.0, "end_s": 20.0},
    ]}))
    (song["ref_dir"] / "sections.json").write_text(json.dumps({
        "confidence": "silver", "source": "bad reference",
        "sections": [{"start_s": 20.0, "end_s": 10.0}],
    }))
    seen_stories: list[dict | None] = []

    def fake_evaluate(reduced: dict, ref_dir: Path, *, story: dict | None = None) -> dict:
        seen_stories.append(story)
        return {"layers": {"bass": {"activity": {"f1": 0.7, "silent_fp_rate": 0.1}}}}

    monkeypatch.setattr("songviz.bench.find_benchmark_songs", lambda _songs_dir: [song])
    monkeypatch.setattr("songviz.bench.ensure_reduced", lambda *_args, **_kwargs: reduced_path)
    monkeypatch.setattr("songviz.bench.output_dir_for_audio", lambda *_args, **_kwargs: out_dir)
    monkeypatch.setattr("songviz.bench.evaluate_reduced", fake_evaluate)

    song_result = evaluate_all_songs(tmp_path)["songs"]["song-ref"]
    assert seen_stories == [None]
    assert song_result["results"]["layers"]["bass"]["activity"]["f1"] == 0.7
    assert song_result["section_evaluation"]["status"] == "not_evaluated_reference_malformed"
    assert song_result["section_evaluation"]["warnings"]


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


def _section_summary(
    *,
    boundary_f1: float,
    under_seg_rate: float,
    source: str = "listening",
    over_seg_ratio: float = 1.0,
    song_id: str = "song-a",
    reference_sha256: str | None = "a" * 64,
) -> dict:
    metric = lambda value: {"mean": value, "min": value, "max": value, "n": 1}
    signature_complete = reference_sha256 is not None
    return {
        "song_count": 1,
        "reference_confidences": ["silver"],
        "reference_sources": [source],
        "boundary_f1_3s": metric(boundary_f1),
        "boundary_f1_05s": metric(boundary_f1),
        "pairwise_f1": metric(boundary_f1),
        "under_seg_rate": metric(under_seg_rate),
        "over_seg_ratio": metric(over_seg_ratio),
        "reference_cohort": {
            "members": [{"song_id": song_id, "sections_json_sha256": reference_sha256}],
            "member_song_ids": [song_id],
            "sections_json_sha256": [reference_sha256],
            "signature_status": "complete" if signature_complete else "unknown_missing_song_id_or_sha256",
            "reference_independence": "unverified" if signature_complete else "unknown_missing_signature",
        },
    }


def test_comparison_detects_section_regressions_for_matching_diagnostic_cohort():
    baseline = {"aggregate": {"sections": {
        "all_references_diagnostic": _section_summary(boundary_f1=0.9, under_seg_rate=0.1),
    }}}
    current = {"aggregate": {"sections": {
        "all_references_diagnostic": _section_summary(
            boundary_f1=0.7, under_seg_rate=0.3, over_seg_ratio=10.0,
        ),
    }}}

    comp = compare_to_baseline(current, baseline)
    section_regressions = [r for r in comp["regressions"] if r["layer"] == "sections"]
    assert {r["metric"] for r in section_regressions} == {
        "boundary_f1_3s", "boundary_f1_05s", "pairwise_f1", "under_seg_rate",
    }
    assert {r["cohort"] for r in section_regressions} == {"all_references_diagnostic"}
    assert all(r["metric"] != "over_seg_ratio" for r in section_regressions)
    assert not comp["incomparable"]


def test_comparison_marks_changed_section_reference_cohort_incomparable():
    baseline = {"aggregate": {"sections": {
        "all_references_diagnostic": _section_summary(boundary_f1=0.9, under_seg_rate=0.1),
    }}}
    current = {"aggregate": {"sections": {
        "all_references_diagnostic": _section_summary(
            boundary_f1=0.2, under_seg_rate=0.8, source="different source",
        ),
    }}}

    comp = compare_to_baseline(current, baseline)
    assert not comp["regressions"]
    assert comp["incomparable"] == [
        "sections[all_references_diagnostic]: reference cohort changed; metrics not compared",
    ]
    assert "Incomparable metrics" in format_comparison_report(comp)


@pytest.mark.parametrize("current_member", [
    {"song_id": "song-b"},
    {"reference_sha256": "b" * 64},
])
def test_comparison_marks_same_count_source_but_changed_section_membership_incomparable(
    current_member: dict[str, str],
):
    baseline = {"aggregate": {"sections": {
        "all_references_diagnostic": _section_summary(boundary_f1=0.9, under_seg_rate=0.1),
    }}}
    current = {"aggregate": {"sections": {
        "all_references_diagnostic": _section_summary(
            boundary_f1=0.2, under_seg_rate=0.8, **current_member,
        ),
    }}}

    comp = compare_to_baseline(current, baseline)
    assert not comp["regressions"]
    assert comp["incomparable"] == [
        "sections[all_references_diagnostic]: exact song/reference hash membership changed; metrics not compared",
    ]


def test_comparison_marks_missing_section_hash_incomparable():
    baseline = {"aggregate": {"sections": {
        "all_references_diagnostic": _section_summary(boundary_f1=0.9, under_seg_rate=0.1),
    }}}
    current = {"aggregate": {"sections": {
        "all_references_diagnostic": _section_summary(
            boundary_f1=0.2, under_seg_rate=0.8, reference_sha256=None,
        ),
    }}}

    comp = compare_to_baseline(current, baseline)
    assert not comp["regressions"]
    assert comp["incomparable"] == [
        "sections[all_references_diagnostic]: reference independence/version unknown "
        "(missing exact song/hash signature); metrics not compared",
    ]


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


def test_format_bench_report_labels_activity_transcription_and_section_reference_groups():
    song = _make_song_result()
    song["results"]["layers"]["bass"]["note_transcription"] = {
        "ref_note_count": 10,
        "note_f1": 0.6,
        "note_f1_octave_invariant": 0.7,
        "onset_f1": 0.8,
        "onset_precision": 0.8,
        "onset_recall": 0.8,
        "mean_onset_error_ms": 15.0,
        "note_precision": 0.6,
        "note_recall": 0.6,
        "pitch_tol_st": 1.0,
        "pitch_accuracy": 0.5,
        "mean_pitch_error_st": 0.5,
        "det_note_count": 12,
        "fragmentation_ratio": 1.2,
    }
    song["results"]["sections"] = _section_result(confidence="bronze", source="audio inference")
    report = format_bench_report({
        "timestamp": "2026-03-23T10:00:00Z",
        "song_count": 1,
        "success_count": 1,
        "errors": [],
        "songs": {"test-song": song},
        "aggregate": _compute_aggregate({"test-song": song}),
    })
    assert "Coarse activity" in report
    assert "Note-level transcription" in report
    assert "Silver/gold-labelled references (independence unverified)" in report
    assert "All-reference diagnostic" in report
    assert "audio inference" in report


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
