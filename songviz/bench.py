"""Benchmark runner: evaluate reduced representations across all benchmark songs.

Usage::

    songviz bench                         # run eval on all benchmark songs
    songviz bench --json                  # machine-readable JSON output
    songviz bench --save-baseline         # save current results as baseline
    songviz bench --baseline baseline.json  # compare against a saved baseline
"""
from __future__ import annotations

import json
import math
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .eval import (
    evaluate_reduced,
    format_report,
    load_reference,
    references_dir_for_song,
    _load_song_index,
    _BENCHMARK_DIR,
)
from .ingest import song_id_for_path
from .paths import (
    analysis_path_for_output_dir,
    output_dir_for_audio,
    reduced_path_for_output_dir,
    story_path_for_output_dir,
)


# ── Reduction from cached stems ──


def ensure_reduced(audio_path: Path, *, force: bool = False) -> Path:
    """Generate reduced.json for a song using cached stems.

    Assumes stems already exist under ``outputs/<song>/stems/``.
    Runs the extraction pipeline (drums, bass, vocals) without rendering video.

    Returns the path to reduced.json.
    """
    from .analyze import analyze_file
    from .pipeline import _build_stem_analyses

    song_id = song_id_for_path(audio_path)
    out_dir = output_dir_for_audio(audio_path, str(song_id))
    reduced_path = reduced_path_for_output_dir(out_dir)

    if reduced_path.exists() and not force:
        return reduced_path

    # Need stems to exist
    stems_dir = out_dir / "stems"
    if not stems_dir.exists():
        raise FileNotFoundError(
            f"Stems not found at {stems_dir}. "
            "Run `songviz stems <audio>` first."
        )

    analysis = analyze_file(str(audio_path))
    # _build_stem_analyses produces reduced.json as a side effect
    _build_stem_analyses(
        audio_path, out_dir, analysis,
        stems_model="htdemucs",
        stems_device="auto",
        stems_force=False,
    )
    return reduced_path


# ── Benchmark orchestration ──


def find_benchmark_songs(songs_dir: Path) -> list[dict[str, Any]]:
    """Find all audio files in songs_dir that have benchmark references.

    Returns a list of dicts: {song_id, audio_path, ref_dir, ref_name}.
    """
    index = _load_song_index()
    if not index:
        return []

    found: list[dict[str, Any]] = []
    audio_files = sorted(
        p for p in songs_dir.iterdir()
        if p.suffix.lower() in (".flac", ".mp3", ".wav", ".ogg", ".m4a")
    )

    for audio_path in audio_files:
        sid = song_id_for_path(audio_path)
        ref_name = index.get(sid)
        if ref_name is None:
            continue
        ref_dir = _BENCHMARK_DIR / "references" / ref_name
        if not ref_dir.is_dir():
            continue
        found.append({
            "song_id": sid,
            "audio_path": audio_path,
            "ref_dir": ref_dir,
            "ref_name": ref_name,
        })

    return found


def _validate_sections_document(data: Any) -> str | None:
    """Return a reason a section document is unsafe for evaluation, if any."""
    if not isinstance(data, dict):
        return "document is not an object"
    sections = data.get("sections")
    if not isinstance(sections, list) or not sections:
        return "no usable non-empty sections list"

    previous_end = -1.0
    for index, section in enumerate(sections):
        if not isinstance(section, dict):
            return f"section {index} is not an object"
        start, end = section.get("start_s"), section.get("end_s")
        if (
            isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
            or not math.isfinite(start) or not math.isfinite(end)
        ):
            return f"section {index} has non-finite or non-numeric start_s/end_s"
        if start < 0 or end <= start:
            return f"section {index} has negative or reversed/empty interval"
        if start < previous_end:
            return f"section {index} overlaps or is out of order"
        previous_end = float(end)
    return None


def _load_benchmark_story(out_dir: Path, *, force_reduce: bool) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Load the canonical story artifact, with a legacy analysis fallback.

    ``story.json`` is the canonical persisted story artifact for this benchmark.
    Older output directories may predate that artifact, so only when it is
    absent do we fall back to ``analysis.json``'s embedded ``story`` object.
    A malformed canonical artifact is reported rather than masked by a fallback.
    """
    story_path = story_path_for_output_dir(out_dir)
    warnings: list[str] = []

    if story_path.exists():
        try:
            story = json.loads(story_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, {
                "status": "not_evaluated_story_malformed",
                "story_source": "story.json",
                "warnings": [f"story.json could not be parsed: {exc}"],
            }
        invalid_reason = _validate_sections_document(story)
        if invalid_reason:
            return None, {
                "status": "not_evaluated_story_malformed",
                "story_source": "story.json",
                "warnings": [f"story.json is unsafe for section evaluation: {invalid_reason}"],
            }
        source = "story.json"
    else:
        # Legacy compatibility only: modern pipeline runs always write story.json.
        analysis_path = analysis_path_for_output_dir(out_dir)
        if not analysis_path.exists():
            return None, {
                "status": "not_evaluated_story_missing",
                "warnings": ["No story.json artifact (or legacy analysis.json fallback) found"],
            }
        try:
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, {
                "status": "not_evaluated_story_malformed",
                "story_source": "analysis.json fallback",
                "warnings": [f"legacy analysis.json could not be parsed: {exc}"],
            }
        story = analysis.get("story") if isinstance(analysis, dict) else None
        invalid_reason = _validate_sections_document(story)
        if invalid_reason:
            return None, {
                "status": "not_evaluated_story_missing",
                "story_source": "analysis.json fallback",
                "warnings": [f"legacy analysis.json story is unsafe for section evaluation: {invalid_reason}"],
            }
        source = "analysis.json fallback"
        warnings.append("story.json is absent; using legacy analysis.json story fallback")

    status: dict[str, Any] = {
        "status": "available",
        "story_source": source,
        "warnings": warnings,
    }
    if force_reduce:
        # ensure_reduced refreshes reduced.json but does not persist story.json.
        status["story_freshness"] = "stale_possible_after_force_reduce"
        status["warnings"].append(
            "--force-reduce refreshed reduced.json; this story artifact was not regenerated and may be stale"
        )
    return story, status


def _section_reference_metadata(ref_dir: Path) -> dict[str, Any]:
    """Read and validate section-reference provenance before evaluation."""
    path = ref_dir / "sections.json"
    if not path.exists():
        return {"available": False}
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return {"available": True, "malformed": True, "warning": str(exc)}
    reference_sha256 = sha256(raw).hexdigest()
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "available": True,
            "malformed": True,
            "sha256": reference_sha256,
            "warning": str(exc),
        }
    if not isinstance(data, dict):
        return {
            "available": True,
            "malformed": True,
            "sha256": reference_sha256,
            "warning": "sections.json is not an object",
        }
    invalid_reason = _validate_sections_document(data)
    if invalid_reason:
        return {
            "available": True,
            "malformed": True,
            "sha256": reference_sha256,
            "source": data.get("source", "unknown"),
            "confidence": data.get("confidence", "unknown"),
            "warning": f"sections.json is unsafe for section evaluation: {invalid_reason}",
        }
    return {
        "available": True,
        "sha256": reference_sha256,
        "source": data.get("source", "unknown"),
        "confidence": data.get("confidence", "unknown"),
    }


def evaluate_all_songs(
    songs_dir: Path,
    *,
    force_reduce: bool = False,
) -> dict[str, Any]:
    """Run eval on all benchmark songs, return consolidated results."""
    songs = find_benchmark_songs(songs_dir)
    if not songs:
        return {"error": "No benchmark songs found", "songs": {}}

    all_results: dict[str, Any] = {}
    errors: list[str] = []

    for song in songs:
        ref_name = song["ref_name"]
        audio_path = song["audio_path"]
        song_id = song["song_id"]

        try:
            reduced_path = ensure_reduced(audio_path, force=force_reduce)
            reduced = json.loads(reduced_path.read_text(encoding="utf-8"))
            out_dir = output_dir_for_audio(audio_path, str(song_id))
            story, section_status = _load_benchmark_story(out_dir, force_reduce=force_reduce)
            ref_metadata = _section_reference_metadata(song["ref_dir"])
            section_status["reference"] = ref_metadata
            # evaluate_reduced loads sections.json itself.  Withhold an otherwise
            # valid story when that reference is malformed so layer evaluation
            # still runs without allowing malformed sections to abort the song.
            story_for_evaluation = None if ref_metadata.get("malformed") else story
            results = evaluate_reduced(reduced, song["ref_dir"], story=story_for_evaluation)

            if "sections" in results:
                section_status["status"] = "evaluated"
                section_status["reference"] = {
                    **ref_metadata,
                    "available": True,
                    "source": results["sections"].get("ref_source", "unknown"),
                    "confidence": results["sections"].get("ref_confidence", "unknown"),
                }
            elif ref_metadata.get("malformed"):
                section_status["status"] = "not_evaluated_reference_malformed"
                section_status["warnings"].append(
                    f"sections.json could not be used: {ref_metadata.get('warning', 'malformed reference')}"
                )
            elif not ref_metadata.get("available"):
                section_status["status"] = "not_applicable_no_section_reference"
            elif story is not None:
                section_status["status"] = "not_evaluated_empty_reference"
                section_status["warnings"].append("sections.json has no usable reference sections")
            all_results[ref_name] = {
                "song_id": song_id,
                "audio_file": audio_path.name,
                "results": results,
                "section_evaluation": section_status,
            }
        except Exception as e:
            errors.append(f"{ref_name}: {e}")
            all_results[ref_name] = {
                "song_id": song_id,
                "audio_file": audio_path.name,
                "error": str(e),
            }

    # Compute aggregate metrics across all successful evals
    aggregate = _compute_aggregate(all_results)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "song_count": len(songs),
        "success_count": len(songs) - len(errors),
        "errors": errors,
        "songs": all_results,
        "aggregate": aggregate,
    }


def _compute_aggregate(all_results: dict[str, Any]) -> dict[str, Any]:
    """Compute aggregate metrics across all songs."""
    agg: dict[str, dict[str, list[float]]] = {}

    for _ref_name, song_data in all_results.items():
        if "error" in song_data:
            continue
        layers = song_data.get("results", {}).get("layers", {})
        for layer_name, layer_data in layers.items():
            if "error" in layer_data:
                continue
            if layer_name not in agg:
                agg[layer_name] = {}

            # Activity F1
            act = layer_data.get("activity")
            if act:
                agg[layer_name].setdefault("activity_f1", []).append(act["f1"])
                agg[layer_name].setdefault("silent_fp_rate", []).append(act["silent_fp_rate"])

            # Octave-invariant
            oi = layer_data.get("octave_invariant", {})
            pc = oi.get("pitch_class", {})
            if pc.get("checked", 0) > 0:
                if "in_scale_pct" in pc:
                    agg[layer_name].setdefault("in_scale_pct", []).append(pc["in_scale_pct"])
                if "root_pc_pct" in pc:
                    agg[layer_name].setdefault("root_pc_pct", []).append(pc["root_pc_pct"])

            rs = oi.get("register_stability", {})
            if rs.get("checked", 0) >= 2:
                agg[layer_name].setdefault("octave_jump_pct", []).append(rs["octave_jump_pct"])

            # Octave-sensitive
            os_ = layer_data.get("octave_sensitive", {})
            pr = os_.get("pitch_range", {})
            if pr.get("checked", 0) > 0:
                agg[layer_name].setdefault("in_range_pct", []).append(pr["in_range_pct"])
                agg[layer_name].setdefault("below_range_pct", []).append(pr["below_range_pct"])

            # Note-level transcription (only present if MIDI reference exists)
            nt = layer_data.get("note_transcription")
            if nt and nt.get("ref_note_count", 0) > 0:
                agg[layer_name].setdefault("note_f1", []).append(nt["note_f1"])
                agg[layer_name].setdefault("note_f1_octave_invariant", []).append(nt["note_f1_octave_invariant"])
                agg[layer_name].setdefault("onset_f1", []).append(nt["onset_f1"])
                agg[layer_name].setdefault("pitch_accuracy", []).append(nt["pitch_accuracy"] * 100.0)
                agg[layer_name].setdefault("fragmentation_ratio", []).append(nt["fragmentation_ratio"])

    # Average each layer metric.
    result: dict[str, Any] = {}
    for layer_name, metrics in agg.items():
        result[layer_name] = {}
        for metric_name, values in metrics.items():
            import numpy as np
            result[layer_name][metric_name] = {
                "mean": round(float(np.mean(values)), 4),
                "min": round(float(np.min(values)), 4),
                "max": round(float(np.max(values)), 4),
                "n": len(values),
            }

    # All section references are diagnostic. Confidence labels describe the
    # references but do not establish that their sources are independent.
    section_entries: list[dict[str, Any]] = []
    for song_data in all_results.values():
        if "error" in song_data:
            continue
        section = song_data.get("results", {}).get("sections")
        if not section:
            continue
        entry = dict(section)
        reference = song_data.get("section_evaluation", {}).get("reference", {})
        entry["_benchmark_song_id"] = song_data.get("song_id")
        entry["_sections_json_sha256"] = reference.get("sha256")
        section_entries.append(entry)

    def summarise_sections(entries: list[dict[str, Any]]) -> dict[str, Any]:
        metrics: dict[str, list[float]] = {}
        for section in entries:
            for name, value in (
                ("boundary_f1_3s", section.get("boundary_f1_3s", {}).get("f1")),
                ("boundary_f1_05s", section.get("boundary_f1_05s", {}).get("f1")),
                ("pairwise_f1", section.get("pairwise_f1", {}).get("f1")),
                ("over_seg_ratio", section.get("over_seg_ratio")),
                ("under_seg_rate", section.get("under_seg_rate")),
            ):
                if isinstance(value, (int, float)):
                    metrics.setdefault(name, []).append(float(value))
        members = sorted(
            (
                {
                    "song_id": entry.get("_benchmark_song_id"),
                    "sections_json_sha256": entry.get("_sections_json_sha256"),
                }
                for entry in entries
            ),
            key=lambda member: (str(member["song_id"]), str(member["sections_json_sha256"])),
        )
        signature_complete = bool(members) and all(
            isinstance(member["song_id"], str) and member["song_id"]
            and isinstance(member["sections_json_sha256"], str) and member["sections_json_sha256"]
            for member in members
        )
        summary: dict[str, Any] = {
            "song_count": len(entries),
            "reference_confidences": sorted({str(e.get("ref_confidence", "unknown")) for e in entries}),
            "reference_sources": sorted({str(e.get("ref_source", "unknown")) for e in entries}),
            "reference_cohort": {
                "members": members,
                "member_song_ids": [member["song_id"] for member in members],
                "sections_json_sha256": [member["sections_json_sha256"] for member in members],
                "signature_status": "complete" if signature_complete else "unknown_missing_song_id_or_sha256",
                # Hash pinning identifies the reference version; it cannot prove
                # that listening/SSM sources were independent.
                "reference_independence": "unverified" if signature_complete else "unknown_missing_signature",
            },
        }
        for name, values in metrics.items():
            import numpy as np
            summary[name] = {
                "mean": round(float(np.mean(values)), 4),
                "min": round(float(np.min(values)), 4),
                "max": round(float(np.max(values)), 4),
                "n": len(values),
            }
        return summary

    if section_entries:
        by_confidence: dict[str, list[dict[str, Any]]] = {}
        for entry in section_entries:
            confidence = str(entry.get("ref_confidence", "unknown")).lower()
            by_confidence.setdefault(confidence, []).append(entry)
        silver_gold = [
            entry for entry in section_entries
            if str(entry.get("ref_confidence", "unknown")).lower() in {"silver", "gold"}
        ]
        result["sections"] = {
            "silver_gold_diagnostic": summarise_sections(silver_gold),
            "all_references_diagnostic": summarise_sections(section_entries),
            "by_reference_confidence": {
                confidence: summarise_sections(entries)
                for confidence, entries in sorted(by_confidence.items())
            },
        }

    return result


# ── Baseline comparison ──


def save_baseline(
    bench_results: dict[str, Any],
    baselines_dir: Path | None = None,
) -> Path:
    """Save benchmark results as a baseline."""
    if baselines_dir is None:
        baselines_dir = _BENCHMARK_DIR / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = baselines_dir / f"baseline_{ts}.json"
    path.write_text(json.dumps(bench_results, indent=2, default=str) + "\n", encoding="utf-8")

    # Also write a "latest" symlink/copy
    latest = baselines_dir / "latest.json"
    latest.write_text(json.dumps(bench_results, indent=2, default=str) + "\n", encoding="utf-8")

    return path


def compare_to_baseline(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Compare current results against baseline, flag regressions.

    Returns a dict with per-metric deltas and regression flags.
    """
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    unchanged: list[str] = []
    incomparable: list[str] = []

    # Thresholds: how much a metric can degrade before it's a regression.
    # Positive = higher is better; negative = lower is better.
    metric_direction: dict[str, str] = {
        "activity_f1": "higher",
        "silent_fp_rate": "lower",
        "in_scale_pct": "higher",
        "root_pc_pct": "higher",
        "octave_jump_pct": "lower",
        "in_range_pct": "higher",
        "below_range_pct": "lower",
        "boundary_f1_3s": "higher",
        "boundary_f1_05s": "higher",
        "pairwise_f1": "higher",
        "under_seg_rate": "lower",
    }
    # Regression threshold (absolute change that triggers a flag)
    regression_threshold = 0.02  # 2% or 0.02 F1

    cur_agg = current.get("aggregate", {})
    base_agg = baseline.get("aggregate", {})

    def compare_metric(
        layer_name: str,
        metric_name: str,
        current_metric: dict[str, Any],
        baseline_metric: dict[str, Any],
        *,
        cohort: str | None = None,
    ) -> None:
        cur_val = current_metric.get("mean")
        base_val = baseline_metric.get("mean")
        if not isinstance(cur_val, (int, float)) or not isinstance(base_val, (int, float)):
            return

        delta = cur_val - base_val
        direction = metric_direction.get(metric_name, "higher")
        is_regression = (
            (direction == "higher" and delta < -regression_threshold)
            or (direction == "lower" and delta > regression_threshold)
        )
        is_improvement = (
            (direction == "higher" and delta > regression_threshold)
            or (direction == "lower" and delta < -regression_threshold)
        )
        entry = {
            "layer": layer_name,
            "metric": metric_name,
            "baseline": round(base_val, 4),
            "current": round(cur_val, 4),
            "delta": round(delta, 4),
            "direction": direction,
        }
        if cohort is not None:
            entry["cohort"] = cohort
        if is_regression:
            regressions.append(entry)
        elif is_improvement:
            improvements.append(entry)
        else:
            suffix = f"[{cohort}]" if cohort else ""
            unchanged.append(f"{layer_name}{suffix}.{metric_name}")

    for layer_name in set(list(cur_agg.keys()) + list(base_agg.keys())):
        if layer_name == "sections":
            continue
        cur_metrics = cur_agg.get(layer_name, {})
        base_metrics = base_agg.get(layer_name, {})

        for metric_name in set(list(cur_metrics.keys()) + list(base_metrics.keys())):
            compare_metric(
                layer_name, metric_name,
                cur_metrics.get(metric_name, {}), base_metrics.get(metric_name, {}),
            )

    def all_reference_section_cohort(aggregate: dict[str, Any]) -> dict[str, Any] | None:
        sections = aggregate.get("sections", {})
        if not isinstance(sections, dict):
            return None
        summary = sections.get("all_references_diagnostic")
        return summary if isinstance(summary, dict) else None

    # Compare only the explicitly all-reference diagnostic cohort. Comparing
    # both it and its confidence subgroups would double-count the same songs.
    cohort = "all_references_diagnostic"
    current_summary = all_reference_section_cohort(cur_agg)
    baseline_summary = all_reference_section_cohort(base_agg)
    if current_summary is None or baseline_summary is None:
        if current_summary is not None or baseline_summary is not None:
            incomparable.append(f"sections[{cohort}]: cohort is missing from one run")
    else:
        current_signature = current_summary.get("reference_cohort", {})
        baseline_signature = baseline_summary.get("reference_cohort", {})
        if (
            not isinstance(current_signature, dict)
            or not isinstance(baseline_signature, dict)
            or current_signature.get("signature_status") != "complete"
            or baseline_signature.get("signature_status") != "complete"
        ):
            incomparable.append(
                f"sections[{cohort}]: reference independence/version unknown "
                "(missing exact song/hash signature); metrics not compared"
            )
        elif current_signature.get("members") != baseline_signature.get("members"):
            incomparable.append(
                f"sections[{cohort}]: exact song/reference hash membership changed; metrics not compared"
            )
        else:
            cohort_keys = ("song_count", "reference_confidences", "reference_sources")
            if any(current_summary.get(key) != baseline_summary.get(key) for key in cohort_keys):
                incomparable.append(f"sections[{cohort}]: reference cohort changed; metrics not compared")
            else:
                # over_seg_ratio is intentionally descriptive only: unlike an F1 or
                # missed-boundary rate, moving toward either direction is not uniformly
                # better across over- and under-segmented outputs.
                for metric_name in ("boundary_f1_3s", "boundary_f1_05s", "pairwise_f1", "under_seg_rate"):
                    compare_metric(
                        "sections", metric_name,
                        current_summary.get(metric_name, {}), baseline_summary.get(metric_name, {}),
                        cohort=cohort,
                    )

    return {
        "has_regressions": len(regressions) > 0,
        "regressions": regressions,
        "improvements": improvements,
        "unchanged_count": len(unchanged),
        "incomparable": incomparable,
    }


# ── Report formatting ──


def format_bench_report(bench_results: dict[str, Any]) -> str:
    """Format consolidated benchmark results as a human-readable report."""
    lines: list[str] = []
    lines.append(f"Benchmark run: {bench_results.get('timestamp', '?')}")
    lines.append(f"Songs: {bench_results.get('success_count', 0)}/{bench_results.get('song_count', 0)} evaluated")

    errors = bench_results.get("errors", [])
    if errors:
        lines.append(f"\nErrors:")
        for e in errors:
            lines.append(f"  - {e}")

    # Per-song results
    for ref_name, song_data in bench_results.get("songs", {}).items():
        lines.append(f"\n{'='*60}")
        lines.append(f"Song: {ref_name} ({song_data.get('audio_file', '?')})")
        lines.append(f"{'='*60}")

        if "error" in song_data:
            lines.append(f"  ERROR: {song_data['error']}")
            continue

        results = song_data.get("results", {})
        lines.append(format_report(results))

        section_status = song_data.get("section_evaluation")
        if section_status:
            status = section_status.get("status", "unknown")
            reference = section_status.get("reference", {})
            ref_detail = ""
            if reference.get("available"):
                ref_detail = (
                    f" | reference={reference.get('confidence', 'unknown')}: "
                    f"{reference.get('source', 'unknown')}"
                )
            lines.append(f"  Section evaluation status: {status}{ref_detail}")
            if section_status.get("story_source"):
                freshness = section_status.get("story_freshness")
                fresh_detail = f" ({freshness})" if freshness else ""
                lines.append(f"    Story: {section_status['story_source']}{fresh_detail}")
            for warning in section_status.get("warnings", []):
                lines.append(f"    WARNING: {warning}")

    # Aggregate
    agg = bench_results.get("aggregate", {})
    if agg:
        lines.append(f"\n{'='*60}")
        lines.append("AGGREGATE (mean across songs)")
        lines.append(f"{'='*60}")
        for layer_name, metrics in agg.items():
            if layer_name == "sections":
                continue
            lines.append(f"\n  {layer_name.capitalize()}:")

            def add_metrics(title: str, names: tuple[str, ...]) -> None:
                present = [(name, metrics[name]) for name in names if name in metrics]
                if not present:
                    return
                lines.append(f"    {title}")
                for metric_name, vals in present:
                    lines.append(
                        f"      {metric_name}: {vals['mean']:.4f} "
                        f"(min={vals['min']:.4f}, max={vals['max']:.4f}, n={vals['n']})"
                    )

            add_metrics(
                "Coarse activity (section occupancy; not note transcription):",
                ("activity_f1", "silent_fp_rate"),
            )
            add_metrics(
                "Pitch / register diagnostics:",
                ("in_scale_pct", "root_pc_pct", "octave_jump_pct", "in_range_pct", "below_range_pct"),
            )
            add_metrics(
                "Note-level transcription (MIDI references only):",
                ("note_f1", "note_f1_octave_invariant", "onset_f1", "pitch_accuracy", "fragmentation_ratio"),
            )

        sections_agg = agg.get("sections")
        if sections_agg:
            lines.append("\n  Sections (boundary/grouping evaluation):")

            def add_section_group(label: str, summary: dict[str, Any]) -> None:
                lines.append(
                    f"    {label}: songs={summary.get('song_count', 0)} "
                    f"| confidence={', '.join(summary.get('reference_confidences', [])) or 'none'}"
                )
                sources = summary.get("reference_sources", [])
                if sources:
                    lines.append(f"      Sources: {'; '.join(sources)}")
                signature = summary.get("reference_cohort", {})
                if signature:
                    lines.append(
                        f"      Reference cohort: {signature.get('signature_status', 'unknown')} "
                        f"| independence={signature.get('reference_independence', 'unknown')}"
                    )
                    song_ids = [str(value) for value in signature.get("member_song_ids", []) if value]
                    hashes = [str(value) for value in signature.get("sections_json_sha256", []) if value]
                    if song_ids:
                        lines.append(f"      Song IDs: {', '.join(song_ids)}")
                    if hashes:
                        lines.append(
                            "      sections.json SHA256: "
                            f"{', '.join(hashes)}"
                        )
                for metric_name in (
                    "boundary_f1_3s", "boundary_f1_05s", "pairwise_f1",
                    "over_seg_ratio", "under_seg_rate",
                ):
                    vals = summary.get(metric_name)
                    if vals:
                        lines.append(
                            f"      {metric_name}: {vals['mean']:.4f} "
                            f"(min={vals['min']:.4f}, max={vals['max']:.4f}, n={vals['n']})"
                        )

            add_section_group(
                "Silver/gold-labelled references (independence unverified) — diagnostic",
                sections_agg.get("silver_gold_diagnostic", {}),
            )
            add_section_group(
                "All-reference diagnostic (includes bronze/inferred references; not independent validation)",
                sections_agg.get("all_references_diagnostic", {}),
            )
            for confidence, summary in sections_agg.get("by_reference_confidence", {}).items():
                add_section_group(f"Diagnostic reference confidence group: {confidence}", summary)

    return "\n".join(lines)


def format_comparison_report(comparison: dict[str, Any]) -> str:
    """Format baseline comparison as a human-readable report."""
    lines: list[str] = []

    if comparison["has_regressions"]:
        lines.append("REGRESSIONS DETECTED:")
        for r in comparison["regressions"]:
            arrow = "↓" if r["direction"] == "higher" else "↑"
            cohort = f"[{r['cohort']}]" if r.get("cohort") else ""
            lines.append(
                f"  {arrow} {r['layer']}{cohort}.{r['metric']}: "
                f"{r['baseline']:.4f} → {r['current']:.4f} "
                f"(delta={r['delta']:+.4f})"
            )
    else:
        lines.append("No regressions detected.")

    if comparison["improvements"]:
        lines.append("\nImprovements:")
        for imp in comparison["improvements"]:
            arrow = "↑" if imp["direction"] == "higher" else "↓"
            cohort = f"[{imp['cohort']}]" if imp.get("cohort") else ""
            lines.append(
                f"  {arrow} {imp['layer']}{cohort}.{imp['metric']}: "
                f"{imp['baseline']:.4f} → {imp['current']:.4f} "
                f"(delta={imp['delta']:+.4f})"
            )

    incomparable = comparison.get("incomparable", [])
    if incomparable:
        lines.append("\nIncomparable metrics (not treated as regressions):")
        lines.extend(f"  - {reason}" for reason in incomparable)

    lines.append(f"\n{comparison['unchanged_count']} metrics unchanged.")

    return "\n".join(lines)
