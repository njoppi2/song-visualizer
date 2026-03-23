"""Benchmark runner: evaluate reduced representations across all benchmark songs.

Usage::

    songviz bench                         # run eval on all benchmark songs
    songviz bench --json                  # machine-readable JSON output
    songviz bench --save-baseline         # save current results as baseline
    songviz bench --baseline baseline.json  # compare against a saved baseline
"""
from __future__ import annotations

import json
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
from .paths import output_dir_for_audio, reduced_path_for_output_dir


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
            results = evaluate_reduced(reduced, song["ref_dir"])
            all_results[ref_name] = {
                "song_id": song_id,
                "audio_file": audio_path.name,
                "results": results,
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

    # Average each metric
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
    }
    # Regression threshold (absolute change that triggers a flag)
    regression_threshold = 0.02  # 2% or 0.02 F1

    cur_agg = current.get("aggregate", {})
    base_agg = baseline.get("aggregate", {})

    for layer_name in set(list(cur_agg.keys()) + list(base_agg.keys())):
        cur_metrics = cur_agg.get(layer_name, {})
        base_metrics = base_agg.get(layer_name, {})

        for metric_name in set(list(cur_metrics.keys()) + list(base_metrics.keys())):
            cur_val = cur_metrics.get(metric_name, {}).get("mean")
            base_val = base_metrics.get(metric_name, {}).get("mean")

            if cur_val is None or base_val is None:
                continue

            delta = cur_val - base_val
            direction = metric_direction.get(metric_name, "higher")

            # Determine if this is a regression
            is_regression = False
            if direction == "higher" and delta < -regression_threshold:
                is_regression = True
            elif direction == "lower" and delta > regression_threshold:
                is_regression = True

            is_improvement = False
            if direction == "higher" and delta > regression_threshold:
                is_improvement = True
            elif direction == "lower" and delta < -regression_threshold:
                is_improvement = True

            entry = {
                "layer": layer_name,
                "metric": metric_name,
                "baseline": round(base_val, 4),
                "current": round(cur_val, 4),
                "delta": round(delta, 4),
                "direction": direction,
            }

            if is_regression:
                regressions.append(entry)
            elif is_improvement:
                improvements.append(entry)
            else:
                unchanged.append(f"{layer_name}.{metric_name}")

    return {
        "has_regressions": len(regressions) > 0,
        "regressions": regressions,
        "improvements": improvements,
        "unchanged_count": len(unchanged),
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

    # Aggregate
    agg = bench_results.get("aggregate", {})
    if agg:
        lines.append(f"\n{'='*60}")
        lines.append("AGGREGATE (mean across songs)")
        lines.append(f"{'='*60}")
        for layer_name, metrics in agg.items():
            lines.append(f"\n  {layer_name.capitalize()}:")
            for metric_name, vals in metrics.items():
                lines.append(
                    f"    {metric_name}: {vals['mean']:.4f} "
                    f"(min={vals['min']:.4f}, max={vals['max']:.4f}, n={vals['n']})"
                )

    return "\n".join(lines)


def format_comparison_report(comparison: dict[str, Any]) -> str:
    """Format baseline comparison as a human-readable report."""
    lines: list[str] = []

    if comparison["has_regressions"]:
        lines.append("REGRESSIONS DETECTED:")
        for r in comparison["regressions"]:
            arrow = "↓" if r["direction"] == "higher" else "↑"
            lines.append(
                f"  {arrow} {r['layer']}.{r['metric']}: "
                f"{r['baseline']:.4f} → {r['current']:.4f} "
                f"(delta={r['delta']:+.4f})"
            )
    else:
        lines.append("No regressions detected.")

    if comparison["improvements"]:
        lines.append("\nImprovements:")
        for imp in comparison["improvements"]:
            arrow = "↑" if imp["direction"] == "higher" else "↓"
            lines.append(
                f"  {arrow} {imp['layer']}.{imp['metric']}: "
                f"{imp['baseline']:.4f} → {imp['current']:.4f} "
                f"(delta={imp['delta']:+.4f})"
            )

    lines.append(f"\n{comparison['unchanged_count']} metrics unchanged.")

    return "\n".join(lines)
