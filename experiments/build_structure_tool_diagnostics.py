#!/usr/bin/env python3
"""Build dashboard-facing structure-tool diagnostics from available outputs.

This does not run external models. It packages everything we already have into
one JSON so the dashboard can show useful candidates before we decide what to
keep: current SongViz sections, novelty peaks, Phrase Spec low-similarity block
starts, old phrase-grid experiment output, and placeholders for external tools.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _song_id(song_path: Path) -> str:
    return song_path.stem


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else None


def _nearest_beat_bar(t: float, beats: np.ndarray, bar_phase: int, beats_per_bar: int = 4) -> dict[str, Any]:
    if beats.size == 0:
        return {}
    idx = int(np.argmin(np.abs(beats - float(t))))
    beat_delta = float(float(t) - beats[idx])
    bar_indices = np.arange(int(bar_phase) % beats_per_bar, beats.size, beats_per_bar, dtype=int)
    if bar_indices.size:
        nearest_bar_pos = int(np.argmin(np.abs(beats[bar_indices] - float(t))))
        nearest_bar_beat_idx = int(bar_indices[nearest_bar_pos])
        nearest_bar_idx = int(nearest_bar_pos)
        delta_to_bar = float(float(t) - beats[nearest_bar_beat_idx])
    else:
        nearest_bar_beat_idx = ""
        nearest_bar_idx = ""
        delta_to_bar = None
    return {
        "nearest_beat_idx": idx,
        "delta_to_beat_s": beat_delta,
        "nearest_bar_beat_idx": nearest_bar_beat_idx,
        "nearest_bar_idx": nearest_bar_idx,
        "delta_to_bar_s": delta_to_bar,
    }


def _bar_phase(bar_diag: dict[str, Any] | None) -> int:
    if not bar_diag:
        return 0
    summary = bar_diag.get("summary", {})
    if "external_consensus_phase" in summary:
        return int(summary["external_consensus_phase"])
    votes: dict[int, int] = {}
    for src in bar_diag.get("sources", []) or []:
        ph = src.get("selected_phase")
        if ph is not None:
            votes[int(ph)] = votes.get(int(ph), 0) + 1
    return max(votes, key=votes.get) if votes else 0


def _songviz_story_tool(analysis: dict[str, Any], beats: np.ndarray, phase: int) -> dict[str, Any]:
    sections = analysis.get("story", {}).get("sections", []) or []
    segments = []
    boundaries = []
    for sec in sections:
        start = float(sec.get("start_s", 0.0))
        end = float(sec.get("end_s", start))
        item = {
            "start_s": start,
            "end_s": end,
            "label": sec.get("label", ""),
            "role": sec.get("role", ""),
            "label_confidence": sec.get("confidence", ""),
            "boundary_confidence": sec.get("novelty_to_prev", ""),
            **_nearest_beat_bar(start, beats, phase),
        }
        segments.append(item)
        if start > 0:
            boundaries.append({
                "time_s": start,
                "type": "section_start",
                "label": sec.get("label", ""),
                "role": sec.get("role", ""),
                "confidence": sec.get("novelty_to_prev", ""),
                **_nearest_beat_bar(start, beats, phase),
            })
    return {
        "id": "songviz_story",
        "name": "SongViz Story",
        "kind": "current sections",
        "status": "available",
        "notes": "Current deterministic SongViz section detector.",
        "segments": segments,
        "boundaries": boundaries,
    }


def _local_peak_indices(values: np.ndarray, *, min_distance: int, limit: int) -> list[int]:
    if values.size < 3:
        return []
    candidates = [
        i for i in range(1, values.size - 1)
        if values[i] >= values[i - 1] and values[i] >= values[i + 1]
    ]
    candidates.sort(key=lambda i: float(values[i]), reverse=True)
    picked: list[int] = []
    for idx in candidates:
        if all(abs(idx - prev) >= min_distance for prev in picked):
            picked.append(idx)
        if len(picked) >= limit:
            break
    return sorted(picked)


def _novelty_peak_tool(analysis: dict[str, Any], beats: np.ndarray, phase: int) -> dict[str, Any]:
    nov = analysis.get("story", {}).get("novelties", {}) or {}
    times = np.asarray(nov.get("times_s", []), dtype=float)
    boundaries = []
    for key, label in (
        ("section_diff", "Section diff"),
        ("novelty_medium", "Novelty 16b"),
        ("novelty_long", "Novelty 32b"),
    ):
        vals = np.asarray(nov.get(key, []), dtype=float)
        n = min(times.size, vals.size)
        if n < 3:
            continue
        vals = vals[:n]
        ts = times[:n]
        thresh = float(np.nanpercentile(vals, 82))
        idxs = [i for i in _local_peak_indices(vals, min_distance=max(8, n // 35), limit=16) if vals[i] >= thresh]
        for idx in idxs:
            t = float(ts[idx])
            if t <= 0:
                continue
            boundaries.append({
                "time_s": t,
                "type": key,
                "label": label,
                "confidence": float(vals[idx]),
                **_nearest_beat_bar(t, beats, phase),
            })
    return {
        "id": "songviz_novelty_peaks",
        "name": "SongViz Novelty Peaks",
        "kind": "boundary proposals",
        "status": "available",
        "notes": "High peaks from section_diff and medium/long novelty curves.",
        "segments": [],
        "boundaries": sorted(boundaries, key=lambda b: float(b["time_s"])),
    }


def _phrase_spec_tools(analysis: dict[str, Any], beats: np.ndarray, phase: int) -> list[dict[str, Any]]:
    tools = []
    stems = analysis.get("story", {}).get("stem_novelties", {}) or {}
    for stem, data in stems.items():
        spans = data.get("block_spans", {}).get("16b", []) or []
        sims = np.asarray(data.get("phrase_spec_sim_16", []), dtype=float)
        x = np.asarray(analysis.get("story", {}).get("novelties", {}).get("times_s", []), dtype=float)
        n = min(sims.size, x.size)
        sims_use = sims[:n]
        x_use = x[:n]
        boundaries = []
        for span in spans:
            start = float(span.get("start_s", 0.0))
            end = float(span.get("end_s", start))
            if end <= start or n == 0:
                continue
            mask = (x_use >= start) & (x_use < end)
            if not np.any(mask):
                continue
            score = float(np.nanmedian(sims_use[mask]))
            if score <= 0.72:
                boundaries.append({
                    "time_s": start,
                    "type": "phrase_spec_16_low_similarity",
                    "label": f"{stem} low sim",
                    "confidence": float(1.0 - score),
                    "similarity": score,
                    **_nearest_beat_bar(start, beats, phase),
                })
        tools.append({
            "id": f"phrase_spec_16_{stem}",
            "name": f"Phrase Spec 16 {stem}",
            "kind": "stem phrase boundary proposals",
            "status": "available",
            "notes": "16-beat Phrase Spec blocks whose current-vs-previous similarity is low.",
            "segments": [],
            "boundaries": boundaries,
        })
    return tools


def _phrase_grid_experiment_tool(exp: dict[str, Any] | None, beats: np.ndarray, trusted_phase: int) -> dict[str, Any]:
    if not exp:
        return {
            "id": "phrase_grid_experiment",
            "name": "Phrase Grid Experiment",
            "kind": "phrase grid",
            "available": False,
            "status": "not found",
            "notes": "No experiments/results/*/results.json found.",
            "segments": [],
            "boundaries": [],
        }
    rec = exp.get("recommended", {}) or {}
    phrase_len_bars = int(rec.get("phrase_length", 8) or 8)
    offset_bars = int(rec.get("phrase_offset", 0) or 0)
    exp_phase = int(rec.get("bar_phase", trusted_phase) or 0)
    beats_per_bar = int(exp.get("config", {}).get("beats_per_bar", 4) or 4)
    boundaries = []
    segments = []
    if beats.size:
        first_beat = exp_phase + offset_bars * beats_per_bar
        step = max(1, phrase_len_bars * beats_per_bar)
        starts = list(range(first_beat, len(beats), step))
        for i, s in enumerate(starts):
            e = starts[i + 1] if i + 1 < len(starts) else min(len(beats) - 1, s + step)
            if s < 0 or s >= len(beats):
                continue
            start_t = float(beats[s])
            end_t = float(beats[e]) if e < len(beats) else start_t
            segments.append({
                "start_s": start_t,
                "end_s": end_t,
                "label": f"{phrase_len_bars} bars",
                "role": f"offset {offset_bars}",
                "label_confidence": exp.get("recommended", {}).get("hybrid_confidence", ""),
                "boundary_confidence": exp.get("recommended", {}).get("hybrid_confidence", ""),
                **_nearest_beat_bar(start_t, beats, trusted_phase),
            })
            if i > 0:
                boundaries.append({
                    "time_s": start_t,
                    "type": "phrase_grid_start",
                    "label": f"{phrase_len_bars} bars",
                    "confidence": exp.get("recommended", {}).get("hybrid_confidence", ""),
                    **_nearest_beat_bar(start_t, beats, trusted_phase),
                })
    return {
        "id": "phrase_grid_experiment",
        "name": "Phrase Grid Experiment",
        "kind": "old experimental phrase grid",
        "status": "available",
        "confidence": rec.get("hybrid_confidence", ""),
        "notes": (
            "Old experiment kept for visual comparison. Its bar-phase choice is "
            "not trusted if it disagrees with external downbeat diagnostics."
        ),
        "segments": segments,
        "boundaries": boundaries,
        "raw_recommended": rec,
    }


def build(song_path: Path, output_dir: Path | None = None) -> dict[str, Any]:
    song_id = _song_id(song_path)
    output_dir = output_dir or Path("outputs") / song_id / "analysis"
    analysis = _load_json(output_dir / "analysis.json")
    if analysis is None:
        raise FileNotFoundError(output_dir / "analysis.json")
    bar_diag = _load_json(output_dir / "bar_alignment_diagnostics.json")
    exp = _load_json(Path("experiments") / "results" / song_id / "results.json")

    beats = np.asarray(analysis.get("beats", {}).get("beat_times_s", []), dtype=float)
    phase = _bar_phase(bar_diag)
    tools = [
        _songviz_story_tool(analysis, beats, phase),
        _novelty_peak_tool(analysis, beats, phase),
        *_phrase_spec_tools(analysis, beats, phase),
        _phrase_grid_experiment_tool(exp, beats, phase),
        {
            "id": "allin1",
            "name": "allin1",
            "kind": "external learned structure model",
            "available": False,
            "status": "not run yet",
            "notes": "Top candidate from research report; run in Python 3.10/3.11 sidecar.",
            "segments": [],
            "boundaries": [],
        },
        {
            "id": "linkseg",
            "name": "LinkSeg",
            "kind": "external repeated-section model",
            "available": False,
            "status": "not run yet",
            "notes": "Candidate for repeated-section topology; likely sidecar env.",
            "segments": [],
            "boundaries": [],
        },
        {
            "id": "songformer",
            "name": "SongFormer",
            "kind": "external functional section model",
            "available": False,
            "status": "not run yet",
            "notes": "Candidate for modern section labels; likely sidecar env.",
            "segments": [],
            "boundaries": [],
        },
    ]
    return {
        "schema_version": 1,
        "song": song_path.name,
        "reference": "analysis.beats.beat_times_s",
        "bar_phase_source": "bar_alignment_diagnostics.external_consensus_phase",
        "bar_phase": phase,
        "tools": tools,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("song", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    payload = build(args.song, args.out_dir)
    out_dir = args.out_dir or Path("outputs") / _song_id(args.song) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "structure_tool_diagnostics.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
