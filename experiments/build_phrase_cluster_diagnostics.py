#!/usr/bin/env python3
"""Build stem-level phrase clustering diagnostics.

The first target is the current best comparator in the dashboard: "Spec Old",
implemented as audibility-weighted log-CQT cosine similarity. This script cuts
each stem on the trusted beat grid, compares every phrase against every other
phrase, and stores the all-pairs similarity matrix for interactive clustering in
the dashboard.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


PHRASE_LENGTHS_BEATS = (8, 16, 32)
DEFAULT_SR = 22050
HOP_LENGTH = 512
AUDIBLE_FLOOR_DB = -36.0
COVERAGE_THRESHOLD = 0.05


def _song_id(song_path: Path) -> str:
    return song_path.stem


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else None


def _load_mono(path: Path, sr_target: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    import librosa

    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != sr_target:
        y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
        sr = sr_target
    return np.asarray(y, dtype=np.float32), sr


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


def _cosine_01(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    n = min(aa.size, bb.size)
    if n == 0:
        return 1.0
    aa = aa[:n]
    bb = bb[:n]
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom < 1e-12:
        return 1.0 if float(np.linalg.norm(aa - bb)) < 1e-12 else 0.0
    return float(np.clip(np.dot(aa, bb) / denom, 0.0, 1.0))


def _weighted_cosine_01(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    ww = np.asarray(weights, dtype=np.float64).ravel()
    n = min(aa.shape[-1], bb.shape[-1], ww.size)
    if n == 0:
        return 1.0
    ww = np.clip(ww[:n], 0.0, 1.0)
    if float(ww.sum()) < 1e-9:
        return 1.0
    return _cosine_01(aa[..., :n] * np.sqrt(ww), bb[..., :n] * np.sqrt(ww))


def _phrase_spans(beat_times: np.ndarray, phase: int, phrase_beats: int) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    start = int(phase) % max(1, phrase_beats)
    median_beat = float(np.median(np.diff(beat_times))) if beat_times.size > 1 else 0.5
    idx = 0
    for s in range(start, len(beat_times), phrase_beats):
        e = s + phrase_beats
        if e > len(beat_times):
            break
        end_s = float(beat_times[e]) if e < len(beat_times) else float(beat_times[-1] + median_beat)
        spans.append({
            "index": idx,
            "start_beat": int(s),
            "end_beat": int(e),
            "start_s": float(beat_times[s]),
            "end_s": end_s,
            "bar_index": int((s - int(phase)) // 4) if s >= int(phase) else -1,
        })
        idx += 1
    return spans


def _stem_phrase_data(stem_path: Path, beat_times: np.ndarray, phase: int) -> dict[str, Any]:
    import librosa

    y, sr = _load_mono(stem_path)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP_LENGTH, center=True)[0]
    q_mag = np.abs(librosa.cqt(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
        n_bins=84,
        bins_per_octave=12,
        fmin=librosa.note_to_hz("C1"),
    ))

    beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=HOP_LENGTH)
    beat_frames = np.unique(np.clip(beat_frames, 0, max(0, min(q_mag.shape[1], rms.size) - 1)))
    if beat_frames.size < 2:
        return {"phrase_lengths": {}}

    q_sync = librosa.util.sync(q_mag, beat_frames, aggregate=np.mean)
    rms_sync = librosa.util.sync(rms.reshape(1, -1), beat_frames, aggregate=np.mean)[0]
    n_beats = min(q_sync.shape[1], rms_sync.size, beat_times.size)
    q_sync = np.log1p(q_sync[:, :n_beats])
    rms_sync = np.asarray(rms_sync[:n_beats], dtype=np.float64)
    bt = beat_times[:n_beats]

    floor = float(rms_sync.max()) * (10.0 ** (AUDIBLE_FLOOR_DB / 20.0)) if rms_sync.size else 0.0
    scale = max(float(rms_sync.max()) - floor, 1e-12) if rms_sync.size else 1.0
    aud = np.clip((rms_sync - floor) / scale, 0.0, 1.0)

    out: dict[str, Any] = {"phrase_lengths": {}}
    for phrase_beats in PHRASE_LENGTHS_BEATS:
        spans = _phrase_spans(bt, phase, phrase_beats)
        n = len(spans)
        sim = np.eye(n, dtype=np.float64)
        for i in range(n):
            si = int(spans[i]["start_beat"])
            ei = int(spans[i]["end_beat"])
            qi = q_sync[:, si:ei]
            ai = aud[si:ei]
            for j in range(i + 1, n):
                sj = int(spans[j]["start_beat"])
                ej = int(spans[j]["end_beat"])
                qj = q_sync[:, sj:ej]
                aj = aud[sj:ej]
                weights = np.maximum(ai, aj)
                if float(weights.max()) < COVERAGE_THRESHOLD:
                    s = 1.0
                else:
                    s = _weighted_cosine_01(qi, qj, weights)
                sim[i, j] = sim[j, i] = s

        prev_sim = [None]
        for i in range(1, n):
            prev_sim.append(float(sim[i, i - 1]))

        out["phrase_lengths"][str(phrase_beats)] = {
            "phrase_beats": phrase_beats,
            "method": "phrase_spec_old_audibility_weighted_cqt_all_pairs",
            "phrases": [
                {
                    **span,
                    "similarity_to_previous": prev_sim[i],
                    "audibility": float(np.mean(aud[int(span["start_beat"]):int(span["end_beat"])])),
                }
                for i, span in enumerate(spans)
            ],
            "similarity_matrix": np.round(sim, 6).tolist(),
        }
    return out


def build(song_path: Path, output_dir: Path | None = None) -> dict[str, Any]:
    song_id = _song_id(song_path)
    output_dir = output_dir or Path("outputs") / song_id / "analysis"
    analysis = _load_json(output_dir / "analysis.json")
    if analysis is None:
        raise FileNotFoundError(output_dir / "analysis.json")
    beat_times = np.asarray(analysis.get("beats", {}).get("beat_times_s", []), dtype=np.float64)
    if beat_times.size < 2:
        raise ValueError("analysis.json has no usable beat grid")

    phase = _bar_phase(_load_json(output_dir / "bar_alignment_diagnostics.json"))
    stems_dir = output_dir.parent / "stems"
    stems: dict[str, Any] = {}
    for stem in ("other", "bass", "drums", "vocals"):
        path = stems_dir / f"{stem}.wav"
        if not path.exists():
            continue
        stems[stem] = _stem_phrase_data(path, beat_times, phase)

    return {
        "schema_version": 1,
        "song": song_path.name,
        "reference": "analysis.beats.beat_times_s",
        "bar_phase": int(phase),
        "beats_per_bar": 4,
        "default_phrase_beats": 16,
        "default_similarity_threshold": 0.84,
        "stems": stems,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("song", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    payload = build(args.song, args.out_dir)
    out_dir = args.out_dir or Path("outputs") / _song_id(args.song) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phrase_cluster_diagnostics.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
