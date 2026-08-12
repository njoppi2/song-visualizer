#!/usr/bin/env python3
"""
Bar-phase and phrase-start detection — experiment driver.

Usage:
    python experiments/align_structure.py --song songs/foo.flac [options]

Options:
    --song PATH           Audio file to analyse (required)
    --stems DIR           Folder with drums.wav / bass.wav / vocals.wav / other.wav
                          (defaults to outputs/<song_id>/stems/)
    --analysis JSON       Existing analysis.json to reuse beat times
                          (defaults to outputs/<song_id>/analysis/analysis.json)
    --out DIR             Output folder (default: experiments/results/<song_id>/)
    --phrase-lengths N    Comma-separated bar counts to test (default: 8,16,32)
    --beats-per-bar N     Time signature numerator (default: 4)
    --methods LIST        Comma-separated subset of: m1,m2,m3,m4,m5,m6 (default: all)
    --no-plots            Skip matplotlib plots
    --madmom              Enable madmom comparison (m6); skipped if not installed
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

# ── project root on path ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from songviz.bar_phase import (
    beats_to_bar_times,
    compute_beat_features,
    hybrid_score,
    madmom_downbeat_compare,
    multi_feature_metrical_score,
    onset_strength_phase_vote,
    phrase_grid_boundary_contrast,
    similarity_phrase_alignment,
)


# ---------------------------------------------------------------------------
# Audio / stem loading
# ---------------------------------------------------------------------------

def _load_mono(path: Path, sr_target: int = 22050) -> tuple[np.ndarray, int]:
    """Load any audio file as mono float32 at sr_target."""
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != sr_target:
        try:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
        except Exception:
            warnings.warn(f"Could not resample {path.name} from {sr} → {sr_target} Hz; using as-is")
            return y, sr
    return y, sr_target


def _song_id(song_path: Path) -> str:
    return song_path.stem


def load_stems(stems_dir: Path) -> dict[str, Path]:
    """Return dict of available stem paths keyed by role (drums/bass/vocals/other/mix)."""
    found: dict[str, Path] = {}
    candidates = {
        "drums": ["drums.wav", "drums.flac"],
        "bass": ["bass.wav", "bass.flac"],
        "vocals": ["vocals.wav", "vocals.flac"],
        "other": ["other.wav", "other.flac"],
        "mix": ["mix.wav", "mix.flac", "mixture.wav", "mixture.flac", "no_vocals.wav"],
    }
    for role, names in candidates.items():
        for name in names:
            p = stems_dir / name
            if p.exists():
                found[role] = p
                break
    return found


# ---------------------------------------------------------------------------
# Beat time loading
# ---------------------------------------------------------------------------

def get_beat_times(analysis_path: Path | None, y_mix: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Return beat times from analysis.json if available, else run librosa beat tracker."""
    if analysis_path is not None and analysis_path.exists():
        with open(analysis_path) as f:
            data = json.load(f)
        bt = data.get("beats", {}).get("beat_times_s")
        if bt and len(bt) > 8:
            print(f"  Using {len(bt)} beat times from {analysis_path.name}")
            return np.array(bt, dtype=np.float64)

    print("  Running librosa beat tracker…")
    import librosa
    _, beat_frames = librosa.beat.beat_track(y=y_mix, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    print(f"  Detected {len(beat_times)} beats")
    return beat_times


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_json(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  JSON → {path}")


def write_csvs(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bar phase scores across methods
    phase_rows = []
    for method_key in ("m1", "m2", "m6"):
        r = results.get(method_key)
        if not r or r.get("skipped"):
            continue
        scores = r.get("scores", {})
        norm = r.get("scores_normalised", scores)
        for p_str, score in scores.items():
            phase_rows.append({
                "method": r.get("method", method_key),
                "phase": p_str,
                "score": score,
                "score_normalised": norm.get(p_str, ""),
            })
    if phase_rows:
        path = out_dir / "bar_phase_scores.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["method", "phase", "score", "score_normalised"])
            w.writeheader()
            w.writerows(phase_rows)
        print(f"  CSV  → {path}")

    # Phrase grid candidates
    for method_key, cand_key in (("m3", "candidates"), ("m4", "candidates")):
        r = results.get(method_key)
        if not r:
            continue
        cands = r.get(cand_key, [])
        if not cands:
            continue
        path = out_dir / f"{method_key}_phrase_candidates.csv"
        fieldnames = list(cands[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(cands)
        print(f"  CSV  → {path}")

    # Hybrid candidates
    r5 = results.get("m5")
    if r5:
        cands = r5.get("candidates", [])
        if cands:
            path = out_dir / "hybrid_candidates.csv"
            flat_rows = []
            for c in cands:
                row = {k: v for k, v in c.items() if k != "component_scores"}
                row.update(c.get("component_scores", {}))
                flat_rows.append(row)
            fieldnames = list(flat_rows[0].keys())
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(flat_rows)
            print(f"  CSV  → {path}")


def write_plots(
    results: dict,
    beat_times_s: np.ndarray,
    features: dict[str, np.ndarray],
    sr: int,
    hop_length: int,
    out_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed; skipping plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Beat-level feature curves grouped by bar position ─────────────────
    bar_phase = results.get("recommended", {}).get("bar_phase", 0)
    beats_per_bar = results.get("config", {}).get("beats_per_bar", 4)

    n_beats = len(beat_times_s)
    positions = (np.arange(n_beats) - bar_phase) % beats_per_bar
    beat_frames = np.round(beat_times_s * sr / hop_length).astype(int)

    feat_cols = [k for k in ("onset_env", "kick_energy", "snare_energy", "bass_onset")
                 if k in features and len(features[k]) > 0]
    if feat_cols:
        fig, axes = plt.subplots(len(feat_cols), 1, figsize=(14, 3 * len(feat_cols)), sharex=True)
        if len(feat_cols) == 1:
            axes = [axes]
        colors = ["#e35cc8", "#f09840", "#f0d860", "#8be0c4"]
        for ax, key, col in zip(axes, feat_cols, colors):
            arr = np.asarray(features[key], dtype=np.float64)
            frame_times = np.arange(len(arr)) * hop_length / sr
            ax.plot(frame_times, arr, color=col, alpha=0.5, linewidth=0.6)
            # Overlay beat positions coloured by bar position
            pos_colors = ["#ff4444", "#4488ff", "#44cc44", "#ffaa00"]
            for pos in range(beats_per_bar):
                idx = np.where(positions == pos)[0]
                bt = beat_times_s[idx]
                bf = np.clip(beat_frames[idx], 0, len(arr) - 1)
                ax.scatter(bt, arr[bf], color=pos_colors[pos], s=12, zorder=3,
                           label=f"pos {pos}" if key == feat_cols[0] else "")
            ax.set_ylabel(key, fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].legend(loc="upper right", fontsize=7, ncol=beats_per_bar)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"Beat features by bar position (phase={bar_phase})", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "beat_features_by_position.png", dpi=120)
        plt.close(fig)

    # ── 2. Bar phase score comparison ─────────────────────────────────────────
    phase_data: dict[str, dict[str, float]] = {}
    for key in ("m1", "m2"):
        r = results.get(key)
        if r and not r.get("skipped"):
            phase_data[r.get("method", key)] = {
                p: float(v) for p, v in r.get("scores_normalised", {}).items()
            }
    if phase_data:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(beats_per_bar)
        width = 0.8 / max(len(phase_data), 1)
        for i, (label, scores) in enumerate(phase_data.items()):
            vals = [scores.get(str(p), 0.0) for p in range(beats_per_bar)]
            ax.bar(x + i * width, vals, width, label=label, alpha=0.8)
        ax.set_xticks(x + width * (len(phase_data) - 1) / 2)
        ax.set_xticklabels([f"Phase {p}" for p in range(beats_per_bar)])
        ax.set_ylabel("Normalised score")
        ax.set_title("Bar phase scores by method")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "bar_phase_scores.png", dpi=120)
        plt.close(fig)

    # ── 3. Boundary novelty curve ─────────────────────────────────────────────
    m3 = results.get("m3")
    if m3 and m3.get("n_bars", 0) > 0:
        from songviz.bar_phase import _build_bar_novelty, _beat_to_frame, _sample_at_beats
        beat_feats_at_beat: dict[str, np.ndarray] = {}
        bframes = np.clip(
            np.round(beat_times_s * sr / hop_length).astype(int), 0, None
        )
        for key, arr in features.items():
            if arr is None or len(arr) == 0:
                continue
            bf = np.clip(bframes, 0, len(arr) - 1)
            beat_feats_at_beat[key] = _sample_at_beats(np.asarray(arr, dtype=np.float64), bf)

        bar_novelty = _build_bar_novelty(beat_feats_at_beat, bar_phase, beats_per_bar)
        bar_times = beats_to_bar_times(beat_times_s, bar_phase, beats_per_bar)
        bar_t = bar_times[:len(bar_novelty)]

        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(np.arange(len(bar_novelty)), bar_novelty, color="#2ca4a4", linewidth=1.2)

        best_per = m3.get("best_per_length", {})
        colors_pl = {"8": "#e35cc8", "16": "#f09840", "32": "#f0d860"}
        n_bars = len(bar_novelty)
        for L_str, cand in best_per.items():
            L, o = int(L_str), cand["offset"]
            b_idx = np.arange(o, n_bars, L)
            for bi in b_idx:
                ax.axvline(bi, color=colors_pl.get(L_str, "#888"), alpha=0.5,
                           linewidth=1.0, linestyle="--",
                           label=f"{L}b offset={o}" if bi == b_idx[0] else "")
        ax.set_xlabel("Bar index")
        ax.set_ylabel("Novelty")
        ax.set_title("Bar novelty + best phrase grid boundaries")
        handles, labels = ax.get_legend_handles_labels()
        seen: set[str] = set()
        unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]  # type: ignore[func-returns-value]
        if unique:
            ax.legend(*zip(*unique), fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "boundary_novelty.png", dpi=120)
        plt.close(fig)

    # ── 4. Adjacent phrase similarity curves ──────────────────────────────────
    m4 = results.get("m4")
    if m4:
        best_cands = m4.get("candidates", [])[:3]
        if best_cands:
            fig, ax = plt.subplots(figsize=(12, 4))
            cols = ["#e35cc8", "#f09840", "#8be0c4"]
            for cand, col in zip(best_cands, cols):
                # Recompute adj sim curve for this candidate for plotting
                L, o = cand["phrase_length"], cand["offset"]
                label = f"L={L}b off={o} (stab={cand['stability_cleanliness']:.3f})"
                # Approximate curve from adj_mean — just plot the scalar as a horizontal reference
                ax.axhline(cand["adj_similarity_mean"], linestyle="--", color=col, alpha=0.7, label=label)
            ax.set_xlabel("Phrase index")
            ax.set_ylabel("Adjacent phrase similarity (mean)")
            ax.set_title("Phrase alignment — top 3 candidates (adj similarity)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / "phrase_adj_similarity.png", dpi=120)
            plt.close(fig)

    print(f"  Plots → {out_dir}/")


def write_markdown(results: dict, out_dir: Path, song_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = results.get("recommended", {})
    lines = [
        f"# Structure alignment — {song_name}",
        "",
        "## Recommended alignment",
        f"- **Bar phase**: {rec.get('bar_phase', '?')}  ",
        f"- **Phrase length**: {rec.get('phrase_length', '?')} bars  ",
        f"- **Phrase offset**: {rec.get('phrase_offset', '?')} bars  ",
        f"- **Hybrid confidence**: {rec.get('hybrid_confidence', 0.0):.4f}  ",
        "",
        "## Bar phase by method",
        "",
        "| Method | Phase | Confidence | Accepted |",
        "| ------ | ----- | ---------- | -------- |",
    ]
    for key, label in (("m1", "Onset vote (M1)"), ("m2", "Multi-feature (M2)"), ("m6", "madmom (M6)")):
        r = results.get(key)
        if not r:
            continue
        if r.get("skipped"):
            lines.append(f"| {label} | — | — | skipped |")
        else:
            lines.append(
                f"| {label} | {r.get('phase', '?')} "
                f"| {r.get('confidence', 0.0):.4f} "
                f"| {'✓' if r.get('accepted', True) else '✗'} |"
            )
    lines += [
        "",
        "## Best phrase grid by method",
        "",
        "| Method | Phrase length | Offset | Score |",
        "| ------ | ------------- | ------ | ----- |",
    ]
    m3 = results.get("m3", {})
    if m3:
        lines.append(
            f"| Boundary contrast (M3) | {m3.get('best_phrase_length', '?')} bars "
            f"| {m3.get('best_offset', '?')} | {m3.get('best_contrast', 0.0):.4f} |"
        )
    m4 = results.get("m4", {})
    if m4 and m4.get("candidates"):
        top = m4["candidates"][0]
        lines.append(
            f"| Similarity alignment (M4) | {top.get('phrase_length', '?')} bars "
            f"| {top.get('offset', '?')} | stab={top.get('stability_cleanliness', 0.0):.4f} |"
        )
    m5 = results.get("m5", {})
    if m5 and m5.get("best"):
        b = m5["best"]
        lines.append(
            f"| Hybrid (M5) | {b.get('phrase_length', '?')} bars "
            f"| {b.get('phrase_offset', '?')} | {b.get('total_score', 0.0):.4f} |"
        )

    # Agreement / disagreement
    phases = {}
    for key in ("m1", "m2", "m6"):
        r = results.get(key)
        if r and not r.get("skipped") and not r.get("error"):
            phases[r.get("method", key)] = r.get("phase", "?")
    unique_phases = set(phases.values())
    lines += ["", "## Agreement summary", ""]
    if len(unique_phases) == 1:
        lines.append(f"✓ All methods agree on bar phase **{next(iter(unique_phases))}**.")
    else:
        lines.append(f"⚠ Methods disagree on bar phase: {phases}")

    conf = results.get("m5", {}).get("confidence", 0.0)
    if conf < 0.05:
        lines.append("⚠ Hybrid confidence is **low** — manual inspection recommended.")
    else:
        lines.append(f"✓ Hybrid confidence: **{conf:.4f}**")

    lines += [
        "",
        "## Notes",
        f"- Beat count: {results.get('beat_count', '?')}  ",
        f"- Tempo: {results.get('tempo_bpm', '?'):.1f} BPM  " if results.get("tempo_bpm") else "",
        f"- Bar count (est.): {results.get('m3', {}).get('n_bars', '?')}  ",
        "",
        "_Generated by experiments/align_structure.py_",
    ]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines))
    print(f"  MD   → {path}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    song_path: Path,
    stems_dir: Path | None,
    analysis_path: Path | None,
    out_dir: Path,
    phrase_lengths: list[int],
    beats_per_bar: int,
    enabled_methods: set[str],
    make_plots: bool,
    run_madmom: bool,
) -> dict:
    import librosa

    sr = 22050
    hop_length = 512

    print(f"\n=== {song_path.name} ===")

    # Load mix
    print("Loading audio…")
    y_mix, _ = _load_mono(song_path, sr)

    # Load stems
    y_drums = y_bass = y_other = None
    if stems_dir is not None and stems_dir.exists():
        stem_paths = load_stems(stems_dir)
        print(f"  Found stems: {list(stem_paths.keys())}")
        if "drums" in stem_paths:
            y_drums, _ = _load_mono(stem_paths["drums"], sr)
        if "bass" in stem_paths:
            y_bass, _ = _load_mono(stem_paths["bass"], sr)
        if "other" in stem_paths:
            y_other, _ = _load_mono(stem_paths["other"], sr)
        if "mix" in stem_paths and y_mix is None:
            y_mix, _ = _load_mono(stem_paths["mix"], sr)
    else:
        print("  No stems dir; using mix only")

    # Beat times
    beat_times_s = get_beat_times(analysis_path, y_mix, sr, hop_length)
    tempo_bpm = 60.0 / float(np.median(np.diff(beat_times_s))) if len(beat_times_s) > 1 else 0.0

    # Features
    print("Computing features…")
    features = compute_beat_features(y_mix, y_drums, y_bass, y_other, sr, hop_length, beat_times_s)
    print(f"  Feature keys: {list(features.keys())}")

    results: dict = {
        "song": song_path.name,
        "beat_count": int(len(beat_times_s)),
        "tempo_bpm": float(tempo_bpm),
        "config": {
            "phrase_lengths": phrase_lengths,
            "beats_per_bar": beats_per_bar,
            "enabled_methods": sorted(enabled_methods),
        },
    }

    # M1 — Onset vote
    if "m1" in enabled_methods and "onset_env" in features:
        print("M1: onset-strength phase vote…")
        results["m1"] = onset_strength_phase_vote(
            beat_times_s, features["onset_env"], sr, hop_length,
            beats_per_bar=beats_per_bar,
        )

    # M2 — Multi-feature
    if "m2" in enabled_methods:
        print("M2: multi-feature metrical score…")
        results["m2"] = multi_feature_metrical_score(
            beat_times_s, features, sr, hop_length,
            beats_per_bar=beats_per_bar,
        )

    # Resolve bar phase from best available method
    bar_phase = 0
    for key in ("m2", "m1"):
        r = results.get(key)
        if r and not r.get("warning"):
            bar_phase = int(r.get("phase", 0))
            break

    # M3 — Phrase boundary contrast
    if "m3" in enabled_methods:
        print("M3: phrase-grid boundary contrast…")
        results["m3"] = phrase_grid_boundary_contrast(
            beat_times_s, features, sr, hop_length, bar_phase,
            phrase_lengths=phrase_lengths, beats_per_bar=beats_per_bar,
        )

    # M4 — Similarity alignment
    if "m4" in enabled_methods:
        print("M4: similarity phrase alignment…")
        results["m4"] = similarity_phrase_alignment(
            beat_times_s, features, sr, hop_length, bar_phase,
            phrase_lengths=phrase_lengths, beats_per_bar=beats_per_bar,
        )

    # M5 — Hybrid
    if "m5" in enabled_methods and "m2" in results and "m3" in results and "m4" in results:
        print("M5: hybrid score…")
        results["m5"] = hybrid_score(results["m2"], results["m3"], results["m4"],
                                     phrase_lengths=phrase_lengths)

    # M6 — madmom (optional)
    if "m6" in enabled_methods and run_madmom:
        print("M6: madmom downbeat comparison…")
        results["m6"] = madmom_downbeat_compare(
            beat_times_s, str(song_path), beats_per_bar=beats_per_bar
        )

    # Build recommendation
    m5 = results.get("m5", {})
    best = m5.get("best") or {}
    results["recommended"] = {
        "bar_phase": best.get("bar_phase", bar_phase),
        "phrase_length": best.get("phrase_length", phrase_lengths[1] if len(phrase_lengths) > 1 else phrase_lengths[0]),
        "phrase_offset": best.get("phrase_offset", 0),
        "hybrid_confidence": float(m5.get("confidence", 0.0)),
        "primary_method": "hybrid (M5)" if best else "multi-feature (M2)",
    }

    # Outputs
    print("Writing outputs…")
    write_json(results, out_dir)
    write_csvs(results, out_dir)
    if make_plots:
        write_plots(results, beat_times_s, features, sr, hop_length, out_dir)
    write_markdown(results, out_dir, song_path.stem)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bar-phase and phrase-start detection experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--song", required=True, type=Path, help="Audio file path")
    p.add_argument("--stems", type=Path, default=None, help="Stem folder")
    p.add_argument("--analysis", type=Path, default=None, help="Existing analysis.json")
    p.add_argument("--out", type=Path, default=None, help="Output folder")
    p.add_argument("--phrase-lengths", default="8,16,32",
                   help="Comma-separated bar counts (default: 8,16,32)")
    p.add_argument("--beats-per-bar", type=int, default=4)
    p.add_argument("--methods", default="m1,m2,m3,m4,m5,m6",
                   help="Methods to run (default: all)")
    p.add_argument("--no-plots", action="store_true", help="Skip plots")
    p.add_argument("--madmom", action="store_true", help="Enable madmom (M6)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    song_path = args.song.resolve()
    if not song_path.exists():
        sys.exit(f"ERROR: song not found: {song_path}")

    song_id = _song_id(song_path)
    root = Path(__file__).resolve().parent.parent

    # Default stems dir
    stems_dir = args.stems
    if stems_dir is None:
        guess = root / "outputs" / f"{song_id}" / "stems"
        if guess.exists():
            stems_dir = guess

    # Default analysis path
    analysis_path = args.analysis
    if analysis_path is None:
        for candidate in (f"{song_id}", f"{song_id}_"):
            guess = root / "outputs" / candidate / "analysis" / "analysis.json"
            if guess.exists():
                analysis_path = guess
                break

    # Output dir
    out_dir = args.out or (root / "experiments" / "results" / song_id)

    phrase_lengths = [int(x) for x in args.phrase_lengths.split(",")]
    enabled = set(args.methods.lower().split(","))

    run(
        song_path=song_path,
        stems_dir=stems_dir,
        analysis_path=analysis_path,
        out_dir=out_dir,
        phrase_lengths=phrase_lengths,
        beats_per_bar=args.beats_per_bar,
        enabled_methods=enabled,
        make_plots=not args.no_plots,
        run_madmom=args.madmom,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
