from __future__ import annotations

from pathlib import Path
from typing import Any

# ── Section color palette (dark-theme friendly) ───────────────────────────────
_SECTION_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]

_DARK_BG = "#111111"
_GRID_COLOR = "#333333"


def _section_color(label: str) -> str:
    idx = 0
    for ch in label:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return _SECTION_COLORS[(idx - 1) % len(_SECTION_COLORS)]


def generate_overview(analysis: dict[str, Any], out_dir: Path) -> Path:
    """
    Generate a 2-panel dark-theme PNG: energy+novelty+repetition lines and
    section timeline bar.  Returns the path to the written PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as _np

    meta = analysis.get("meta", {})
    beats = analysis.get("beats", {})
    envelopes = analysis.get("envelopes", {})
    story = analysis.get("story", {})

    times = envelopes.get("times_s", [])
    loudness = envelopes.get("loudness", [])
    novelties = story.get("novelties", {})
    nov_times = novelties.get("times_s", times)
    nov_short = novelties.get("novelty_short", [])
    nov_medium = novelties.get("novelty_medium", [])
    nov_long = novelties.get("novelty_long", [])
    rep = novelties.get("repetition", [])
    section_diff = novelties.get("section_diff", [])
    sections = story.get("sections", [])
    ev = story.get("events", {})
    drop_times = ev.get("drop_times_s", [])
    ev_silences = ev.get("silences", [])
    ev_transitions = ev.get("stem_transitions", [])
    ev_candidates = ev.get("boundary_candidates", [])
    beat_times = beats.get("beat_times_s", [])
    tempo = beats.get("tempo_bpm", 0.0)
    duration = meta.get("duration_s", max(times) if times else 0)
    n_sections = len(sections)
    n_drops = len(drop_times)

    fig, (ax_lines, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 7), dpi=120, facecolor=_DARK_BG,
        gridspec_kw={"height_ratios": [5, 0.8]},
    )
    for ax in (ax_lines, ax_bot):
        ax.set_facecolor(_DARK_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID_COLOR)

    # ── Lines panel: Energy + Novelty curves + Repetition ────────────────────
    ax = ax_lines
    for sec in sections:
        ax.axvspan(sec["start_s"], sec["end_s"],
                   color=_section_color(sec["label"]), alpha=0.12)
    for bt in beat_times:
        ax.axvline(bt, color="#555555", lw=0.4, alpha=0.5)
    for dt in drop_times:
        ax.axvline(dt, color="#ff4444", lw=1.2, linestyle="--", alpha=0.85)

    if times and loudness:
        _loud = _np.array(loudness)
        _k = _np.ones(22) / 22.0
        ax.plot(times, _np.convolve(_loud, _k, mode="same"),
                color="#2ca4a4", lw=1.2, label="Energy")
    if nov_times and rep:
        ax.plot(nov_times, rep, color="#888888", lw=1.0, alpha=0.6, label="Repetition")
    if nov_times and nov_short:
        ax.plot(nov_times, nov_short, color="#e35cc8", lw=1.0, alpha=0.75, label="Novelty 4b")
    if nov_times and nov_medium:
        ax.plot(nov_times, nov_medium, color="#f09840", lw=1.1, alpha=0.85, label="Novelty 16b")
    if nov_times and nov_long:
        ax.plot(nov_times, nov_long, color="#f0d860", lw=1.3, alpha=1.0, label="Novelty 32b")
    if nov_times and section_diff:
        ax.plot(nov_times, section_diff, color="#ffffff", lw=1.8, alpha=0.9,
                drawstyle="steps-post", label="Section diff")

    ax.set_xlim(0, duration)
    ax.set_ylim(-0.05, 1.10)
    ax.set_ylabel("Amplitude (norm.)", color="#cccccc")
    ax.yaxis.label.set_color("#cccccc")
    ax.tick_params(colors="#888888", labelbottom=False)
    ax.grid(True, color=_GRID_COLOR, lw=0.5, alpha=0.6)
    ax.legend(loc="upper right", facecolor="#222222", edgecolor="#555555",
              labelcolor="#cccccc", fontsize=9)
    n_ev = len(ev_silences) + len(ev_transitions)
    ax.set_title(
        f"{tempo:.1f} BPM  ·  {duration:.1f} s  ·  {n_sections} sections"
        f"  ·  {n_drops} drops  ·  {n_ev} events",
        color="#dddddd", fontsize=11, pad=6,
    )

    # ── Section timeline bar ──────────────────────────────────────────────────
    _STEM_CLR = {"drums": "#e35cc8", "bass": "#f09840",
                 "vocals": "#f0d860", "other": "#8be0c4"}

    ax = ax_bot
    for sec in sections:
        lbl = sec["label"]
        s0, s1 = sec["start_s"], sec["end_s"]
        col = _section_color(lbl)
        ax.barh(0, s1 - s0, left=s0, height=0.6, color=col, alpha=0.85,
                edgecolor="#111111", linewidth=0.5)
        ax.text((s0 + s1) / 2, 0, lbl, ha="center", va="center",
                color="#eeeeee", fontsize=8, fontweight="bold")
        for sub in sec.get("subsections", [])[1:]:
            sub_t = sub.get("start_s", 0.0)
            if abs(sub_t - s0) > 0.5:
                ax.axvline(sub_t, color="#eeeeee", linewidth=0.7,
                           linestyle=":", alpha=0.6, ymin=0.2, ymax=0.8)

    # Silence overlays
    for s in ev_silences:
        ax.axvspan(s["start_s"], s["end_s"], color="#222222", alpha=0.55,
                   ymin=0.05, ymax=0.95, zorder=4)

    # Stem entry (▲ below bar) / exit (▼ above bar) triangles
    for tr in ev_transitions:
        y_pos = -0.42 if tr["kind"] == "enter" else 0.42
        marker = "^" if tr["kind"] == "enter" else "v"
        ax.scatter(tr["time_s"], y_pos, marker=marker, s=22,
                   color=_STEM_CLR.get(tr["stem"], "#cccccc"),
                   edgecolor="#111111", linewidth=0.4, zorder=5)

    # Discarded boundary candidates — thin dashed white ticks
    for c in ev_candidates:
        ax.axvline(c["time_s"], color="#cccccc", lw=0.7,
                   linestyle=(0, (2, 2)), ymin=0.1, ymax=0.9,
                   alpha=0.55, zorder=3)

    ax.set_xlim(0, duration)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)", color="#888888")
    ax.tick_params(axis="x", colors="#888888")
    ax.set_title("Sections  (▲ stem-enter  ▼ stem-exit)", color="#888888", fontsize=9, pad=3)

    plt.tight_layout(pad=0.8)
    out_path = out_dir / "overview.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, facecolor=_DARK_BG)

    import json as _json
    fig.canvas.draw()
    fig_h = fig.get_figheight() * fig.dpi
    x0_px, y1_disp = ax_lines.transData.transform((0.0, 1.0))
    x1_px, y0_disp = ax_lines.transData.transform((duration, 0.0))
    (out_dir / "overview_axes.json").write_text(_json.dumps({
        "plot_x0_px": int(round(x0_px)),
        "plot_x1_px": int(round(x1_px)),
        "plot_y0_px": int(round(fig_h - y1_disp)),
        "plot_y1_px": int(round(fig_h - y0_disp)),
    }))

    plt.close(fig)
    return out_path


def generate_stem_novelty_overviews(analysis: dict[str, Any], out_dir: Path) -> list[Path]:
    """
    Generate one overview PNG per stem showing energy + 3 novelty curves + section bar.
    Returns list of paths written (empty if no stem_novelties in analysis).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as _np

    stem_novelties = analysis.get("story", {}).get("stem_novelties", {})
    if not stem_novelties:
        return []

    story = analysis.get("story", {})
    meta = analysis.get("meta", {})
    envelopes = analysis.get("envelopes", {})
    times = envelopes.get("times_s", [])
    sections = story.get("sections", [])
    duration = meta.get("duration_s", max(times) if times else 0)
    nov_times = story.get("novelties", {}).get("times_s", times)
    _glob_ev = story.get("events", {})
    _glob_silences = _glob_ev.get("silences", [])
    _glob_transitions = _glob_ev.get("stem_transitions", [])
    _STEM_CLR2 = {"drums": "#e35cc8", "bass": "#f09840",
                  "vocals": "#f0d860", "other": "#8be0c4"}

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    _STEM_METHODS = [
        ("cqt", "CQT", "#5b9cf6"),
        ("phrase", "Phrase", "#9b6eea"),
        ("phrase_spec", "Phrase Spec", "#e0a800"),
        ("phrase_env", "Phrase Env", "#75b843"),
        ("phrase_amp", "Phrase Amp", "#0ea5e9"),
        ("onset", "Onset", "#ff9966"),
    ]

    for stem_name, metrics in stem_novelties.items():
        energy = metrics.get("energy", [])

        fig, (ax_lines, ax_bot) = plt.subplots(
            2, 1, figsize=(14, 6), dpi=120, facecolor=_DARK_BG,
            gridspec_kw={"height_ratios": [5, 0.8]},
        )
        for ax in (ax_lines, ax_bot):
            ax.set_facecolor(_DARK_BG)
            for spine in ax.spines.values():
                spine.set_edgecolor(_GRID_COLOR)

        # Lines panel: energy + CQT/Phrase/Onset similarity and novelty curves
        for sec in sections:
            ax_lines.axvspan(sec["start_s"], sec["end_s"],
                             color=_section_color(sec["label"]), alpha=0.12)
        if energy:
            e = _np.array(energy)
            k = _np.ones(22) / 22.0
            ax_lines.plot(nov_times[: len(e)], _np.convolve(e, k, mode="same"),
                          color="#2ca4a4", lw=1.2, label="Energy")

        for prefix, label, col in _STEM_METHODS:
            sim = metrics.get(f"{prefix}_sim_16", [])
            nov = metrics.get(f"{prefix}_nov_16", [])
            t_sim = nov_times[: len(sim)]
            t_nov = nov_times[: len(nov)]
            if sim:
                ax_lines.plot(t_sim, sim, color=col, lw=1.2, alpha=0.9,
                              label=f"{label} similarity 16b")
            if nov:
                ax_lines.plot(t_nov, nov, color=col, lw=1.2, alpha=0.9,
                              linestyle="--", label=f"{label} novelty 16b")

        ax_lines.set_xlim(0, duration)
        ax_lines.set_ylim(-0.02, 1.05)
        ax_lines.set_ylabel("Similarity / Novelty", color="#cccccc")
        ax_lines.yaxis.label.set_color("#cccccc")
        ax_lines.tick_params(colors="#888888", labelbottom=False)
        ax_lines.grid(True, color=_GRID_COLOR, lw=0.5, alpha=0.6)
        ax_lines.legend(loc="upper right", facecolor="#222222", edgecolor="#555555",
                        labelcolor="#cccccc", fontsize=8, ncol=2)
        ax_lines.set_title(
            f"Stem: {stem_name}  ·  {duration:.1f}s  ·  solid=similarity  dashed=novelty",
            color="#dddddd", fontsize=11, pad=6,
        )

        # Section bar
        for sec in sections:
            lbl, s0, s1 = sec["label"], sec["start_s"], sec["end_s"]
            ax_bot.barh(0, s1 - s0, left=s0, height=0.6,
                        color=_section_color(lbl), alpha=0.85,
                        edgecolor="#111111", linewidth=0.5)
            ax_bot.text((s0 + s1) / 2, 0, lbl, ha="center", va="center",
                        color="#eeeeee", fontsize=8, fontweight="bold")

        # Silence overlays
        for s in _glob_silences:
            ax_bot.axvspan(s["start_s"], s["end_s"], color="#222222", alpha=0.55,
                           ymin=0.05, ymax=0.95, zorder=4)

        # Own-stem transitions only
        own_color = _STEM_CLR2.get(stem_name, "#cccccc")
        for tr in _glob_transitions:
            if tr["stem"] != stem_name:
                continue
            y_pos = -0.42 if tr["kind"] == "enter" else 0.42
            marker = "^" if tr["kind"] == "enter" else "v"
            ax_bot.scatter(tr["time_s"], y_pos, marker=marker, s=22,
                           color=own_color, edgecolor="#111111",
                           linewidth=0.4, zorder=5)

        ax_bot.set_xlim(0, duration)
        ax_bot.set_ylim(-0.5, 0.5)
        ax_bot.set_yticks([])
        ax_bot.set_xlabel("Time (s)", color="#888888")
        ax_bot.tick_params(axis="x", colors="#888888")
        ax_bot.set_title("Sections  (▲ enter  ▼ exit)", color="#888888", fontsize=9, pad=3)

        plt.tight_layout(pad=0.8)
        out_path = out_dir / f"overview_{stem_name}.png"
        fig.savefig(out_path, dpi=120, facecolor=_DARK_BG)
        plt.close(fig)
        written.append(out_path)

    return written


def generate_stems_overview(stems_dir: Path, story: dict[str, Any], out_dir: Path) -> Path | None:
    """
    Generate a 2-panel stems overview PNG (per-stem envelopes + section heatmap).
    Returns None if stems WAVs don't exist.
    """
    stem_names = ["drums", "bass", "vocals", "other"]
    stem_paths = {n: stems_dir / f"{n}.wav" for n in stem_names}
    existing = {n: p for n, p in stem_paths.items() if p.exists()}
    if not existing:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa

    sections = story.get("sections", [])
    stem_colors = {
        "drums": "#e15759",
        "bass": "#4e79a7",
        "vocals": "#59a14f",
        "other": "#f28e2b",
    }

    # Compute RMS envelopes for each stem
    hop_length = 512
    sr_target = 22050
    envelopes: dict[str, tuple[list[float], list[float]]] = {}
    for name in stem_names:
        if name not in existing:
            continue
        y, sr = librosa.load(existing[name], sr=sr_target, mono=True)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        n = len(rms)
        times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop_length).tolist()
        rms = np.asarray(rms, dtype=np.float32)
        mx = float(rms.max())
        if mx > 0:
            rms = rms / mx
        envelopes[name] = (times, rms.tolist())

    # Compute per-section mean RMS for heatmap
    n_sections = len(sections)
    heatmap = np.zeros((len(stem_names), max(1, n_sections)), dtype=np.float32)
    for si, name in enumerate(stem_names):
        if name not in envelopes:
            continue
        t_arr, v_arr = envelopes[name]
        t_np = np.array(t_arr, dtype=np.float32)
        v_np = np.array(v_arr, dtype=np.float32)
        for ci, sec in enumerate(sections):
            mask = (t_np >= sec["start_s"]) & (t_np < sec["end_s"])
            if mask.any():
                heatmap[si, ci] = float(v_np[mask].mean())

    if sections:
        duration = sections[-1]["end_s"]
    elif envelopes:
        duration = max(max(t) for t, _ in envelopes.values())
    else:
        duration = 1.0

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 7), facecolor=_DARK_BG,
        gridspec_kw={"height_ratios": [3, 2]},
    )
    for ax in (ax_top, ax_bot):
        ax.set_facecolor(_DARK_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID_COLOR)

    # Top: per-stem envelopes with section shading
    ax = ax_top
    for sec in sections:
        ax.axvspan(sec["start_s"], sec["end_s"],
                   color=_section_color(sec["label"]), alpha=0.10)
    for name in stem_names:
        if name not in envelopes:
            continue
        t_arr, v_arr = envelopes[name]
        ax.plot(t_arr, v_arr, color=stem_colors.get(name, "#ffffff"),
                lw=1.0, label=name, alpha=0.9)

    ax.set_xlim(0, duration)
    ax.set_ylim(-0.05, 1.10)
    ax.set_ylabel("RMS (norm.)", color="#cccccc")
    ax.yaxis.label.set_color("#cccccc")
    ax.tick_params(colors="#888888")
    ax.grid(True, color=_GRID_COLOR, lw=0.5, alpha=0.6)
    ax.legend(loc="upper right", facecolor="#222222", edgecolor="#555555",
              labelcolor="#cccccc", fontsize=9)
    ax.set_title("Stem RMS envelopes (normalized)", color="#dddddd", fontsize=11, pad=6)

    # Bottom: heatmap
    ax = ax_bot
    im = ax.imshow(
        heatmap, aspect="auto", cmap="inferno", vmin=0, vmax=1,
        extent=[0, n_sections, len(stem_names) - 0.5, -0.5],
        interpolation="nearest",
    )
    ax.set_yticks(range(len(stem_names)))
    ax.set_yticklabels(stem_names, color="#cccccc", fontsize=9)
    ax.set_xticks(range(n_sections))
    sec_labels = [s["label"] for s in sections]
    ax.set_xticklabels(sec_labels, color="#cccccc", fontsize=8)
    ax.set_xlabel("Section", color="#888888")
    ax.tick_params(colors="#888888")
    ax.set_title("Mean RMS per stem × section", color="#888888", fontsize=9, pad=3)
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.01)
    cbar.set_label("Mean RMS", color="#888888")
    cbar.ax.yaxis.set_tick_params(color="#888888")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#888888")

    plt.tight_layout(pad=0.8)
    out_path = out_dir / "stems_overview.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, facecolor=_DARK_BG)
    plt.close(fig)
    return out_path


def generate_analysis_readme(
    analysis: dict[str, Any],
    out_dir: Path,
    *,
    has_stems: bool,
    has_lyrics: bool,
    has_overview_png: bool,
    has_stems_png: bool,
) -> Path:
    """
    Write analysis/README.md — no matplotlib needed.
    """
    meta = analysis.get("meta", {})
    beats = analysis.get("beats", {})
    story = analysis.get("story", {})
    sections = story.get("sections", [])
    drop_times = story.get("events", {}).get("drop_times_s", [])
    beat_times = beats.get("beat_times_s", [])
    song_id = meta.get("song_id", "unknown")
    duration = meta.get("duration_s", 0.0)
    tempo = beats.get("tempo_bpm", 0.0)
    n_beats = len(beat_times)
    n_sections = len(sections)
    n_drops = len(drop_times)

    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w(f"# Analysis: {song_id}")
    w()
    w("## Metadata")
    w()
    w("| Field | Value |")
    w("| --- | --- |")
    w(f"| song_id | `{song_id}` |")
    w(f"| duration | {duration:.2f} s |")
    w(f"| tempo | {tempo:.1f} BPM |")
    w(f"| beat count | {n_beats} |")
    w(f"| section count | {n_sections} |")
    w(f"| drop candidates | {n_drops} |")
    w()

    # File index
    files = [
        ("analysis.json", "Full analysis output (envelopes, beats, story)"),
        ("story.json", "Story sub-object: sections, tension curve, events"),
        ("README.md", "This file — human-readable summary"),
    ]
    if has_overview_png:
        files.append(("overview.png", "Signal overview + section timeline (dark theme)"))
    if has_stems_png:
        files.append(("stems_overview.png", "Stem RMS envelopes + per-section heatmap"))

    w("## Files in this folder")
    w()
    w("| File | Description |")
    w("| --- | --- |")
    for fname, desc in files:
        w(f"| `{fname}` | {desc} |")
    w()

    # Sections table
    w("## Sections")
    w()
    w("| Label | Start (s) | End (s) | Duration (s) |")
    w("| --- | --- | --- | --- |")
    for sec in sections:
        s0 = sec["start_s"]
        s1 = sec["end_s"]
        w(f"| {sec['label']} | {s0:.2f} | {s1:.2f} | {s1 - s0:.2f} |")
    w()

    # Drop candidates
    w("## Drop Candidates")
    w()
    if drop_times:
        for t in drop_times:
            w(f"- {t:.3f} s")
    else:
        w("_None detected._")
    w()

    # Signal glossary
    w("## Signal Glossary")
    w()
    w("| Signal | Source | Description |")
    w("| --- | --- | --- |")
    w("| **energy** | librosa RMS | Normalized loudness envelope [0,1]; tracks overall volume |")
    w("| **harmonic novelty** | chroma CQT frame-to-frame L2 flux | Chord/tonality change rate [0,1]; peaks at harmonic transitions |")
    w("| **timbre novelty** | MFCC flux (excl. coeff 0) | Instrumentation/texture change rate [0,1]; peaks when arrangement shifts |")
    w("| **repetition strength** | mean SSM similarity to non-adjacent beats | How repetitive this moment is vs the rest [0,1]; high = chorus/hook, low = bridge |")
    w("| **beats** | librosa beat_track | Beat timestamps in seconds (tempo-locked grid) |")
    w("| **sections** | MFCC agglomerative clustering | Coarse A/B/C… chapter labels (min 7 s each) |")
    w()

    # Stems
    w("## Stems")
    w()
    if has_stems:
        w("Demucs stem separation has been run. WAVs are in `../stems/`:")
        w("- `drums.wav` — rhythmic percussive component")
        w("- `bass.wav` — low-frequency melodic bass")
        w("- `vocals.wav` — lead vocal")
        w("- `other.wav` — everything else (guitars, keys, etc.)")
    else:
        w("_Stems not generated. Run `songviz stems <audio>` to separate._")
    w()

    # Lyrics
    w("## Lyrics")
    w()
    if has_lyrics:
        w("Lyrics alignment has been run. See `../lyrics/alignment.json` for word timestamps.")
        w("Generated with `songviz lyrics <audio>` (Whisper word-level alignment).")
    else:
        w("_Lyrics not generated. Run `songviz lyrics <audio>` (requires `pip install -e '.[lyrics]'`)._")
    w()

    # Re-generate
    w("## How to Re-generate")
    w()
    w("```bash")
    w("# Re-run analysis (rewrites this folder)")
    w("python3 -m songviz analyze <audio_path>")
    w()
    w("# Full pipeline with stems layout")
    w("python3 -m songviz render <audio_path> --layout stems4")
    w()
    w("# Lyrics alignment")
    w("python3 -m songviz lyrics <audio_path>")
    w("```")

    out_path = out_dir / "README.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def generate_lyrics_diagnostic(
    alignment: dict[str, Any],
    audio_path: Path,
    out_dir: Path,
    *,
    chunk_dur: float = 18.0,
) -> dict[str, Any]:
    """Generate a lyrics alignment diagnostic: plot + numerical quality stats.

    Overlays word boundaries on the vocal RMS envelope so timing issues are
    visually obvious.  Also computes per-word onset metrics.

    Returns a dict with:
        png_path: Path to the saved diagnostic PNG (or None)
        total_words, well_aligned, in_silence, significantly_early: int counts
        details: list of dicts for words that start in silence
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import librosa  # type: ignore
    import numpy as np

    segments = alignment.get("segments", [])
    if not segments:
        return {"png_path": None, "total_words": 0, "well_aligned": 0,
                "in_silence": 0, "significantly_early": 0, "details": []}

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    hop = int(0.01 * sr)  # 10ms frames
    frame_len = hop * 4
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop, center=True)[0]
    times_rms = np.arange(len(rms)) * (hop / sr)
    frame_hz = 100

    def _rms_at(t: float) -> float:
        f = max(0, min(len(rms) - 1, int(round(t * frame_hz))))
        return float(rms[f])

    def _rms_max_in(t1: float, t2: float) -> float:
        f1 = max(0, int(round(t1 * frame_hz)))
        f2 = min(len(rms), int(round(t2 * frame_hz)))
        return float(rms[f1:f2].max()) if f2 > f1 else 0.0

    # ── Numerical audit ──────────────────────────────────────────────────
    total_words = 0
    in_silence = 0
    sig_early = 0
    details: list[dict[str, Any]] = []

    for si, seg in enumerate(segments):
        for w in seg.get("words", []):
            ws, we = float(w["start_s"]), float(w["end_s"])
            dur = we - ws
            if dur < 0.04:
                continue
            total_words += 1
            start_rms = _rms_at(ws)
            peak_rms = _rms_max_in(ws, we)
            if peak_rms < 1e-6:
                continue
            ratio = start_rms / peak_rms
            if ratio >= 0.10:
                continue  # well-aligned
            in_silence += 1
            # Find actual voice onset within word span
            f_start = max(0, int(round(ws * frame_hz)))
            f_end = min(len(rms), int(round(we * frame_hz)))
            threshold = peak_rms * 0.15
            delay = 0.0
            for f in range(f_start, f_end):
                if float(rms[f]) >= threshold:
                    delay = f / frame_hz - ws
                    break
            if delay > 0.10:
                sig_early += 1
            details.append({
                "seg": si, "word": w["word"], "start_s": ws, "end_s": we,
                "start_rms": round(start_rms, 6), "peak_rms": round(peak_rms, 6),
                "ratio": round(ratio, 4), "delay_to_onset": round(delay, 3),
            })

    well_aligned = total_words - in_silence

    # ── Plot ──────────────────────────────────────────────────────────────
    lyric_chunks: set[int] = set()
    for seg in segments:
        for w in seg.get("words", []):
            lyric_chunks.add(int(float(w["start_s"]) // chunk_dur))
            lyric_chunks.add(int(float(w["end_s"]) // chunk_dur))
    lyric_chunks_sorted = sorted(lyric_chunks)

    n_panels = len(lyric_chunks_sorted)
    if n_panels == 0:
        return {"png_path": None, "total_words": total_words,
                "well_aligned": well_aligned, "in_silence": in_silence,
                "significantly_early": sig_early, "details": details}

    fig, axes = plt.subplots(n_panels, 1, figsize=(22, 2.8 * n_panels), squeeze=False)

    for ax_i, chunk_i in enumerate(lyric_chunks_sorted):
        ax = axes[ax_i, 0]
        t_start = chunk_i * chunk_dur
        t_end = (chunk_i + 1) * chunk_dur
        mask = (times_rms >= t_start) & (times_rms < t_end)
        if not mask.any():
            continue
        ax.fill_between(times_rms[mask], rms[mask], alpha=0.35, color="steelblue")
        ax.set_xlim(t_start, t_end)
        local_max = float(rms[mask].max())
        ax.set_ylim(0, local_max * 1.25)

        for seg in segments:
            for w in seg.get("words", []):
                w_s, w_e = float(w["start_s"]), float(w["end_s"])
                if w_e < t_start or w_s > t_end:
                    continue
                ax.axvspan(w_s, w_e, alpha=0.13, color="orange")
                ax.axvline(w_s, color="red", lw=0.7, alpha=0.6)
                ax.axvline(w_e, color="green", lw=0.5, alpha=0.4, ls="--")
                mid = (w_s + w_e) / 2
                ax.text(mid, local_max * 1.08, w["word"], ha="center", va="bottom",
                        fontsize=7, rotation=50, color="darkred", fontweight="bold")

        ax.set_title(f"{t_start:.0f}s – {t_end:.0f}s", fontsize=10, loc="left")
        ax.set_ylabel("Vocal RMS")

    axes[-1, 0].set_xlabel("Time (s)")
    tool = alignment.get("metadata", {}).get("alignment_tool", "?")
    fig.suptitle(
        f"Lyrics Diagnostic — {well_aligned}/{total_words} well-aligned, "
        f"{in_silence} in-silence, {sig_early} >0.1s early  [{tool}]",
        fontsize=12, y=1.002,
    )
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "alignment_diagnostic.png"
    fig.savefig(str(png_path), dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {
        "png_path": png_path,
        "total_words": total_words,
        "well_aligned": well_aligned,
        "in_silence": in_silence,
        "significantly_early": sig_early,
        "details": details,
    }


def generate_all(
    analysis: dict[str, Any],
    out_dir: Path,
    *,
    stems_dir: Path | None = None,
    has_lyrics: bool = False,
) -> dict[str, Path | None]:
    """
    Orchestrate generation of all analysis artifacts.
    Silently skips PNGs if matplotlib is not installed.
    Always writes README.md.
    """
    analysis_dir = out_dir / "analysis"
    overview_path: Path | None = None
    stems_path: Path | None = None

    try:
        overview_path = generate_overview(analysis, analysis_dir)
    except ImportError:
        pass  # matplotlib not installed

    if stems_dir is not None and stems_dir.is_dir():
        story = analysis.get("story", {})
        try:
            stems_path = generate_stems_overview(stems_dir, story, analysis_dir)
        except ImportError:
            pass
        try:
            generate_stem_novelty_overviews(analysis, analysis_dir)
        except ImportError:
            pass

    has_stems = stems_dir is not None and stems_dir.is_dir() and any(
        (stems_dir / f"{n}.wav").exists() for n in ("drums", "bass", "vocals", "other")
    )

    readme_path = generate_analysis_readme(
        analysis,
        analysis_dir,
        has_stems=has_stems,
        has_lyrics=has_lyrics,
        has_overview_png=overview_path is not None,
        has_stems_png=stems_path is not None,
    )

    return {
        "readme": readme_path,
        "overview": overview_path,
        "stems_overview": stems_path,
    }
