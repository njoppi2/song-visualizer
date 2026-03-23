"""Shared render pipeline: analyze → (stems) → render → viz.

Both cli.py and ui.py delegate to run_render_pipeline so the stems analysis
loop and render dispatch are defined only once.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from .analyze import analyze_audio
from .features import (
    bass_note_events_basic_pitch,
    bass_pitch_hz,
    drums_band_energy_3,
    drums_band_energy_3_from_components,
    other_chroma_12,
    vocals_note_events_basic_pitch,
    vocals_pitch_hz,
)
from .paths import (
    analysis_path_for_output_dir,
    reduced_path_for_output_dir,
    story_path_for_output_dir,
    video_path_for_output_dir,
)
from .reduction import (
    _REDUCED_SCHEMA_VERSION,
    estimate_key_scale,
    extract_bass_notes,
    extract_drum_hits,
    extract_drum_hits_fallback,
    extract_vocal_notes,
)
from .render import RenderConfig, render_mp4, render_mp4_stems4
from .stems import ensure_demucs_stems, ensure_drumsep_components


def _write_analysis_artifacts(analysis: dict[str, Any], out_dir: Path) -> None:
    """Persist analysis.json and story.json to the output directory."""
    analysis_path = analysis_path_for_output_dir(out_dir)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    story_p = story_path_for_output_dir(out_dir)
    story_p.parent.mkdir(parents=True, exist_ok=True)
    story_p.write_text(json.dumps(analysis.get("story", {}), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_stem_analyses(
    audio_path: Path | str,
    out_dir: Path,
    analysis: dict[str, Any],
    *,
    stems_model: str,
    stems_device: str,
    stems_force: bool,
) -> dict[str, dict[str, Any]]:
    """Run Demucs stem separation and analyse each stem, returning stem_analyses dict."""
    stems = ensure_demucs_stems(
        audio_path,
        out_dir=out_dir,
        model=stems_model,
        device=stems_device,
        force=stems_force,
    )

    stem_analyses: dict[str, dict[str, Any]] = {}
    hop_length = 512
    frame_length = 2048

    # Key estimation is disabled — Krumhansl-Kessler profiles on Demucs stems
    # consistently estimate the wrong key, and scale snapping with the wrong key
    # degrades pitch-class accuracy.  Once a reliable key estimator is available,
    # re-enable this block and pass key_scale_pcs to extract_bass_notes().
    key_scale_pcs: list[int] | None = None

    for name, stem_path in stems.stems.items():
        y, sr = librosa.load(stem_path, sr=22050, mono=True)
        y = np.asarray(y, dtype=np.float32)
        a = analyze_audio(y, int(sr), hop_length=hop_length, frame_length=frame_length)
        # Share the mix story/sections so all quadrants follow the same narrative.
        a["story"] = analysis.get("story", {})

        feats: dict[str, Any] = {}
        if name == "drums":
            heuristic = drums_band_energy_3(y, int(sr), hop_length=hop_length, n_fft=frame_length)
            feats["drums_bands_3_heuristic"] = heuristic
            feats["drums_bands_3"] = heuristic

            drumsep = ensure_drumsep_components(stem_path, out_dir=out_dir)
            comp_audio: dict[str, np.ndarray] | None = None
            if drumsep is not None:
                comp_audio = {
                    comp: np.asarray(librosa.load(str(cp), sr=int(sr), mono=True)[0], dtype=np.float32)
                    for comp, cp in drumsep.components.items()
                }
                drumsep_energy = drums_band_energy_3_from_components(comp_audio, int(sr), hop_length=hop_length)
                if drumsep_energy.size > 0:
                    feats["drums_bands_3_drumsep"] = drumsep_energy
                    feats["drums_bands_3"] = drumsep_energy

            # ── Drum hit extraction (reduced representation) ──
            beat_times = analysis.get("beats", {}).get("beat_times_s")
            if drumsep is not None and comp_audio is not None:
                drum_hits = extract_drum_hits(
                    comp_audio, int(sr), hop_length=hop_length, beat_times_s=beat_times,
                )
            else:
                drum_hits = extract_drum_hits_fallback(
                    y, int(sr), hop_length=hop_length, n_fft=frame_length,
                    beat_times_s=beat_times,
                )
            feats["drum_hits"] = drum_hits

            # Write/update reduced.json (unified reduced representation)
            reduced_path = reduced_path_for_output_dir(out_dir)
            reduced_path.parent.mkdir(parents=True, exist_ok=True)
            reduced: dict[str, Any] = {}
            if reduced_path.exists():
                try:
                    reduced = json.loads(reduced_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            reduced["schema_version"] = _REDUCED_SCHEMA_VERSION
            reduced["drums"] = drum_hits
            reduced_path.write_text(
                json.dumps(reduced, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
        elif name == "bass":
            feats["pitch_hz"] = bass_pitch_hz(y, int(sr), hop_length=hop_length, frame_length=frame_length)
            try:
                feats["note_events"] = bass_note_events_basic_pitch(str(stem_path))
            except Exception:
                pass

            # ── Bass note extraction (reduced representation) ──
            beat_times = analysis.get("beats", {}).get("beat_times_s")
            bass_notes = extract_bass_notes(
                note_events=feats.get("note_events"),
                pitch_hz=feats.get("pitch_hz"),
                y=y, sr=int(sr), hop_length=hop_length,
                beat_times_s=beat_times,
                scale_pcs=key_scale_pcs,
            )
            feats["bass_notes"] = bass_notes

            # Write/update reduced.json
            reduced_path = reduced_path_for_output_dir(out_dir)
            reduced_path.parent.mkdir(parents=True, exist_ok=True)
            reduced: dict[str, Any] = {}
            if reduced_path.exists():
                try:
                    reduced = json.loads(reduced_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            reduced["schema_version"] = _REDUCED_SCHEMA_VERSION
            reduced["bass"] = bass_notes
            reduced_path.write_text(
                json.dumps(reduced, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
        elif name == "vocals":
            feats["pitch_hz"] = vocals_pitch_hz(y, int(sr), hop_length=hop_length, frame_length=frame_length)
            try:
                feats["note_events"] = vocals_note_events_basic_pitch(str(stem_path))
            except Exception:
                pass

            # ── Vocal note extraction (reduced representation) ──
            beat_times = analysis.get("beats", {}).get("beat_times_s")
            vocal_notes = extract_vocal_notes(
                note_events=feats.get("note_events"),
                pitch_hz=feats.get("pitch_hz"),
                y=y, sr=int(sr), hop_length=hop_length,
                beat_times_s=beat_times,
            )
            feats["vocal_notes"] = vocal_notes

            # Write/update reduced.json
            reduced_path = reduced_path_for_output_dir(out_dir)
            reduced_path.parent.mkdir(parents=True, exist_ok=True)
            reduced: dict[str, Any] = {}
            if reduced_path.exists():
                try:
                    reduced = json.loads(reduced_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            reduced["schema_version"] = _REDUCED_SCHEMA_VERSION
            reduced["vocals"] = vocal_notes
            reduced_path.write_text(
                json.dumps(reduced, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
        elif name == "other":
            feats["chroma_12"] = other_chroma_12(y, int(sr), hop_length=hop_length, n_fft=frame_length)
        if feats:
            a["features"] = feats

        if name != "drums":
            a["beats"]["beat_times_s"] = []
            a["beats"]["tempo_bpm"] = 0.0
        stem_analyses[name] = a

    return stem_analyses


def run_render_pipeline(
    audio_path: Path | str,
    out_dir: Path,
    analysis: dict[str, Any],
    layout: str,
    cfg: RenderConfig,
    *,
    stems_model: str = "htdemucs",
    stems_device: str = "auto",
    stems_force: bool = False,
    alignment: dict[str, Any] | None = None,
) -> Path:
    """Full render pipeline: analyze → (stems) → render → viz.

    Writes analysis.json, story.json, video.mp4 and static PNGs under out_dir.
    Returns the path of the rendered video.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_analysis_artifacts(analysis, out_dir)

    canonical_mp4 = video_path_for_output_dir(out_dir)

    if layout == "mix":
        render_mp4(analysis=analysis, audio_path=audio_path, out_path=canonical_mp4, cfg=cfg, alignment=alignment)
    elif layout == "stems4":
        stem_analyses = _build_stem_analyses(
            audio_path, out_dir, analysis,
            stems_model=stems_model,
            stems_device=stems_device,
            stems_force=stems_force,
        )
        render_mp4_stems4(
            stem_analyses=stem_analyses,
            duration_s=float(analysis["meta"]["duration_s"]),
            audio_path=audio_path,
            out_path=canonical_mp4,
            cfg=cfg,
            alignment=alignment,
        )
    else:
        raise AssertionError(f"Unknown layout: {layout!r}")

    from .viz import generate_all
    generate_all(
        analysis,
        out_dir,
        stems_dir=out_dir / "stems",
        has_lyrics=(out_dir / "lyrics" / "alignment.json").exists(),
    )

    return canonical_mp4
