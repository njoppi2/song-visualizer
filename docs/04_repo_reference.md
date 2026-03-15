# SongViz Repository Reference

This document is an execution-oriented map of the repository.
For current status/priorities, use `docs/03_working_state.md`.

## High-level goal
SongViz aims to understand and visualize what is happening in a song over time — section structure, tension arcs, repetition, arrangement changes, and how melody/rhythm/bass/harmony interact. The current main output is a music-reactive MP4 video, but the core of the project is the analysis pipeline that derives interpretable musical signals from audio. The toolchain is intentionally deterministic, modular (ingest → separate → analyze → reduce → render), and runnable via `python -m songviz render` or the provided `Makefile` shortcuts. See `docs/01_roadmap.md` for the phased plan.

## Read order (for new contributors and LLMs)
1. `README.md` (project direction + quickstart)
2. `docs/01_roadmap.md` (phased plan)
3. `docs/03_working_state.md` (what is implemented now)
4. `docs/04_repo_reference.md` (this file — repo map)
5. `docs/06_reduced_representation.md` (current phase design)
6. `docs/05_lyrics_playbook.md` (if working on lyrics)
7. `docs/research/lyrics_syncing_research.md` (background only)

## Repository layout
- `songviz/` — Python package and CLI. Key modules: `ingest` (path normalization, decoding), `analyze` (beat/loudness/onset), `story` (sections, tension, drop candidates), `render` (frame generation + ffmpeg), `stems` (Demucs wrappers), `lyrics` (Whisper alignment, `alignment.json` I/O), `viz` (analysis PNGs + README generation), `ui` (terminal picker), `tidy` (outputs cleanup), and `cli` (`songviz` command definitions).
- `songs/` — curated input directory (gitignored) where you drop FLAC/MP3/WAV files for analysis/rendering.
- `outputs/` — gitignored workspace for per-song artifacts; see "Outputs" below.
- `docs/` — design/roadmap/architecture/working-state + this reference + lyrics playbook.
- `tests/` — `pytest` cases that assert the analysis JSON shape/keys and `song_id` stability.
- `Makefile` — sets up `.songviz/venv` (with optional stems extras) and exposes `make ui`.

## Core workflow
1. **Ingest** `songs/<file>` by normalizing the path, hashing it into a stable `song_id`, and (if necessary) decoding to a working WAV at 22.05 kHz mono.
2. **Analyze** with `librosa`: tempo, beat times, RMS loudness envelope, and normalized onset strength (shared hop size), plus `story.compute_story()` which yields tension, sections, and drop events.
3. **Story** output merges envelopes + tension with narrative metadata: rough `A/B/C…` labels, `tension` curve (smoothed energy/brightness), and `events.drop_times_s`. These are written to `analysis/story.json` for human inspection.
4. **Render** either the default `mix` layout or a 2×2 `stems4` grid (requires Demucs) using `RenderConfig` (width, height, fps, seed, audio codec/bitrate). Frames are painted with layered visuals controlled by beats, loudness, and onset strength; ffmpeg then muxes the selected audio codec (default `mp3`) into `video.mp4`.
5. **UI / tidy / stems** wrappers: `songviz ui` surfaces an interactive picker of `songs/`, `songviz tidy` hides legacy files under `.songviz/`, and `songviz stems` precomputes Demucs outputs used by the stems layout.

## CLI entrypoints
| Command | Purpose | Key options |
| --- | --- | --- |
| `songviz analyze <audio>` | write `analysis/analysis.json` + `analysis/story.json` without rendering video | `--out` to duplicate the JSON elsewhere |
| `songviz render <audio>` | run analysis + render + mux audio into `outputs/<song_id>/video.mp4` | `--layout mix|stems4`, `--seed`, `--fps`, `--audio-codec aac|mp3`, `--audio-bitrate`, `--stems-model`, `--stems-device`, `--stems-force` |
| `songviz stems <audio>` | run Demucs (via `ensure_demucs_stems`) and dump WAV stems | `--model`, `--device`, `--force` |
| `songviz ui` | terminal interface to pick a song from `songs/` and render it with `make ui`-like defaults | same rendering args as `render`, plus `--songs-dir`, `--outputs-dir` |
| `songviz lyrics <audio>` | run lyrics alignment (LRCLIB + backend routing + auto calibration) and write `outputs/<song_id>/lyrics/alignment.json` | `--artist`, `--title` (override ID3 tags), `--language`, `--model` (tiny/base/small/medium/large), `--backend auto|whisper|stable_whisper|whisperx`, `--no-auto-calibrate`, `--force` |
| `songviz tidy` | move stray files/legacy folders under `outputs/` into hidden `.songviz/*` areas | `--outputs-dir`, `--dry-run` |

The package also exposes the console entry: `python -m songviz <subcommand>`. `pip install -e .` or `pip install -e '.[stems]'` registers the console script `songviz` for convenience.

## Inputs and outputs organization
- **Inputs**: drop songs under `songs/` (the interactive `ui` scans this directory); keep FLAC/MP3/WAV files here per the user-requested "song-per-folder" convention.
- **Per-song output folder**: `outputs/<song_id>/` (song names + hashes stay separated). Inside each folder:
  - `analysis/analysis.json` — meta (song_id, duration, sample rate, `analysis.version`), `beats` (tempo, beat times), `envelopes` (hop_s, times_s, loudness, onset_strength normalized to [0,1]), and whatever extra `features`/`stems` data current code emits.
  - `analysis/story.json` — story sections, tension curve, drop events, and metadata about the features used to build the story.
  - `analysis/README.md` — auto-generated human-readable summary: metadata table, sections, drop candidates, signal glossary, re-generate instructions. Always written after `analyze` or `render`.
  - `analysis/overview.png` — 2-panel dark-theme plot: signal envelopes (loudness/onset/tension) + section timeline bar. Written when `matplotlib` is installed (`pip install -e '.[viz]'`).
  - `analysis/stems_overview.png` — 2-panel plot: per-stem RMS envelopes + per-section RMS heatmap. Written when `stems/*.wav` exist and `matplotlib` is installed.
  - `video.mp4` (or a user-specified `--out` path) — 30 fps MP4 with the calculated visuals and muxed audio; stems layouts optionally reference `outputs/<song_id>/stems/` when `--layout stems4` is used.
  - `stems/` (optional) — Demucs output WAVs (`drums`, `bass`, `vocals`, `other`) plus `stems.json` metadata when separation has run.
  - `lyrics/alignment.json` (optional) — backend word-level alignment (whisper/whisperx) with optional auto-offset calibration; written by `songviz lyrics`; requires `pip install -e '.[lyrics]'` (`.[lyricsx]` for whisperx backend).

## Storytelling signals (see `songviz/story.py`)
- **Sections** are detected via **SSM (self-similarity matrix) + checkerboard novelty** boundary detection (min section length: 12 s for SSM, 15 s for agglomerative fallback). Each section gets a **role-based label** — `intro`, `build`, `payoff`, `valley`, `contrast`, `outro` — assigned from blended similarity (MFCC 40% + chroma 20% + energy 40%), rms_slope, contrast gating, and build boost heuristics. Letters `A`, `B`, `C`, … are derived from role + acoustic similarity clustering, so recurring similar sections share a letter.
- **Tension** is a weighted blend of smoothed RMS, onset strength, and spectral centroid, normalized into `[0,1]`, and intended to mark buildups/drops.
- **Events** include `drop_times_s` (sharp tension drops) and `buildups` (each entry has `buildup_start_s`, `buildup_peak_s`, `drop_time_s`).
- Fallback chain: SSM → tension valley (songs >90 s with ≤2 sections) → agglomerative (exception).
- Every renderer can consume `story` output to adjust visuals (background gradients per section, tightened composition during buildups, crossfade hints, etc.).

## Lyrics implementation status
- **Implemented**: full pipeline — `songviz lyrics <audio>` → `alignment.json`; word overlay in rendered video.
- Uses a 6-tier fallback chain with three backends (`stable_whisper`, `whisperx`, `whisper`); forced alignment via `stable_whisper` is the preferred path.
- Requires `pip install -e '.[lyrics]'` (openai-whisper + stable-ts + mutagen); `.[lyricsx]` adds whisperx.
- Module: `songviz/lyrics.py`; output: `outputs/<song_id>/lyrics/alignment.json`.
- See `docs/05_lyrics_playbook.md` for the full pipeline specification, tier definitions, and output contract.

## Research & documentation references
- `docs/01_roadmap.md` → phased plan (what is done, what is next, what is future).
- `docs/02_architecture.md` → pipeline shape, key concepts, CLI/design notes.
- `docs/03_working_state.md` → implementation status (updated constantly).
- `docs/05_lyrics_playbook.md` → canonical lyrics implementation and output contract.
- `docs/06_reduced_representation.md` → design for the current phase: converting features into discrete musical events.
- `experiments/README.md` → separation backend experiments and decisions (DrumSep integration rationale).
- `docs/research/voice-notes-research.md` → pitch extraction tool survey (pYIN, Basic Pitch, CREPE).
- `docs/research/song-story-research.md` → structural segmentation tool survey (SSM, MSAF, novelty detection).
- `docs/research/lyrics_syncing_research.md` → lyric/phoneme alignment tool comparison.

## Running and testing
- Setup: `python3 -m venv .songviz/venv && pip install -e .` (optional `.[stems]` for Demucs, `.[lyrics]` for Whisper, `.[lyricsx]` for whisperx backend, `.[viz]` for matplotlib analysis PNGs).
- Quick render: `python3 -m songviz render songs/<track>.flac --out outputs/demo.mp4`.
- UI: `make ui` or `python3 -m songviz ui --layout stems4`.
- Lyrics: `python3 -m songviz lyrics songs/<track>.flac` (requires `.[lyrics]`; add `--backend whisperx` and install `.[lyricsx]` to use forced alignment).
- Test: `pytest -q` (validates analysis JSON shape, `song_id` determinism, and lyrics alignment contract).
- Clean: `python3 -m songviz tidy` to move old artifacts into hidden folders.
