# SongViz Repository Reference

This document is an execution-oriented map of the repository.
For current status/priorities, use `docs/03_working_state.md`.

## High-level goal
SongViz converts a local song file into an MP4 whose abstract visuals respond to beats, loudness, and story cues. The toolchain is intentionally deterministic, modular (ingest → analysis → story → render), and runnable via `python -m songviz render` or the provided `Makefile` shortcuts.

## Read order (for new contributors and LLMs)
1. `README.md`
2. `docs/03_working_state.md`
3. `docs/04_repo_reference.md` (this file)
4. `docs/05_lyrics_playbook.md` (if working on lyrics)
5. `docs/research/lyrics_syncing_research.md` (background only)

## Repository layout
- `songviz/` — Python package and CLI. Key modules: `ingest` (path normalization, decoding), `analyze` (beat/loudness/onset), `story` (sections, tension, drop candidates), `render` (frame generation + ffmpeg), `stems` (Demucs wrappers), `lyrics` (Whisper alignment, `alignment.json` I/O), `ui` (terminal picker), `tidy` (outputs cleanup), and `cli` (`songviz` command definitions).
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
| `songviz lyrics <audio>` | run Whisper alignment and write `outputs/<song_id>/lyrics/alignment.json` | `--language`, `--model` (tiny/base/small/medium/large), `--force` |
| `songviz tidy` | move stray files/legacy folders under `outputs/` into hidden `.songviz/*` areas | `--outputs-dir`, `--dry-run` |

The package also exposes the console entry: `python -m songviz <subcommand>`. `pip install -e .` or `pip install -e '.[stems]'` registers the console script `songviz` for convenience.

## Inputs and outputs organization
- **Inputs**: drop songs under `songs/` (the interactive `ui` scans this directory); keep FLAC/MP3/WAV files here per the user-requested "song-per-folder" convention.
- **Per-song output folder**: `outputs/<song_id>/` (song names + hashes stay separated). Inside each folder:
  - `analysis/analysis.json` — meta (song_id, duration, sample rate, `analysis.version`), `beats` (tempo, beat times), `envelopes` (hop_s, times_s, loudness, onset_strength normalized to [0,1]), and whatever extra `features`/`stems` data current code emits.
  - `analysis/story.json` — story sections, tension curve, drop events, and metadata about the features used to build the story.
  - `video.mp4` (or a user-specified `--out` path) — 30 fps MP4 with the calculated visuals and muxed audio; stems layouts optionally reference `outputs/<song_id>/stems/` when `--layout stems4` is used.
  - `stems/` (optional) — Demucs output WAVs (`drums`, `bass`, `vocals`, `other`) plus `stems.json` metadata when separation has run.
  - `lyrics/alignment.json` (optional) — Whisper word-level alignment; written by `songviz lyrics`; requires `pip install -e '.[lyrics]'`.

## Storytelling signals (see `songviz/story.py`)
- **Sections** are MFCC-driven clusters labelled `A`, `B`, `C`, … with neighbor merging (min length 7 s), offering coarse story chapters.
- **Tension** is a weighted blend of smoothed RMS, onset strength, and spectral centroid, normalized into `[0,1]`, and intended to mark buildups/drops.
- **Events** include `drop_times_s` candidate timestamps where the tension slope sharply falls.
- Every renderer can consume `story` output to adjust visuals (background gradients per section, tightened composition during buildups, crossfade hints, etc.).

## Lyrics implementation status
- **Implemented**: `songviz lyrics <audio>` (Whisper word-timestamps → `alignment.json`).
- Module: `songviz/lyrics.py` — public API: `align_lyrics`, `load_alignment`, `lyric_activity_at`, `lyric_signals_for_timeline`.
- Output: `outputs/<song_id>/lyrics/alignment.json` (see contract in `docs/05_lyrics_playbook.md`).
- Requires optional dep: `pip install -e '.[lyrics]'`.
- Canonical implementation spec: `docs/05_lyrics_playbook.md`.
- Long-form research (non-default): `docs/research/lyrics_syncing_research.md`.
- Not yet done: MFA path, pYIN pitch summary, render visual integration.

## Research & documentation references
- `docs/00_project_brief.md` → high-level goal and roadmap principles.
- `docs/01_roadmap.md` → strategic backlog (not an execution source of truth).
- `docs/02_architecture.md` → pipeline diagram and CLI/design notes.
- `docs/03_working_state.md` → implementation status (updated constantly).
- `docs/05_lyrics_playbook.md` → canonical lyrics implementation and output contract.
- `docs/research/voice-notes-research.md` → ideas for how to "tell a story" and map narrative/scene logic onto sound.
- `docs/research/lyrics_syncing_research.md` → detailed recommendations and tool comparison for lyric/phoneme alignment.
- `docs/research/deep-research-report.md` and `docs/research/song-story-research.md` → supplemental studies captured by earlier collaborators.

## Running and testing
- Setup: `python3 -m venv .songviz/venv && pip install -e .` (optional `.[stems]` for Demucs, `.[lyrics]` for Whisper).
- Quick render: `python3 -m songviz render songs/<track>.flac --out outputs/demo.mp4`.
- UI: `make ui` or `python3 -m songviz ui --layout stems4`.
- Lyrics: `python3 -m songviz lyrics songs/<track>.flac` (requires `.[lyrics]`).
- Test: `pytest -q` (validates analysis JSON shape, `song_id` determinism, and lyrics alignment contract).
- Clean: `python3 -m songviz tidy` to move old artifacts into hidden folders.
