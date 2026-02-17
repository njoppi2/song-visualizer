# Working State (Source of Truth)

Use this file for the current runtime state and near-term priorities.
Backlog/future ideation lives in `docs/01_roadmap.md`.

## Start here
- Runtime status and commands: this file
- Repo and module map: `docs/04_repo_reference.md`
- Canonical lyrics implementation path: `docs/05_lyrics_playbook.md`
- Lyrics research notes (non-default): `docs/research/lyrics_syncing_research.md`

## What is implemented
- Python package scaffold: `songviz/` with module entrypoint (`python -m songviz`).
- CLI:
  - `songviz analyze <audio>` writes `outputs/<song_name>/analysis/analysis.json`.
  - `songviz render <audio> --out outputs/demo.mp4` writes:
    - `outputs/<song_name>/analysis/analysis.json`
    - `outputs/<song_name>/analysis/story.json`
    - `outputs/<song_name>/video.mp4` (always)
    - plus a copy/hardlink at `--out` if provided
  - `songviz stems <audio>` writes `outputs/<song_name>/stems/{drums,bass,vocals,other}.wav` and `outputs/<song_name>/stems/stems.json` (requires Demucs).
  - `songviz ui` provides a simple interactive picker for songs in `songs/` to render/regenerate videos.
- Packaging via `pyproject.toml` with an optional console script: `songviz ...` after `pip install -e .`.
- Analysis:
  - tempo + beat times
  - loudness (RMS) envelope normalized to [0,1]
  - onset strength normalized to [0,1]
- Story:
  - coarse section segmentation (`A/B/C/...`) from MFCC clustering
  - per-frame `tension` curve aligned to envelope frames
  - simple `drop_times_s` event candidates
- Renderer:
  - 30 fps (default)
  - visuals driven by loudness + onset + beat flashes
  - deterministic with `--seed`
  - ffmpeg muxes original audio into MP4
  - `--audio-codec aac|mp3` (default `mp3`)
  - `--layout stems4` renders a 2x2 stem grid with stem-specific visuals
- Minimal pytest coverage for analysis keys/array lengths and `song_id` stability.
- Lyrics alignment:
  - `songviz lyrics <audio>` writes `outputs/<song_id>/lyrics/alignment.json`.
  - Whisper-based (word-level timestamps + confidence); uses vocals stem when present.
  - `lyrics.load_alignment()` / `lyrics.lyric_activity_at()` / `lyrics.lyric_signals_for_timeline()` expose derived signals.

## Lyrics status
- **Implemented**: `songviz lyrics <audio>` runs Whisper with word timestamps and writes `outputs/<song_id>/lyrics/alignment.json`.
- Uses vocals stem (`outputs/<song_id>/stems/vocals.wav`) when available; falls back to full mix.
- Requires optional dep: `pip install -e '.[lyrics]'` (pulls `openai-whisper`).
- Key public API in `songviz/lyrics.py`: `align_lyrics()`, `load_alignment()`, `lyric_activity_at()`, `lyric_signals_for_timeline()`.
- The output contract and full pipeline spec remain in `docs/05_lyrics_playbook.md`.
- Do not treat `docs/research/lyrics_syncing_research.md` as an execution default.
- **Not yet done**: MFA forced alignment, pYIN pitch summary, story/render visual integration.

## Current priorities
- Improve story signal quality:
  - reduce section jitter and micro-sections
  - add explicit buildup/climax/drop event extraction
- Deepen lyrics → render integration (lyrics pipeline is done; rendering still ignores it):
  - use `lyric_signals_for_timeline()` to drive word-flash visuals in the renderer
  - add `--lyrics` flag to `songviz render` to load alignment.json automatically
- Improve story-driven rendering:
  - stronger chapter transitions at boundaries
  - clearer buildup vs drop visual grammar

## How to run locally
- `pip install -e .`
- For stems: `pip install -e '.[stems]'` (Demucs + TorchCodec)
- For lyrics: `pip install -e '.[lyrics]'` (openai-whisper)
- `python3 -m songviz --help`
- `python3 -m songviz analyze path/to/song.flac`
- `python3 -m songviz render path/to/song.flac --out outputs/demo.mp4`
- `python3 -m songviz render path/to/song.flac --layout stems4` (requires Demucs; creates stems if missing)
- `python3 -m songviz stems path/to/song.flac`
- `python3 -m songviz lyrics path/to/song.flac` (requires `.[lyrics]`; uses vocals stem if present)
- `python3 -m songviz ui` (or `make ui`)
  - `make ui` defaults to `--layout stems4` (override with `make ui UI_LAYOUT=mix`)
- `python3 -m songviz tidy` (optional cleanup of legacy output layout)
- `pytest -q` (or `pip install -e '.[test]' && pytest -q`)

## Known issues
- Rendering uses ffmpeg:
  - if `ffmpeg` is on PATH (or available as `$VIRTUAL_ENV/bin/ffmpeg`), SongViz uses it
  - otherwise SongViz tries a user-space ffmpeg binary via `imageio-ffmpeg` (first run may download it)

## Dev notes
- Keep outputs out of git
- Local dev artifacts are grouped under `.songviz/` (venv, pytest cache, egg-info)
- Prefer small commits after each milestone
