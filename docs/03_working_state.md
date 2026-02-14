# Working State (keep updated!)

## What is implemented
- Python package scaffold: `songviz/` with module entrypoint (`python -m songviz`).
- CLI:
  - `songviz analyze <audio>` writes `outputs/<song_id>/analysis.json`.
  - `songviz render ...` is still a stub (not implemented yet).
- Packaging via `pyproject.toml` with an optional console script: `songviz ...` after `pip install -e .`.
- Analysis (v0):
  - tempo + beat times
  - loudness (RMS) envelope normalized to [0,1]
  - onset strength normalized to [0,1]
- Minimal pytest coverage for analysis keys/array lengths and `song_id` stability.

## What is next
- Implement `songviz render` to create an MP4 (frames + mux original audio via ffmpeg).
  - must be deterministic given `--seed`
  - must render at 30 fps with layered visuals (loudness + onsets + beat flashes)

## How to run locally
- `pip install -e .`
- `python -m songviz --help`
- `python -m songviz analyze path/to/song.flac`
- `pytest -q` (or `pip install -e '.[test]' && pytest -q`)

## Known issues
- `songviz render` is not implemented yet (no MP4 output).

## Dev notes
- Keep outputs out of git
- Prefer small commits after each milestone
