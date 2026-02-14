# Working State (keep updated!)

## What is implemented
- Python package scaffold: `songviz/` with module entrypoint (`python -m songviz`).
- CLI skeleton with `analyze` and `render` subcommands (not implemented yet).
- Packaging via `pyproject.toml` with an optional console script: `songviz ...` after `pip install -e .`.

## What is next
- Implement ingest utilities:
  - stable `song_id` (content hash)
  - audio loading/resampling
- Implement `songviz analyze` to write `outputs/<song_id>/analysis.json`.
- Implement `songviz render` to create an MP4 (frames + mux original audio via ffmpeg).

## How to run locally
- `pip install -e .`
- `python -m songviz --help`

## Known issues
- `songviz analyze` and `songviz render` raise `NotImplementedError` until v0 work is completed.

## Dev notes
- Keep outputs out of git
- Prefer small commits after each milestone
