# Working State (keep updated!)

## What is implemented
- Python package scaffold: `songviz/` with module entrypoint (`python -m songviz`).
- CLI:
  - `songviz analyze <audio>` writes `outputs/<song_id>/analysis.json`.
  - `songviz render <audio> --out outputs/demo.mp4` writes:
    - `outputs/<song_id>/analysis.json`
    - `outputs/<song_id>/video.mp4` (always)
    - plus a copy/hardlink at `--out` if provided
- Packaging via `pyproject.toml` with an optional console script: `songviz ...` after `pip install -e .`.
- Analysis (v0):
  - tempo + beat times
  - loudness (RMS) envelope normalized to [0,1]
  - onset strength normalized to [0,1]
- Renderer (v0):
  - 30 fps (default) video
  - layered visuals driven by loudness + onset strength + beat flashes
  - deterministic given `--seed`
  - uses ffmpeg to mux original audio into the MP4
- Minimal pytest coverage for analysis keys/array lengths and `song_id` stability.

## What is next
- Smoke-test MP4 output on a machine with ffmpeg installed.
- Polish error handling and defaults based on real renders.

## How to run locally
- `pip install -e .`
- `python -m songviz --help`
- `python -m songviz analyze path/to/song.flac`
- `python -m songviz render path/to/song.flac --out outputs/demo.mp4`
- `pytest -q` (or `pip install -e '.[test]' && pytest -q`)

## Known issues
- Rendering uses ffmpeg:
  - if `ffmpeg` is on PATH (or available as `$VIRTUAL_ENV/bin/ffmpeg`), it will use that
  - otherwise it will try a user-space ffmpeg binary via `imageio-ffmpeg` (first run may download it)

## Dev notes
- Keep outputs out of git
- Prefer small commits after each milestone
