# Working State (keep updated!)

## What is implemented
- Python package scaffold: `songviz/` with module entrypoint (`python -m songviz`).
- CLI:
  - `songviz analyze <audio>` writes `outputs/<song_name>/analysis/analysis.json`.
  - `songviz render <audio> --out outputs/demo.mp4` writes:
    - `outputs/<song_name>/analysis/analysis.json`
    - `outputs/<song_name>/video.mp4` (always)
    - plus a copy/hardlink at `--out` if provided
  - `songviz ui` provides a simple interactive picker for songs in `songs/` to render/regenerate videos.
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
  - audio codec can be selected via `--audio-codec aac|mp3` (default: mp3 for browser-friendliness)
  - can also write VS Code preview files:
    - `outputs/<song_name>/preview.webm` (VP8+Vorbis)
    - `outputs/<song_name>/previews/*` (optional variants via `--vscode-preview-formats all`)
- Minimal pytest coverage for analysis keys/array lengths and `song_id` stability.

## What is next
- Smoke-test MP4 output on a machine with ffmpeg installed.
- Polish error handling and defaults based on real renders.

## How to run locally
- `pip install -e .`
- `python -m songviz --help`
- `python -m songviz analyze path/to/song.flac`
- `python -m songviz render path/to/song.flac --out outputs/demo.mp4`
- If your browser shows a muted/disabled volume icon, try: `python -m songviz render path/to/song.flac --audio-codec mp3`
- If VS Code preview is silent, try:
  - `outputs/<song_name>/preview.webm`
  - or generate a set of alternatives: `python -m songviz render path/to/song.flac --vscode-preview-formats all`
- `python -m songviz ui` (or `make ui`)
- `python -m songviz tidy` (optional: cleans `outputs/` by moving legacy dirs/loose files into hidden folders)
- `pytest -q` (or `pip install -e '.[test]' && pytest -q`)

## Known issues
- Rendering uses ffmpeg:
  - if `ffmpeg` is on PATH (or available as `$VIRTUAL_ENV/bin/ffmpeg`), it will use that
  - otherwise it will try a user-space ffmpeg binary via `imageio-ffmpeg` (first run may download it)

## Dev notes
- Keep outputs out of git
- Local dev artifacts are grouped under `.songviz/` (venv, pytest cache, egg-info)
- Prefer small commits after each milestone
