# SongViz

Generate a music-reactive video from an audio file.

## Quickstart
1) Install Python 3.10+
2) Install ffmpeg (recommended for MP4 output). If not available, SongViz will attempt to use a user-space ffmpeg binary via `imageio-ffmpeg`.
3) Install deps:
   - `pip install -e .`
4) Run:
   - `python -m songviz render path/to/song.flac --out outputs/demo.mp4`

## Interactive UI
- Put songs in `songs/`
- Run: `python -m songviz ui`
- Or: `make ui`

## Outputs
SongViz writes per-song artifacts under `outputs/<song_name>/`:
- `video.mp4`
- `analysis/analysis.json`

## Notes about copyrighted audio
Use your own purchased/downloaded tracks locally. Do not commit copyrighted audio into public repositories.

For local organization, put audio files under `songs/` (gitignored).
