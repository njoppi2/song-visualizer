# SongViz

Generate a music-reactive video from an audio file.

## Quickstart
1) Install Python 3.10+
2) Install ffmpeg (required for MP4 output)
3) Install deps:
   - `pip install -e .`
4) Run:
   - `python -m songviz render path/to/song.flac --out outputs/demo.mp4`

## Outputs
SongViz writes per-song artifacts under `outputs/<song_id>/` including `analysis.json`.

## Notes about copyrighted audio
Use your own purchased/downloaded tracks locally. Do not commit copyrighted audio into public repositories.
