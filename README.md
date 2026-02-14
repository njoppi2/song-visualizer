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

If you want `outputs/` to stay clean, run `python -m songviz tidy` to move old layout folders and loose files into hidden subfolders.

## Audio in MP4
By default SongViz encodes audio as MP3-in-MP4 to maximize "it plays somewhere" compatibility on Linux. If you prefer the standard MP4 audio codec, use AAC.

If you prefer AAC (more standard for MP4), or if you want to experiment:
- `python -m songviz render songs/my.flac --audio-codec aac --audio-bitrate 128k`
- `python -m songviz render songs/my.flac --audio-codec mp3 --audio-bitrate 128k`

### VS Code preview (Linux)
VS Code's bundled "Media Preview" has limitations on Linux:
- `.mp4` preview does not support AAC audio tracks.
- `.webm` preview supports VP8 video.

SongViz can also write `outputs/<song_name>/preview.webm` (VP8+Vorbis). In VS Code, open `preview.webm` to hear audio.

By default this preview is auto-enabled when running from VS Code's integrated terminal. You can control it explicitly:
- `python -m songviz render songs/my.flac --vscode-preview`
- `python -m songviz render songs/my.flac --no-vscode-preview`

## Notes about copyrighted audio
Use your own purchased/downloaded tracks locally. Do not commit copyrighted audio into public repositories.

For local organization, put audio files under `songs/` (gitignored).

Dev/runtime artifacts (venv, caches) are grouped under `.songviz/` (gitignored).
