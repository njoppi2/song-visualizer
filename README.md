# SongViz

**Complement and intensify the experience of listening to a song through visual stimulus.**

The goal is to make you *feel* music more deeply: the build-up should feel more intense, the drop should hit harder, a quiet bridge might go to black so the return of the drums hits you visually too. Every high, every low, every shift in tension — the visuals should amplify what your ears already sense but your eyes don't yet see.

This isn't about decorative waveforms or spectrum analyzers. It's about understanding the *story* of a song — its sections, tension arcs, repetition, dynamics — and translating that understanding into visuals that move with the music in a meaningful way. A kick drum could alternate left and right. A sudden drop in energy could strip all visuals away. Multiple displays could each serve a different function. The creative possibilities are wide, but they all depend on one thing: deeply understanding what's happening in the music.

### Strategy

To get there, we work bottom-up:

1. **Simplify the song** — strip away timbral complexity, reduce it to an "8-bit" skeleton of melody contour, rhythm hits, bass movement, and harmonic motion. If you can still recognize the song from this simplified version, the structural core is preserved.
2. **Understand the structure** — identify sections (intro, build, drop, bridge, chorus, outro), map tension arcs, detect repetition and contrast. This is the "story" of the song.
3. **Map structure to visuals** — translate structural understanding into creative visual decisions. This layer is where artistic intent lives: *what should a build-up look like? What does silence look like?*

We're currently focused on layers 1 and 2. Layer 3 is future work, but everything we build is in service of it.

### Pipeline

- **Separation** (Demucs) isolates musical layers (drums, bass, vocals, other) so we can analyze them independently
- **Feature extraction** captures per-stem musical content (pitch tracks, drum hits, chroma, energy envelopes)
- **Reduced representation** converts features into discrete musical events — note onsets, drum hits, pitch contours — stripped of timbre
- **Story / structural analysis** identifies sections, roles, tension, and repetition patterns
- **Rendering** visualizes the analysis as video — the current primary output, evolving toward the full creative vision above

See `docs/01_roadmap.md` for the phased plan and `docs/06_reduced_representation.md` for the reduced representation design.

## Start Here (Humans + LLMs)
- Project roadmap and phases: `docs/01_roadmap.md`
- Current runtime status and priorities: `docs/03_working_state.md`
- Repo map and command reference: `docs/04_repo_reference.md`
- Reduced-representation design (next phase): `docs/06_reduced_representation.md`
- Canonical lyrics implementation path: `docs/05_lyrics_playbook.md`
- Lyrics research and alternatives (non-default): `docs/research/lyrics_syncing_research.md`

## Quickstart
1) Install Python 3.10+ and ffmpeg
2) Install deps: `pip install -e '.[stems]'`
3) Drop songs into `songs/`
4) Run: **`make ui`** — picks a song interactively, separates stems, and renders `outputs/<song>/video.mp4`

Override layout: `make ui UI_LAYOUT=mix` (skips stem separation, faster).

## Make targets
- `make ui` (or `python3 -m songviz ui`) shows a terminal picker, separates the selected track into stems, runs the story-aware analysis pipeline, and renders a stems-grid video.
- `make render` is shorthand for running `python3 -m songviz render` with a handful of defaults; it produces the per-song `analysis/analysis.json`, `story.json`, and `video.mp4` files under `outputs/`.
- `make analyze` or `python3 -m songviz analyze` only generates `analysis/*.json` (including `story.json`) so you can inspect beats, envelopes, sections, and tension without rendering a video.

## Outputs
SongViz writes per-song artifacts under `outputs/<song_name>/`:
- `video.mp4`
- `analysis/analysis.json`
- `analysis/story.json`
- `stems/` (optional; written by `python3 -m songviz stems ...`)
- `lyrics/alignment.json` (optional; written by `python3 -m songviz lyrics ...`; requires `pip install -e '.[lyrics]'`)

If you want `outputs/` to stay clean, run `python3 -m songviz tidy` to move old layout folders and loose files into hidden subfolders.

## Audio in MP4
By default SongViz encodes audio as MP3-in-MP4 to maximize "it plays somewhere" compatibility on Linux. If you prefer the standard MP4 audio codec, use AAC.

If you prefer AAC (more standard for MP4), or if you want to experiment:
- `python3 -m songviz render songs/my.flac --audio-codec aac --audio-bitrate 128k`
- `python3 -m songviz render songs/my.flac --audio-codec mp3 --audio-bitrate 128k`

## Stems (Optional)
If you install the optional Demucs dependency, SongViz can separate a track into stems:

```bash
python3 -m songviz stems songs/my.flac
```

This writes WAV stems under `outputs/<song_name>/stems/`:
- `drums.wav`, `bass.wav`, `vocals.wav`, `other.wav`

You can also render a 2x2 stems grid video (one quadrant per stem):

```bash
python3 -m songviz render songs/my.flac --layout stems4
```

## Lyrics (Optional)
Install the lyrics extras (Whisper + mutagen for ID3 tag reading):

```bash
pip install -e '.[lyrics]'
python3 -m songviz lyrics songs/my.flac
```

Optional extra backend (whisperx):

```bash
pip install -e '.[lyricsx]'
python3 -m songviz lyrics songs/my.flac --backend whisperx
```

This writes `outputs/<song_name>/lyrics/alignment.json` with word-level timestamps. The alignment uses a 6-tier fallback chain that combines LRCLIB text with audio-based timing from one of three backends (`stable_whisper`, `whisperx`, or `whisper`), with automatic global offset calibration enabled by default. See `docs/05_lyrics_playbook.md` for the full pipeline specification and output contract.

LRCLIB lookup uses ID3/Vorbis tags automatically. Override them with `--artist`/`--title` if tags are missing.

`make ui` runs lyrics alignment automatically before rendering (cached; re-run with `--force`).

Options:
- `--artist "Name"` — artist name (overrides ID3 tag)
- `--title "Name"` — track title (overrides ID3 tag)
- `--language en` — language code for Whisper (default: en)
- `--model small` — Whisper model size: tiny, base, small, medium, large (default: small)
- `--backend auto` — alignment backend: auto, whisper, stable_whisper, whisperx (default: auto)
- `--no-auto-calibrate` — disable automatic global timing calibration
- `--force` — re-run alignment even if a cached file exists

To display lyrics as a word overlay in the rendered video:

```bash
python3 -m songviz lyrics songs/my.flac
python3 -m songviz render songs/my.flac --lyrics
```

### VS Code note (Linux)
VS Code's bundled media preview can be unreliable for some video/audio codecs. If a rendered MP4 is silent (or fails to load) inside VS Code, open it in a media player instead:
- `xdg-open outputs/<song_name>/video.mp4`
- or `mpv outputs/<song_name>/video.mp4`

## Notes about copyrighted audio
Use your own purchased/downloaded tracks locally. Do not commit copyrighted audio into public repositories.

For local organization, put audio files under `songs/` (gitignored).

Dev/runtime artifacts (venv, caches) are grouped under `.songviz/` (gitignored).