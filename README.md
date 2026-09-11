# SongViz

**Resuming or starting a new agent? Read [CONTINUE.md](CONTINUE.md) first.**
It is the single handoff for current progress, known failures, the next experiment,
delegation policy and artifact locations. [AGENTS.md](AGENTS.md) directs coding
agents to the same entry point.

**Complement and intensify the experience of listening to a song through visual stimulus.**

The goal is to make you *feel* music more deeply: the build-up should feel more intense, the drop should hit harder, a quiet bridge might go to black so the return of the drums hits you visually too. Every high, every low, every shift in tension — the visuals should amplify what your ears already sense but your eyes don't yet see.

This isn't about decorative waveforms or spectrum analyzers. It's about understanding the *story* of a song — its sections, tension arcs, repetition, dynamics — and translating that understanding into visuals that move with the music in a meaningful way. A kick drum could alternate left and right. A sudden drop in energy could strip all visuals away. Multiple displays could each serve a different function. The creative possibilities are wide, but they all depend on one thing: deeply understanding what's happening in the music.

### Intended end product

SongViz should be an **automatic visual director for a song**. The user runs one
command with an audio file; the system analyzes the music, creates a saved visual
direction plan, and renders a video with the original audio. An LLM may help plan
the direction, but a particular model or provider is not part of the product goal.

The director should use whole-song context to choose what deserves attention at
each moment: which layers to emphasize or hide, how to treat them, and when to
change composition, movement, color, or visual density. An instrument is not tied
to one permanent animation. Repeated musical ideas can reuse recognizable visual
motifs; an entrance, breakdown, return, or unexpected change can redirect attention
or transform an established motif. Variation should serve the music, not become
random effect switching. Showing every detected instrument is not the objective.

Song structure is not limited to one level of discrete sections. Verse/chorus
labels can anchor identity while smaller and larger, potentially overlapping
transitions and graded changes guide the visuals within and across those parts.
Local change, familiarity and visual importance are distinct; every detected
peak need not become a section boundary or a visual cut.

**Current gap:** the repo has analysis, heuristic story signals, reactive rendering,
and a first experimental rule-based directing plan for a cached passage. It does
not yet have the full-song director or LLM planning. Existing production render
commands are unchanged. Earlier drum/pulse previews are timing and visual-vocabulary
experiments, not the final product specification. Try the new directing experiment
using the commands in [Directing prototype](docs/10_directing_prototype.md).
The proposed architecture and unresolved decisions live in
[Architecture](docs/02_architecture.md); the next experiment is in the
[Roadmap](docs/01_roadmap.md).

Current musical-understanding work: the [boundary and recurrence review](docs/13_structure_review.md)
compares old/new structural logic with original audio and role-independent
16/32-beat passage pairs. The first user section annotations now feed a
[dimension-separated development evaluation](docs/16_structural_evaluation.md):
musical identity, arrangement variation and transition intervals are represented
separately, with acoustic pattern/level and local/historical context diagnostics.
This is not yet a reliable automatic identity or transition detector. The first
[short-scale change/transition experiment](docs/17_local_structure.md) now adds
inspectable local-change points and bounded energy-dip candidates. It remains
experimental: important chorus variations and some transitions are still missed.

You can now [define your own sections](docs/14_section_annotation.md) from a blank
timeline, with free labels and optional independent annotation layers. The first
export is preserved unchanged; it guides development without being silently
promoted to high-confidence ground truth.

### Strategy

We develop the analysis and visual experience together:

1. **Establish a trustworthy baseline** — compare short audio/video passages, audit reference annotations, and preserve concrete feedback as regression cases or artistic preferences.
2. **Direct a passage across a meaningful musical change** — test selective focus, omission, and evolving visual treatments with an inspectable plan, rather than polish a fixed instrument-to-animation mapping.
3. **Improve the musical understanding that matters** — evaluate timing, activity, repetition, and transitions using audio features, reduced musical events, or both. Add harmony and timbre when a demonstrated visual use case needs them.

The existing pipeline already includes stems, lyrics, story analysis, reduced drum/vocal/bass events, sonification, and rendering. User feedback supports the experimental regular pulse and separate drum indicators on the reviewed excerpts. A small directing-layer prototype is now ready for review before changing production behavior. Reduced representation is a useful hypothesis and diagnostic; full note transcription is not a prerequisite for useful visuals.

### Pipeline

- **Separation** (Demucs) isolates musical layers (drums, bass, vocals, other) so we can analyze them independently
- **Feature extraction** captures per-stem musical content (pitch tracks, drum hits, chroma, energy envelopes)
- **Reduced representation** converts features into discrete musical events — note onsets, drum hits, pitch contours — stripped of timbre
- **Story / structural analysis** identifies sections, roles, tension, and repetition patterns
- **Visual direction (experimental)** chooses focus, omissions, treatments, motifs, and transitions in a saved plan; the first cached-passage policy is rule-based, not a full-song director
- **Rendering** visualizes the analysis as video — the current primary output, evolving toward the full creative vision above

See `docs/01_roadmap.md` for the current milestones and `docs/06_reduced_representation.md` for the historical reduced representation design.

## Project navigation

- **Current work, handoff and active team assignments:** [CONTINUE.md](CONTINUE.md).
  This is the only mutable current-status document.
- **Rules for coding agents:** [AGENTS.md](AGENTS.md). For two-account work, read
  the [collaboration protocol](docs/22_collaboration_protocol.md) after the handoff.
- **Product, quickstart and stable orientation:** this README.
- **Durable evidence, designs and past experiments:** `docs/`; these documents do
  not override `CONTINUE.md`'s current queue.

Useful starting references:

- Multiscale change responses and listening-feedback findings: [Change episodes](docs/21_change_episodes.md)
- Guided audio examples and optional feedback: [Listening examples](docs/19_listening_examples.md)
- Target architecture, current gaps, and open design questions: `docs/02_architecture.md`
- Project roadmap and phases: `docs/01_roadmap.md`
- Current runtime status and priorities: `docs/03_working_state.md`
- First restart review, reference audit, and preview instructions: `docs/07_restart_review.md`
- First feedback, timing evidence, and controlled rhythm comparison: `docs/08_rhythm_feedback.md`
- First artistic opening preview and review instructions: `docs/09_visual_passage.md`
- Plan-driven transition/return prototype, commands and limits: `docs/10_directing_prototype.md`
- User section feedback and separated structural evaluation: `docs/15_section_feedback.md`, `docs/16_structural_evaluation.md`
- Short-scale local changes and transition candidates: `docs/17_local_structure.md`
- Repo map and command reference: `docs/04_repo_reference.md`
- Reduced-representation design (historical): `docs/06_reduced_representation.md`
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
