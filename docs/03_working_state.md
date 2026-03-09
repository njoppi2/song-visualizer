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
  - Shared render orchestration extracted into `songviz/pipeline.py` (used by both cli and ui).
- Packaging via `pyproject.toml` with an optional console script: `songviz ...` after `pip install -e .`.
- Analysis:
  - tempo + beat times
  - loudness (RMS) envelope normalized to [0,1]
  - onset strength normalized to [0,1]
- Story:
  - coarse section segmentation with **motif-aware labels** (same letter reused for recurring sections, e.g. A B A C A)
  - per-frame `tension` curve aligned to envelope frames
  - `drop_times_s` event detection (wired to renderer: full-frame flash on drop)
  - **buildup detection**: buildup windows exported to story + renderer shows extra rays and intensity bar during buildups
  - `min_len_s` raised to 15 s; adjacent same-label sections merged via `_merge_same_label_sections`; drop detection requires ≥ 0.6 peak tension in preceding 2 s.
- Renderer:
  - 30 fps (default)
  - visuals driven by loudness + onset + beat flashes
  - **section crossfade** at boundaries (1.5 s smoothstep blend)
  - deterministic with `--seed`
  - ffmpeg muxes original audio into MP4
  - `--audio-codec aac|mp3` (default `mp3`)
  - `--layout stems4` renders a 2x2 stem grid with stem-specific visuals
  - `_VisualizerBase` base class shared by `Visualizer` (mix) and `StemQuadVisualizer` (stems4)
  - Drop flash is now a two-phase sharp white spike (decay 60 ms) + accent afterglow (decay 300 ms); section boundaries have a 150 ms color wash in the incoming palette; buildup fraction drives orb radius swell (+12% at peak).
- Minimal pytest coverage for analysis keys/array lengths and `song_id` stability.
- Analysis visualization:
  - `songviz analyze <audio>` and `songviz render <audio>` now auto-generate `analysis/overview.png` (2-panel dark-theme: envelopes + section timeline) and `analysis/README.md` (metadata table, sections, signal glossary, re-generate instructions).
  - `analysis/stems_overview.png` is also written when `stems/*.wav` exist (stem RMS envelopes + per-section heatmap).
  - Requires optional dep: `pip install -e '.[viz]'` (pulls `matplotlib`); README is always written even without it.
  - Module: `songviz/viz.py` — public API: `generate_overview`, `generate_stems_overview`, `generate_analysis_readme`, `generate_all`.
- Lyrics alignment + render integration:
  - `songviz lyrics <audio>` writes `outputs/<song_id>/lyrics/alignment.json`.
  - Backend-based (whisper/whisperx word-level timestamps + confidence); uses vocals stem when present.
  - Automatic global timing calibration is enabled by default (large offsets ≥100ms require improvement ≥0.005 to filter noise-level correlation changes).
  - Pipeline v7: phoneme-aware `lead_in_s` — `_initial_phoneme_class()` maps word-initial letter(s) to articulatory class; per-class lead-in from `_PHONEME_CLASS_LEAD_IN_S` (5–55ms). Populates `w["phones"]` with `[{"class":…,"source":"text_heuristic"}]`. Details include `phoneme_class` and `lead_in_s` per snapped word. v6 onset_detect-based snapping (spectral flux via `librosa.onset.onset_detect`) retained as base; v5 RMS function preserved as `_snap_words_to_vocal_onset_rms` for rollback.
  - `lyrics.load_alignment()` / `lyrics.lyric_activity_at()` / `lyrics.lyric_signals_for_timeline()` expose derived signals.
  - `songviz render <audio> --lyrics` loads `alignment.json` and renders a **full-line overlay**: active word in accent color, remaining words dimmed (mix layout: bottom-center; stems4: vocals quadrant bottom-center).
- `cli.py` has a local `_copy_or_link` helper (intentional — separate from `stems.py`'s internal copy logic).

## Lyrics status
- **Implemented**: `songviz lyrics <audio>` runs the fallback chain below and writes `outputs/<song_id>/lyrics/alignment.json`.
- **Fallback chain** (best to worst):
  1. LRCLIB synced + **forced alignment** → `lrclib+stable_whisper_forced`: uses `stable_whisper.load_model().align()` to align the known LRCLIB text directly to audio, producing 1:1 word timestamps without lossy difflib matching.
  2. LRCLIB synced + backend transcribe+merge → `lrclib+stable_whisper_timing` / `lrclib+whisperx_timing` / `lrclib+whisper_timing`: backend transcribes independently, then difflib matches words to LRCLIB text (fallback when FA fails).
  3. LRCLIB synced + no backend → `lrclib_synced`: proportional word timing within each LRC line.
  4. LRCLIB plain lyrics + backend → `stable_whisper+lrclib_prompt` / `whisper+lrclib_prompt`.
  5. No metadata or no LRCLIB match → pure backend (`stable_whisper` / `whisper` / `whisperx`).
  6. Auto calibration applies a **pre-merge** global offset (for lrclib paths: calibrates raw backend output before merge so LRCLIB segment boundaries are preserved; for pure-whisper paths: applies post-merge).
- Default model: `small` (was `base`); default auto backend order: `whisperx` > `stable_whisper` > `whisper`.
- `stable-ts` (stable_whisper backend) now included in `.[lyrics]` dep — uses DTW on mel spectrogram for much more accurate word boundaries than vanilla Whisper attention weights.
- Uses vocals stem when present; falls back to full mix for backend paths.
- Requires optional dep: `pip install -e '.[lyrics]'` (openai-whisper + stable-ts + mutagen). `pip install -e '.[lyricsx]'` adds whisperx backend.
- Key public API in `songviz/lyrics.py`: `align_lyrics()`, `load_alignment()`, `lyric_activity_at()`, `lyric_signals_for_timeline()`.
- The output contract and full pipeline spec remain in `docs/05_lyrics_playbook.md`.
- Do not treat `docs/research/lyrics_syncing_research.md` as an execution default.
- **Render integration done**: `--lyrics` flag on `songviz render` draws active word as text overlay.
- **Manual corrections workflow**:
  - `songviz lyrics-tap <audio>` — tap-along session: play audio, press space at each word onset, writes corrections.yaml, auto-applies to alignment.json.
  - `songviz lyrics-template <audio>` — generate blank corrections.yaml for manual YAML editing.
  - `songviz lyrics-correct <audio>` — apply corrections.yaml + print quality stats (mean/median error, pct within 100ms/200ms, systematic offset).
  - `songviz lyrics-preview <audio>` — render fast lyrics-only video (960x270 @ 15fps) for timing verification.
  - Corrections auto-reapplied on `songviz lyrics --force` reruns.
  - Module: `songviz/tap.py` (tap session + mapping logic), corrections logic in `songviz/lyrics.py`.
- **Not yet done**: pYIN pitch summary, `lyrics-aligner` fallback (wav2vec2).

## Current priorities
- Deepen lyrics integration (pipeline + render done; remaining):
  - pYIN pitch summary per word
  - `lyrics-aligner` fallback (wav2vec2, pip-installable)
- Story improvements:
  - reduce section jitter and micro-sections further
- Render improvements:
  - stronger chapter transition visuals (currently: smoothstep crossfade; want: more dramatic)

## How to run locally

### Gerar um vídeo (caminho mais curto)
```
pip install -e '.[stems,viz]'   # Demucs + TorchCodec + matplotlib
# coloque músicas em songs/
make ui                      # picker interativo → stems → video.mp4
make ui UI_LAYOUT=mix        # sem stems, mais rápido
```

### Comandos individuais
- `python3 -m songviz --help`
- `python3 -m songviz analyze path/to/song.flac`
- `python3 -m songviz render path/to/song.flac` (ou `--layout stems4`, `--lyrics`, etc.)
- `python3 -m songviz stems path/to/song.flac` (requer `.[stems]`)
- `python3 -m songviz lyrics path/to/song.flac` (requer `.[lyrics]`; `--backend whisperx` requires `.[lyricsx]`; `--artist`/`--title` overrides ID3 tags)
- `python3 -m songviz lyrics-tap path/to/song.flac` (tap-along session to correct word timing)
- `python3 -m songviz lyrics-template path/to/song.flac` (generate blank corrections.yaml)
- `python3 -m songviz lyrics-correct path/to/song.flac` (apply corrections + print quality stats)
- `python3 -m songviz lyrics-preview path/to/song.flac` (fast lyrics-only video preview)
- `python3 -m songviz tidy` (limpeza de outputs antigos)
- `pytest -q` (ou `pip install -e '.[test]' && pytest -q`)

## Known issues
- Rendering uses ffmpeg:
  - if `ffmpeg` is on PATH (or available as `$VIRTUAL_ENV/bin/ffmpeg`), SongViz uses it
  - otherwise SongViz tries a user-space ffmpeg binary via `imageio-ffmpeg` (first run may download it)
  - final MP4 writes are atomic: renderer encodes to a temp file in the output directory and only replaces `video.mp4` after ffmpeg exits successfully (prevents partial/corrupt final files on failed/interrupted renders)

## Dev notes
- Keep outputs out of git
- Local dev artifacts are grouped under `.songviz/` (venv, pytest cache, egg-info)
- Prefer small commits after each milestone
