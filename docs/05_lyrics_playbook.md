# Lyrics Playbook (Canonical)

Use this document as the default implementation path for lyrics in SongViz.
If this playbook conflicts with research notes, follow this file.

## Scope and status
- **Implemented**: `songviz lyrics <audio>` → `outputs/<song_id>/lyrics/alignment.json` with word-level timestamps.
- **Render integration done**: `songviz render --lyrics` draws the active word as a text overlay; `make ui` runs this automatically.
- **Alignment chain** (6 tiers, best to worst):
  1. `lrclib+stable_whisper_forced` — forced alignment via `stable_whisper.load_model().align()` (1:1 word mapping, no difflib)
  2. `lrclib+stable_whisper_timing` / `lrclib+whisperx_timing` / `lrclib+whisper_timing` — backend transcribes independently, difflib merges words to LRCLIB text
  3. `lrclib_synced` — proportional word timing within each LRC line (no backend needed)
  4. `stable_whisper+lrclib_prompt` / `whisper+lrclib_prompt` — LRCLIB plain text fed as prompt to backend
  5. Pure backend: `stable_whisper` / `whisper` / `whisperx` (no LRCLIB match)
  6. Auto calibration: pre-merge global offset for lrclib paths (preserves segment boundaries); post-merge for pure-backend paths
- Background options and theory live in `docs/research/lyrics_syncing_research.md`.

## Source of truth
- Runtime/project status: `docs/03_working_state.md`
- Pipeline and repo map: `docs/04_repo_reference.md`
- Lyrics implementation path: `docs/05_lyrics_playbook.md` (this file)

## Default pipeline (as implemented)
1. **Audio source** — vocals stem (`outputs/<song_id>/stems/vocals.wav`) preferred; full mix as fallback. Used only for Whisper paths.
2. **Metadata** — artist + title read from ID3/Vorbis tags via `mutagen`; `--artist`/`--title` CLI flags override.
3. **LRCLIB lookup** — `GET https://lrclib.net/api/get?artist_name=X&track_name=Y&duration=Z` (stdlib urllib, no auth). Returns synced LRC and/or plain lyrics, or None on miss/timeout.
4. **Alignment** — decided by what LRCLIB returned and which backend is available. The preferred path is **forced alignment** via `stable_whisper` (`stable_whisper.load_model().align()`), which maps known LRCLIB text directly to audio without lossy difflib matching. Default backend order: `whisperx` > `stable_whisper` > `whisper`; default model: `small`.
   - **Synced LRC + forced alignment** → `lrclib+stable_whisper_forced` (best quality).
   - **Synced LRC + backend transcribe+merge** → `lrclib+stable_whisper_timing` / `lrclib+whisperx_timing` / `lrclib+whisper_timing` (fallback when FA fails).
   - **Synced LRC, no backend** → `lrclib_synced` (proportional word timing).
   - **Plain lyrics only + backend** → `stable_whisper+lrclib_prompt` / `whisper+lrclib_prompt`.
   - **No LRCLIB match** → pure backend (`stable_whisper` / `whisper` / `whisperx`).
5. **Auto calibration** — estimate and apply a global offset from vocal-energy vs word-activity correlation when confidence is sufficient.
5a. **Onset snapping** — after calibration, `_snap_words_to_vocal_onset()` nudges each word start forward to the nearest spectral onset detected by `librosa.onset.onset_detect` (spectral flux). Lead-in before the onset is **per-word** based on the word-initial phoneme class (see table below), so plosive-initial words get more pre-roll than vowel-initial ones. The `phones` field on each word is populated with `[{"class": <class>, "source": "text_heuristic"}]`.

| Class | Lead-in | Examples |
|---|---|---|
| plosive | 55 ms | blue, time, come |
| affricate | 45 ms | child, jump |
| fricative | 25 ms | sing, the, she |
| nasal | 20 ms | moon, know |
| approximant | 20 ms | run, write |
| vowel | 5 ms | apple, open |
| unknown | 40 ms | punctuation-only tokens |

The class is determined by `_initial_phoneme_class(word)` — a pure-text heuristic with silent-letter exceptions (`kn→nasal`, `wr→approximant`, `th/sh→fricative`, `ps/pn/pt/gh→vowel`, etc.). No external deps.

6. **Output** — write `alignment.json` under `outputs/<song_id>/lyrics/`.
7. **Render** — `load_alignment()` loads the file; `lyric_activity_at(t)` returns active word/segment; `_draw_lyric_overlay()` in `render.py` paints the word on each frame.

## Fallback policy
- LRCLIB is the primary text source; backend words provide audio-based timestamps.
- `auto` backend routing prefers `whisperx` when installed, then `stable_whisper` (stable-ts), then `whisper`.
- Forced alignment (`stable_whisper`) is attempted first for synced LRC; transcribe+merge is the fallback.
- Pure backend transcription is the last resort when no metadata or LRCLIB match exists.

## Required output contract
Write one file: `outputs/<song_id>/lyrics/alignment.json`.

Minimum fields:
- `metadata`: `song_id`, `language`, `alignment_tool`, `created_utc`
- `segments[]`: phrase or line windows
- `segments[].words[]`: `word`, `start_s`, `end_s`, `confidence`
- `segments[].words[].phones[]`: populated by onset snapping with `[{"class": <phoneme_class>, "source": "text_heuristic"}]`; full `ph`/`start_s`/`end_s`/`confidence` fields reserved for future MFA path
- `quality_flags[]`: optional reliability markers (`possible_melisma`, `backing_vocals_overlap`)
- `pitch_summary`: per-word aggregates (`median_hz`, `mean_voiced_prob`) when available

## Implementation order (for contributors/LLMs)
Steps 1–4 complete.
5. Add pYIN-based word-level pitch summary.
6. Add MFA forced alignment path (better phoneme-level accuracy for clean vocals).
7. Add fallback path to `lyrics-aligner` (wav2vec2).

## Non-goals for this phase
- Perfect semantic understanding of lyrics.
- Multi-language auto-routing between aligners.
- Full phoneme accuracy guarantees on melismatic vocals.
