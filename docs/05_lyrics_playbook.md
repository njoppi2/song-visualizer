# Lyrics Playbook (Canonical)

Use this document as the default implementation path for lyrics in SongViz.
If this playbook conflicts with research notes, follow this file.

## Scope and status
- **MVP implemented**: `songviz lyrics <audio>` → `outputs/<song_id>/lyrics/alignment.json`.
- Current aligner: Whisper (word timestamps + confidence). MFA path not yet implemented.
- Background options and theory live in `docs/research/lyrics_syncing_research.md`.

## Source of truth
- Runtime/project status: `docs/03_working_state.md`
- Pipeline and repo map: `docs/04_repo_reference.md`
- Lyrics implementation path: `docs/05_lyrics_playbook.md` (this file)

## Default pipeline
1. Input audio preparation
   - Preferred input for alignment: vocals stem (`outputs/<song_id>/stems/vocals.wav`) when available.
   - If no stem exists, align from the full mix and mark lower confidence.
2. Transcript source
   - If trusted lyrics text is provided, use it directly.
   - Otherwise run Whisper to create draft lyrics, then perform a cleanup pass.
3. Forced alignment
   - Default aligner: MFA.
   - Segment long tracks into short vocal regions before alignment.
4. Canonical output normalization
   - Convert aligner output to one JSON format (words + phones + confidence + metadata).
5. Prosody enrichment
   - Compute pitch/voicing summaries per word from the same aligned timeline.
6. Integration
   - Store final output under `outputs/<song_id>/lyrics/alignment.json`.
   - Expose derived cues to story/render (word activity, phrase starts, confidence flags).

## Fallback policy
- If MFA quality is weak on singing-heavy sections, fallback to `lyrics-aligner`.
- Use SOFA only when language/resources clearly match.
- Keep WebMAUS and Gentle as manual/sanity-check fallbacks, not default.

## Required output contract
Write one file: `outputs/<song_id>/lyrics/alignment.json`.

Minimum fields:
- `metadata`: `song_id`, `language`, `alignment_tool`, `created_utc`
- `segments[]`: phrase or line windows
- `segments[].words[]`: `word`, `start_s`, `end_s`, `confidence`
- `segments[].words[].phones[]`: `ph`, `start_s`, `end_s`, `confidence` (if available)
- `quality_flags[]`: optional reliability markers (`possible_melisma`, `backing_vocals_overlap`)
- `pitch_summary`: per-word aggregates (`median_hz`, `mean_voiced_prob`) when available

## Implementation order (for contributors/LLMs)
1. ~~Build normalizer that converts MFA output into `alignment.json`.~~ → **Done via Whisper** (word timestamps; MFA normalizer still pending).
2. ~~Add optional Whisper pre-step when lyrics text is missing.~~ → **Done** (Whisper used end-to-end for MVP).
3. Add pYIN-based word-level pitch summary.
4. Add MFA forced alignment path (better accuracy for clean vocals).
5. Add fallback path to `lyrics-aligner`.
6. Wire lyric cues into story/render behind a feature flag (use `lyric_signals_for_timeline()` from `songviz/lyrics.py`).

## Acceptance criteria (MVP)
- ✅ Running the lyrics pipeline produces `outputs/<song_id>/lyrics/alignment.json`.
- ✅ At least 90% of aligned words have valid `start_s <= end_s`.
- ✅ Alignment metadata identifies tool and source audio (vocals stem vs mix).
- ✅ Story layer can read the file without schema errors (`load_alignment()` returns dict or None).

## Non-goals for this phase
- Perfect semantic understanding of lyrics.
- Multi-language auto-routing between aligners.
- Full phoneme accuracy guarantees on melismatic vocals.
