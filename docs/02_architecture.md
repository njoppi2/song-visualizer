# Architecture

This file describes the pipeline shape and key concepts.
For current implementation status, see `docs/03_working_state.md`.
For the phased plan, see `docs/01_roadmap.md`.

## Key concepts

- **Story**: the structural narrative of a song — section boundaries and roles (intro, build, payoff, valley, contrast, outro), tension arc, repetition patterns, energy dynamics. Not a text narrative; a model of how the song's structure unfolds over time.
- **Separation**: splitting audio into stems (vocals, bass, drums, other) so each musical layer can be analyzed independently. A means to better analysis, not an end in itself.
- **Features**: per-stem signals extracted from audio — vocal pitch track, bass pitch track, drum band energy, chroma. Currently frame-level (one value per ~23ms hop). Used today to drive the renderer.
- **Reduced representation** (next phase): discrete musical events derived from features — note onsets/offsets, drum hits, chord labels — stripped of timbre. The goal is a simplified structural skeleton that is easier to analyze and compare across sections. See `docs/06_reduced_representation.md`.
- **Timbre**: *how* notes sound (tone color, texture, instrument character). Deliberately excluded from the reduced representation. Can be layered back from original audio later as a separate descriptor.

## Pipeline
1) **ingest**
   - normalize input path
   - hash file contents into a stable `song_id`
   - decode to mono WAV at 22.05 kHz
2) **separate** (optional)
   - stems: vocals/drums/bass/other via Demucs
   - drum sub-components: kick/snare/toms/hh/ride/crash via DrumSep (optional post-pass)
3) **analyze**
   - beats + tempo
   - envelopes (RMS loudness + normalized onset strength)
   - per-stem features: vocal pitch, bass pitch, drum band energy, other chroma
   - structural segmentation (SSM + checkerboard novelty; role-based labels)
   - story signals: tension curve, drop candidates, buildup windows
4) **reduce** (planned — see `docs/06_reduced_representation.md`)
   - convert frame-level features into discrete musical events
   - drum hits, note events, chord labels
   - beat-quantized summaries for section comparison
5) **lyrics** (optional)
   - query LRCLIB for human-verified synced lyrics
   - refine word timestamps via Whisper/stable-ts/whisperx backend
   - write `outputs/<song_id>/lyrics/alignment.json`
6) **render**
   - generate frames at configured fps/resolution
   - section-aware background gradients (crossfade at boundaries)
   - beat flash, tension buildup bar, drop strobe
   - optional lyric word overlay (mix: bottom-centre; stems4: vocals quadrant)
   - mux original audio into MP4 via ffmpeg

## Output structure
```
outputs/<song_id>/
  analysis/
    analysis.json
    story.json
    overview.png          # signal envelopes + section timeline (requires [viz])
    stems_overview.png    # per-stem RMS heatmap (requires [viz] + stems)
    README.md             # auto-generated human-readable summary
  video.mp4
  stems/                  # optional; drums.wav, bass.wav, vocals.wav, other.wav
  lyrics/
    alignment.json        # optional; word-level timestamps
```

`song_id` is the first 16 hex chars of the SHA-256 of the file contents (stable across machines).

## analysis.json schema
Top-level:
- `meta`: song_id, duration_s, sample_rate, created_at
- `beats`: tempo_bpm, beat_times_s
- `envelopes`: hop_s, times_s, loudness (normalized), onset_strength (normalized)
- `story`: sections, tension, events — see story.json

## CLI commands
- `songviz analyze <audio>` — write `analysis/*.json` without rendering
- `songviz render <audio>` — analyze + render + mux into `video.mp4`
- `songviz stems <audio>` — run Demucs and dump WAV stems
- `songviz lyrics <audio>` — run lyrics alignment pipeline
- `songviz ui` — interactive terminal picker
- `songviz tidy` — move legacy output files into hidden subfolders

Full option reference: `docs/04_repo_reference.md`.
Lyrics pipeline contract: `docs/05_lyrics_playbook.md`.
