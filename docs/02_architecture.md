# Architecture

This file describes the pipeline shape. For current implementation status, see `docs/03_working_state.md`.

## Pipeline
1) ingest
   - normalize input path
   - hash file contents into a stable `song_id`
   - decode to mono WAV at 22.05 kHz
2) separate (optional)
   - stems: vocals/drums/bass/other via Demucs
3) analyze
   - beats + tempo
   - envelopes (RMS loudness + normalized onset strength)
   - structural segmentation (MFCC-based A/B/C… labels; same letter = recurring motif)
   - story signals: tension curve, drop candidates, buildup windows
4) lyrics (optional)
   - query LRCLIB for human-verified synced lyrics
   - refine word timestamps via Whisper/stable-ts/whisperx backend
   - write `outputs/<song_id>/lyrics/alignment.json`
5) render
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
