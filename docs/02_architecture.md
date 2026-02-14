# Architecture

## Pipeline
1) ingest
   - normalize input path
   - decode to a working WAV (consistent sample rate)
2) separate (optional, v1+)
   - stems: vocals/drums/bass/other
3) analyze
   - beats + tempo
   - envelopes (overall + per-stem)
   - optional structural segmentation (v2+)
   - derive storyboard events (v3+)
4) render
   - generate frames (fps, duration)
   - render scenes based on analysis timeline
   - mux original audio back into MP4

## Output structure
outputs/<song_name>/
  analysis/
    analysis.json
  video.mp4
  # future:
  # stems/ (v1+)
  # render/frames/ (optional, behind a flag)

`song_id` should be stable (hash of file contents or file name + size + mtime). It is stored in `analysis.json` metadata.

## analysis.json schema (v0-v2)
Top-level:
- meta:
  - song_id, original_path (optional), sr, duration_s, created_at
- beats:
  - tempo_bpm
  - beat_times_s: [..]
- envelopes:
  - hop_s, times_s: [..]
  - loudness: [..]          # overall RMS or LUFS proxy
  - onset_strength: [..]    # normalized
- stems (v1+):
  - names: ["vocals","drums","bass","other"]
  - energy: {stem_name: [..]}

- sections (v2+):
  - [{start_s, end_s, label}]   # label can be "A","B","C" etc.

Notes:
- Keep arrays same length using a common hop size.
- Normalize signals for stable visuals.

## Scene system
Renderer should be decoupled:
- Scene gets (t, analysis, rng) and returns drawing primitives.
- Provide 3–5 built-in scenes with different motion behaviors.
- Map sections -> scenes via deterministic assignment (hash label -> scene preset).

## CLI commands (planned)
- `songviz analyze <audio> --out outputs/<song_name>/analysis/analysis.json`
- `songviz render <audio> --out out.mp4 [--analysis existing.json]`
- `songviz separate <audio> --out stems_dir` (optional)
