# Roadmap

Phased plan for SongViz. Each phase builds on the previous one.
For current implementation status, see `docs/03_working_state.md`.

## Phase 0: Core pipeline (done)
- Ingest, analyze (beats, loudness, onset), render abstract visuals, mux into MP4
- Deterministic rendering with seed option
- CLI: `songviz analyze`, `songviz render`

## Phase 1: Stem separation + per-stem features (done)
- Demucs 4-stem separation (vocals, bass, drums, other)
- Per-stem feature extraction: vocal pitch (pYIN), bass pitch (pYIN), drum band energy (3-band), other chroma (12-bin)
- DrumSep integration: 6-component drum decomposition (kick, snare, toms, hh, ride, crash) as optional post-pass on Demucs drum stem
- stems4 render layout (2x2 stem grid)

## Phase 2: Lyrics alignment (done)
- LRCLIB + Whisper/stable-ts/whisperx fallback chain
- Word-level timestamps with auto calibration and onset snapping
- Word overlay in rendered video
- Manual correction workflow (tap-along, YAML editing, preview)

## Phase 3: Story / structural analysis (done)
- SSM + checkerboard novelty segmentation
- Role-based section labels (intro, build, payoff, valley, contrast, outro)
- Tension curve, drop detection, buildup windows
- Subsections with energy descriptors

## Phase 4: Reduced representation (current phase)
- Convert continuous per-stem features into discrete musical events
- Goal: a simplified, low-timbre representation that preserves the structural core (melody contour, rhythm skeleton, harmonic motion, section changes, dynamics)
- Separation quality matters because cleaner stems produce cleaner discrete events
- Incremental build order:
  1. Drum hit events (from DrumSep components)
  2. Vocal note events (from pYIN pitch track)
  3. Bass note events (same approach)
  4. Harmony / chord labels (deferred)
- Detailed design: `docs/06_reduced_representation.md`
- Separation experiments and decisions: `experiments/README.md`

## Phase 5: Analysis from reduced representation (future)
- Feed discrete events into story module for more robust section similarity
- Arrangement-level analysis: what changes between sections (instruments added/dropped, density shifts, register changes)
- Sonification of reduced representation for listening validation

## Phase 6: Timbral descriptors (future)
- Extract timbral information from original audio as a separate layer
- Spectral envelope, formant features, instrument classification
- The reduced representation captures *what notes are played when*; timbre captures *how they sound*
- Layer timbre back onto the structural skeleton only where it adds analytical value

## Phase 7: Richer visualization (future)
- Render from discrete musical events instead of signal-level curves
- Note shapes, chord colors, arrangement diagrams
- Stronger section transition visuals
- The renderer becomes a consumer of musical-level analysis, not signal-level features
