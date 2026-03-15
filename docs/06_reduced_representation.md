# Reduced-Representation Pipeline — Planning Document

## 1. Current direction

SongViz currently operates on raw waveforms: Demucs separates a song into four stems (vocals, bass, drums, other), and feature extractors produce per-frame signals from those stems — vocal pitch, bass pitch, drum band energy, chroma. These features drive the renderer, but they're still dense time-series tied to the audio sample rate. There is no intermediate representation that captures the *musical content* of a song in a compact, analysis-friendly form.

The next phase is to build that layer: a **reduced representation** that strips timbre and production detail down to a structural skeleton — something closer to a lead sheet than a waveform.

This does *not* replace the current pipeline. Separation and feature extraction stay as they are. The reduced representation sits between them and the renderer/story modules, providing a simpler signal to reason about.

### What we have today

| Stem    | Feature              | Shape          | What it captures                         |
|---------|----------------------|----------------|------------------------------------------|
| vocals  | `pitch_hz` (pYIN)    | (N,)           | Fundamental frequency per frame, NaN=unvoiced |
| vocals  | `note_events` (basic-pitch) | list of dicts | start_s, end_s, midi, velocity          |
| bass    | `pitch_hz` (pYIN)    | (N,)           | Same as vocals, bass range               |
| drums   | `drums_bands_3`      | (N, 3)         | kick/snare/hats energy per frame         |
| other   | `chroma_12`          | (N, 12)        | Pitch-class distribution per frame       |

Plus global signals: RMS envelope, onset strength, beat times, tempo, story sections, tension curve, drop events, buildup windows.

These features are computed in `pipeline.py:_build_stem_analyses()` and consumed by `render.py` for visualization. No downstream module currently interprets them as *music* (notes, chords, rhythm patterns). The render treats them as visual intensity curves.

### Prior manual experiments

Three hand-made WAV files exist for The Great Magician:
- `stemmix-8bit.wav` — all stems mixed down to 8-bit 11025 Hz mono (lo-fi "chiptune" quality)
- `pitch-only.wav` — sine waves at the detected pitch track frequencies (16-bit 22050 Hz)
- `pitch-minimal.wav` — similar to pitch-only with additional processing (16-bit 22050 Hz)

These were one-off explorations, not code-generated. They inform the direction but aren't part of any pipeline.


## 2. Core hypothesis and rationale

**Hypothesis:** A song can be simplified into a reduced representation that removes most timbral detail while preserving enough structural core (melody contour, rhythm skeleton, harmonic motion, section changes) to make downstream analysis — and ultimately visualization — easier, more interpretable, and more robust.

**Rationale:**

Our current features are noisy in practice. Vocal pitch tracks have unvoiced gaps and octave jumps. Drum band energy is a continuous envelope, not discrete hits. Chroma is a 12-dimensional smear. Each feature is useful individually for driving visuals, but none of them provides a clean, musically interpretable signal that the story module or a future arrangement-aware renderer could reason about.

A reduced representation that converts these continuous signals into discrete musical events — notes with onset/offset, drum hits with identity and timing, chord labels per beat — would:

1. **Make section similarity more meaningful.** Comparing sequences of (note, duration) pairs or chord progressions is more robust than comparing raw chroma frames, which is why our SSM-based segmentation sometimes merges verses and choruses that sound similar in timbre but differ in melody.

2. **Enable arrangement-level analysis.** With discrete events we can answer questions like "does the chorus add a harmony vocal?" or "does the bridge drop the drums?" — questions the current pipeline can't address.

3. **Provide a sonifiable intermediate form.** A reduced representation can be rendered as audio (sine waves, simple synth patches, MIDI) — making it audible for debugging and validation without needing the original stems.

4. **Decouple visual design from feature extraction.** The renderer could work from a compact event list instead of frame-level arrays, simplifying the render code and making it possible to design visuals at the musical level (note shapes, chord colors) rather than the signal level (amplitude curves).

**Important constraint:** Timbre is not permanently discarded. It can be re-attached later as a second layer (e.g., spectral envelope, formant features, instrument classification). The reduced representation captures *what notes are played when*; timbre captures *how they sound*. We build the first layer now.


## 3. What the simplified version should preserve

These are the structural elements that must survive reduction — if any of these is destroyed, the representation is too lossy.

### Melody contour
- Which pitch is being sung/played at each moment (quantized to semitones is fine)
- Note onsets and offsets (when a note starts and ends)
- Phrasing: gaps between notes (breath marks, rests) should be preserved as silence, not interpolated away

### Rhythm skeleton
- Beat positions (already available from analysis)
- Drum hit identity and timing: kick hits, snare hits, hi-hat pattern
- Onset density: whether a section is sparse or dense

### Harmonic motion
- Chord identity per beat or half-beat (e.g., "Am", "F", "C", "G")
- Harmonic rhythm: how often the chord changes (once per bar vs. twice per beat)
- Key/mode context if detectable

### Section boundaries and roles
- Already computed by the story module — the reduced representation should be *compatible* with section information, not duplicate it
- But section identity should be verifiable from the reduced representation alone (i.e., two sections labeled "A" should produce similar reduced representations)

### Dynamics
- Relative loudness trajectory (soft → loud → soft)
- Not the exact RMS waveform, but the broad shape: verse is quiet, chorus is loud, bridge drops


## 4. What it can remove

These elements are explicitly *not* required in the first-pass reduced representation. They can be layered back in later.

### Timbre and texture
- Vocal formant quality, vibrato shape, breathiness
- Guitar vs. synth vs. piano distinction in the "other" stem
- Drum kit character (tight snare vs. loose snare, bright cymbals vs. dark)
- Production effects: reverb, delay, distortion, compression

### Stereo information
- Everything is already mono in our pipeline

### Microstructure
- Sub-beat ornamentation, grace notes, pitch bends within a single note
- Hi-hat open/close variation (the DrumSep gives us ride/crash separately, but open vs. closed hi-hat within the hh component is microstructure)

### Exact loudness values
- We keep the *shape* of the dynamics, not the absolute levels
- A reduced representation that says "verse at 0.3, chorus at 0.8" is fine; the exact RMS value at frame 4,217 is not needed

### Lyrics content
- Word timing is handled by the lyrics module. The reduced representation is purely musical.
- However, voiced/unvoiced segmentation from the vocal pitch track implicitly encodes when lyrics are present.


## 5. Candidate simplified-representation designs

### Design A: "Stem event lists" — discrete events per stem

Convert each stem's continuous features into a list of discrete events:

```
vocals: [{onset_s, offset_s, midi_note, velocity}, ...]
bass:   [{onset_s, offset_s, midi_note, velocity}, ...]
drums:  [{onset_s, component: "kick"|"snare"|"hh"|..., velocity}, ...]
other:  [{onset_s, offset_s, chord_label, confidence}, ...]
```

**How to build it:**
- Vocals: already have `note_events` from basic-pitch (MIDI note events with onset/offset/velocity). Also have pYIN pitch track — could be converted to note events via simple onset detection + sustained-pitch grouping.
- Bass: apply same note-event extraction (basic-pitch or pYIN → note events).
- Drums: peak-pick each DrumSep component's RMS envelope → hit events with timestamps and velocity.
- Other: run a chord detection algorithm (e.g., `librosa.feature.chroma_cqt` → template matching, or a dedicated chord model like `autochord` or `madmom.features.chords`).

**Pros:**
- Closest to a musical score / MIDI representation
- Directly sonifiable (MIDI → synth)
- Compact: a 4-minute song might have ~200 vocal notes, ~50 bass notes, ~500 drum hits, ~80 chord changes = well under 1000 events total
- Easy to compare across sections for similarity

**Cons:**
- Note segmentation is imperfect — basic-pitch already exists but has false positives on bleed; pYIN-to-notes requires onset detection logic we don't have yet
- Chord detection is a non-trivial addition (no current code for it)
- Multiple extraction methods means multiple potential failure modes

### Design B: "Beat-quantized feature matrix" — one row per beat

Quantize all features to the beat grid and produce a single matrix:

```
Per beat: [vocal_midi, bass_midi, kick_vel, snare_vel, hh_vel, chroma_0..11, rms]
         → 18 dimensions per beat
```

**How to build it:**
- Use beat times from analysis as the time grid
- For pitch: take the median pitch (in MIDI) within each beat window
- For drums: take the peak energy of each DrumSep component within each beat window
- For chroma: take the mean chroma vector within each beat window
- For dynamics: take the mean RMS within each beat window

**Pros:**
- Single unified representation (one matrix)
- Naturally aligned to musical time
- Good for section similarity (cosine distance between beat vectors)
- Simple to compute from existing features — no new extraction, just resampling

**Cons:**
- Loses sub-beat timing (e.g., syncopated snare hits, vocal melisma)
- Still a matrix, not events — harder to sonify or reason about as "music"
- 18 dimensions is compact but not as interpretable as named events
- Fast passages (16th-note hi-hats) get averaged away

### Design C: "Piano-roll grid" — fixed time resolution, binary activations

A 2D binary (or soft) activation grid at ~100ms resolution:

```
Rows: MIDI notes 24–96 (bass C1 to soprano C7) + 6 drum components
Cols: time steps at 10 Hz (100ms)
Values: 0 or 1 (or velocity 0.0–1.0)
```

**How to build it:**
- Vocals + bass: quantize pitch track to nearest MIDI note, activate corresponding row
- Drums: peak-detect each component, activate corresponding drum row
- Other: could map strongest chroma bins to pitch rows, or leave as a separate chord track

**Pros:**
- Familiar piano-roll format — easy to visualize and debug
- Fixed resolution makes comparison trivial (pixel-level diff)
- Can be rendered as an image for visual inspection

**Cons:**
- Wasteful: most cells are zero (sparse), yet the grid is large (~73 rows × ~2400 cols for 4 min)
- "Other" stem doesn't map cleanly to single MIDI notes
- Loses note identity (which stem is which) unless we use separate grids or color channels
- Hard to sonify cleanly (polyphonic MIDI from a grid needs note-off logic)

### Design D: "Hybrid" — events for pitched stems, beat-grid for drums and harmony

Combine the best of A and B:

```
vocals: note events [{onset_s, offset_s, midi, velocity}]
bass:   note events [{onset_s, offset_s, midi, velocity}]
drums:  per-beat hit vector [kick_vel, snare_vel, hh_vel, ride_vel, crash_vel, toms_vel]
harmony: per-beat chord label + chroma summary
dynamics: per-beat RMS
```

**How to build it:**
- Vocals/bass: note-event extraction (basic-pitch or pYIN → notes)
- Drums: quantize DrumSep component peaks to beat grid
- Harmony: beat-synchronous chroma → template-match chords (or just keep raw beat-chroma for now)
- Dynamics: beat-synchronous RMS

**Pros:**
- Uses the most natural representation for each stem type (events for melody, grid for rhythm)
- Incrementally buildable: start with just drum hits + vocal notes, add harmony later
- Each component is independently useful — you don't need all four to start getting value
- Sonification is straightforward: MIDI for pitched stems, drum machine for hits

**Cons:**
- Mixed format (events + grid) requires careful alignment logic
- More complex schema than a single matrix


## 6. Best first representation to try

**Recommendation: Design D (Hybrid), built incrementally.**

The key insight is that we don't need to build the entire reduced representation at once. Each stem's reduction is independently useful and independently testable. The incremental path:

### Phase 1: Drum hits (lowest risk, highest immediate value)

We already have DrumSep components with clean per-component isolation. Converting RMS peaks into discrete hit events is straightforward signal processing (peak-picking on the RMS envelope of each component). This gives us:

```json
{"drums": [
  {"t": 0.52, "component": "kick", "velocity": 0.9},
  {"t": 1.04, "component": "snare", "velocity": 0.7},
  {"t": 0.52, "component": "hh", "velocity": 0.4},
  ...
]}
```

**Why first:** DrumSep already solved the hard part (component isolation). Peak-picking is well-understood. The result is immediately verifiable by ear (sonify as drum machine hits). And drum patterns are the most reliable structural marker — a chorus usually has a different drum pattern than a verse.

### Phase 2: Vocal note events (medium risk, high value)

We already have two extraction paths:
- `vocals_note_events_basic_pitch()` — produces note events directly, but requires the optional `basic-pitch` dependency
- `vocals_pitch_hz()` (pYIN) — always available, but needs conversion from continuous pitch to discrete notes

Build a simple pYIN-to-notes converter: detect onsets (frame-to-frame pitch change or silence→voiced transition), group sustained same-semitone frames into notes, emit events. Compare against basic-pitch output where available.

### Phase 3: Bass note events (low risk after Phase 2)

Same approach as vocals — `bass_pitch_hz()` already exists, apply the same pitch-to-note converter. Bass lines are typically simpler (fewer notes, longer durations) so this should be easier than vocals.

### Phase 4: Harmony / chord track (higher risk, deferred)

Chord detection from the "other" stem is the least mature part. Options:
- Simple template matching on beat-synchronous chroma (Krumhansl key profiles → nearest triad)
- External library (e.g., `madmom.features.chords`, `autochord`)
- Defer entirely and keep raw beat-chroma as a proxy

This phase can wait until phases 1–3 are validated.


## 7. Concrete plan for the next step

### Immediate goal: `songviz/reduction.py` — drum hit extraction

Build a module that converts DrumSep component WAVs into discrete drum hit events.

**Input:** DrumSep component paths (from `ensure_drumsep_components()`) or fallback to the mixed drum stem.

**Output:** A list of hit events, serialized to `outputs/<song_id>/analysis/drum_hits.json`:

```json
{
  "schema_version": 1,
  "hop_s": 0.023,
  "hits": [
    {"t": 0.52, "component": "kick", "velocity": 0.91},
    {"t": 0.52, "component": "hh", "velocity": 0.38},
    {"t": 1.04, "component": "snare", "velocity": 0.72},
    ...
  ],
  "beat_pattern": {
    "beats": [0.0, 0.52, 1.04, ...],
    "per_beat": [
      {"beat_idx": 0, "hits": [{"component": "kick", "velocity": 0.91}, {"component": "hh", "velocity": 0.38}]},
      ...
    ]
  }
}
```

**Algorithm sketch:**
1. Load each DrumSep component WAV at 22050 Hz mono
2. Compute RMS envelope (hop_length=512, same as existing features)
3. Peak-pick: `librosa.util.peak_pick()` or `librosa.onset.onset_detect()` on each component's onset strength envelope
4. Extract velocity from the RMS value at each detected peak (normalize per-component to [0, 1])
5. Merge all component hits into a single timeline, sorted by time
6. Optionally quantize hits to the nearest beat for the `beat_pattern` summary

**Fallback (no DrumSep):** Run `librosa.onset.onset_detect()` on the mixed drum stem and classify hits by frequency band (reuse the 3-band heuristic), producing coarser `"kick"/"snare"/"hats"` labels.

**Validation:**
- Sonify the hit events as click track (kick=low click, snare=mid click, hh=high click) and listen against the original
- Compare hit count and timing against `librosa.onset.onset_detect()` on the full drum stem
- Run on all 4 available songs to check consistency

**Integration:**
- Called from `pipeline.py:_build_stem_analyses()` when processing the drums stem
- Result stored under `a["features"]["drum_hits"]` alongside existing `drums_bands_3`
- New file: `songviz/reduction.py` (keep separate from `features.py` to signal different abstraction level)

### After drum hits are validated

- **Vocal note events:** Build pitch-to-note converter in `reduction.py`, compare against basic-pitch, integrate into pipeline
- **Bass note events:** Same approach, simpler signals
- **Sonification module:** `songviz/sonify.py` — render reduced representation as audio (sine waves + drum clicks) for listening validation
- **Section similarity upgrade:** Feed discrete events into story.py section comparison instead of raw MFCC/chroma

### What NOT to do in this phase

- Do not reopen separator model experiments
- Do not build a chord detection module yet
- Do not modify the renderer to consume reduced representations yet (that comes after the representation is validated)
- Do not aim for perfect note segmentation — "good enough to hear the melody" is the bar
- Do not add external ML dependencies for this phase (basic-pitch is already optional; drum hit detection should use only librosa)
