# First restart review — 2026-09-09

Status: first user review received; Milestone 1 complete. Raw feedback is preserved
in `benchmark/feedback/restart-02.json`, verified against the manifest SHA-256
`2ec2e4c72b2a4b007b8e315c8024785cf4a682ca679df3f9394a942761558a20`.
See `08_rhythm_feedback.md` for triage and the controlled rhythm comparison.

## Review package

Open `outputs/reviews/restart-02/index.html` locally, or serve that directory:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory outputs/reviews/restart-02
```

Then open `http://127.0.0.1:8765/`. The page buffers short clips when served over
HTTP so seeking works even with the standard-library server. The page also works
as a local HTML file. Notes must be downloaded before reloading, or provided in
chat; they are not automatically written into the repo.

| Clip | Original-song interval | Review focus |
| --- | --- | --- |
| 01-opening | 0–22s | Pulses and first instrument entries |
| 02-transition | 130–148s | Whether visual intensity fits the musical change |
| 03-return | 158–178s | Timing and character of the next change |

All three are from the local Feel Good Inc file. They are exploratory development
examples, not a holdout set or evidence of generalization. The soundtrack is the
original audio. Each video combines the current mix renderer with a diagnostic
panel: measured mix/drum/bass RMS, cached raw drum events, beats, and section
boundaries. Purple boundaries and displayed roles are algorithm predictions.
RMS uses one shared scale within each excerpt; separate excerpts have different
scales. Stem energy is not proof of independent instrument activity.

Optional audio players contain the same excerpt's separated drums, separated
bass, and pitched clicks at the raw detected drum times. Players are independent
and start at the excerpt beginning; only one plays at a time. Clicks intentionally
avoid sonification's quantization, velocity floors, and bass octave shifts.

## What this baseline establishes

- The source audio SHA-256 matches the cached analysis song ID and Demucs input
  metadata. This verifies the associated source file, not stem fidelity.
- Visuals are rendered now from cached analysis; extraction has not been rerun.
- The selected cached analysis was created on 2026-05-01 UTC. Its embedded story
  matches the separate story file. Its extraction commit/configuration is unknown.
- Cached reduced data reports drums=`heuristic`, bass=`crepe`, vocals=`basic_pitch`.
  This differs from older notes describing the primary template drum path.
- The historical cached `reduced.wav` is not used for these comparisons because
  its relationship to the current reduced JSON is unverified.
- Source/configuration hashes, source-code snapshots, input JSON snapshots,
  current render settings, and output hashes are in the review manifest/inputs.
  Cached byte snapshots permit baseline reconstruction without pretending the
  old extraction process itself is reproducible.

## Reference audit — unresolved, not silently corrected

1. **Conflicting instrument-entry descriptions.** `feel-good-inc/sections.json`
   describes 0–12.7s as keyboard only, with no drums/bass. `drums.json` instead marks
   drums active from 2.5s; `vocals.json` marks vocals active from 5s. The references
   cannot all support the broad "keyboard only" description as written.
2. **Inconsistent pitch expectations.** `bass.json` describes F1 in the riff but
   omits pitch class F (5) from `scale_pcs`. Its MIDI reference includes F (5) and
   C-sharp (1), both outside that array. An in-scale score cannot establish correct
   transcription while these expectations disagree. The dominant pitch class
   also need not equal the harmonic root.
3. **MIDI timing is unverified.** Drum/vocal reference MIDIs declare roughly 84
   BPM; the cached audio beat estimate is about 92.29 BPM. The bass reference
   declares 120 BPM and ends at 222.39s, beyond the source duration of 221.17s.
   MIDI tempo metadata alone does not prove misalignment (timestamps may have
   been retimed), but these files have no verified alignment record here. Treat
   onset scores as provisional until attacks are compared to the actual audio.
4. **Circular/mixed structural references.** Feel Good Inc and Do I Wanna Know
   cite listening plus SSM analysis; Die For You explicitly derives its boundaries
   from SSM/energy without independent listening. Confidence labels do not prove
   independence. All current section aggregates are diagnostic, including silver.
5. **Cached output does not confirm old completion claims.** Feel Good Inc has
   eight cached sections versus seven reference sections. The cached 137.86–165.67s
   interval is labeled `build`, despite earlier notes saying it was corrected to
   `valley`. No new perceptual verdict on that interval is claimed here.

Keep existing references unchanged until the relevant excerpt is reviewed. Record
the reason and provenance for any later amendment; do not change a reference to
improve an algorithm's score.

## Reproduction and next step

### Technical verification and cached evaluation

- 74 benchmark/evaluation tests and 5 review/render tests passed. Source-clock
  tests cover excerpt offsets, sample preservation, stereo RMS, and unquantized
  drum timing. No ML extraction was needed for these checks.
- All three videos contain both audio and video streams of the expected duration
  (22/18/20 seconds). Cut PCM samples match the source exactly. Decoded AAC has
  zero-offset source correlation above 0.996 in each clip; output checksums match
  the manifest. These checks establish technical alignment, not perceived quality.
- Browser checks verified all three durations, seeking, conversion of excerpt
  2.25s to original-song 132.25s, and feedback JSON linked to the manifest hash.
- Cached evaluation completed for all five benchmark songs, with sections included
  and no run errors. Results and reproducibility inputs are in
  `outputs/reviews/restart-02/evaluation/`. Comparing that result with itself
  reports no regressions, improvements, or incomparable cohorts.

| Cached song | Section boundary F1 ±3s | F1 ±0.5s | Reference label |
| --- | --- | --- | --- |
| Feel Good Inc | 0.9231 | 0.9231 | silver, mixed listening/SSM |
| Do I Wanna Know | 0.8571 | 0.7143 | silver, mixed listening/SSM |
| Feeling This | 1.0000 | 1.0000 | silver |
| Shy Away | 0.7692 | 0.6154 | silver |
| Die For You | 0.9231 | 0.7692 | bronze, inferred |

All scores above are diagnostics against the existing, unaudited references.
They supersede neither independent listening nor a current extraction run.
For Feel Good Inc, coarse drum activity F1 is 1.0000 while MIDI onset F1 is
0.2537; bass onset F1 is 0.1373 and vocals 0.0848. MIDI alignment is unverified,
so these discrepancies motivate inspection rather than a claim of measured
transcription accuracy.

### Commands

Generate into a new directory (the builder refuses to overwrite a review):

```bash
.songviz/venv/bin/python experiments/build_review.py --out outputs/reviews/restart-next
.songviz/venv/bin/python experiments/evaluate_cached.py --out outputs/reviews/restart-next/evaluation
```

The second command checks that every benchmark song has cached reduced data and
refuses to start extraction. It writes results, a report, and evaluation input/code
snapshots separately from the perceptual review manifest. Reference independence
and historical extraction revision remain unknown.

The first review supplied observations for all three clips: unexplained pulse
irregularity, apparent off-beat timing, and unreadable drum-hit indicators. These
establish the first concrete acceptance criteria without requiring more detailed
user annotation upfront. The next review isolates pulse timing and gives kick,
snare/clap, and hi-hat their own indicators; see `08_rhythm_feedback.md`.
