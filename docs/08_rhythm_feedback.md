# Rhythm feedback and controlled candidate — 2026-09-09

Status: second review received. Written feedback supports the regular pulse's
alignment and separate percussion indicators on these development excerpts.
The user clarified in chat that the confusing pulse was the cached version,
not the regular candidate. No production analysis, stems, reference
annotations, or reduced caches were regenerated or overwritten.

## Second review: interpretation and next step

`benchmark/feedback/rhythm-review-01.json` preserves the second export verbatim
from `songviz-rhythm-feedback (1).json`. Its manifest hash matches the selected
rhythm comparison below; all three observation timestamps belong to their clips.
The original manifest remains immutable and still records review as pending at
generation time. This feedback and triage record the subsequent review state.

The written opening and transition notes explicitly say the **regular pulse
follows the song beats very well**. The opening also says kick, snare and hi-hat
appear to show what is happening, with snare the easiest to clap along with. The
return note refers back to the previous answers rather than adding a specific
event correction. This is positive local perceptual evidence, not independent
ground truth, full-song approval, or evidence of generalization.

Two qualifications must survive future summaries:

- All preference dropdowns were left `unreviewed`. Record the positive written
  judgments, not three explicit A/B preference votes; `variant_viewed=regular`
  only records the selected variant at export.
- The opening's initially ambiguous "the pulse" complaint referred to the
  **cached version**, as the user subsequently clarified in chat. See
  `benchmark/feedback/rhythm-review-01-clarification.md`. The regular candidate
  was not the target of that complaint; no further clarification is needed.

The next step at the time of this review was one artistic opening preview using the existing
candidate timing and independent percussion. Keep a subdued base pulse, make
snare/clap the clearest accent, and distinguish kick motion from lighter hi-hat
detail. The user authorized building this passage with cheaper-model delegation;
the particular visual design is still a proposal, not an accepted design. Retain this diagnostic comparison as a
regression reference. Do not retune the grid just because snare is easier to clap
to, promote the constant-tempo experiment to a general tracker, or mark Milestone
2 complete before an actual visual passage is reviewed. Reserve other songs for
later generalization checks before tuning against them.

Update 2026-09-10: that opening preview was built (`09_visual_passage.md`). The
subsequent user clarification redirects the next implementation to an automatic
visual-directing prototype, not further tuning of fixed instrument effects. See
`01_roadmap.md` for the current queue and `02_architecture.md` for open decisions.

## Feedback and acceptance criteria

`benchmark/feedback/restart-02.json` preserves the user's downloaded feedback
verbatim. Its manifest hash matches `outputs/reviews/restart-02/manifest.json`.
The observation timestamps identify reviewed moments or clip starts; they are not
precise annotations of every erroneous beat. Keep interpretation here, not in the
raw feedback file.

| Passage | Observation | Concrete next check |
| --- | --- | --- |
| Opening, 0–22s | Pulse feels disconnected, not merely delayed; drum lines are unreadable | A repeatable clap-along pulse; distinct component indicators |
| Transition, 130–148s | Apparent systematic offset plus slight interval changes | Compare grid phase and spacing against sharp source attacks |
| Return, 158–178s | No obvious kicks/claps initially; pulse seems to speed up then slow | Keep pulse separate from drum activity; inspect local interval variation |

These observations were specific enough to proceed without asking the user to
annotate individual attacks. They combine a technical timing issue with a visual
legibility requirement. The preferred pulse level and perceived alignment still
need listening; detector measurements do not resolve those choices alone.

## Measured evidence

`experiments/inspect_rhythm.py` reads the existing analysis and DrumSep snare/hi-hat
audio. Strong snare transients repeat about every **866.273ms**. Fitting that
repetition on 30–110s gives a two-pulses-per-anchor candidate of **138.525 BPM**
(433.136ms), versus the cached estimate of about **92.285 BPM**.

| Reviewed interval | Cached adjacent pulse range | Cached nearest-snare median | Candidate nearest-snare median |
| --- | --- | --- | --- |
| 0–22s | 534–859ms | 175.78ms | 0.52ms |
| 130–148s | 557–673ms | 180.50ms | 0.99ms |
| 158–178s | 534–673ms | 170.70ms | 1.81ms |

These are timing residuals to independently detected **prominent separated-snare
attacks**, not beat accuracy, audible synchronization scores, or complete drum
ground truth. Nearest-grid residuals are affected by pulse density. The important
additional evidence is consistent anchor repetition and phase, not the residual
improvement alone. Every second candidate pulse is the prominent snare position;
intervening pulses need not contain a snare. A 69.262 BPM half-time interpretation
is also possible. No bar/downbeat position is inferred.

The fit uses 70 anchors with 95.7% within 30ms. Outside its fitting interval,
median anchor residual is 1.40ms and p95 is 47.64ms; extra/ghost attacks remain.
The reviewed passages lie outside the fitting interval, but are development data,
**not untouched holdouts**: full-song diagnostics and feedback informed this
experiment. Global source thresholds also inspect the whole track. We have not
established generalization to another song, variable tempo, or live drumming.

Waveform and spectrogram close-ups show the separated snare's transient evidence.
The lead viewed the close-up and return plots, and a Luna worker inspected both
independently. Its review agreed on regular proposed spacing and irregular cached
spacing, while correctly declining to infer sustained musical acceleration from
the broad plot. Component bleed/separation latency and exact perception remain
unverified. A whole-song picture cannot resolve millisecond timing by itself.

## What changed for this experiment

- `songviz/beat_grid.py`: experimental constant-pulse fit with explicit metrical
  assumptions and rejection when evidence is sparse or insufficiently regular.
  It is **not wired into the production beat tracker**.
- `songviz/percussion.py`: separate component attack candidates, without using or
  snapping to a beat grid. It retains conservative `hits` separately from weaker
  `faint_hits`, with raw RMS/strength and threshold metadata, not confidence claims.
- `experiments/build_rhythm_review.py`: six 60 FPS clips, two variants of each
  original excerpt. Only the pulse grid differs between variants. Audio, layout,
  and percussion events are identical. Kick, snare/clap and hi-hat have separate
  indicators; the pulse can continue while percussion is quiet. This diagnostic
  layout is not a proposal to replace the artistic renderer with a signal chart.
- `experiments/templates/rhythm_review.html`: same-position switching, isolated
  component audio, evidence images, and feedback export linked to the new manifest.

A Terra worker implemented the bounded percussion sidecar and tests. Lead review
found that its initial detector over-reported RMS fluctuations, including 62 snare
candidates during the quiet interval. That version was not used for the review.
After smoothing, local prominence gating and stronger realistic tests, the
rendered prominent set has zero kick and zero snare hits during 137.856–165.674s,
with 26 hi-hat candidates. Weak evidence remains inspectable in `faint_hits` rather
than silently becoming visible drum hits. This is a conservative display choice:
quiet real hits may be omitted and separator labels can be wrong.

## Review and reproduce

Open `outputs/reviews/rhythm-review-01/index.html`, or serve it:

```bash
python3 -m http.server 8766 --bind 127.0.0.1 --directory outputs/reviews/rhythm-review-01
```

Open `http://127.0.0.1:8766/`. Try the opening first. Switch pulse variants and report
which is easier to clap along with, then whether kick and snare/clap are legible.
Use **Download feedback** or reply in chat. Notes are not persisted until exported.
This is a labeled diagnostic comparison, not a blinded preference experiment.

Generate new packages without overwriting existing ones:

```bash
.songviz/venv/bin/python experiments/inspect_rhythm.py --out outputs/reviews/rhythm-evidence-next
.songviz/venv/bin/python experiments/build_rhythm_review.py --evidence outputs/reviews/rhythm-evidence-next --out outputs/reviews/rhythm-review-next
```

Selected manifest SHA-256:
`2f9aae4e87c179bfeca0994056fb64e1f9a590a74d43587e128a62f31de0dfe8`.
It records input/code snapshots, source hashes, output hashes, comparison scope,
and pending user review. `timing.json` preserves the fitted grid and measurements.
Local audio/video and generated figures remain under ignored `outputs/`.

## Verification and next decision

- 91 focused tests passed: beat grid, percussion, controlled renderer, original
  review clock, benchmark and evaluation. Synthetic checks include drifting tempo
  rejection, missing/extra anchors, real sample-hop timing, bipolar drum decays,
  background noise, independent simultaneous hits and unchanged percussion when
  only the pulse grid changes.
- All six videos have 60 FPS and expected 22/18/20s durations. Their source PCM
  cuts exactly match the original file. Decoded A/B audio is identical per clip;
  zero-offset source correlation is 0.999311, 0.998385 and 0.996397 respectively.
  Output checksums match the manifest. Frame sampling still quantizes visual
  events by up to 16.67ms; sub-2ms grid residuals are not sub-2ms display precision.
- Browser checks covered all durations, paused/playing same-position switching,
  seeking, original-song timestamp conversion (2.25s → 132.25s), separated-audio
  loading, and feedback manifest linkage. Automated export data was intercepted
  for inspection, not saved as user feedback.

If the pulse is perceptually better, integrate that behavior into one visual
passage and reserve new songs before further tuning. If it still feels wrong,
first distinguish pulse level/phase from event display latency and instrument
omissions. Do not add harmony or replace the separation stack on the basis of
this local timing experiment. Milestone 2 remains open until the user judges the
passage, regardless of how regular the candidate looks to an agent.
