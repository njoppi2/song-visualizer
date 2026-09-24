# Transition evidence from existing components

Team 1, 2026-09-22. The raw Agnes listening judgment is authoritative; the earlier
claim of a definite detector miss was too strong. Doc 31 corrects the timing
interpretation without changing feedback or frozen predictions.

## Fixed diagnostic design

Run the existing `songviz.local_structure.detect_local_structure` control and
`songviz.local_structure_variants.detect_local_structure_variant` with
`variant="separate_channels"`, using their unchanged defaults. Both already
existed before the Agnes feedback. Doc 18 contains the earlier development
comparison: separate-channel thresholds add many proposals and were not adopted
as a salience model. This is application to a newly judged passage, not a new
algorithm or another parameter search.

Use SHA-256-verified CQT/RMS caches from `structure-evaluation-03` and
`arrangement-continuity-reserve-features-01`, verifying their bound sources.
Compare the recomputed development control with the frozen `local-structure-02`
prediction data. No source extraction, model inference, audio upload or paid
compute. Refuse to overwrite output directories and preserve code/config/input
fingerprints in the new `transition-components-01` package.

Keep spectral-pattern contrast and RMS contrast separate at 2, 4 and 8 beats.
For each of the four existing development excerpts and Agnes's 24-second clip,
report nearest-reference-anchor component scores, each component's maximum and
all tied maximum locations within the excerpt, and the unchanged policies'
candidate times and feature support. The development prompts/fixed anchors are
not precise events. Agnes's approximate 12-second point is only a visual guide.
For Agnes, report only anchors whose full ±8-beat support fits the original
93.294263–117.294263s reserve; any reported candidate's full support must also
fit. Whole-track computation is necessary for unchanged thresholds and floors,
but do not expose other Agnes predictions.

Overlay already-frozen level-only anchors to show the relationship to doc 31.
Do not score an overlap as a correct event, an empty nearest anchor as a miss,
or a larger contrast as greater musical importance. Log-CQT contrast is not
perfectly gain-invariant, section identity, melody identity or a probability.
Feature support describes input context, not detected transition duration.
This is an exploratory comparison on seen data; no matching tolerance, new
threshold, automatic section naming or production promotion is introduced.
The Agnes beat grid still lacks an independent source hash; association rests
on cache location and verified source-bound stem metadata. Beat accuracy is
unvalidated. Beat pooling and CQT frame/window support limit timing precision.

## Decision boundary

Inspect whether existing evidence supplies a useful candidate near the newly
reported change and whether the non-change passage also attracts candidates.
If the evidence is present but the distinction is poor, the next problem is
selection of musically important changes, not another detector for one excerpt.
If it is absent, document which representation is weak before selecting a new
one. Do not claim generalization from these five seen passages.

## Completed diagnostic — 2026-09-22

Accepted execution: `outputs/reviews/transition-components-01/`. No policy is
promoted. The default development result reproduces the frozen control exactly
after JSON normalization, including all curves and candidates. The control has
16 changes plus two dips; separate channels has 39 changes plus the same two
dips. These are existing results, not new improvements.

The Agnes control reports no candidate within the eligible reserve. Separate
channels reports one pattern candidate at song **98.382948s**, elapsed **5.088685s**,
supported by 2/4/8-beat responses. It does not select a candidate near the human's
approximate elapsed 12s. The prior level-only anchor remains elapsed 10.336395s.
No exact hit/miss tolerance is introduced by this diagnostic.

At the nearest beat to the human point (song **105.395374s**, elapsed **12.101111s**):

| Window on each side | Pattern contrast | RMS contrast |
| --- | ---: | ---: |
| 2 beats | 0.348645 | 0.119867 |
| 4 beats | 0.296116 | 0.226486 |
| 8 beats | 0.206381 | 0.303787 |

The 2-beat bass/other pattern contrasts are 0.343756/0.353534, while their RMS
contrasts are 0.109810/0.077889. Drums have very little pattern contrast
(0.000695). Thus the saved representation contains spectral-change evidence
beyond volume near the reported transition. It does not identify a verse,
pre-chorus, melody change or important transition automatically.

The decisive counterexample remains the human's non-change passage. Separate
channels proposes **five** changes in 16–26s, including one at 20.196880s.
Its 2-beat pattern maximum is **0.702409**, versus **0.569821** for the Agnes
eligible excerpt; at the fixed non-change inspection anchor the 2/4-beat values
are 0.363154/0.465729, also above the Agnes reference-anchor values. Raising raw
pattern strength to musical importance would preserve this contradiction.
Neither control nor separate channels proposes a change in the drum-entry
excerpt, where the frozen level rule supplies two anchors. The broad breakdown
has three changes and one dip under both policies; the verse-ending excerpt has
one separate-channel candidate at 121.983912s. Candidate counts are not counts
of independent musical events or automatic false positives.

Conclusion: useful acoustic components exist, but the present selection rules
do not establish which changes deserve a major visual response. Do not add a
new detector or lower thresholds to recover this one Agnes example. Keep the
failed continuity gate rejected and the laughter/model branches parked.

Validation actually performed:

- **8 focused tests passed**, one existing ddtrace warning. Tests cover source
  mismatch/no overwrite, differing manifest path roots, independent component
  extrema/ties, full support and common reserve masking, and real detector schema.
- Lead independently recomputed **60 per-stem records** across five reference
  anchors and three scales, checked their top-two aggregates/support bounds,
  all seven output hashes, and both reserve output masks.
- The five-panel plot was visually inspected. Dotted black lines are inspection
  guides; purple vertical lines are the frozen level-only anchors. Their overlap
  with a curve is not a validated event match.
- The first runner attempt stopped before output creation because dictionary
  source paths were incorrectly resolved relative to the package. The lead
  corrected source/output roots and added a regression test. The subsequent
  execution completed. Frozen source packages were never edited.

| Artifact | SHA-256 |
| --- | --- |
| `transition-components-01/manifest.json` | `b5d10e0deacb69485de860b189abb684cc79ceadb204b4534ac2063da7e4d79d` |
| `transition-components-01/records.json` | `db1e42b75319a494b201df9dc76cbb8cc1c867997a296f98c11464e5642871b6` |

```bash
.songviz/venv/bin/python -m pytest -q tests/test_transition_components.py
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 60s \
  .songviz/venv/bin/python experiments/diagnose_transition_components.py \
  --repo . --output outputs/reviews/transition-components-01
```

The completed directory is frozen: use another directory if a justified future
run is necessary. The copied `design.md` records the design before results.

## Next reference

Prepared one original 24-second excerpt from the middle of the cached Arctic
Monkeys track using only source duration, without inspecting its predictions.
The human has been asked whether the music continues the same idea or noticeably changes, roughly
when and how. Approximate prose is sufficient. Existing methods/configuration
are already frozen; no new rule is chosen from the answer. This is an additional
song reference before designing a musical-importance comparison. Historical
exposure is unknown, so it is not a certified untouched holdout. Record the
answer verbatim in a new intake artifact before selecting another experiment.

Package: `outputs/reviews/transition-components-listening-01/`. Source frames
[5,477,090, 6,535,490), 44,100 Hz stereo PCM-24, song bounds
124.197052154195–148.197052154195s. The lead independently verified exact int32
PCM equality to original source frames, 24-second length, source/clip/code/README
hashes and frozen method binding. Actual Chromium native playback passed with
duration 24, paused false, readyState 4. The existing local review server serves
the clip on port 8770; no new UI was built.

Clip SHA-256: `e2d60d10437f6b79664f155b6302679dc153aa056453ff731e61f4bd842df5ef`.
Manifest SHA-256: `df1f8f56b148501e4cffcdd0d179edf8e78d0a4d50ad0d2cee5c69b050f43ac5`.
The manifest is immutable and records feedback absent at creation; any eventual
response belongs in a new intake artifact, not an edit to this frozen package.

## Arctic Monkeys feedback and fixed-method check

Raw feedback received: "it kinda of the same idea, but at aroun 17s more elements
are introduced". The exact message is saved in
`outputs/reviews/transition-components-arctic-intake-01/human-feedback.json`.
This supports a listener-described arrangement addition within a mostly
continuing musical idea, at approximate song time **141.197052s**. It does not
identify the added instruments, establish precise onset or rate visual importance.
The earlier Agnes new-part interpretation remains a separate type of judgment.

Before examining this track's candidate outputs, registration fixes the already
implemented control, separate-channel and sustained-activity policies, with
unchanged defaults. The sustained-activity policy is the doc-18 method, not the
doc-31 two-scale/two-anchor persistence rule or rejected continuity gate. No
semantic continuity classifier is introduced. The human judgment is evaluation
context; methods predate it, but this run occurs after feedback and is not a
blinded prediction. Do not interpret a nearest timestamp as a validated match.

Reuse the original source-bound cached stems and cached analysis beat grid,
fingerprinted in `registration.json` alongside the unchanged extractor and
detector modules. Full-track CQT/RMS extraction is necessary for unchanged
floors/thresholds; report only the original 24s excerpt's shared ±8-beat eligible
anchors and candidates with full support inside the excerpt. Save features for
reuse. The cached grid's independent source binding/beat accuracy remain
unverified; source-bound stem metadata plus current byte hashes document cache
association, not independently certified separation lineage. No new separation,
learned model, paid service or source-audio upload is part of this check.

### Completed Arctic comparison — reviewed 2026-09-23

The run completed before the usage-limit interruption; it was not rerun on
resumption. Accepted numeric package:
`outputs/reviews/transition-components-arctic-arrangement-01/`.
The unchanged sustained-activity policy proposes an `other`-stem increase at
song **141.525624s**, elapsed **17.328571s**, supported at both 4 and 8 beats.
This is consistent in timing and direction with the listener's approximate
17s addition. It does not automatically establish a named instrument, exact
onset, musical continuity, section identity or visual importance.

At that anchor:

| Per-side window | Other-stem median RMS before → after | Relative level contrast | Left/right persistence |
| --- | --- | ---: | --- |
| 4 beats | 0.001701 → 0.039290 | 0.956700 | 100% / 100% |
| 8 beats | 0.001296 → 0.042075 | 0.969208 | 100% / 100% |

The before medians lie below the heuristic activity floor (0.004131), which
explains the policy's `entrance` descriptor. This is not proof of perceptual
silence before the change. Drum/vocal median level contrasts are only
0.0423/0.0347 at four beats and 0.0170/0.0276 at eight beats. Their spectral
pattern contrasts are also small. That supplies background-continuity evidence
compatible with the human's description; it is not a trained identity judgment.
The bass activity estimate depends on scale and does not qualify as a persistent
change at either scale. Do not infer that bass is the added element.

All three policies share **18 eligible anchors** in this clip:

| Existing policy | Candidate elapsed times (s) |
| --- | --- |
| Control | 13.103, 15.216 |
| Separate channels | 13.103, 15.216, 16.609 |
| Sustained activity plus control | 9.573, 13.103, 15.216, 17.329 |

There are no dip proposals in the reported window. The earlier extra candidates
are unscored, not automatically false positives. The sustained candidate's full
input support is song **135.883175–147.168073s** (elapsed **11.686–22.971s**),
not a detected transition duration. No matching tolerance or precision/recall
score was introduced after seeing the response.

This adds a useful continuing-idea arrangement case on another song. It supports
retaining the existing level/activity evidence separately from spectral change;
it does not justify promoting either to a universal section detector. Together
with Agnes and the original four cases, there are now six seen listening
references across three tracks. The next bounded design task is to define
separate acceptance criteria for layer additions, new-part changes and ordinary
variation across all six, before introducing another decision rule. No new
listening task or large-model run is required for that design.

Validation: **5 focused tests passed**, one existing ddtrace warning, before the
successful run. The lead independently verified 14 input and 10 output hashes
plus registration, 12 per-stem spectral/mean-level records, eight per-stem
median/persistence records, their aggregates, all three output masks and complete
candidate supports. Features contain 380 intervals, 84 CQT bins per stem. The
CPU extraction/comparison/plot stage took **5.924s**, with process peak RSS
**510,532 KiB**, using two threads. Cached-grid provenance limitations remain.

Review found a plot-only issue: the variant API retains auxiliary activity
curves even under separate-channel selection, so the first figure showed an
unused curve in that panel. The corrected figure and code snapshot are in
`outputs/reviews/transition-components-arctic-plot-02/`; generated solely from
saved records, with no feature extraction or detector rerun. Original plots
remain frozen. The runner now limits that curve to the activity policy and
labels all panels. Registration byte-count support and source metadata binding
were also checked during lead review before the successful run.

| Artifact | SHA-256 |
| --- | --- |
| Intake feedback | `7f7580eaccdf04308299d47b1b3ee3212850d6580a3d04aecf3c3ae538d40950` |
| Registration | `f89df8cdc5733dd100e46aee4dbcfc9a07d088adcdbfcf8fa0fa8cbd46366384` |
| Numeric package manifest | `29935b555b8d5d22111ff05c8ebbe2a5a270de4529c01e572901dafbb7c78d2e` |
| Features | `0c474eaad61528ed8f1bfac6e18a9900cb3b2ee60254cc96878b745381074c64` |
| Records | `46d801c27399b54aebaae2a7be5936947b503aaa77f75228d57669c98e375b16` |
| Corrected plot manifest | `383c728f6426520e4946b78e747a0367b5b2b4933e3480f32c46d9285676a7a7` |

```bash
.songviz/venv/bin/python -m pytest -q tests/test_arctic_arrangement.py
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 timeout 180s \
  .songviz/venv/bin/python experiments/evaluate_arctic_arrangement.py --repo . \
  --registration outputs/reviews/transition-components-arctic-intake-01/registration.json \
  --output outputs/reviews/transition-components-arctic-arrangement-01
```

These commands document completed work, not instructions to overwrite/repeat it.
