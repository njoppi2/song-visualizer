# Arrangement change with continuing material

## Fixed experiment — 2026-09-20

The human authorized the broader arrangement/continuity task after parking the
laughter-specific branch. This experiment adds one missing distinction to the
existing sustained-level detector: a changing layer alongside other layers whose
ordered acoustic patterns and levels remain similar. It is an automatic candidate
generator, not a claim that acoustic continuity proves musical identity or importance.

Use saved, source-bound beat CQT/RMS features from `structure-evaluation-03`.
Do not feed reference notes, section identities or named examples into inference.
Run the entire existing numeric grid once; join the four development excerpts
only after predictions are frozen. No new model, separation, annotation page or
production detector changes. Team 1 owns design/review; Terra implements.

## Prospective reserve, before candidate scores

Reserve the middle 24 seconds of Agnes — MILK: song seconds
**[93.294263,117.294263)**, selected from duration metadata alone. Neither audio,
feature values nor event judgments from this passage informed these rules. It is
prospectively withheld from this experiment's design/tuning; absence of prior
listening elsewhere in the project is not proven. Existing cached stems may be
used only after source association and hashes are checked. Record any provenance
gap rather than silently treating an old cache as verified. No reference labels
are available for this passage, so frozen predictions alone cannot establish
accuracy/generalization. Never substitute a more favorable passage after results.

Metadata inspection found no bound Agnes CQT/RMS cache. Its `stems.json` binds
the existing Demucs htdemucs outputs to source SHA-256
`ac23d00df4892e03b8df1696927d1b066bcc147122112b7154239ea3dbb43fbb`.
Freeze a fresh CPU-only feature extraction of the four full cached stems using
their stored analysis beat grid: mono 22,050 Hz, hop 512, absolute CQT with 84 bins
from C1, log1p of mean CQT per floor-quantized beat interval; mean frame RMS with
frame length 2048. These match the development extractor. The full-track numerical
extraction preserves the whole-track audibility floor; only predictions whose
entire two-anchor/two-scale support lies inside the fixed reserve are reported.
Do not inspect or use scores outside the reserve. Save requested/quantized timing,
source/stem/meta/analysis hashes, actual versions and extractor snapshot. The
cached beat grid has no independent source-hash record; its association relies
on the same-song cache directory and verified stem metadata, and beat accuracy
is unvalidated. Centered/windowed features are offline support, not exact onsets.

## Fixed rules and ablation

Use half-windows of 4 and 8 complete beats at every supported anchor. For each
stem, audibility floor is the existing offline `max(0.02 * track_max_rms, 1e-8)`.
Retain this dependency and the full left/right source support in every output.
The preexisting beat grid and its phase limitations remain input assumptions.

A sustained layer-level change uses the existing rule: at least one side's median
RMS exceeds its floor; `1 - min(median_left, median_right)/max(...) >= 0.5`;
at least 75% of each side's beats lie on their respective side of the midpoint
of the two medians. Preserve increase/decrease and whether the median crosses
the activity floor. Crossing that proxy floor is not proof of physical onset.

An unchanged layer supporting continuity must meet all of:

- At least 75% of beats on each side exceed the audibility floor.
- Median-level relative difference is at most 0.25.
- Ordered cosine similarity is at least 0.90 on flattened, corresponding
  frequency-by-beat windows, after `expm1` reverses the saved log1p CQT.
  Nonpositive norms provide no evidence; shared silence never supports continuity.
- It is not a qualifying changed layer at that scale.

Two or more unchanged stems, including at least one of bass/other, are required.
This checks shared background pattern evidence, not whether a solo, vocal melody
or formal section has the same identity. A common repeating accompaniment can
continue across a meaningful section change; that failure mode stays explicit.

Both comparison arms use the same persistence rule: the same changed stem and
direction must qualify at both scales at the anchor and its next adjacent anchor.
Only anchors whose next sample supports both scales are evaluated. The level-only
arm ignores continuity. The candidate additionally requires the unchanged-layer
rule at both scales at both anchors. This isolates the effect of the added gate.
Retain each scale/anchor record and every accepted/rejected stem. No selected
peak suppression, threshold search, best separator selection or human-label input.

Outputs keep `level_change`, `continuity_support` and `arrangement_candidate`
separate. No candidate is an explicit negative result only for this operational
gate: it does not assert no perceived development. Pattern support is a hypothesis
about continuation, not a section label. Multiple simultaneous changes and
withdrawals remain visible even when the continuation gate rejects them.

## Acceptance and stopping

Evaluate the four existing cases descriptively, including all anchors in their
fixed focus bands and their full excerpts. Declare before reading predictions:

1. Drum-entry focus: retain at least one drum-increase candidate with continuity
   support. Do not assert bass entry or resolve verse/chorus naming by inference.
2. Within-passage focus: zero arrangement candidates. Report whether this improves
   on level-only false proposals; matching zeroes are not an improvement.
3. Retain all results for the broad transition and vocal-change excerpts. Their
   musical extent/behavior requirements remain unscored by this detector, and
   rejection there is not evidence that nothing happens.
4. Both 1 and 2 are needed for narrow development success. These are seen cases;
   even a pass is not a generalization or musical-importance result. Existing
   benchmark rubrics are not rewritten to award broader semantic credit.

Synthetic tests must exercise layer increase with unchanged patterned background,
shared silence, uniform gain changes, and changed/reordered background with matched
energy; verify that invalid/nonfinite/misaligned inputs are rejected. These verify
the mechanism, not real-song accuracy. Independently reproduce source/output
fingerprints and numerical decisions. Export a small result table and saved-score
plot, without building another application. Stop after the fixed comparison;
if it fails, record which assumption failed before proposing any next experiment.

## Implementation and evidence

Implemented. Owned implementation paths are
`experiments/arrangement_continuity.py`,
`experiments/run_arrangement_continuity.py`,
`experiments/extract_arrangement_reserve.py`,
`tests/test_arrangement_continuity.py`, and fresh
`outputs/reviews/arrangement-continuity-*/` packages. Root updates this experiment
record and `CONTINUE.md`; frozen source packages remain unchanged.

## Completed comparison — reviewed 2026-09-21

**The added continuity gate fails the fixed development screen and is rejected
for promotion.** One valid comparison completed in
`outputs/reviews/arrangement-continuity-01/`. Predictions were written and
fingerprinted before the evaluator opened the benchmark. Thresholds, persistence
rules and the reserved passage were unchanged. The account limit interrupted the
final handoff, not the completed computation; the human requested continuation
after reset. No extraction or comparison was repeated to resume.

| Existing focus | Eligible anchors | Level-only proposals | With continuity gate |
| --- | ---: | ---: | ---: |
| Within-passage non-change | 4 | 0 | 0 |
| Drum increase | 4 | 2 | 0 |
| Verse ending | 6 | 0 | 0 |
| Broad transition | 8 | 5 | 0 |

The drum-increase criterion fails. Both arms are quiet at the non-change focus,
so there is no improvement there. Over the full development grid, 482 anchors
were eligible: 50 level-only proposals and zero gated proposals. These are
overlapping anchor counts, not counts of independent musical events. The broad
transition's extent and the vocal-change interpretation remain unscored. Zero
proposals there do not establish absence of development.

The prospective Agnes reserve has 38 eligible anchors, one level-only proposal
and zero gated proposals. It has no human reference, so those counts are **not
an accuracy or generalization result**. The recorded predictions are now seen;
the passage cannot be treated as an untouched reserve for later tuning.

### Why the known drum increase was rejected

The level-only arm retained drum-layer increases at anchors 169/170
(78.670281s and 79.103417s). The following are numerical findings from their
saved 4/8-beat and neighboring-anchor records, not a new listening annotation:

- Bass falls below the operational activity floor in the relevant windows; it
  cannot supply the required unchanged-layer evidence. This is not proof of
  physical bass absence or resolution of the user's uncertainty about bass entry.
- The `other` stem remains active and its median level is similar, but its
  ordered CQT cosine is approximately 0.746–0.811, below the fixed 0.90 threshold.
- Vocal ordered cosine is approximately 0.347–0.468; some longer-window median
  differences also exceed 0.25. It likewise supplies no unchanged-layer evidence.
- No development-song 8-beat sample passes the two-unchanged-layer rule; just one
  4-beat sample does. The four-comparison final gate consequently never passes.

The arithmetic audit found no implementation mismatch. The failed assumption is
that this strict short-window similarity gate is a useful required condition for
the reviewed continuation. The scores alone do not identify whether phrase
evolution, alignment or the representation causes each mismatch. Do not lower
thresholds until this example passes, discard the known drum increase, or claim
that musical continuity is absent. Keep level-change evidence and continuity
uncertainty separate. No new production/directing policy was adopted.

### Validation and provenance

Terra implemented the numerical module and the separate reserve extractor. Luna
reviewed the design/runner before execution; the lead inspected actual source.
Pre-output repairs covered final-anchor indexing, strict midpoint comparisons,
nonfinite inverse-log input and meaningful silence/reordered-pattern test fixtures.
The runner's proposed “false proposals reduced” field was renamed to a neutral
candidate-count comparison: rejected anchors are not independently labelled false
events. This wording correction did not change the frozen numeric rules.

- **6 focused tests passed, 1 existing ddtrace warning.** They cover a layer
  increase with stable background, uniform gain, shared silence, reordered spectral
  patterns with matched RMS, invalid inputs/edge support and strict midpoint ties.
  Compilation and `git diff --check` passed. No broad suite/model rerun occurred.
- Agnes's one CPU-only feature extraction took 2.981s on the extractor timer,
  with 466,924 KiB peak RSS. All four existing stems were reused; no separation or
  learned model ran. It saved 466 beat boundaries, 465 intervals, and four
  84-by-465 CQT plus 465-value RMS arrays. The 180-second timeout did not fire.
- The lead verified all 12 reserve source/snapshot/output fingerprints and exact
  cached/requested/floor-quantized grid parity. The beat-grid source-association
  and accuracy qualifications above remain; extraction cannot validate beat phase.
- Independent saved-data arithmetic reproduced **4,208 per-stem records and 520
  eligible anchors**, including medians, strict support fractions, cosine values,
  each gate, cross-scale/neighbor persistence and full support bounds. All 13
  comparison output fingerprints passed. This establishes execution fidelity,
  not musical correctness.
- The saved comparison plot was visually inspected. All eligible rejected anchors
  are retained alongside proposals; reports do not show only selected successes.

Evidence:

- Frozen rules: `outputs/reviews/arrangement-continuity-registration-01/protocol.md`.
- New reserve features: `outputs/reviews/arrangement-continuity-reserve-features-01/`.
- [Comparison report](../outputs/reviews/arrangement-continuity-01/report.md),
  [plot](../outputs/reviews/arrangement-continuity-01/comparison.png), full predictions
  and reference joins: `outputs/reviews/arrangement-continuity-01/`.
- Independent audit: `outputs/reviews/arrangement-continuity-audit-01/`.

| Record | SHA-256 |
| --- | --- |
| Frozen protocol | `06bc4bb0aef7658871ef59b4b68a601b3ed92c08175f905763f230bc54ca2307` |
| Numerical module | `f2948373e71939f9a338845d516f70e4482a4ed59d209ee649aa968e5fb55e93` |
| Executed comparison runner | `f90d58d30444e07fe713154b5694e8a967591dd908fc2a9d84f034213c0bbe1c` |
| Reserve feature manifest | `8a46795f3d4890b6fa694d5be8b992d4f47093e50adf44e95394496880b682a5` |
| Comparison manifest | `f077d3025280a6aab5eb3330ba9ed226d718e26b396442ac6dac00c5f43a4a70` |
| Independent arithmetic result | `1c69f4d2c6db7d9c66452f2f2b21d26bcfa8ff16024520c53352d33119994691` |

Commands actually performed (completed packages must not be overwritten):

```bash
.songviz/venv/bin/python -m pytest -q tests/test_arrangement_continuity.py
timeout 180s env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 .songviz/venv/bin/python \
  experiments/extract_arrangement_reserve.py \
  --output outputs/reviews/arrangement-continuity-reserve-features-01
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 60s \
  .songviz/venv/bin/python experiments/run_arrangement_continuity.py --repo . \
  --reserve-package outputs/reviews/arrangement-continuity-reserve-features-01 \
  --output outputs/reviews/arrangement-continuity-01
.songviz/venv/bin/python outputs/reviews/arrangement-continuity-audit-01/check_saved.py
```

## Listening judgment intake (completed below)

A source-identical, unnormalized PCM-24 clip and a neutral question are ready in
`outputs/reviews/arrangement-continuity-listening-01/`. The clip is exactly 24s,
48,000 Hz stereo; nearest-frame bounds are
[93.29427083333333,117.29427083333333). Independent round-trip PCM equality and
actual Chromium native playback/duration passed. Manifest:
`b22123695cb25f88f2c1e6ac0d10a292aaea44966f2cc1f587a25faaa51895f2`.

Ask whether the arrangement noticeably changes or mostly continues, with a brief
description of anything entering/leaving or becoming fuller/sparser. Do not show
the candidate time as a suggested answer. No exact timestamps, annotation UI or
laughter labels are required. Preserve the response as new raw feedback and
compare it with the already-frozen output before selecting a follow-up method.
The question was sent to the human on continuation; the answer is recorded
below. No further annotation is pending for this comparison.

### Human reserve judgment received — 2026-09-22

The human reports a transition to a new part at approximately elapsed 12 seconds,
around song time **105.294263s**, with a tentative interpretation of verse to
pre-chorus. The raw statement is preserved in
`outputs/reviews/arrangement-continuity-listening-01/human-feedback.json`.
Its intake binding is
`outputs/reviews/arrangement-continuity-listening-01/human-feedback-manifest.json`.
This is approximate listening evidence: it does not establish exact onset,
section identity, or an exhaustive label.

The original intake interpretation called this a concrete miss. A subsequent
timing audit corrects that overstatement: around elapsed 12.10s, the
nearest eligible anchor has `level_change=false`, `continuity_support=false` and
`arrangement_candidate=false`; the reserve's only level-only proposal is earlier,
at elapsed **10.336s** (song 103.630658s), for an `other`-stem increase. The
continuity-gated arm has no proposals anywhere in the reserve. The earlier
level proposal uses source windows spanning **100.124444–107.578050s** (elapsed
**6.830181–14.283787s**), which include the human's approximate transition time.
Its anchor is **1.663605s earlier** than the reported point. Feature support is
not detected transition extent, so this overlap establishes neither a hit nor
an event boundary. No matching tolerance was preregistered. The gate's failed
development criterion remains valid; the broader task of recognizing a new
section differs from its narrower continuing-material arrangement hypothesis.
Do not retune thresholds on this example. [Doc 32](32_transition_components.md)
inspects the existing pattern/level algorithms before proposing a new method.
Agnes is now seen development material; independent references are required
before claiming generalization.
