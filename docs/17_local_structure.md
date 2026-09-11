# Local-change and transition candidates

This is the next development experiment after `16_structural_evaluation.md`, not
a replacement for the production section detector or a complete song hierarchy.
The implementation uses cached acoustic features, with no human labels as
detector inputs. The existing 16/32-beat recurrence evidence is context for
evaluation, not a gate that erases changes when material is familiar.

## Fixed first hypothesis

`songviz/local_structure.py` compares adjacent windows of 2, 4 and 8 reviewed
beats at each supported beat boundary. Per-stem spectral-pattern change and
RMS/activity change are separate. The largest up to two stem contributions are
averaged per channel, so one instrument can matter without every stem changing.
The proposal score takes the larger channel. This is a heuristic acoustic
contrast, not novelty probability, section identity or musical importance.

Each scale uses a median/MAD threshold with a fixed floor and deterministic
peak selection. Cross-scale suppression avoids duplicating nearby change
proposals; it does not merge or impose a minimum length on user sections.
The configuration is serialized and chosen before comparing this run with the
development annotations. There is no search for a target count of 19 spans.

Transition proposals use a separate, deliberately narrow hypothesis: a short
energy depression with audible preceding and following context. The energy
proxy is the root sum of squared stem RMS, not a resynthesized mix envelope.
Interior values must remain below a fraction of both flank medians, with actual
entry/recovery edges and sufficient elevated flank support. Candidate durations
are 2–12 beats, supported by four-beat flanks. These are measured dip intervals,
not fixed-width rectangles drawn around a boundary peak.

Silence, a permanent level step and a terminal fade do not provide a bounded
dip-and-recovery interval. Fills, gradual ramps, timbral transitions without an
energy dip, edge transitions without complete context and sub-two-beat events
may be missed. Failure to propose a dip does not invalidate the user's transition.

## Timing and unknowns

The cached grid covers only its measured beat endpoints; no edge beats or
bar phase are invented. At 138.5 BPM, a beat is roughly 0.43s. Feature frames use
the parent's floor-quantized 512-sample grid at 22050 Hz, slightly earlier than
reported requested beats. Sub-beat endpoint precision is not established.

Unsupported window positions and absent shared-pattern evidence remain null.
Scores stay bounded but are not calibrated probabilities. Proposal support and
`available_at_s` expose required right-side context. Whole-song audibility
calibration also uses future audio, so these are offline proposals, not a causal
realtime onset detector.

When peaks from different scales support one proposal, individual support
windows are retained and availability covers their union, not just the winning
shorter scale. Per-stem evidence identifies the primary scale separately.

## Evaluation and inspection

`songviz/local_structure_evaluation.py` reports all annotated boundaries, each
nearest new proposal and legacy cut, and both directions of transition-interval
overlap comparisons. IoU and signed endpoint errors describe overlap, without
an accepted correctness tolerance or a one-to-one accuracy claim. Every
prediction is retained so increased proposal density is visible. An unannotated
candidate is not automatically a false positive, and a smaller nearest distance
with more candidates does not establish better precision.

Same-group variation marks, other named groups and unknown identities stay
distinct. Raw human motif assertions and explicit analyst transition/variation
interpretations retain their original provenance and unspecified certainty.

For each change, persisted 16/32-beat context describes a following full phrase
when supported. Its window offset and end time matter: it is not instantaneous
surprise at the change. No new identity, verse or chorus label is inferred.

The review page compares user layers, unchanged legacy sections, new local
changes, energy-dip intervals and per-scale curves. Native playback references
the existing original WAV with HTTP range support; there is no Blob-audio fetch,
audio duplication, new annotation assignment, or external model call at runtime.

```bash
.songviz/venv/bin/python experiments/build_local_structure_review.py \
  --out outputs/reviews/local-structure-02
```

Ready review: `http://127.0.0.1:8770/local-structure-02/` while the range server
serves `outputs/reviews`. If needed, start it with
`.songviz/venv/bin/python -m songviz.review_server`. Version 01 is preserved as
the initial development package; version 02 includes validation and UI hardening,
with the same fixed detector policy and candidate timestamps.

Use a new directory for every run; existing/input packages are never overwritten.
The builder verifies numeric feature/reference/timing/recurrence fingerprints,
original audio and source provenance, snapshots code, then writes predictions,
evaluation, review data, report and manifest. Existing story caches, annotations,
recurrence results and director/render behavior remain unchanged.

## Completion criteria and next decision

### Fixed-policy development result

The first full-song run produced **16 local-change candidates and two energy-dip
intervals**, compared with five internal legacy cuts. Examples near user marks
include 33.191s versus 33.445s and 203.847s versus 203.759s. The former single
long ending now has additional proposed changes, but proposals do not yet form
a section partition or say which musical idea is returning.

Both dip proposals overlap one interpreted reference transition each:

| User interval (s) | Proposed interval (s) | IoU | Limitation |
| --- | --- | --- | --- |
| 61.368–64.855 | 63.944–65.243 | 0.235 | Captures a late sub-interval, not the full breakdown |
| 92.816–95.398 | 93.830–95.563 | 0.571 | Starts roughly one second after the user mark |

The other three interpreted transitions have no overlapping dip proposal.
Crucially, the default change policy misses the low/high-energy chorus marks
at 78.890s and 165.534s, despite the old detector having nearby cuts. It also
misses the verse-2 outro start around 124.295s and the low-energy chorus return
around 158.591s. These are material gaps, so this policy is **not promoted over
the existing detector**. Current candidates are dominated by RMS/activity
changes; a global combined-contrast threshold can leave other musically useful
changes below threshold. That is a diagnosis to test, not proof that lowering a
threshold will solve the problem.

For example, at the nearest grid point to 78.890s, the 2/4/8-beat arrangement
changes are approximately 0.640/0.615/0.650, below thresholds
0.833/0.922/0.781. Near 165.534s the corresponding values are
0.773/0.700/0.636. The feature evidence is present; this first threshold/peak
policy does not select it. These observations are development diagnostics, not
new correctness thresholds.

The immediate next experiment should compare separately calibrated channels and
sustained stem-activity changes, with the current run frozen as a control. It
must retain the chorus-variation examples as explicit regressions and evaluate
the whole proposal set. Broader transition spans will need more than the deepest
energy trough; do not stretch them to annotation endpoints by hand.

Verify synthetic stationary/step/dip/return cases, unsupported edges, intervals
and score invariants, package/input integrity, full regressions, and browser
playback. Then record the fixed-policy full-song results including misses and
proposal burden. Automated checks are not human listening validation.

Use the resulting evidence to choose the next change: more precise endpoints,
non-dip transition hypotheses or a revised proposal policy. Do not automatically
promote every local change to a section cut. A separate identity/variation
grouping step is still needed before the director can rely on these proposals.
Feel Good Inc remains development data; holdout musical quality is unverified.

## Verified handoff

- Full suite: **551 passed**, with 123 existing dependency/fixture warnings.
- Final package: 16 consumed-input, eight snapshot and four output fingerprints
  verified; all 18 parent-source fingerprints still match. Raw annotations,
  cached stories, stems and original audio remain unchanged.
- Predictions in review 02 are byte-for-byte equivalent JSON data to review 01:
  the UI/validation hardening did not retune the detector.
- Browser smoke passed native original-audio loading/playback, bounded audition,
  mobile layout, and no page/console errors. Additional isolated checks passed
  plot-coordinate seeking, retry without duplicate keyboard listeners, numeric
  channels, null preservation and safe rendering of malicious-looking labels as
  text. The final timeline was visually inspected; motif colors repeat and long
  labels are clipped with complete tooltip text.

Browser smoke command:

```bash
SONGVIZ_PLAYWRIGHT_MODULE=/home/njoppi2/.npm/_npx/04489aa25f0b609a/node_modules/playwright \
  node experiments/check_local_structure_review.cjs \
  http://127.0.0.1:8770/local-structure-02/
```
