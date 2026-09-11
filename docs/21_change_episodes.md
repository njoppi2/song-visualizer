# Multiscale change-response episodes

This experiment follows the four listening judgments in
[20_listening_feedback.md](20_listening_feedback.md). It builds an inspectable
representation of how acoustic changes develop over time and overlap across
instruments/scales. It does not assign perceived importance or musical identities.

## Fixed first hypothesis

Use the exact per-stem spectral-pattern and RMS/activity contrasts already
defined in `local_structure_variants.py`, on the verified cached 2/4/8-beat grid.
For every stem, scale and channel, compute a high threshold as the larger of
0.20 and median + 2.5 scaled MAD, capped at one. A lower release threshold is
half the high threshold. A connected region above the release threshold becomes
an episode only if it contains a sample strictly above the high threshold.

Keep separate channels, stems and scales, including their overlaps. A region's
peak is its earliest maximum. Its start/end describe the **windowed contrast
response**, with complete contributing left/right audio support recorded
separately. Gaps in evidence split regions. Edge-censored regions are flagged.
No threshold search uses the four listening judgments or the 19 annotated spans.

`support_start_s`/`support_end_s` include known below-release neighbours needed
to confirm an episode's edges. `response_support_*` cover above-release samples
alone. `available_at_s` records the local feature-support end, **not streaming
availability**: thresholds and audibility calibration additionally use the whole
track. The cached beat-feature extraction has its own frame/grid provenance.

This is a first extent hypothesis with a known limitation: a sharp level step
can produce a wide windowed response. The response start is not a verified
musical onset, and the release edge is not verified settling. Explicit physical
onset/settled times remain unknown. Compare those response intervals to the
old narrow energy dips and human spans descriptively, without calling IoU a
musical-transition accuracy measure.

An independent pre-implementation check of a stationary signal jumping from
RMS 1 to 4 at beat 32 shows this limitation concretely: the fixed policy's
response ranges are [31,34), [29,36), and [25,39) for 2/4/8-beat windows.
The underlying change is instantaneous. The broader range therefore cannot
by itself establish a longer transition or greater musical importance.

## Evidence for changing instrumental roles

At each episode peak, retain before/after context for all stems:

- Mean RMS and fraction of beats above the existing per-stem audibility floor.
- Fraction of summed stem RMS power, a relative acoustic contribution proxy.
- Concentration of the averaged cached log-CQT distribution, a coarse spectral
  descriptor rather than a validated pitch/tonality measure.
- Adjacent-beat spectral-pattern change where both beats have usable evidence.

These measurements let the review distinguish an instrument becoming quiet,
remaining active with a different pattern, or changing its relative contribution.
Neither energy share nor spectral concentration certifies musical prominence.
Vocal function (speech, singing, laughter, lead/background) and perceived
importance remain unknown in automatic output. The user's voice/laughter note
is displayed as human evidence, not reused as an automatic prediction.

## Required development cases

- Around 79s: localized addition of instrumentation within a continuing idea;
  do not silently infer a new verse/chorus identity or confirm uncertain bass entry.
- Around 21s: retain the user's `none` judgment even if a strong accompaniment
  response remains. This checks the distinction between acoustic change and
  perceived importance.
- Around 124s: inspect vocal evidence as well as accompaniment, preserving the
  `subtle` selection and the note that it may be stronger. An accompaniment-only
  candidate cannot automatically explain the observed vocal role change.
- Around 61–65s: compare the response extent, prior energy dip and existing human
  transition span. The later `broad` judgment supplies no new exact endpoints.

Also evaluate all five interpreted transition spans and report episode counts
and duration distributions by stem/channel/scale. Keep unfamiliar material and
returning identity separate from local response. Feel Good Inc is development
data; this experiment introduces no production/director promotion.

## Development findings

The fixed detector produces 259 overlapping responses: bass 61, drums 80,
other 59, vocals 59. These counts include separate scales/channels and are not
259 independent musical events. The four excerpts contain 3 / 17 / 11 / 26
overlapping responses, respectively. Density remains a problem for direction.

| Interpreted human span (s) | Overlapping responses | Best response IoU | Best frozen dip IoU |
| --- | ---: | ---: | ---: |
| 5.350–6.157 | 0 | — | 0 |
| 30.467–33.445 | 13 | .529 | 0 |
| 61.368–64.855 | 23 | .745 | .235 |
| 92.816–95.398 | 13 | .674 | .571 |
| 187.441–189.926 | 15 | .657 | 0 |

Best overlap selects retrospectively among many candidates; greater coverage
does not prove better physical endpoints or musical accuracy. The short first
transition is still missed. No acceptance tolerance was introduced.

- Drum entry: a 2-beat drum-level response covers 78.237–79.970s, but its
  earliest maximum is 78.237s, before the annotated 78.890s change. Response
  coverage and peak timing must be evaluated separately.
- Within-passage: 17 responses coexist with the user's `none` judgment. The
  nearest peak is a vocal-pattern response at 20.197s. Acoustic change remains
  insufficient to choose visual emphasis.
- Verse ending: the automatically nearest episode is **bass pattern** at
  125.449s, scale 8. Its all-stem context shows vocal RMS-power share dropping
  from 17.38% to 3.69%, with active fraction 1 on both sides. That agrees with
  continued voice plus reduced relative contribution, but does not identify
  laughter or a lost leading role. The nearest vocal response peaks earlier,
  at 122.417s. The descriptor result depends on which episode supplies context.
- Broad breakdown: overlapping responses cover more of the human interval than
  the old narrow dip; the closest individual peak (other-pattern, 63.077s)
  still describes only a small development. There is no inferred hierarchy.

**Next bounded experiment:** compute the before/after role-context descriptors
on the entire beat grid, independently of thresholded episode selection. Inspect
the same four cases, especially vocals around 124s and the perceived non-change
around 21s. Keep all scales, quiet/unknown handling, signed trends and explicit
window support. Determine whether context is stable across nearby anchors;
do not tune a semantic classifier or add automatic visual cuts from these data.
This separates a missing candidate from missing descriptive evidence. New music
and held-out judgments are still required before general-quality claims.

## Validation and handoff

Terra implemented detector/tests and review builder/page/tests in disjoint scopes.
Lead review corrected release-confirmation support, inaudible spectral context,
source binding and presentation of the selected episode's exact windows.
The detector's 16 focused tests and builder's three tests passed; real-data
checks passed for all 24 curves and all 259 interval/support/power-share
invariants. Final full suite: **614 passed, 123 warnings** (15.71s). A subsequent
comment-only correction clarified annotation isolation; no executable logic
changed after that full-suite run. Tests do not establish musical validity.

Final artifact: **`outputs/reviews/change-episodes-01/`**, served at
<http://127.0.0.1:8770/change-episodes-01/>. All **97** declared source/snapshot/
output fingerprints, HTML derivation and byte-preserved note contents verified.
Page size: 146,531 bytes. Browser smoke passed native original audio, bounded
playback, four-case navigation, readonly notes, mobile layout and no page errors.
Additional checks passed seek/retry, selected bass context and frozen-dip
visibility. Desktop/mobile screenshots were captured in `/tmp/`; mobile was
visually inspected. Its dense bands remain a technical review, not a proposed
final music-video layout. No new feedback form or user action is required.

Both Terra workers are closed. A temporary root-directory test server created
during delegated checking was stopped; final browser checks used the existing
localhost ranged server. No prior package, source audio or feedback was replaced.

The review uses native original audio and existing readonly notes, with no new
feedback form. Complete graded curves are retained in `episodes.json`; excerpts
and their all-stem context are in `review.json`/`index.html`. `evaluation.json`
contains all five reference spans and unchanged dip comparisons. `manifest.json`
binds source records, code/feedback snapshots and generated files. No separation
or audio copying is needed. Packages refuse overwrite.

Reproduce into a **new** directory from the repository root:

```bash
.songviz/venv/bin/python experiments/build_change_episode_review.py \
  --out outputs/reviews/change-episodes-01
.songviz/venv/bin/python -m pytest -q --disable-warnings
SONGVIZ_PLAYWRIGHT_MODULE=/home/njoppi2/.npm/_npx/04489aa25f0b609a/node_modules/playwright \
  node experiments/check_change_episode_review.cjs \
  http://127.0.0.1:8770/change-episodes-01/
```

Use the existing ranged review server on 8770; if absent, start
`.songviz/venv/bin/python -m songviz.review_server`. Do not expose the filesystem
root to serve an experiment. Source audio and cached outputs remain gitignored.
