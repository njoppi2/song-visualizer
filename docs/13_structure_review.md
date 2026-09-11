# Boundary and recurrence review — 2026-09-10

This step adds an audio-linked review and two targeted algorithm fixes. It does
not replace the whole structural model or certify musical section identities.

## What changed

### One transition per agreeing detector pair

`_score_and_filter_boundaries` previously retained both the SSM timestamp and a
nearby energy-valley timestamp even when it classified them as agreement.
Its downstream two-second deduplication allowed the pair at 76.93 and 79.09s to
become a roughly 2.16-second section.

SSM/energy candidates now match one-to-one, closest-first within the existing
five-second agreement window, with deterministic ties. A matched pair emits the
SSM time. Independent candidates keep their existing acceptance rules; we did
not impose a blanket minimum length that would erase genuine short transitions.
Choosing SSM timing is a reviewable policy, not proof it is perceptually exact.
Energy-only acceptance and heuristic role assignment still need evaluation.

### Novelty: missing history is not surprise

The old lag implementation inserted zero similarity where no past beat existed,
then independently rescaled each lag row and each novelty curve. This made the
opening look maximally novel without supporting past context and distorted the
meaning of similarity across lags.

The new path keeps clipped cosine similarity on its original 0–1 scale, excludes
unavailable lag comparisons, and does not apply per-song min/max stretching.
No-history output is a neutral zero placeholder with an explicit validity mask:
`story.novelties.history_valid` and per-stem `novelty_history_valid`.
`story.meta.novelty_method` and `boundary_method` version these changes.

This remains nearest-past-beat novelty over 4/16/32 beats, using whatever history
is available, not complete-phrase surprise. Global novelty still uses the other
stem when supplied; per-stem curves remain separate. Similarity is not a
probability. Offline feature synchronization/interpolation is not a causal
live-audio detector. These changes affect fresh `compute_story` calls; existing
cached values, reference annotations and previous review packages are untouched.

### Repeated passages independently of role labels

`songviz/recurrence.py` compares ordered, complete 16- and 32-beat log-CQT windows
from the four stems, sampled every four beats on the exact requested grid.
Each audible stem contributes spectral cosine similarity multiplied by its
mean-RMS agreement. A stem present on only one side lowers the match; shared
silence contributes no positive evidence. Windows cannot overlap or extrapolate
beyond the last supplied beat endpoint.

The review selects a distant candidate return at each scale and a lower-scoring
contrast. This is a diagnostic retrieval method, not a classifier: no threshold
turns those scores into confirmed repeated sections. It does not use role labels,
verify bar phase, allow tempo warping or identify transposed repeats. The existing
section A/B letters are still role-derived; the new candidates are intentionally
separate and not yet wired into the director.

## Review and reproduction

Ready package: `outputs/reviews/structure-review-03/index.html` (playback repair).
With the local server running, open
[the structure review](http://127.0.0.1:8770/structure-review-03/).
`structure-review-01` is a retained development draft with superseded playback
handling; **02** is preserved but its Blob-based audio loader failed in the
user's browser. Use **03** for feedback. Questions, analysis and audio are unchanged
from the current 02 package, including the user's intervening PCM audio repair.
Earlier timing/directing reviews are unchanged.

The review presents four boundary contexts, three A/B passage comparisons, paired
clickable full-song timelines and timestamped free notes. The final long section
may contain missed changes: listen anywhere and mark them, not only near proposed
boundaries. All answers begin unreviewed; acoustic scores are optional evidence.

Use **Download feedback JSON** when finished. The export includes the exact
manifest hash, question IDs, original song times, optional corrected boundaries,
answers and notes. Notes are not persisted until downloaded. Confirmed human
observations will become evaluation references only after feedback is received;
the algorithm's suggestions are not promoted to ground truth automatically.

```bash
.songviz/venv/bin/python experiments/build_structure_review.py \
  --out outputs/reviews/structure-review-04
.songviz/venv/bin/python -m songviz.review_server \
  --directory outputs/reviews --port 8770
```

Choose a new directory; the builder refuses to overwrite an existing one. It
expects the preserved `structure-grid-01` parent review by default. All old
source/cache and baseline output hashes are checked before use. Review audio is
the decoded original PCM with sample-for-sample equality, no gain or timing edits.

The package records code/input snapshots and fingerprints, candidate story,
raw SSM/energy/fused boundary evidence, full recurrence pairs, plots, question
data and single-change comparisons. No audio upload, stem separation, beat
refitting or runtime LLM calls are needed. Terra implemented boundary fusion and
its tests; Luna implemented the review page; the lead handled novelty, recurrence,
integration and end-to-end verification.

## Measured differences, not a quality claim

With identical reviewed timing, source and stems:

- Previous: seven sections, including the 76.93–79.09s split.
- Candidate: six sections; boundaries at approximately 6.99, 64.37, 79.09,
  138.00 and 165.72s. The final span still has a questionable `outro` role.
- The context-free opening novelty spike disappears. Later changes remain;
  smaller scores are not automatically better scores.
- Novelty-only regeneration preserves previous sections exactly.
- Boundary-only regeneration matches candidate sections exactly and preserves
  previous global novelty within 1e-6. The combined candidate matches novelty-only
  global curves within that tolerance.

Visual inspection supports treating the broad texture/energy changes around
64–79 and 138–166s as useful review targets. It does not establish which exact
timestamp a listener considers the boundary. The model has not supplied a
listening annotation or assumed that all energy changes are new sections.

Feel Good Inc remains development data. No holdout musical improvement is claimed.
Next, ingest the small listening review, separate confirmed boundaries from
within-section events, and test recurrence changes on independent passages before
using them to drive recurring visual identities or surprise effects.

## Original implementation verification (before subsequent playback repairs)

- Full suite: **449 passed**, 123 warnings, using
  `.songviz/venv/bin/python -m pytest -q --disable-warnings`.
- Local Chromium smoke test passed: actual A/B seek positions, bounded stops,
  user-seek cancellation, editable/preserved notes, invalid-time rejection,
  downloaded JSON contents and manifest hash, mobile overflow and audio-failure
  recovery. Reproduce with `node experiments/check_structure_review.cjs URL`;
  set `SONGVIZ_PLAYWRIGHT_MODULE` to an existing Playwright installation if it is
  not on Node's module path. This check does not install dependencies.
- All seven source/cache/stem hashes, ten input snapshot hashes and nine output
  hashes in the final package were verified. Current implementation files match
  the saved snapshots. The HTML matches its template, embedded data and manifest
  hash exactly. Feedback manifest SHA-256:
  `e87475be1c4b2b2796fd322998534902e4c7c3698f06663fae3de9b358f91369`.
- The secondary MCP browser could load the page but failed to fetch the large
  WAV; the cause was not established. The local Chromium end-to-end test passes
  against the actual served package. HTTP playback buffers the full lossless WAV
  (about 75 MiB) before enabling seeking; this is a local review, not a
  bandwidth-optimized streaming product.

## Playback repair

The reported failure was reproduced in the previously failing browser: the WAV
returned HTTP 200 and could be read completely as an ArrayBuffer, but
`response.blob()` rejected with `TypeError: Failed to fetch`. The player never
received a source. A hard reload or changing WAV encoding did not resolve that
loader failure. Its underlying browser Blob-storage cause is not established.

Playback now gives the original WAV URL directly to the native audio element.
`songviz.review_server` provides real HTTP byte ranges for seeking, so the player
does not depend on fetching a whole song into a JavaScript Blob first. It binds
only to localhost and streams selected file ranges in bounded chunks. Native
load errors retain editable notes and report error details; retries reload the
source. The plain `python -m http.server` command is no longer the recommended
server for this page because it does not support the required range responses.

To repair an existing package without re-running any music analysis:

```bash
.songviz/venv/bin/python experiments/repair_structure_review.py \
  --parent outputs/reviews/structure-review-02 \
  --out outputs/reviews/structure-review-03
```

Use a fresh output directory. The command verifies parent output/snapshot hashes,
preserves the old package, retains original analysis-code snapshots, adds separate
playback-code snapshots, and assigns the repaired package a new feedback manifest.

Repair verification: **459 tests passed** (123 warnings). The final 03 page also
passed the local Chromium playback/seek/retry/export smoke test. In the formerly
failing MCP browser, native audio now loads with duration 221.173333s and no
media error; clicking Play A starts at the expected 86.89987s with its bounded
stop retained. This verification covers the actual packaged page, not just a
template override. Parent audio, questions, predictions and recurrence hashes
match byte-for-byte. New feedback manifest SHA-256:
`53670ae4f04814fa26436cbd1e1e151872f61f70a747ffed1b80ad0f854730af`.

Terra implemented the native loader, browser checks and range server/tests; the
lead reproduced the failure and handled repackaging and final integration.
