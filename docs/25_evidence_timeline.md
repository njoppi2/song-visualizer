# Full-song evidence timeline

Team 2 implementation record for the analysis-first listening interface assigned
on 2026-09-13. The integration owner is Team 1; only Team 1 updates the live
checkpoint in `CONTINUE.md` after review.

## Scope and boundary

The page is a read-only adapter over frozen `role-context-02`,
`structure-evaluation-03`, `listening-examples-01`, and the original WAV bound
by `structure-review-03`. It provides one native full-song audio cursor, a
click-to-seek context plot, the four existing excerpt shortcuts, selectable
stem/metric/2-4-8-beat local context, and earlier/later recurrence listening at
the existing 16/32-beat scales.

It does not run a model, separate audio, create feedback, generate an inferred
musical role or vocal function, select event boundaries, or promote a director
policy. RMS-power share is displayed only as relative acoustic contribution.
Missing support and null metrics are rendered as unavailable rather than held
forward. User notes remain source-bound frozen records; factual display text is
tied to the displayed record and analyst limitations are marked separately.

## Inputs

- Original WAV: `outputs/reviews/structure-review-03/original.wav`, SHA-256
  `7d800828527fb8812dc6433361d1d72ad1f7cdba368b9c51dce4abc954f7a5c1`.
  Its source FLAC is `657af9333edafbb6cf8cc573446650773d1169b0117c784106d1895cf802cb44`.
- Full local context: `role-context-02/role-context.json`: 499 requested beat
  anchors and 495/491/483 complete samples at 2/4/8 beats.
- Recurrence: 6,903 saved 16-beat pairs and 5,995 saved 32-beat pairs. Every
  paired-audition interval is taken directly from the referenced saved spans;
  the earlier passage precedes the later one.
- Listening shortcuts: existing 16–26, 57–70, 74–85 and 119–132s excerpts and
  their already-recorded raw notes. No new annotation is requested.

The builder rejects absent, duplicate, mismatched, or modified consumed records
and refuses an existing output directory. It snapshots its source/template/check
code and frozen parent manifests, records all output hashes, and derives the
HTML exactly from the snapshotted template and `evidence-timeline.json` hash.

## Returned original package and validation

The original candidate was [evidence-timeline-05](http://127.0.0.1:8770/evidence-timeline-05/),
at `outputs/reviews/evidence-timeline-05/` (about 11 MB). Its manifest SHA-256
is `53fa424c5ffe0ce7f3fcbd9009a1ed72e78170c72f46ee846240905d4b2db6c6`.
It contains 103 verified frozen/current source records, eight code/input
snapshots, and hashes for its adapted JSON and derived HTML. The page retains
499 aligned anchors, all three local scales, 121/117 complete recurrence/history
contexts at 16/32 beats, 24 deterministic saved recurrence listening pairs, and
16 explicit unsupported-edge probes.

Commands actually run:

```bash
.songviz/venv/bin/python -m pytest -q tests/test_evidence_timeline_builder.py --disable-warnings
SONGVIZ_PLAYWRIGHT_MODULE=/home/njoppi2/.npm/_npx/04489aa25f0b609a/node_modules/playwright \
  node experiments/check_evidence_timeline.cjs http://127.0.0.1:8770/evidence-timeline-05/
.songviz/venv/bin/python -m pytest -q --disable-warnings
git diff --check
```

Focused builder tests passed **3 tests, 1 warning**. The browser check passed native WAV
loading and declared source identity, plot seek, all four shortcuts, actual
stem/metric/scale data changes, unavailable-edge/no-carry-forward behavior,
earlier/later exact recurrence intervals, simulated audio error/retry and no
overflow at 320/375/768px. All source/snapshot/output fingerprints and exact
HTML derivation from the snapshotted template plus `evidence-timeline.json` hash
were independently rechecked. Desktop and mobile captures were visually
inspected at `/tmp/songviz-evidence-timeline-05-desktop.png` and
`/tmp/songviz-evidence-timeline-05-mobile.png`.

Two development packages are intentionally retained: `evidence-timeline-01`
contains an initial template JavaScript syntax failure caught by the first live
browser run; `-02` used the corrected page but exposed a browser-checker relative
URL normalization fault; `-03` passed interaction but began at an unsupported
edge, making the opening context unhelpful. None was overwritten. Version 04
preserves the first supported context while still exposing edges as unavailable
when selected.

The user then said the graph was not understandable. Version 05 is a separate,
frozen clarity pass: it opens on voice RMS rather than bass, names the selected
stem/metric, adds time ticks and axis/unit wording, a cyan-measure/gold-cursor/
unavailable-gap legend, explains measured zero versus a missing value, and
adds a three-step reading guide. No data, source audio, measurements, claims or
interaction semantics changed. The browser check now asserts the initial voice
selection and axis/legend explanation. The standalone Team 2 validations above
passed for version 05.

An attempted shared-worktree suite after this pass produced **715 passed,
3 failed, 123 warnings**. The failures are confined to Team 1's in-progress
`tests/test_music_representation_builder.py`: its synthetic runtime fixture
still lacks Team 1's newly required `closest eligible chunk center; earlier
chunk wins exact ties` ownership declaration. It fails before the expected
tamper assertions. Team 2 did not edit those Team 1 paths; relay this exact
fixture/clock mismatch to Team 1. The prior 712-pass suite applies to the
version-04 workspace before those Team 1 in-progress changes.

## P1–P5 repair handoff — complete, awaiting Team 1 integration — 2026-09-14

Team 1 returned version 05 after reproducing five interface/support defects in
its independent review. The repaired candidate is
[evidence-timeline-10](http://127.0.0.1:8770/evidence-timeline-10/), at
`outputs/reviews/evidence-timeline-10/`. Its manifest SHA-256 is
`a2b2b55fbb00b0bdd8859c4bd88be25336fa433590ef72010bed611a04c91fa1`.
All earlier packages, including version 05 and repair-development packages
06–09, remain immutable evidence; version 10 is the sole returned candidate.

No measurement, source record, original audio binding, adapted timeline JSON or
claim changed. `evidence-timeline.json` is byte-identical to version 05
(`aad36cd2828c95aba7ce7675aeb878422f8f7704dcd94ca94d068a46994848fe`).
The repair is confined to page interaction, disclosure and browser coverage.

| Finding | Repair in version 10 |
| --- | --- |
| P1 — plot click inversion | The click handler maps client coordinates through the SVG screen transform, then inverts the actual 40–870 plot area in the 900-unit viewBox and clamps to the song bounds. The real-browser check clicks 0s, 60s and the duration at desktop, plus 60s/duration at 320px. |
| P2 — stale range slider | One cursor-sync path updates the range input during `timeupdate`, native `seeking`/`seeked`, rendering and programmatic seek. Range values are explicitly converted from strings before clamping, so a slider seek no longer becomes 0s. |
| P3 — old bounded audition survives new seek | Ordinary API, plot, slider and native seeks clear the prior `stopAt`/excerpt state; bounded listening still preserves its own endpoint and pauses exactly there. An external seek also replaces the stale “Tocando …” status with the current cursor time. |
| P4 — history appears outside support | A history value is selected only when the cursor lies in its saved posterior source span. Among overlapping spans, the record with the closest saved `available_at_s` is chosen. The page shows source span, record index, completion/availability time, cursor relationship, offline status, and whether the independently selected pair is the same window. Outside every saved span, including the song tail, it states unavailable and explicitly says no nearest record was carried forward. |
| P5 — opening anchor wording changes | The first supported local anchor is stored during initialization and the guide always distinguishes that fixed data start from the gold current-audio cursor. |

Repair-development packages are retained deliberately: 06 exposed an overly
strict 50 ms browser assertion despite normal coarse `timeupdate` scheduling;
07 exposed the actual string-range-to-zero seek defect; 08 exposed an overly
restrictive first history-support interpretation during inspection; and 09
exposed stale audition wording after an external seek. They are not accepted
packages and were not overwritten.

### Validation actually run

```bash
.songviz/venv/bin/python -m songviz.review_server
.songviz/venv/bin/python -m pytest -q tests/test_evidence_timeline_builder.py --disable-warnings
SONGVIZ_PLAYWRIGHT_MODULE=/home/njoppi2/.npm/_npx/04489aa25f0b609a/node_modules/playwright \
  node experiments/check_evidence_timeline.cjs http://127.0.0.1:8770/evidence-timeline-10/
SONGVIZ_PLAYWRIGHT_MODULE=/home/njoppi2/.npm/_npx/04489aa25f0b609a/node_modules/playwright \
  node outputs/reviews/music-representation-integration-review-01/verify_display.cjs \
  http://127.0.0.1:8770/evidence-timeline-10/ --out /tmp/songviz-display-10.json
SONGVIZ_PLAYWRIGHT_MODULE=/home/njoppi2/.npm/_npx/04489aa25f0b609a/node_modules/playwright \
  node outputs/reviews/music-representation-integration-review-01/browser_review.cjs \
  /tmp/songviz-page-browser-10 http://127.0.0.1:8770/evidence-timeline-10/
git diff --check
```

- Focused builder tests: **3 passed, 1 warning**.
- The expanded real-browser checker passed native source/playback, exact plot
  ticks/endpoints, slider/native synchronization, every ordinary-seek origin,
  exact bounded stop, history source-span/tail behavior, four shortcuts,
  recurrence listening, unavailable local edges, retry and 320/375/768px
  no-overflow layouts.
- The existing integration display verifier matched all **96** DOM metric
  combinations to their sources with no failures; its unavailable-edge probe
  remained unavailable with no carry-forward.
- The independent diagnostic observed a 60s plotted click at 60.069s, playback
  cursor/slider at 40.703/40.486s (one normal coarse `timeupdate` interval),
  range stop exactly at 58s, ordinary seek continuing at 100.462s, and the song
  tail explicitly unavailable for history. Final desktop and 320px captures
  were visually inspected at `/tmp/songviz-page-browser-10/`.
- A direct candidate audit verified all **113** manifest records, current
  source/snapshot equality for the builder/template/checker, and exact HTML
  derivation from the snapshotted template plus timeline SHA.

### Limits and next action

This is still an offline, frozen baseline-evidence page; it does not establish
musical role, importance, semantic identity or a director policy. Mobile remains
a long diagnostic page by design, though it has no horizontal overflow. Team 1
should now independently inspect version 10 and either accept it or return a
specific remaining finding. Team 2 must not update `CONTINUE.md` or integrate a
model schema.
