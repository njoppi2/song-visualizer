# Explicit structural grids — 2026-09-10

Historical step report. The subsequent logic changes and audio-linked review are
documented in [Boundary and recurrence review](13_structure_review.md); the
`structure-grid-01` artifacts described here remain unchanged.

The beat-grid follow-up to `11_beat_grid_audit.md` is implemented. Structural
analysis can consume the reviewed pulse without independently tracking beats.
This completes the timing-input/provenance step, not reliable section detection,
musical recurrence or calibrated novelty.

## Contract and scope

`songviz.story.compute_story` accepts optional `beat_times_s` and
`beat_grid_source` keywords. `songviz/structure_grid.py` validates the supplied
grid and resolves feature-frame alignment once. Main SSM section features,
multi-scale novelty and per-stem diagnostics receive that shared grid.

- Supplied times must be finite, strictly increasing, inside the audio duration,
  and distinct at the feature-frame resolution. Invalid grids raise; they do not
  silently switch to tracked or pseudo-beats.
- `story.meta.beat_grid` records exact requested seconds, effective seconds,
  frame indices, source, SHA-256 hashes of both time lists, sample rate, hop,
  endpoint policy, explicit-input status and fallback reason.
- Times are floored to feature frames (512 / 22050 = approximately 23.22 ms in
  this experiment). Frame zero and the last feature frame are inserted as needed.
  These endpoints are feature support, not detected beats or downbeats.
- Without explicit input, the internal tracker remains available. Short grids
  and tracker errors use the approximately half-second fallback, now recorded.
- `story.meta.section_method` and `section_error` expose whether the main section
  path failed. If it does, the existing agglomerative fallback remains; supplied
  timing is retained for novelty, not replaced. Successful SSM runs can still
  contain existing tension-based boundary refinements.

Production CLI options, cached analyses, reduction timing and the standalone
phrase-cluster experiment are unchanged. Production callers still use the internal
tracker unless they explicitly pass a grid. This is not a global beat-tracker
replacement or a verified bar/downbeat grid.

## Regenerate the controlled comparison

From the repository root, with the existing source, stems and verified rhythm
review present:

```bash
.songviz/venv/bin/python experiments/regenerate_structure.py \
  --out outputs/reviews/structure-grid-02
```

Choose a new output directory each time; the command refuses to overwrite one.
It uses `outputs/reviews/rhythm-review-01` by default, verifies review inputs,
and computes the full song twice with the same current code and four existing
stems. Only the supplied grid differs. It guards against beat-tracker calls,
requires all four stem diagnostics, saves source/code/input fingerprints and
rechecks that original audio, stems and caches are unchanged. It does not render
a video, separate stems, refit beats or call an LLM.

Completed package: `outputs/reviews/structure-grid-01/`.

- `cached-grid/` and `reviewed-grid/`: complete new `story.json` and `beat-grid.json`.
- `comparison.json`: predicted sections and grid/method provenance.
- `sections-comparison.png`: paired predicted section timelines.
- `novelty-comparison.png`: paired 4/16/32-beat lag-novelty curves.
- `inputs/` and `manifest.json`: snapshots and integrity evidence.

The fresh cached-grid run is the controlled baseline, **not** the original
historical story. Comparing only an old cache with a new run would conflate code
changes and timing changes. The historical internal grid remains unknown.

## Observed results and limits

Both runs used the SSM path without a grid fallback or section exception and
produced drums, bass, vocals and other-stem diagnostics. Exact requested grids
contain 335 and 499 times respectively; endpoint insertion yields 337 and 500
effective frame times. Original source/cache/stem hashes match
before and after regeneration; saved input and output hashes were also verified.

The fresh cached-grid run produced eight sections; the reviewed-grid run produced
seven. This is a difference, not an improvement score. Several boundaries near
79, 138 and 166 seconds remain close while roles and other splits change. The
reviewed run still contains a roughly 2.16-second section at 76.93–79.09s and a
long final span labeled `outro` from 165.72s. Those predictions need scrutiny;
they are not certified song structure.

Novelty curves also change, but each scale is independently normalized and the
opening dominates their range. They still implement nearest recent similarity
over lag windows, not a learned or calibrated measure of musical surprise.
Section letters remain derived from role plus similarity, not independent A/B
recurrence identities. Timeline colors distinguish segment order, not recurrence.

Next: review a small set of actual section boundaries and repeated passages
against the song, then evaluate boundary/recurrence changes on this explicit
grid. Keep pulse approval separate from downbeat phase, section correctness and
novelty quality; do not tune all three at once or expand directing on unverified
labels. The new tests cover grid validation, exact/effective hashes, tracker
bypass, fallback reporting and propagation through `compute_story`.

Verification: `.songviz/venv/bin/python -m pytest -q --disable-warnings` reports
425 passed (123 warnings); 16 tests exercise the new grid contract. Terra wrote
the grid tests, and the lead reviewed/integrated them and ran the full suite.
The experiment's refusal to overwrite an existing directory was also checked.
