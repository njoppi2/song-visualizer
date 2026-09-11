# Separate structural dimensions — development implementation

Implemented 2026-09-10 after the free-form annotations in
`15_section_feedback.md`. This completes the scoped representation/evaluation
step, not automatic identification of all song parts or a new segmentation model.

## Contracts

- `songviz/structure_annotations.py` validates editor exports and creates a
  `songviz-structural-development-reference`. Every exact interval, label, note
  and certainty is retained. Nonempty motif names create positive, layer-scoped
  identity groups. Blank motifs remain unknown; different names are not labeled
  negative examples. Independent layers need not form a hierarchy.
- `benchmark/feedback/section-editor-02.interpretation.json` contains explicit
  analyst variation/transition interpretations, bound to the raw export's hash.
  These are not additional user annotations. No generic label parser guesses
  meaning. The five explicit transition/breakdown spans are marked as transitions;
  whether the bridge introduction is also a transition remains unknown.
- `songviz/recurrence.py` keeps the original combined acoustic score unchanged,
  but adds bounded ordered spectral pattern similarity over shared active stems,
  separate RMS/activity agreement, and per-stem evidence. This allows pattern
  continuity despite a level change or added instrument, without claiming that
  any remaining shared stem proves the same musical identity.
- Each fixed window also gets local pattern/arrangement change and historical
  pattern novelty (one minus the best comparable earlier pattern similarity).
  History excludes overlapping/future windows. Missing comparable evidence is
  null, not zero novelty or maximal surprise. Scores use the entire target
  window, with `available_at_s` at its end, and a whole-song audibility floor:
  this is offline evidence, not a causal realtime change-onset signal.
- `songviz/structure_evaluation.py` compares those dimensions with development
  examples. It records boundaries and their interpretation, exact transition
  intervals, legacy-section overlaps, phrase support, repeat-pair summaries and
  separate local/historical context summaries. Unsupported cases stay visible.
  Identity windows may cross adjacent low/high-energy spans carrying the same
  explicit motif; their variation is unknown instead of forcing one variant.

The raw 19 spans are never collapsed into a coarse target segmentation. Adjacent
spans with the same explicit motif provide contiguous identity support and
distinguish local variation examples from separated returns; this does not invent
musical parents or modify intervals. Legacy role-derived letters are not identity
predictions.

## Reproduce

Run from the repository root with the existing original audio, cached stems,
reviewed beat grid and review packages available:

```bash
.songviz/venv/bin/python experiments/evaluate_structure_feedback.py \
  --interpretations benchmark/feedback/section-editor-02.interpretation.json \
  --out outputs/reviews/structure-evaluation-03
```

Choose a new output directory for a rerun: existing directories are refused.
Omit `--interpretations` for only explicit user assertions, with unknown variation
and transition fields. The acoustic extractor has no annotation argument and
always analyzes the same full-song, fixed-grid windows, independent of labels.

Ready artifact: `outputs/reviews/structure-evaluation-03/report.md`.
Versions 01/02 are retained development runs; 03 adds dimension-specific window
support, allowing identity comparisons across same-motif subdivisions. Version 01
also predates bounded-score/context-report hardening. Use 03; this is a diagnostic
report, not another annotation assignment.

Artifacts include normalized `reference.json`, `evaluation.json`, decomposed
`recurrence.json`, requested/frame-quantized `timing.json`, reusable
`features.npz`, unchanged `legacy-sections.json`, the report, input/code snapshots,
and a manifest with source/output fingerprints and legacy-control results.
The builder verifies the original/editor audio, editor manifest/data, raw
feedback, previous predictions, source/stems and duration. It verifies all
consumed inputs again before finalizing. No separation, section regeneration,
audio copy, external model call, cache replacement or render change is required.

## Observations on Feel Good Inc

The 16/32-beat control preserves all 12,898 existing recurrence pairs and their
per-stem scores. This isolates the new dimensions from changes in audio, grid,
feature extraction or the original combined formula.

At 16 beats, a selected low-/high-energy chorus return has pattern similarity
about 0.812 and arrangement similarity about 0.512 (66.11s versus 173.53s window
starts; shared active evidence from `other` and `vocals`). It illustrates the
distinction, not reliable identity classification. All-pair summaries and
shared-stem support are reported alongside these explicitly selected maxima.
Neither channel is a calibrated probability or a perceptual surprise measure.

Of 121 sixteen-beat windows, 53 cross user marks; of 117 thirty-two-beat windows,
89 cross marks. Such windows cannot be assigned one within-span variation, but
can still support identity when they stay inside adjacent spans of the same
explicit motif. The corrected evaluation therefore retains 32-beat chorus and
verse-2 return evidence rather than interpreting subdivisions as missing identity
support. The short returning low-energy chorus has no fully contained 16-beat
window on this stride/phase. This limits variation-specific evidence, not the
existence of the chorus. Labels never shift or resize extraction windows.

## Verification and remaining work

Focused tests cover raw/mapped validation, unknowns, layer independence, precise
short spans, acoustic pattern versus arrangement, entry/exit/silence, A/B/A local
versus historical context, no future/overlapping reference windows, bounded
scores, cross-variation identity support, regression controls and isolated package
integrity. Run:

```bash
.songviz/venv/bin/python -m pytest -q
```

Final verification: **511 tests passed** (123 dependency/fixture warnings).
All 18 consumed-input, nine snapshot and seven output fingerprints matched in
the final package; the raw feedback copy is byte-identical and all 12,898 legacy
pair/per-stem scores are unchanged (maximum absolute error zero). Every context
reference precedes and does not overlap its target, with bounded scores and
end-of-window availability recorded.

Automatic identity clustering, variation change-points and transition-interval
detection are **not** implemented by this step. Existing story predictions and
director behavior remain unchanged. No listening approval or holdout improvement
is claimed, and no universal timing tolerance has been selected.

Next: develop shorter-scale local-change/transition proposals alongside the
longer recurrence evidence. Keep identity separate from changes in arrangement,
evaluate interval support explicitly, and use this song as development data.
Reserve new listening examples before claiming generality. Do not tune toward
exactly 19 coarse sections or interpret all unmatched marks as same-level errors.
