# User section annotations — 2026-09-10

Raw export: `benchmark/feedback/section-editor-02.json`, preserved byte-for-byte
from `/home/njoppi2/Downloads/songviz-sections (1).json`.
SHA-256: `dd100b34d43ece3dbe500b297f0fb4320db71e3eebb9dc705959145e54aa8b2f`.

The export's manifest matches `outputs/reviews/section-editor-02/manifest.json`.
Both original-source and review-audio hashes, duration and annotation state were
validated against the package and its audio, using the editor's state validator.
This is human-authored feedback, not an algorithm-derived reference. No existing
benchmark reference, detector output or musical algorithm was changed during
ingestion. Certainty is `unspecified` throughout; that must not be silently
upgraded to high confidence or interpreted as rejection of the annotations.
There are no per-span or global written notes in this export.

## What the user supplied

One layer, **Song parts**, with 19 contiguous labeled spans. The labels include
intro, transitions/breakdowns, initial verse/pre-verse, verse 1, verse 2 and its
outro, low/high-energy choruses, intro to bridge, bridge and the final outro.

Explicit repeated-idea groups:

- `chorus`: 64.86–92.82s and 158.59–187.44s, each with a low/high-energy split.
- `verse 2`: 95.40–138.00s and 189.93–217.97s, each with an outro subdivision.
- `bridge`: 138.00–158.59s, split into an introduction and the bridge itself.
- `verse 1`: 33.44–61.37s. The user did not give verse 1 and verse 2 a common
  motif identifier; shared vocabulary alone is not evidence of equivalence.

Several marked transition spans are short: 0.81s for the initial transition and
approximately 2.49–3.49s for the breakdowns. These are intentional annotations,
not short-section errors to eliminate automatically. Empty motif fields make no
positive or negative claim about recurrence.

## Comparison with the current candidate

Compared against the unchanged `structure-review-03/candidate-story.json`:
the detector has six spans/five internal boundaries; the user has 19 spans/18
internal boundaries. Each predicted internal boundary has a distinct user mark
within 0.833s (nearest-mark distances, not a validated matching tolerance).

| Detector boundary | Nearest user boundary | User label beginning there |
| --- | --- | --- |
| 6.989s | 6.157s | initial verse / pre-verse |
| 64.366s | 64.855s | low energy chorus |
| 79.087s | 78.890s | high energy chorus |
| 137.996s | 137.999s | intro to bridge |
| 165.721s | 165.534s | high energy chorus |

Thus the detector catches some large changes but leaves many user-described
changes unrepresented. This does not mean all 18 user marks should become
boundaries at one coarse level: the annotation intentionally mixes transition
spans, major musical parts and variations within a part.

The detector's final `outro` spans 165.72–221.17s. The user instead marks a high
energy chorus, a breakdown, a verse 2 return and its outro, then the song's final
outro only at 217.97–221.17s. This is a substantive structural/role mismatch,
not just a few milliseconds of timing error.

## Interpretation and next implementation target

These are inferences from the supplied labels/groups, not extra user annotations:

1. Separate **musical identity** from **arrangement/energy variation**. A low- and
   high-energy chorus can share identity while warranting a visual change.
2. Represent **transitions as intervals**, not only instantaneous cuts. A short
   breakdown has its own duration and visual opportunity.
3. Distinguish **local change** from **unfamiliar material**. A returning chorus
   can be familiar at song scale and still produce a strong local change.

The next scoped implementation should preserve these separate dimensions in
the structural/evaluation representation and use the explicit repeat groups as
development examples. Do not first tune a single threshold to produce exactly
19 sections. Leave a formal hierarchy, timing tolerance and equivalence between
other named parts open. No further annotations are required to start that work.

Update: this scoped representation/evaluation step is now implemented in
`16_structural_evaluation.md`. The raw export remains unchanged. Analyst
interpretations have a separate hash-bound sidecar, not inferred user certainty.

These examples are development data. Holdout quality and calibrated novelty
remain unverified; human terminology is useful guidance without requiring a
universal verse/chorus taxonomy for every song.
