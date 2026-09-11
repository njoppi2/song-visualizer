# Free-form section annotation

Open [the section editor](http://127.0.0.1:8770/section-editor-02/)
(`outputs/reviews/section-editor-02/index.html`). The range server on port 8770
serves `outputs/reviews`. Version 01 is a retained development build; use 02.

The user wants to define their own interpretation, not only accept/reject a
detector's suggestions. This editor starts blank: a whole-song unlabeled span,
with no algorithm-predicted boundaries, roles or repeated-part identities.

## Working model

Each named layer partitions the song into contiguous spans. Add a boundary to
split a span, adjust the boundary's timestamp, or remove it to merge neighbors.
Labels, repeated-idea identifiers and notes are free text. Certainty can remain
unspecified or be marked clear/uncertain. Blank spans are not confirmed musical
sections: they are simply areas not yet described.

Start with one layer, **Song parts**. Additional layers can describe smaller
changes or another interpretation. Their boundaries may cross those of other
layers; the editor does not assume a strict hierarchy, a common musical vocabulary,
or that every meaningful event belongs at one structural level. Within one layer,
spans cannot overlap or leave gaps. Use another layer for overlapping levels and
notes for ambiguity that the current interval model does not express.

This is a deliberately simple annotation contract, not a final theory of song
structure. The resulting observations will inform what our detector and director
should distinguish. No detector tuning or new automatic labels are part of this
step, and downloading annotations does not silently add them to the benchmark.

## Workflow

1. Play or seek through the original song; the waveform is only a navigation aid.
2. Mark a boundary at the current song time (button or **B** outside text fields),
   or enter a precise time manually. Boundaries are not snapped to inferred beats.
3. Select a span and give it your own label. Optionally reuse a repeated-idea label
   across passages, add notes, and mark uncertainty.
4. Add/rename a layer if a second level of detail is useful. You do not have to
   finish every span or annotate every layer.
5. Download **songviz-sections.json** and send its path back. The same file can be
   imported to continue later.

Undo/redo applies to annotation changes. Removing a boundary preserves text from
both neighboring spans; review the merged label/notes if they differ. Browser
localStorage is a convenience draft, not the sharable backup: storage may be
blocked or cleared. Download the JSON before changing browsers or moving the
review. No annotations or audio are uploaded or posted to the server.

## Export contract

The JSON uses `kind: songviz-section-annotations`, schema version 1, the package
manifest SHA-256 and both review-WAV and original-source hashes. It includes:

- Source song title and duration, with absolute original-song seconds.
- Named layers and stable segment IDs with start/end times.
- Free-form `label`, `motif` (repeated idea), `notes`, and `certainty` fields.
- Global notes and the current layer/selection for editing continuity.

Imports validate audio identity, duration and annotation structure before
replacing the current document. Same-audio exports from another package version
may be imported, but are exported again against the current package manifest.
This supports editor revisions without treating unrelated audio as equivalent.

## Reproduction

The builder consumes only verified audio identity, duration and waveform data
from the existing review. It does not import predicted sections or reanalyze audio.

```bash
.songviz/venv/bin/python experiments/build_section_editor.py \
  --out outputs/reviews/section-editor-03
.songviz/venv/bin/python -m songviz.review_server \
  --directory outputs/reviews --port 8770
```

The default parent is `structure-review-03`. Choose a new output directory if one
already exists; existing packages are never overwritten. Playback uses the native
audio element and the proven byte-range server, not full-song JavaScript Blobs.

Implementation: `experiments/section_editor_state.js` owns immutable annotation
operations and validation; `experiments/templates/section_editor.html` owns the
editor; `experiments/build_section_editor.py` creates the isolated package with
source/code fingerprints. Terra wrote the editor and state module; the lead
handled packaging, integration and verification.

## Verification

- Python suite: **463 passed** (123 warnings).
- State module: **8 Node tests passed**, including immutable edits, noncontiguous
  or invalid input rejection, cross-layer boundaries and imported-ID collision
  prevention without browser UUID support.
- `experiments/check_section_editor.cjs` passed against the final served page:
  blank start, edits/merge text retention, undo/redo, independent layers,
  shortcut safety, native playback/seeking/retry, local draft reload, JSON
  export/hash and import validation, mobile layout, blocked storage and recovery
  of unreadable drafts. Invalid embedded data cannot be exported.
- The formerly failing MCP browser loads the final audio successfully. The page
  starts blank, and the mark-boundary button remains in the sticky player.
- Five input snapshot hashes and three output hashes match; the HTML matches its
  snapshot, embedded editor data and manifest hash. Audio matches the parent WAV
  byte-for-byte. Export manifest SHA-256:
  `7f1befc68052be07cca3828f32b1b0df4c616e2128f59cb3fc35adf0433cfe32`.

Reproduce state tests with `node --test tests/section_editor_state.test.cjs`.
For browser checks, use `node experiments/check_section_editor.cjs URL`, setting
`SONGVIZ_PLAYWRIGHT_MODULE` to an existing Playwright installation when needed.
These checks do not install dependencies or create human reference annotations.
