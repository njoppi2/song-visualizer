# Beat-grid audit — 2026-09-10

Initial scope: one small read-only audit of Feel Good Inc's cached timing and the then-current
section/novelty code paths. No extraction, rendering, beat fitting, cache changes
or algorithm changes were performed during that audit. The subsequently requested
implementation is complete; see [Explicit structural grids](12_structure_grid.md).

## Measured cache comparison

Source cache: `outputs/Gorillaz - Feel Good Inc (featuring De La Soul)/analysis/`.
Reviewed pulse: `outputs/reviews/rhythm-review-01/timing.json`.

| Measurement | Cached `analysis.beats` | Reviewed regular candidate |
| --- | --- | --- |
| Beat count | 335 | 499 |
| Tempo | 92.285 BPM | 138.525 BPM |
| Median beat interval | 0.650159s | 0.433136s |
| 16 × median interval | 10.403s | 6.930s |
| 32 × median interval | 20.805s | 13.860s |

The cached beat object exactly matches the timing review's baseline. The reviewed
candidate is separate and has not replaced that cache. Counts also reflect
different start/end coverage, not just tempo. The window durations above are
nominal calculations from median spacing, not measured durations of every window
on the irregular cached grid. A feature called "16 beats" therefore cannot be
assumed comparable across these grids.

## Code paths and provenance gap (before the follow-up implementation)

- `songviz/story.py::_beat_sync_features` calls `librosa.beat.beat_track` internally
  on the source mix and adds endpoint frames. If fewer than eight beats are found,
  it uses approximately half-second pseudo-beats. `compute_story` calls this
  helper; it does not accept the reviewed beat times as an input.
- Section self-similarity and boundary novelty use that internal grid. The
  multi-scale lag novelty also uses it when available, although its audio
  features can come from the "other" stem. Its separate fallback is a half-second
  grid. The implemented novelty scales are 4, 16 and 32 beats.
- The inspected historical `story.json` does not serialize the internal beat
  times or a beat-grid identifier. Its embedded copy in `analysis.json` is exactly
  equal, but that does not establish which grid generated its features. The
  historical extraction revision is unverified; current source code describes
  today's path, not proof of the historical run.
- `experiments/build_phrase_cluster_diagnostics.py` explicitly reads top-level
  `analysis.beats.beat_times_s` for its 8/16/32-beat comparisons. Running that
  script against this cache would use the old grid, not the reviewed candidate.
  `phrase_cluster_diagnostics.json` and `structure_tool_diagnostics.json` were
  absent from this particular analysis directory; no claim is made about copies
  elsewhere.

Conclusion: the top-level grid mismatch is verified; the historical internal
section/novelty grid is **unknown**, not proven identical to the old top-level
grid. Approving a new pulse has not automatically repaired structural features.

## Follow-up (now implemented separately)

Give the structural analysis an explicit beat-grid input and persist its exact
times, source/hash and fallback status. Then regenerate structural diagnostics
into a new experiment directory using the reviewed candidate. Do not simply
rename or rescale old novelty arrays: their beat-synchronous feature windows
depend on the input grid. Downbeat/bar phase remains a separate unverified choice.

The reviewed constant-tempo pulse has local listening support, not cross-song
validation or verified bar alignment. Preserve the historical caches for a
controlled comparison before judging section boundaries or novelty improvements.

The follow-up now supplies and records the structural grid and regenerates both
timing hypotheses with the same current code. See `12_structure_grid.md` for
results and remaining limitations; this does not recover the historical grid.

## Audited input SHA-256

- `analysis.json`: `f466b3599801d93c03e2d1bd96792769d2bc27c898e05fec8bf00faae25536bf`
- `story.json`: `5dafd953d30f2b765b0a2a2a9521c5aae499249f6b87f72f6bd50d5d8a655fd7`
- Review `timing.json`: `fe0c7b8ed310fbfdf3cf640b7d845513d70cbedfbe83dced06a3f37b2e594b86`
