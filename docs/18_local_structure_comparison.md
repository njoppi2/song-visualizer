# Local-change policy comparison

Follow-up to [17_local_structure.md](17_local_structure.md). This experiment
compares two specific explanations for missed changes; it does not implement
a new section partition, identify returning motifs or widen transition intervals.

The user clarified during implementation that a single discrete section layer
is insufficient: small and large transitions and degrees of novelty coexist.
This confirms the scope, rather than requiring a new partitioning algorithm in
this experiment. Keep the full per-scale curves and support, not only selected
peaks. Musical identity labels are optional anchors; graded, potentially
overlapping developments must remain usable without turning each into a cut.

## Fixed experiment design

The control is `outputs/reviews/local-structure-02/`. Recompute its predictions
from the identical verified `structure-evaluation-03` numeric cache and require
exact JSON-data equality before interpreting new results. Original audio,
reviewed grid, human feedback, recurrence and energy-dip logic stay unchanged.

| Variant | Change from control | Question |
| --- | --- | --- |
| `control` | None | Is the frozen result reproducible? |
| `separate_channels` | Independent pattern and arrangement thresholds/peaks | Does the larger arrangement distribution conceal useful pattern changes? |
| `sustained_activity` | Add persistent per-stem RMS changes to control proposals | Are stable entrances/exits or level changes missed by the mean-contrast threshold? |
| `combined` | Add the same activity proposals to the separate-channel policy | Do the sources complement each other, and at what proposal burden? |

Parameters chosen before running these variants against the development labels:
per-channel floor 0.20, median plus 2.5 scaled MAD, contrast windows 2/4/8 beats;
activity windows 4/8 beats, median relative change at least 0.50, and at least
75% of each window on its respective side of the midpoint between medians.
The strongest qualifying stem can propose an event. The existing audibility
floor and two-beat peak spacing remain. There is no parameter search to fit
19 spans or specifically recover the chorus timestamps.

Contrast candidates are ranked by relative threshold exceedance, not by treating
pattern and arrangement raw magnitudes as directly comparable. Activity is
additive: nearby evidence supports an existing base timestamp rather than moving
it. All merged source windows must be retained and availability must cover their
union. Scores and thresholds remain heuristic, not calibrated probabilities.

## Required checks and interpretation

- Synthetic stationary/silent signals, spectral-only changes, level steps,
  sustained entrances/exits, returns, transient spikes and alternating pulses.
- Explicit plateau timing, unavailable edges, finite/null output, support unions,
  deterministic selection, unchanged inputs and unchanged dip intervals.
- Frozen-control equality, source/cache/grid integrity, label-free generation,
  refusal to overwrite artifacts, safe review text and native original-audio
  playback/seek/bounded audition on desktop and mobile.
- All annotated boundaries versus all variants, not just successful excerpts;
  candidate counts/density and reverse nearest-reference distances. No adopted
  matching tolerance, one-to-one accuracy claim or automatic false-positive label.

Inspect the previously missed 78.889763s and 165.533910s chorus variations,
124.295069s verse-2 outro and 158.590911s chorus return. Preserve the useful
evidence near 33.445s and 203.759s. More candidates mechanically reduce nearest
distance; that alone does not establish better musical discrimination.

Persistence rejects an isolated spike but is not semantic importance: vocal
phrases and instrumental articulation may produce many candidates. Median-based
windows can produce plateaus and early timestamps; leftmost selection is not
sample-accurate onset localization. Whole-track floors/thresholds and right-side
windows make this an offline detector. The two narrow dip intervals are a fixed
control, not a solution to full transition extents.

Feel Good Inc remains development data. No automatic production promotion or
claim of held-out quality is justified by this comparison alone.

## Results and handoff

The fixed policies produce the following development results. None is promoted
to production. More local evidence is useful, but it is not a better section
partition or a validated salience measure.

| Variant | Changes | Changes/minute | Energy-dip intervals |
| --- | --- | --- | --- |
| Control | 16 | 4.34 | 2 |
| Separate channels | 39 | 10.58 | 2 |
| Sustained activity + control | 69 | 18.72 | 2 |
| Combined | 79 | 21.43 | 2 |

Nearest candidate minus human mark, in seconds (descriptive, many-to-one):

| Human mark | Control | Separate channels | Sustained activity | Combined |
| --- | --- | --- | --- | --- |
| Chorus energy change 78.890 | -14.080 | -14.080 | +0.214 | +0.214 |
| Verse-2 outro 124.295 | +14.148 | -2.311 | +0.288 | +0.288 |
| Chorus return 158.591 | -14.084 | -0.224 | +0.210 | -0.224 |
| Chorus energy change 165.534 | -21.027 | -0.670 | -0.236 | -0.670 |

The activity policy exposes nearby evidence at all four earlier misses, without
moving/removing the control's timestamps. Its primary evidence is drums at
79.103s, a decrease in the `other` stem at 124.583s, vocals at 158.800s, and drums
at 165.298s. A nearby acoustic proposal is not proof that it expresses the same
musical event the user intended. Especially at 124.295s, do not retrospectively
claim the detected stem change explains the user's whole outro interpretation.

Separate channels alone still miss the first chorus energy change. They add
many pattern changes within the opening vocal passage. Activity adds 53 exact
timestamps to control; the combined result adds 63. These unannotated proposals
are not automatically false positives: many may be real local details that do
not warrant the same visual response as a broad transition.

Combination is not automatically better: the deliberate preference for an
existing contrast timestamp merges the 165.298s activity evidence into a
164.864s contrast candidate, increasing that example's nearest timing error.
Raw support timestamps remain accessible. Both useful control examples near
33.445s and 203.759s are retained. The earlier opening and breakdown endpoint
misses remain; the two dip intervals and their limited overlap are unchanged.

### Next decision

Do not continue raising/lowering thresholds to obtain a section count. The next
bounded analysis step should characterize **multiscale change episodes** from
the retained curves and per-stem evidence: possible onset/peak/settling extents,
direction and contributing layers, without requiring a flat section partition
or a single importance score. Preserve short developments within broader ones
and the source evidence when they overlap. Treat it as an explicit hypothesis
to compare with the narrow dip control, not a pre-established hierarchy.

Keep the four chorus/verse examples as regressions, inspect all five interpreted
transition spans, and include dense within-phrase proposals when judging what
should actually affect direction. Separate later identity/familiarity reasoning
from local activity; neither the 69 nor the 79 proposals is a finished director
input policy. Small visual examples can test graded emphasis alongside this work.

## Review package and verification

Final package: `outputs/reviews/local-structure-comparison-02/`, served at
`http://127.0.0.1:8770/local-structure-comparison-02/` while the review server
is running. Version 01 is preserved as a development package but embeds complete
curves/recurrence data in its page; use **02**. Version 02 keeps the complete
per-variant curves and recurrence context in `variants/<name>/predictions.json`
and `evaluation.json`, while the review page only embeds what it renders.

The page presents all 19 user spans, the four proposal timelines, proposal burden,
every signed nearest-boundary comparison, readable source/channel evidence and
links to complete JSON. It calls candidates acoustic changes, not accepted
sections or a winner. The user’s labels, certainty and explicit motif groups
remain separate from analyst variation/transition fields. Native original audio,
ranged seeking and short bounded audition are used; no Blob fetch or copied audio.

Completed verification:

- Focused variant/comparison suite: **35 passed, 1 warning**.
- Full suite after final package/payload work: **586 passed, 123 warnings**.
- All **78** manifest source, snapshot and output fingerprints matched after the
  final build. The page was regenerated exactly from its snapshotted template and
  `review.json`; the page payload excludes full recurrence context.
- Browser smoke passed native audio loading/playback, bounded audition, plot
  seeking, retry followed by one five-second keyboard seek, readable proposal
  burden, null/unknown rendering, mobile width and no page/console errors.

No cached audio, stems, parent evaluation, raw feedback, frozen control, existing
detector or production rendering was changed. The chosen fixed policies and their
development-only outcomes remain recorded above.
