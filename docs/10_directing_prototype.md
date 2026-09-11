# First directing prototype — 2026-09-10

This is a local, deterministic rule-based prototype, not an LLM-backed director
or a claim of full-song musical understanding. It implements the first saved-plan
path from the revised roadmap. Artistic acceptance remains pending.

## Run it

```bash
.songviz/venv/bin/python experiments/build_directed_review.py --out outputs/reviews/directed-next
```

The command verifies the selected `rhythm-review-01` package, reads existing
source stems, measures energy, generates and validates a direction plan, then
renders the directed and fixed-comparison videos. Defaults are the local Feel
Good Inc file and 130–178s. `--start` and `--end` can select another span covered
by those inputs; this is not yet a general audio-file CLI. No source separation,
beat fitting or note extraction is rerun. No runtime model calls or uploads occur.

Replay from saved evidence and audio without planning or stem analysis:

```bash
.songviz/venv/bin/python experiments/build_directed_review.py --replay outputs/reviews/directed-review-01 --out outputs/reviews/directed-replay-next
```

To try an edited plan, copy it to a separate file, then pass `--plan path/to/edited-plan.json`
with `--replay`. The edit must retain the excerpt interval and signal hash; invalid
plans fail before rendering. Neither operation overwrites the original package.
Replay uses the current renderer code; source snapshots identify the version used
for a prior artifact. Matching code/configuration is needed for matching pixels.

## What was implemented

- `songviz/direction.py`: JSON-compatible version-1 plans, signal/plan validation,
  automatic activity-gap/energy policy, and a fixed always-on comparator.
- `songviz/directed_render.py`: generic ring, ribbon, ticks and rails treatments
  usable by any layer. Plans control visibility, focus, treatment, gain, palette,
  motif identity and scene crossfades. Primary focus occupies the center; support
  geometry is smaller. Event responses and completed-block energy are causal.
- `experiments/build_directed_review.py`: input verification, energy measurement,
  planning/rendering/replay, source snapshots, provenance, evidence image and page.
- `experiments/templates/directed_review.html`: same-position comparison, an
  expandable plan/evidence view, timestamped feedback, and local/HTTP playback.

The plan contains absolute-time contiguous segments, visible/hidden layers,
bounded gains, supported treatment/palette IDs, focus and motif IDs, transition
durations, reasons and evidence references. It is bound to the exact signals by
a content hash. Validation rejects gaps/overlaps, unavailable layer names,
unsupported treatments, invalid focus/gains, unknown evidence references, altered
signal content and times outside coverage. It checks the contract, not whether a
reason is musically persuasive. Reasons and summaries remain inspectable.

Terra implemented the renderer and renderer tests. Luna implemented the review
page and builder tests. The lead implemented the policy/contract, evidence and
package builder, reviewed worker code, and integrated the result. Review caught
and corrected an overlapping-layer crossfade issue, faint-signal overvisibility,
and a vacuous hidden-layer test. Delegation is recorded, not asserted to be an
optimal or measured billing configuration.

## Selected evidence and resulting plan

Main-percussion gaps are inferred from the song-wide prominent kick/snare event
stream. A gap must last at least 4s; its sparse span begins 350ms after the last
main hit and ends at the next. This tail allowance and threshold are explicit
development policy, not musical section boundaries. Sparse spans need not be
silent: hi-hat and other layers can continue.

Bass, vocals and other use stereo-preserving RMS over completed 100ms blocks,
normalized by each stem's whole-song p95 and capped at one. These are relative
activity measurements, not cross-stem absolute loudness, note events or confidence.
The renderer uses sample-and-hold; energy changes may lag by up to 100ms.

| Song interval | Focus | Policy evidence / choice |
| --- | --- | --- |
| 130–137.917s | Snare | Prominent percussion present; bass is supporting; other/vocals omitted |
| 137.917–165.735s | Other stem | No prominent kick/snare; other has highest mean normalized energy; percussion hidden despite some hi-hat evidence |
| 165.735–178s | Vocals | Percussion returns and mean vocal activity exceeds the 0.65 threshold; vocals lead, percussion supports, bass/other hidden |

Warm `percussion-groove` motif identity is reused before and after the gap; the
sparse texture gets a cool ribbon treatment. This is recurrence of activity and
visual vocabulary, not verified melodic/harmonic motif matching. A fixed comparator
uses the same source audio, timing, signals and treatment vocabulary, but holds
focus/visibility/gains/palette constant. This isolates direction choices as a
bundle; it does not isolate one individual palette or gain parameter.

The policy is implemented from input data; these times are its output, not
hand-authored boundaries. However, the song, excerpt and thresholds are development
choices informed by prior reviews. This is not a holdout result. No cached section
role was assumed correct, and no musical surprise detector is claimed.

## Review and limitations

Selected package: `outputs/reviews/directed-review-01/index.html`. Serve locally:

```bash
python3 -m http.server 8768 --bind 127.0.0.1 --directory outputs/reviews/directed-review-01
```

Open `http://127.0.0.1:8768/`. Compare **Directed sequence** and **Always-on comparison**.
Ask whether attention follows the music and what should be emphasized or omitted.
Feedback exports as `songviz-direction-feedback.json` with original-song timestamps
and a manifest hash. Notes are not saved until downloaded; chat feedback also works.

The evidence plot shows vocal activity rising within the long sparse span before
percussion returns. This coarse planner may therefore foreground vocals later
than desired. It currently summarizes each gap/non-gap span rather than detecting
all phrase-level entrances. Treat that as a concrete open review question, not
proof that the three-state direction tells the whole story. The "other" stem is
not a confidently identified instrument, and separation can contain bleed.

Further limitations: no LLM integration, no automatic full-song style development,
no reliable surprise/semantic-role detection, no independent generalization test,
and a deliberately small geometric vocabulary. An edited human plan is supported
for exploration but cannot count as automatic planner output. Production render
commands and cached analysis remain unchanged. Milestone 2 needs user review;
technical success alone does not complete it.

## Verification of the selected package

- 127 focused tests passed across the new directing modules and prior rhythm,
  percussion, review, benchmark and evaluation paths. Tests cover contract
  rejection, data-derived boundaries, clock translation, hidden-layer behavior,
  treatment changes, independent scene crossfades and source-input isolation.
- Both comparison videos contain 2,880 frames at 60 FPS and last 48 seconds.
  The PCM cut exactly matches the original source interval; decoded A/B audio is
  identical. Zero-offset AAC/source correlation is 0.997218. Visual event sampling
  still has up to 16.67ms quantization, separate from the 100ms energy block lag.
- The full `--replay` path generated `directed-replay-01` without planning/stem
  analysis. Plan, signals, PCM, directed MP4 and fixed MP4 all matched the selected
  package byte-for-byte under the same code/configuration.
- Lead image inspection covered the measured activity plot and rendered samples
  from all three focus spans. Browser checks covered duration, same-position
  paused/playing switching, retry after a simulated fetch failure, timestamp
  conversion (local 15s → song 145s), bounds/empty feedback checks and manifest
  linkage. Test exports were intercepted, not saved as user feedback.
- Input snapshots/output checksums verified. Selected manifest SHA-256:
  `320a058b12a358acf505afc714117169c00f2637802ad5829ae367c50c13f040`.
