# SongViz agent entry point

Read [CONTINUE.md](CONTINUE.md) before project work. It is the single maintained
handoff for current status, product intent, known failures, the next experiment,
artifact locations and validation commands. Follow its links for task-specific
details; do not reconstruct the current queue from historical phase numbers.

When work is split across independent accounts, also read
[docs/22_collaboration_protocol.md](docs/22_collaboration_protocol.md). The active
team assignment, owned paths and integration owner are in `CONTINUE.md`; the
protocol is durable procedure, not another status index. Do not edit another
team's owned paths or the current checkpoint unless you are the named integration
owner. A counterpart may be offline: leave a compact durable request and tell the
human exactly which team to run, rather than blocking on an assumed live channel.

For implementation tasks, the user's preferred workflow is a lead handling
experimental design and final review, delegating bounded implementation/tests to
Terra agents when available. Delegation is explicitly requested for that work.
Use focused context and disjoint write scopes; inspect actual changes/results,
not just agent summaries. Model choice is not proof of correctness.

Preserve existing/untracked changes, raw feedback, source audio and frozen review
packages. Verify provenance before reusing caches. New experiments get new output
directories; do not overwrite controls or promote candidates silently.

At a meaningful handoff, update CONTINUE.md with the completed result, failures,
validation actually performed and one concrete next step. Keep detailed evidence
in the linked experiment document. Do not create another competing status index.
