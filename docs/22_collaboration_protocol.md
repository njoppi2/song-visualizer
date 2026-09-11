# Two-team collaboration protocol

This protocol is for two independent Codex/Astra teams working in the same
SongViz repository, with a human relaying messages between them. It also works
when only one team is running. The repository is the shared workspace; the
human is the coordination channel. No custom orchestration service is needed.

## Entry order and source of truth

At the start of every work session, read in this order:

1. `AGENTS.md` for repository-level operating rules.
2. `CONTINUE.md` for the current status, product intent, known failures, next
   step, artifact locations and validation commands.
3. The task-specific documents linked by `CONTINUE.md`, including the latest
   experiment document when the task concerns an experiment.
4. The relevant code, tests, manifests and existing artifacts, after checking
   their provenance and the working-tree state.

`CONTINUE.md` is the sole mutable current-status and queue document. Do not
create a second status index, queue, or competing handoff. Detailed evidence
belongs in the task's experiment document or an explicitly requested durable
document. A protocol document is not permission to rewrite `CONTINUE.md` for
routine communication; update it only at a meaningful handoff, as required by
`AGENTS.md`.

## Default ownership

Ownership is exclusive for the duration of a task and is recorded in the
human-relayed request or in the current `CONTINUE.md` handoff:

- **Team 1 — lead/integration owner:** defines the question and acceptance
  boundary, selects the task and files in scope, resolves competing results,
  reviews actual diffs and artifacts, runs or authorizes final validation, and
  makes the final promotion or handoff decision. Team 1 owns `CONTINUE.md` and
  integration unless the handoff explicitly says otherwise.
- **Team 2 — bounded counterpart:** performs the independently assigned
  implementation, analysis, or tests within the named scope; records evidence,
  failures and limitations; and returns a compact result to Team 1. Team 2
  does not alter Team 1's files, queue, controls, or conclusions unless those
  paths are explicitly assigned.

For a task led by Team 2, the same rule applies with the labels swapped: the
request must name the integration owner and the exact write scope. Ownership
must never be inferred from which team notices a file first. A file or artifact
has one owner at a time; there is no concurrent ownership and therefore no
merge-conflict workflow between the two teams.

Use disjoint write scopes. If the requested work would touch an owned path,
stop before editing and ask the human to relay a reassignment or a completed
handoff. Reading shared files is fine; simultaneous edits to the same file are
not.

## Direct messages and durable handoffs

A direct message is a short, human-relayed request or reply used for the
current bounded task. It is not repository state and must not be treated as an
instruction that survives context loss. The receiver acknowledges the scope
before editing.

A durable handoff is information needed by a later session or by a team that
will continue after the current team stops. The integration owner records it
in `CONTINUE.md` at a meaningful handoff, with links to the detailed
experiment/evidence document and immutable artifact paths. Include what was
actually done, validation actually run, failures/limitations, ownership and
one concrete next step. Do not use chat history, a new queue file, or a GitHub
Actions run as the durable handoff.

## Compact request/reply template

The human should relay this exact compact shape. Keep one request to one
bounded scope.

```text
REQUEST
owner: Team 1 | Team 2
to: Team 1 | Team 2
goal: <one observable outcome>
read: <files/docs to read first>
write_scope: <exact files/dirs allowed; none if read/test only>
do_not_touch: <owned files, controls, source, other scopes>
validation: <exact commands/checks and evidence expected>
handoff: direct | durable-in-CONTINUE
ask_human_if: <specific condition requiring counterpart or decision>

REPLY
status: accepted | blocked | complete
changed: <paths, or none>
result: <one-sentence outcome>
evidence: <commands, artifact paths, hashes or concise findings>
limitations: <known uncertainty/failure, or none>
next: <integration owner action, or explicit human relay request>
```

The reply reports observed results, not merely an agent's assertion. If the
work is incomplete, use `blocked` only for a real dependency and state what
the human must relay. A durable reply is summarized in `CONTINUE.md` by the
integration owner; the full evidence stays in its linked document.

## When to continue and when to ask for the counterpart

Keep progressing when the task is within the assigned write scope, inputs are
available and provenance is clear, the other team is not needed for a decision,
and the next action cannot overwrite an owned control or artifact. One team
running is normal: it may complete its bounded work, run relevant checks and
leave a durable handoff for the absent team.

Ask the human to run or contact the counterpart when the next step requires
the counterpart's exclusive files, independent review, a decision outside the
current acceptance boundary, unresolved ownership, or a result that must be
compared independently before promotion. Also ask when a required input,
credential, external action, or human judgment is missing. Do not wait for an
absent team merely to preserve symmetry, and do not make a counterpart's
decision by assumption.

## Integration, Git and artifacts

The named integration owner inspects the actual diff, tests and generated
outputs, then integrates only after the bounded result is accepted. Git is the
shared audit trail: use normal branches/status/diff/log and small, descriptive
commits when the owner or user requests commits. Never reset, checkout over,
clean, delete, or overwrite another team's work; preserve unrelated existing
and untracked changes. A clean commit is not proof of correctness.

Generated artifacts are evidence, not automatically source. Create new
experiment output directories, preserve controls, verify manifests and hashes,
and never silently promote a candidate. GitHub Actions artifacts are temporary
transport/build outputs and are unsuitable as canonical project state. The
canonical record is the repository's reviewed source plus the durable
`CONTINUE.md` handoff and linked, provenance-bound artifacts. Do not silently
commit generated artifacts; commit them only when the task explicitly defines
them as repository deliverables and the integration owner has reviewed their
provenance and size.
