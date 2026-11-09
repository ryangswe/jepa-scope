---
name: parallel-work
description: Rules for concurrent work — worktrees for code-writing workers that run at the same time, single-instance resources, and running beside another lead session (branch and plan ownership, shared files, version bumps). Load before dispatching workers that run at the same time, before planning an execution strategy, or when another lead is active.
---

# Parallel work

## Concurrency and worktrees

- Tasks that could overlap are not a schedule: the plan says which ones actually run at the same
  time, in the table below.
- **Every concurrent code-writing worker gets its own worktree**, including when the lead and one
  worker both write. Different files do not make one checkout safe: the branch, index, working
  tree, generated files and running commands are shared. Read-only workers, sequential work and a
  single writer need none.
- Dispatch a worktree worker only once its inputs are committed at its base ref or provisioned in
  the worktree, and record the base commit. **A new worktree lacks every untracked and gitignored
  file**: dependencies, build output, local databases, scratch data. `CLAUDE.md` lists what this
  repository's worktrees are missing.
- **Single-instance resources** cannot be shared by concurrent work: a dev server's port, a local
  database file, a GPU, a browser profile, the owner's live checks. `CLAUDE.md` names this
  repository's. A choice not to use worktrees must account for these and for branch isolation,
  not just for file overlap.
- Stage explicit paths. Never `git add .` or a whole directory in the main checkout; untracked
  drafts get swept in.
- **A lead started outside the repository folder** (for example in a parent workspace folder)
  does not get the repository's `.claude/agents` types: `planner`, `executor` and `reviewer`
  are "not found". Dispatch `general-purpose` (or `Plan` for read-only planning), set the model
  the agent file names, and make the brief's first line "read `.claude/agents/<name>.md` first".

## In the plan

Add this to a plan whose tasks will run at the same time; a sequential plan leaves it out. Name
only the tasks that actually overlap.

| Task | Concurrent with | Worker/session | Writes files? | Base ref | Isolation | Shared mutable resources | Integration |
|---|---|---|---:|---|---|---|---|
| T2 | T3 | Executor | Yes | `<branch/SHA>` | Worktree | None | Cherry-pick |
| T3 | T2 | Research worker | No | `<branch/SHA>` | Read-only shared checkout | None | Report only |

For each task in the table, confirm:

- **Reason:** why running it at the same time is useful or necessary.
- **Isolation:** main checkout, read-only shared checkout, worktree, or an external environment.
- **Base ref:** the exact branch or commit.
- **Tracked inputs available:** yes, or how they are provisioned.
- **Shared resources:** none, or the datasets, caches, GPUs, ports, databases or accounts it uses.
- **Output ownership:** the directories, files or artifacts this task owns.
- **Integration method:** direct, merge, cherry-pick or artifact handoff.

## Parallel lead sessions

More than one lead session can run at once (for example one on research beside one on the product).

- Each lead owns its plan and its branch. It does not commit to another lead's branch or edit
  another lead's plan; anything cross-cutting goes to the owner.
- Files every lead touches (`CLAUDE.md`, `docs/decisions.md`, release notes, shared contracts)
  change in small, separate commits. Merge the default branch in just before opening the PR. In a
  decisions-index conflict keep both sides' lines, newest first.
- A shared-contract change can break another lead's branch: say so in the PR description.
- One branch at a time bumps a given version; check the release notes on the default branch first.
- A lead claims the single-instance resources it needs in its plan; if another active plan already
  claims one, ask the owner.
