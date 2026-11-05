---
name: parallel-work
description: Rules for concurrent work — worktrees for code-writing workers that run at the same time, single-instance resources, and running beside another lead session (branch and plan ownership, shared files, version bumps). Load before dispatching workers that run at the same time, before planning an execution strategy, or when another lead is active.
---

# Parallel work

## Concurrency and worktrees

- A parallel group in a plan is eligibility, not a schedule: the plan says which tasks actually
  run at the same time.
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
