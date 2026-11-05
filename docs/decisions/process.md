# Decisions — Process & tooling

Full rationale for this theme's decisions, newest first. The lean, auto-loaded index is
`docs/decisions.md`.

---

## 2026-11-05 — Adopted the workflow kit at the full level, with its process files, the roadmap and this record published in the public repo

**Status:** live, indexed

**Context.** jepa-scope is the first of three planned tools and will run over many sessions. Until
now its plan lived only in the conversation that produced it. The owner asked for the plan to be
written down and for the same written workflow used in their other projects.

**Decision.** The repository uses the workflow kit (github.com/ryangswe/claude-workflow) at the
**full** level: a committed `.claude/` with the planner, architect, executor, reviewer and Explore
agents, the checkpoint, handoff and parallel-work skills, and a branch guard. It also has this
decisions record, plans in `docs/plans/`, and `.claude/workflow/check.py` keeping them consistent,
which CI runs. The kit's managed files are updated with `workflow-kit sync`; `CLAUDE.md` and these
docs are the repository's own. The owner approved publishing the process files, `docs/roadmap.md`
and this record in the public repository.

**Rejected.**
- *The light level.* It has no plans or planner, and this is a multi-session programme.
- *Keeping the process files private.* The owner chose to build in public. The files describe how
  the work is done and contain nothing sensitive.

---

## 2026-11-05 — CI installs uv.lock versions as constraints with CPU torch; the real-model test stays opt-in and runs locally

**Status:** live, indexed

**Context.** The locked Linux torch pulls about 3 GB of CUDA packages that a CI runner cannot use.
A first version of the workflow installed fresh versions instead and so tested a different
environment from the lockfile's.

**Decision.** `.github/workflows/ci.yml` checks the lockfile, exports it as constraints, and
installs it with `uv pip install --torch-backend cpu`. Every package except torch and torchvision
is pinned to its locked version, and those two use the CPU build. uv is pinned to 0.13.6, the
version used locally when the lockfile was written. The real-model test needs about 1.3 GB of
weights and is meaningful only on natural video, so it stays opt-in and is run locally before
changes to the model, embedding or decoding code.

**Rejected.**
- *`uv sync --locked` in CI.* It downloads the CUDA packages on every uncached run.
- *Changing the lockfile to CPU torch on Linux.* That would give Linux users with NVIDIA GPUs a
  CPU-only install.
