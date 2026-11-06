# Decisions — Process & tooling

Full rationale for this theme's decisions, newest first. The lean, auto-loaded index is
`docs/decisions.md`.

---

## 2026-11-06 — Milestone plans get an adversarial review by a second model and then an outside review before approval; measured results get an outside review before they become public claims

**Status:** live, indexed

**Context.** Week 1's outside review found claims stated more confidently than the evidence
supported, after the work was done. Weeks 3 and 4's plan then went through a fresh adversarial
review before any code existed, and it found two planned tests that could not fail and one built
on a false assumption about how windows and the timeline relate.

**Decision.** Reviews scale with how hard a mistake is to undo.
- Every milestone plan: a fresh adversarial review by a different model from the one that wrote
  it, a revision, an outside review (a model from another provider) of the revised plan, then the
  owner's approval, all before code changes.
- Measured results that will become public claims (README numbers, launch material): an outside
  review of the evidence first. For results inside a milestone, only when they are borderline or
  surprising; for launch material, always.
- Design calls mid-task, a pass bar set mid-milestone, and a root cause after two failed fixes: a
  second-model review, without asking the owner first.
- Routine code changes: the existing fresh-context `reviewer` pass on every change, nothing more.

**Rejected.**
- *An outside review of every pull request.* Each round costs the owner time to hand over and
  bring back, and routine changes are covered by the fresh-context reviewer.
- *Outside reviews only after a milestone ends.* A plan's mistakes are cheapest to fix before
  code exists.

---

## 2026-11-05 — Milestones follow the roadmap's weekly order; Week 2's scope landed in Week 1, so Weeks 3 and 4 come next

**Status:** live, indexed

**Context.** After Week 1's scaffold, the next milestone was the owner's choice among the
roadmap's candidates.

**Decision.** Milestones follow the roadmap's planned weekly order, finishing a week's remaining
steps before moving on. Week 2's scope (inference, the grid, the overlay, the file format and the
CLI) landed in Week 1, so Weeks 3 and 4 come next.

**Rejected.**
- *Re-choosing the order at each milestone.* The weekly order already reflects the roadmap's
  reasoning; changing it is a roadmap edit, not a per-milestone choice.

---

## 2026-11-05 — Cloud GPUs within the spending cap of an owner-approved plan need no further approval: log the price, run a watchdog, delete the pod

**Status:** live, indexed

**Context.** Training `eb_jepa`'s Two Rooms model on the laptop CPU projected to about 17 hours.
The owner has a RunPod account with credit and said cloud GPUs are fine if the cost is not
excessive.

**Decision.** Paid compute outside an approved plan's spending cap stays owner-only. Within the
cap, a session may create cloud GPUs without asking again, provided the GPU type and hourly
price from a live read go in the plan's log before each pod is created, a watchdog stops the pod
if the session is lost, and the pod is deleted when the job ends. `CLAUDE.md` carries the rule.
The Week 1 plan's cap was $25; it spent about $1.65.

**Rejected.**
- *Asking before every pod.* The cap and the logging already bound the cost, and asking again
  adds owner time for no control.

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
