# jepa-scope

jepa-scope is an open-source (Apache-2.0, public at `ryangswe/jepa-scope`) local inspector for JEPA
video encoders. `jepa-scope clip.mp4` embeds a video with Meta's pre-trained V-JEPA 2 ViT-L on a
laptop and opens a browser viewer with PCA feature colours, a surprise overlay and a surprise
timeline. It is the first of three planned tools; the programme is in `docs/roadmap.md`.

## Layout

- `src/jepa_scope/`: the Python package. `video.py` decodes (PyAV, rotation, downscale),
  `model.py` wraps the encoder and predictor, `embed.py` turns a video into a `.latent` file,
  `server.py` serves it, `cli.py` is the entry point, `grid.py` owns the token layout.
- `web/`: the Vite, React and TypeScript viewer (WebGL2 overlay in `web/src/overlay.ts`). It builds
  into `src/jepa_scope/static/` (git-ignored), which the wheel ships.
- `tests/`: fast tests on a tiny random V-JEPA 2; `test_real_model.py` is opt-in (real weights).
- `docs/`: `roadmap.md`, the decisions record and plans.
- **Shared contracts that change together:** the `.latent` file (`latent_file.py`: tensor names,
  shapes, header keys, `FORMAT_VERSION`) with `embed.py` and `server.py`; and the `/api/meta` JSON
  (`server.py`) with `web/src/types.ts`.

## Commands (run from the repository root)

- Install: `uv sync` and `cd web && npm ci`
- Lint: `uv run ruff check src tests && uv run ruff format --check src tests`;
  viewer `cd web && npx oxlint --type-aware src`
- Typecheck (viewer): `cd web && npx tsc -b`
- Test: `uv run pytest -q`. Real weights (about 1.3 GB download, then cached):
  `JEPA_SCOPE_MODEL_TESTS=1 uv run pytest tests/test_real_model.py -s`
- Build: `./scripts/build.sh` (viewer, wheel, and a check that the wheel contains the viewer)
- Run: `uv run jepa-scope clip.mp4`, or `uv run jepa-scope view x.latent --no-browser` beside
  `cd web && npm run dev` for viewer work
- Workflow check: `python3 .claude/workflow/check.py`

## Working agreements

- **Never change files on the default branch** (`main`/`master`). Branch first
  (`git checkout -b <slug>`). `.claude/hooks/guard-default-branch.py` blocks edits, `git commit`
  and pushes to it as a backstop, but does not judge other shell writes (`sed -i`, `>>`).
- Record decisions a later session could plausibly reopen (direction, architecture, contracts,
  security, rejected alternatives): one line in the auto-loaded index @docs/decisions.md and the
  full entry in its `docs/decisions/<theme>.md`. The index header sets the format and budget.
- **Size the task first, and say which lane in one line.** A *small change* (a few files, easy to
  revert, and no change to a shared contract, security or privacy posture, or product behaviour)
  takes the short lane: branch, change, `/checkpoint`, one `reviewer` pass, then a pull request
  whose description ends with a `Decision:` line if one was made. No plan, no decisions-record
  entry, no closeout. Anything bigger, or a small change that grows, takes the full process.
  A `Decision:` line is promoted to the decisions record only if a later session could reopen it.
- For large milestones, write the plan to `docs/plans/YYYY-MM-DD-<slug>.md` from
  `docs/plans/_template.md`, so a fresh session can execute it from the file alone. Keep its
  `Status:` line and the active list in `docs/plans/README.md` current.
- **Output-changing edits bump `OUTPUT_VERSION`** in `src/jepa_scope/__init__.py` (energy,
  preprocessing, windowing, PCA). It keys the cache; without a bump users see stale results.
- **A signal that runs is not a signal that works.** Wrong JEPA preprocessing or energy code
  produces plausible numbers with no error. Check changes to `model.py`, `embed.py` or `video.py`
  against the real-model test on natural video, not only the fast tests.
- **Test media and model weights never enter the repo.** Use the Kinetics clips the opt-in test
  downloads, or local files outside the checkout.
- **This repository is public.** Write docs, commits and PRs for outside readers: full sentences,
  no references to private repositories or internal tracking.
- `CLAUDE.md` loads in every session and is capped at 14 KB by the workflow check. Always-on text
  is for a rule every session needs; add some only by cutting some. Procedures go in a skill, area
  rules in `.claude/rules/` with a `paths:` header.

## Orchestration and delegation

For non-trivial work the lead session is the orchestrator: it owns the goal, the architecture, the
task graph, integration and every exchange with the owner. Workers are the agents in
`.claude/agents/` (`planner`, `architect`, `executor`, `reviewer`, `Explore`); their checklists
and report formats live there.

- **Milestone specs: do not edit files until the architecture and the first small change are
  explained and approved.** Inspect read-only first.
- Decompose work into tasks with dependencies, success criteria and verification. **Price the
  work in the owner's hours**: they are the scarcest resource.
- Work directly when a task is small, tightly coupled or cheaper to do than to brief. Delegate when
  work can run concurrently, is self-contained, would flood the lead's context, or needs a fresh
  judgment. Every worker gets a bounded task and ends with SUCCESS, ESCALATE or BLOCKED; on
  ESCALATE or BLOCKED, stop the tasks that depend on it and update the plan.
- Present the task graph, delegation and approval gates **before dispatching code-writing
  workers**, unless the owner has authorised autonomous work. Read-only workers need no approval.
  Ask before anything that changes product behaviour, architecture, compatibility, security or
  privacy posture, production data, scope or cost.

### Owner-only actions (each time, never implied by plan approval)

- Merging into the default branch.
- Publishing a release: tags, GitHub releases, PyPI uploads.
- Changing repository settings or visibility, or posting about the project anywhere public.
- Paid compute outside the spending cap of a plan the owner approved. Within that cap, cloud
  GPUs (RunPod or similar) need no further approval, provided each pod's GPU type and hourly
  price from a live read go in the plan's log before it is created, a watchdog stops it if the
  session is lost, and it is deleted when the job ends.

### Models

- **The lead runs on Opus at high effort**, chosen at the start of a session and not switched
  during it. Each model has its own prompt cache, so a switch re-reads the whole conversation
  uncached; ask `architect`, or start a session. Each agent's model and effort are set in its
  agent file; when a plan task names a stronger model, dispatch it on that one.
- **Keep the lead's context small.** The lead does not read a decisions theme file or a plan over
  ~30 KB itself; `Explore` fetches the entry or task it needs.

### Concurrency

- Before running workers at the same time, or beside another lead session, load the
  `parallel-work` skill.
- Every concurrent code-writing worker gets its own worktree. Stage explicit paths; never
  `git add .` or a whole directory.
- Single-instance resources: the viewer's default port 8765 (pass `--port` in a second
  worktree), and the Apple GPU (MPS) when two sessions embed at once. The Hugging Face cache
  (`~/.cache/huggingface`) and the result cache (`~/.cache/jepa-scope`) are shared and safe.
- A new worktree lacks `.venv`, `web/node_modules` and the built viewer: run `uv sync`,
  `cd web && npm ci && npm run build`.

### Review, integration and closeout

- Run `/checkpoint` (the `checkpoint` skill) after each integrated step.
- **Two strikes:** when the same test, typecheck or build error survives two fix attempts, stop
  patching. Ask `reviewer` (or `architect` when the cause is unclear) for a root-cause read before
  a third attempt; a worker reports ESCALATE instead.
- Before declaring non-trivial work done, run `reviewer` with fresh context on the plan and the
  diff. Passing tests are not proof. Fix the material findings, re-run the checks.
- Ending a session mid-task, switching topic, or past ~300–400k tokens: run `/handoff` (the
  `handoff` skill). It records state in the repository and prints a short prompt for the next
  session. Don't let a conversation be the only place work's state lives.
- Weekly, at a handoff, or when `.claude/worktrees/` piles up: run `/gc` (the `gc` skill) to remove
  worktrees and branches whose work is already in the default branch.
- **Close out:** the plan's `Status:` line and the active list; the execution log compressed into
  the completion summary; decision lines that meet the index bar; a short summary for the owner.

<!-- Workflow kit (full). `workflow-kit sync` updates the managed files under .claude/; this file
is the repository's own. -->
