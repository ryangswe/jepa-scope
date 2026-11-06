# Decisions index

One short line per **live** decision, so every session sees the settled surface cheaply and does
not re-litigate it. **This index is auto-loaded every session**, so every byte is paid by every
conversation. The *why*, the rejected alternatives and every past decision live in the per-theme
detail files under `docs/decisions/` named in each heading. **Before reopening or contradicting a
decision, read its detail entry**: `rg -A2 '^## 20'` on the theme file lists every entry with its
status; read that entry only, not the file.

**What earns an entry:** a choice a future session could plausibly reopen: direction,
architecture, contracts, security and privacy posture, licensing, or an alternative that was tried
and rejected. **Not an entry:** experiment results, details settled inside an approved plan, and
anything the code already shows.

**Format and budget** (checked by `.claude/workflow/check.py`):
- `- YYYY-MM-DD — <decision>`, newest first under its theme, at most 250 characters (aim for about
  150).
- Each detail entry opens with `**Status:**` `live, indexed`, `live, not indexed`, `superseded by
  <date>`, `withdrawn` or `record` (results only). Update it when its line goes. An entry covered
  by a line of another date says `live, indexed — under the <date> line`.
- Add the full entry (context, decision, rejected alternatives) to the matching
  `docs/decisions/<theme>.md` as `## YYYY-MM-DD — <decision>`, newest first, `---` between
  entries. Start a new theme file only if none fits, and add its section here.
- The whole index stays under 20 KB. Near it, prune lines first.

---

## Method · `docs/decisions/method.md`

- 2026-11-06 — Planning protocol: a Markov planning state as one fixed-shape tensor, one-step stateless predict, model-owned energy, summed objective by default
- 2026-11-05 — Surprise = mean L1 between predictor output and layer_norm(target), with the context frames encoded on their own; never the HF full-clip forward
- 2026-11-05 — Windows of 16 frames at 8 fps, overlapping by half: context is the first half, every later tubelet is predicted exactly once
- 2026-11-05 — Any output-changing edit bumps OUTPUT_VERSION, which keys the result cache

## Product & data · `docs/decisions/product.md`

- 2026-11-06 — The planning protocol is drafted here (`src/jepa_scope/planning/`, `docs/planning-protocol.md`) and moves to latent-plan when that project starts
- 2026-11-05 — Apache-2.0, public at ryangswe/jepa-scope; V-JEPA 2 weights are downloaded at run time, never redistributed
- 2026-11-05 — `.latent` = safetensors plus a validated JSON header; the viewer refuses a video whose SHA-256 differs from the header
- 2026-11-03 — jepa-scope is the first tool; the order of the other two is a proposal; tools target a model-agnostic protocol rather than V-JEPA 2

## Process & tooling · `docs/decisions/process.md`

- 2026-11-06 — Milestone plans get a second-model adversarial review, then an outside review, before approval; public result claims get an outside review first
- 2026-11-05 — Cloud GPUs within the spending cap of an owner-approved plan need no further approval: log the price, run a watchdog, delete the pod
- 2026-11-05 — Milestones follow the roadmap's weekly order; Week 2's scope landed in Week 1, so Weeks 3 and 4 come next
- 2026-11-05 — Adopted the workflow kit at the full level, with its process files, the roadmap and this record published in the public repo
- 2026-11-05 — CI installs uv.lock versions as constraints with CPU torch; the real-model test stays opt-in and runs locally
