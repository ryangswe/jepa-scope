# Decisions — Product & data

Full rationale for what jepa-scope is, what it stores and how it is published, newest first. The
lean, auto-loaded index is `docs/decisions.md`.

---

## 2026-11-06 — The planning protocol is drafted here (`src/jepa_scope/planning/`, `docs/planning-protocol.md`) and moves to latent-plan when that project starts

**Status:** live, indexed

**Context.** The roadmap's Week 1 included running `eb_jepa`'s Two Rooms example and drafting
the planning protocol, which belongs to latent-plan, the third tool, and does not exist yet.

**Decision.** The draft lives in `src/jepa_scope/planning/` (protocol, reference CEM, `eb_jepa`
adapter) with its spec in `docs/planning-protocol.md`. It depends only on torch; `eb_jepa` is
imported only when the adapter is built and is never a dependency. The code moves to latent-plan
when that project starts.

**Rejected.**
- *Starting the latent-plan repository now.* Its plan, UI and audience are not decided; a draft
  in an existing tested package is cheaper to change.

---

## 2026-11-05 — Apache-2.0, public at ryangswe/jepa-scope; V-JEPA 2 weights are downloaded at run time, never redistributed

**Status:** live, indexed

**Context.** The project's purpose is an open developer layer for JEPA models, so it is built in
public.

**Decision.** The code is Apache-2.0 and the repository is public. The tool downloads V-JEPA 2
weights from Hugging Face when it first runs, and neither the repository nor the wheel contains
them. Meta releases V-JEPA 2 under MIT, according to the model card and the
`facebookresearch/vjepa2` repository. Test media is never committed either: the opt-in test
downloads Kinetics sample clips at run time, and `.gitignore` blocks video files.

**Rejected.**
- *Bundling a small checkpoint or sample video.* It is a licensing and repository-size burden with
  no benefit, since the first run downloads what it needs.

---

## 2026-11-05 — `.latent` = safetensors plus a validated JSON header; the viewer refuses a video whose SHA-256 differs from the header

**Status:** live, indexed

**Context.** Latents carry no record of their layout, model or preprocessing. Pairing them with the
wrong video or settings draws plausible but wrong overlays.

**Decision.** A `.latent` file is a plain safetensors file with one JSON header under the metadata
key `jepa_scope` (`FORMAT_VERSION` 1). The header records the model, grid, window, preprocessing
recipe and hash, crop rectangle, source video hash, size and rotation, and the run. `load` validates
the header keys and every tensor shape against the grid and the window count. `jepa-scope view`
refuses a video whose size or SHA-256 differs from the header. Files are written to a temporary
name and renamed, so an interrupted write never leaves a half file behind.

**Rejected.**
- *NumPy `.npz` or a directory of files.* Safetensors reads single tensors without loading the
  whole file, has no code-execution path, and has readers for Rust and JavaScript, which later
  tools may need.
- *Trusting the video path in the header.* Videos move and get re-encoded.

---

## 2026-11-03 — jepa-scope is the first tool; the order of the other two is a proposal; tools target a model-agnostic protocol rather than V-JEPA 2

**Status:** live, indexed

**Context.** `docs/roadmap.md` proposes three solo-sized tools: jepa-scope (a latent inspector),
jepa-probe (few-shot heads behind an HTTP API) and latent-plan (a planning runtime). The owner
asked for jepa-scope to be scaffolded first.

**Decision.** jepa-scope comes first because it needs only pre-trained weights, has the clearest
demo, and produces the latent format and viewer shell the other two reuse. Tools target a protocol
(the `.latent` contract now, and `encode`, `predict` and `energy` for planning later), so new
weights from Meta, AMI Labs or others become an adapter rather than a rewrite. The order of
jepa-probe and latent-plan stays a proposal until a milestone for one of them is planned.

**Rejected.**
- *Starting with latent-plan.* It carries the most research risk, because planning quality depends
  on the model, and it has the smallest audience today.
- *Training a model.* That is out of reach for a solo developer and unnecessary for a developer
  layer.
