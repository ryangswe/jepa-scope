# Decisions — Method

Full rationale for how jepa-scope computes what it shows, newest first. The lean, auto-loaded
index is `docs/decisions.md`.

---

## 2026-11-06 — Planning protocol: a Markov planning state as one fixed-shape tensor, one-step stateless predict, model-owned energy, summed objective by default

**Status:** live, indexed

**Context.** The 2026-11-03 product decision said the tools target a model-agnostic protocol
rather than V-JEPA 2. Week 1 drafted that protocol and checked it against `eb_jepa`'s Two Rooms
model. latent-plan will build on it, with V-JEPA 2-AC as its second model, so its shape is costly
to change later.

**Decision.** A world model has `latent_shape`, `action_dim`, `encode(obs)`, `predict(z, a)` and
`energy(z, z_goal)`. `z` is a planning state: one fixed-shape tensor holding everything the next
prediction needs, which may be more than the latent `energy` compares (for V-JEPA 2-AC, its token
context and the arm's pose). `predict` takes one step and keeps no state. `encode` accepts the
model's own observation type. `plan_cost` sums the energy over all `T + 1` rollout states by
default, with `"last"` as a planner setting. The owner approved the signatures, and later the
Fable architect's changes (planning state, `encode(obs: Any)`, the objective setting).

**Evidence.** Through an adapter, rollouts and costs match `eb_jepa`'s own exactly, and the
reference CEM succeeded in 16 of 20 episodes against 18 of 20 for `eb_jepa`'s planner on the
same episodes (details in `docs/planning-protocol.md`). vjepa2's planner (`notebooks/utils/
mpc_utils.py`) starts from one image and grows its context from its own predictions, so its
state is Markov and fits a stateless `predict`.

**Rejected.**
- *An explicit history argument or an `initial_state(obs_history)` hook.* Neither target model's
  planner uses an observation history, and a fixed history length does not hold the pose.
- *An opaque state object with a separate latent.* `cem_plan` expands `z0` with tensor
  operations, so it would also need an `expand_state` operation; two new operations before any
  adapter needs them.
- *A summed objective hard-coded into `plan_cost`.* V-JEPA 2-AC's planner scores only the final
  frame, and a parity check against it needs the same objective.
- *Deferred, not rejected:* `latent(z) -> [B, *latent]` for drawing latents along a plan, to be
  decided when latent-plan is planned.

---

## 2026-11-05 — Surprise = mean L1 between predictor output and layer_norm(target), with the context frames encoded on their own; never the HF full-clip forward

**Status:** live, indexed

**Context.** Surprise is the per-patch prediction error for the second half of each window. Two
mistakes each produce a flat, useless signal while running without any error, and both were made
and caught during Week 1.

**Decision.** `Encoder.predict` runs the encoder on the context frames only, then calls the
predictor with token-id masks (the ids double as RoPE positions). `Encoder.surprise` compares that
prediction with `F.layer_norm` of the full-clip encoder output for the target tokens, using mean
absolute error. This matches `forward_target` and `loss_fn` in `app/vjepa/train.py` of
`facebookresearch/vjepa2`.

**Evidence.** In one measured run on an archery clip spliced onto a bowling clip, surprise stayed
between about 0.57 and 0.62 within each scene and rose to 0.65–0.67 for the four tubelets after
the cut. `tests/test_real_model.py` reproduces that clip but asserts only the ordering: the four
tubelets after the cut score higher than every other tubelet. Without the layer norm, targets had
a standard deviation of about 3.3 against about 0.65 for predictions, and the timeline was flat.

**Rejected.**
- *`VJEPA2Model.forward` with `context_mask`.* It encodes the whole clip and masks afterwards, so
  context tokens have already attended to the future. `test_prediction_ignores_the_future` keeps a
  negative control showing this path leaks.
- *L1 against raw encoder output.* It does not match the training objective and gave a flat line.
- *Correcting the per-horizon rise.* Within each window, tubelets further from the context are
  about 0.01 harder to predict, which shows as a small sawtooth on the timeline. It is documented
  in the README rather than corrected, to keep the raw energy.

---

## 2026-11-05 — Windows of 16 frames at 8 fps, overlapping by half: context is the first half, every later tubelet is predicted exactly once

**Status:** live, indexed

**Context.** The viewer needs one continuous surprise timeline, while the model predicts within
fixed windows.

**Decision.** Defaults are 8 fps and 16-frame windows (2 seconds, 2,048 tokens on ViT-L) with a
stride of half a window. `embed.timeline_owner` maps every tubelet to exactly one window. The first
half of the first window is context only and has no surprise. Sources slower than the requested
frame rate are refused, because each window would silently cover more time than the header says.

**Rejected.**
- *64-frame windows,* the checkpoint's training length. They give 8,192 tokens, and attention
  cost grows with the square of that count, which is too slow on a laptop.
- *Non-overlapping windows.* Half of every window would have no surprise value.

---

## 2026-11-05 — Any output-changing edit bumps OUTPUT_VERSION, which keys the result cache

**Status:** live, indexed

**Context.** `jepa-scope clip.mp4` caches results in `~/.cache/jepa-scope`. A cache that ignores
code changes keeps showing old results. That is exactly how a pre-fix flat surprise line would
survive an upgrade.

**Decision.** `OUTPUT_VERSION` in `src/jepa_scope/__init__.py` is part of the cache key and is
recorded in each file's header. A cached file from another output version is re-embedded. The key
also covers the video hash, model, frame rate, window, length limit, device, dtype and whether
patches are kept.

**Rejected.**
- *Keying on the package version.* It changes for reasons unrelated to output and can stay the
  same across output changes.
