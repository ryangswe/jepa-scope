# The planning protocol (draft)

This document describes a small, model-agnostic interface for planning with a JEPA world model.
A planner written against it can plan with any world model that implements three operations,
whatever its architecture. The code is in `src/jepa_scope/planning/`. It is the second tool in
the programme described in `docs/roadmap.md`. It is a draft: the signatures may change until the
protocol has been checked against more than one model.

## What a world model implements

A world model is any object with two attributes and three methods. Shapes put the batch first:
`B` is the batch size, `A` is the action size, and `*latent` is the model's latent shape, which
can have any number of dimensions.

| Member | Shape | Meaning |
|---|---|---|
| `latent_shape` | `tuple[int, ...]` | The shape of one latent state, without the batch dimension. |
| `action_dim` | `int` | The size of one action vector, `A`. |
| `encode(obs)` | `[B, *obs]` to `[B, *latent]` | Turns observations into latent states. |
| `predict(z, a)` | `[B, *latent]` and `[B, A]` to `[B, *latent]` | Predicts the latent state one action later. |
| `energy(z, z_goal)` | `[B, *latent]` twice to `[B]` | Scores how far each state is from its goal. |

The three operations have these rules.

- `predict` takes exactly one step and keeps no hidden state between calls. Everything the model
  needs in order to predict must be inside `z`. A recurrent model whose hidden state is its
  latent (like the eb_jepa Two Rooms model) satisfies this directly.
- `energy` is non-negative, is zero when `z` equals `z_goal`, and is lower when `z` is closer to
  the goal. Its scale belongs to the model: energies can be compared within one model, never
  across two models.
- `encode` is used for the start observation and for the goal observation. Planners never call
  it on predicted states.

`WorldModel` in `protocol.py` is a `typing.Protocol`, so a model does not inherit from anything.
`isinstance(model, WorldModel)` checks that the members exist, not their shapes.

## Rolling out and scoring a plan

Two functions are built on the three operations, and planners use only these two.

`rollout(model, z0, actions)` applies `predict` once per action. `z0` is `[B, *latent]` and
`actions` is `[B, T, A]`. It returns `[B, T + 1, *latent]`, with `z0` at index 0 and the state
after action `t` at index `t + 1`.

`plan_cost(model, z0, actions, z_goal)` is the cost of an action sequence. It is the sum of
`energy` to the goal over all `T + 1` states of the rollout, including the start state, and it
returns `[B]`. This is the objective eb_jepa uses for Two Rooms (`sum_all_diffs: true`). Summing
over every state, rather than scoring only the last one, rewards reaching the goal early and
gives every step of the plan a direct signal. It is also less sensitive to prediction error, which
grows with the number of steps. eb_jepa's README reports a Two Rooms success rate of 97% for its
MPPI planner with the summed cost and 89% with only the last state's cost. The start state adds the same amount to every candidate plan, so
including it does not change which plan is best.

## The reference planner: the Cross-Entropy Method

`cem_plan(model, z0, z_goal, config, steps_left=None, generator=None)` in `cem.py` searches for an
action sequence with the Cross-Entropy Method. It plans from one start state (`z0` and `z_goal` are
`[1, *latent]`) and returns a `CEMResult` holding the planned actions `[T, A]` and the lowest
sampled cost in each iteration.

Each iteration works as follows.

1. It samples `num_samples` action sequences from a Gaussian with a per-step, per-dimension mean
   and standard deviation. The mean starts at zero and the standard deviation at `var_scale`.
2. If `max_norm` is set, it rescales every action to a norm of at most `max_norm`.
3. It scores every sequence with `plan_cost`.
4. It takes the `num_elites` lowest-cost sequences and sets the new mean and standard deviation
   to their mean and their unbiased standard deviation.

After the last iteration, the planned sequence is the elite mean, not the best sample seen. The
horizon is `plan_length`, or the steps left in the episode when `steps_left` is given and smaller.
Every call starts afresh from a zero mean. A caller that plans again after each executed action
(model-predictive control) gets eb_jepa's behaviour by executing the first action of each plan.

`CEMConfig`'s defaults are eb_jepa's settings for Two Rooms (`planning_cem.yaml`): a horizon of
90 actions, 20 iterations, 200 samples, 20 elites, an initial standard deviation of 1.5 and an
action norm limit of 2.45. The planner mirrors `CEMPlanner.plan` in eb_jepa's `eb_jepa/planning.py`
at commit `966e61e` step by step, including its norm-rescaling formula, so that the same model can
be planned natively and through the protocol and the two can be compared. Pass a
`torch.Generator` for a reproducible plan.

## Adding a model

To plan with a new model, write an adapter class with the five members in the table above, then
pass an instance to `cem_plan`. The adapter usually wraps the model's own encoder and predictor
and picks an energy, often the mean squared distance between latents. Check it in two ways before
trusting a plan. First, check that `rollout` through the adapter gives the same latents as the
model's own multi-step prediction. Second, check that planning through the adapter succeeds as
often as the model's own planner does on the same episodes.

## Adapters

### eb_jepa Two Rooms

`EbJepaTwoRooms` in `eb_jepa_two_rooms.py` wraps the action-conditioned JEPA in
`examples/ac_video_jepa` of `facebookresearch/eb_jepa` (checked at commit `966e61e`). In that
task, a dot must cross a wall through a door to reach a goal position, and the model sees
two-channel 65 × 65 images. Its latent is a 512-dimensional vector and its actions are 2D
displacements. The three operations map onto the model as follows.

- `encode` applies eb_jepa's Two Rooms `Normalizer` (a per-image min-max scaling followed by
  fixed per-channel statistics) to a raw `[B, 2, 65, 65]` observation, as `DotWall` renders it.
  It then runs the Impala encoder on the result as a one-frame clip and flattens its
  `[B, 512, 1, 1, 1]` output to `[B, 512]`.
- `predict` runs one step of the GRU predictor, with the action as the input and the latent as
  the hidden state, and then applies the encoder's final LayerNorm, which the predictor shares.
  The normalised output is the next latent. eb_jepa feeds it back as the next hidden state, so
  the latent carries everything the next step needs and `predict` is stateless, as the protocol
  requires. This is exactly one step of eb_jepa's `JEPA.unroll` in autoregressive mode.
- `energy` is the mean squared difference over the 512 dimensions. It equals eb_jepa's
  `ReprTargetDistMPCObjective` for one state. eb_jepa's README writes this cost with an L2 norm,
  but its code takes the mean of the squares, and the adapter follows the code.

`EbJepaTwoRooms.from_run(folder)` reads a run folder written by eb_jepa's training script. It
rebuilds the encoder and predictor from the run's config, the way the training script builds
them. It loads only their weights from `latest.pth.tar` (or another checkpoint you name), with
`torch.load(weights_only=True)` and strict key matching, so a wrong key mapping fails loudly
instead of leaving weights random. `EbJepaTwoRooms.untrained(folder)` builds the same
architecture with fresh weights, as a control.

eb_jepa is not a dependency of jepa-scope and is not on PyPI. To use the adapter, install both
into one environment. The check below used Python 3.12 on macOS (Apple silicon) and these steps.

```bash
git clone https://github.com/facebookresearch/eb_jepa && cd eb_jepa && git checkout 966e61e
uv venv --python 3.12 ../.venv
uv pip install --python ../.venv/bin/python torch==2.6.0 torchvision==0.21.0 einops fire \
  omegaconf pyyaml tqdm gymnasium scikit-learn pandas matplotlib seaborn opencv-python \
  imageio wandb pytest av
uv pip install --python ../.venv/bin/python -e . --no-deps
uv pip install --python ../.venv/bin/python -e <path to jepa-scope> --no-deps
```

`--no-deps` skips `decord`, which eb_jepa declares but the Two Rooms example never imports, and
which has no wheel for macOS on Apple silicon. On a Linux server, use `opencv-python-headless` in
place of `opencv-python`.

## Checked against

**eb_jepa Two Rooms, native planner.** The reference model was trained with eb_jepa's own
`examples/ac_video_jepa` script at commit `966e61e` (random wall positions, seed 1, 12 epochs, on
one NVIDIA L40S in about 1.6 hours). eb_jepa's own CEM evaluation (`planning_cem.yaml`, 20
episodes of up to 200 steps) then succeeded in 18 of 20 episodes (90%). With 20 episodes, a
success rate near 90% has a standard error of about 7 points. eb_jepa's README reports 96 ± 2%
for CEM, where each figure averages three training seeds and the last three checkpoints and
± 2 is their spread. One run and one checkpoint is noisier than that, so 90% is consistent with
it.

eb_jepa counts an episode as a success by the agent's distance to the goal after the last allowed
step, not when it first reaches the goal, and the numbers here use the same rule.

**Through the protocol.** Pending; the adapter and its parity and planning checks come next.

## Open points

- **Predictors with a context window.** V-JEPA 2-AC predicts from several past frames. The
  stateless `predict` can carry that history inside `z`, but whether that is clean enough, or
  whether the protocol needs an explicit history argument, is untested.
- **Stochastic predictors.** `predict` returns one next state. Models that predict a distribution
  would need samples or an expected energy.
- **Energy across models.** Energies are only comparable within one model. Comparing planners
  across models would need a calibrated scale. jepa-scope's surprise view meets the same problem
  when it compares prediction errors between clips.
- **Goals per sample.** `plan_cost` takes one goal per batch row. Planning towards several goals
  at once works by repeating the start state, which costs memory.
- **Action spaces.** The protocol says only how long an action is. Bounds, discrete actions and
  units are left to the planner's settings, such as `max_norm`.
