# The planning protocol (draft)

This document describes a small, model-agnostic interface for planning with a JEPA world model.
A planner written against it can plan with any world model that implements three operations,
whatever its architecture. The code is in `src/jepa_scope/planning/`. The protocol is layer E of
the toolchain described in `docs/roadmap.md` and belongs to latent-plan, the third of its tools;
it is drafted here and moves there when that project starts.

**Status: experimental.** The protocol has been checked against one model, eb_jepa's Two Rooms
world model. Treat the interface as provisional until a second adapter (V-JEPA 2-AC is the
intended one) reproduces its own model's native rollouts. The signatures may change until then.

## What a world model implements

A world model is any object with two attributes and three methods. They work on a **planning
state** `z`: one tensor of fixed shape that holds everything the model needs to predict the next
step. For many models the planning state is simply the latent. For others it holds more, such as
earlier predicted frames, past actions or a robot arm's pose, and the model's `energy` decides
which part to compare with the goal. Shapes put the batch first: `B` is the batch size, `A` is
the action size, and `*state` is the shape of one planning state, which can have any number of
dimensions.

| Member | Shape | Meaning |
|---|---|---|
| `state_shape` | `tuple[int, ...]` | The shape of one planning state, without the batch dimension. |
| `action_dim` | `int` | The size of one action vector, `A`. |
| `encode(obs)` | `B` observations to `[B, *state]` | Turns observations into planning states. |
| `predict(z, a)` | `[B, *state]` and `[B, A]` to `[B, *state]` | Predicts the planning state one action later. |
| `energy(z, z_goal)` | `[B, *state]` twice to `[B]` | Scores how far each state is from its goal. |

The three operations have these rules.

- `encode` accepts whatever observation type the model needs: an image tensor, or an image
  together with a robot's pose, for example. Planners pass observations through without looking
  inside them. `encode` is used for the start observation and for the goal observation, and
  planners never call it on predicted states.
- `predict` takes exactly one step and keeps no hidden state between calls. Everything the model
  needs in order to predict must be inside `z`. A recurrent model whose hidden state is its
  latent (like the eb_jepa Two Rooms model) satisfies this directly. A model that predicts from
  several frames keeps them inside `z`.
- `energy` is non-negative and lower when `z` is closer to the goal. It is zero when the parts of
  `z` that the model compares equal those of `z_goal`, so a goal state can leave the parts that
  are not compared (such as a pose) as placeholders. Its scale belongs to the model: energies can
  be compared within one model, never across two models.

`WorldModel` in `protocol.py` is a `typing.Protocol`, so a model does not inherit from anything.
`isinstance(model, WorldModel)` checks that the members exist, not their shapes.

### Rules every adapter follows

Planners rely on these, and the toy models in `tests/test_planning_protocol.py` check them.

- **Inputs are read-only.** `predict` and `energy` never write into the tensors they are given,
  and a state they returned earlier never changes later. Planners branch many candidates from one
  start state, often as an expanded view that shares memory, so writing into it would corrupt
  every branch.
- **Inputs may be expanded or non-contiguous.** Copy before writing, as the packed-context toy
  model does.
- **Batch rows are independent.** Each row's prediction and energy depend only on that row. No
  normalisation or reduction may run across the batch. Rows can be at different points of a
  rollout, so a packed state that records how much of itself is valid must be read row by row.
- **A packed state has a declared capacity and refuses to overflow.** If the state holds a
  context sized for a maximum number of steps, the adapter documents that number, and `predict`
  raises an error when a state is already full. It never truncates old steps, drops new ones or
  resizes the tensor. Callers keep the planning horizon within the capacity.
- **Inference only.** The protocol promises derivative-free inference: planners call it without
  gradients, and nothing guarantees that gradients flow through an adapter. The adapter puts its
  own modules in evaluation mode and moves inputs to its own device and dtype. `predict` is
  deterministic; stochastic predictors are an open point.

### Actions

`action_dim` gives only the length of an action vector. Every adapter documents what an action
means: the order of its coordinates, their units or normalisation, how much time one action
covers (including any repeat), its bounds, and how a planned action is converted into the action
an environment executes. A planner's settings must respect those bounds; `energy` does not.

## Rolling out and scoring a plan

Two functions are built on the three operations, and planners use only these two.

`rollout(model, z0, actions)` applies `predict` once per action and keeps every state. `z0` is
`[B, *state]` and `actions` is `[B, T, A]`. It returns `[B, T + 1, *state]`, with `z0` at index 0
and the state after action `t` at index `t + 1`. It is for callers that need the trajectory;
planners do not.

`plan_cost(model, z0, actions, z_goal, objective="sum")` is the cost of an action sequence, and it
returns `[B]`. It keeps only the current state, so its memory does not grow with the number of
actions. The objective is a setting of the planner, not of the model, and has two values.

- `"sum"`, the default, adds up the `energy` to the goal of all `T + 1` states, including the
  start state, one step at a time in the dtype `energy` returns. It gives every step of the plan
  a cost, which favours plans that reach the goal early and stay there. It is the objective
  eb_jepa uses for Two Rooms (`sum_all_diffs: true`), and eb_jepa's README reports a Two Rooms
  success rate of 97% for its MPPI planner with it against 89% with only the last state's cost.
  That supports it for this task, not as a general rule: a plan that must detour away from the
  goal is penalised for the detour, and a model's errors add up across the sum as well. The start
  state adds the same amount to every candidate, so including it does not change which plan wins.
- `"last"` scores only the final state. Other native planners use it, V-JEPA 2-AC's among them,
  and comparing a model with its native planner needs the same objective.

## The reference planner: the Cross-Entropy Method

`cem_plan(model, z0, z_goal, config, steps_left=None, generator=None)` in `cem.py` searches for an
action sequence with the Cross-Entropy Method. It plans from one start state (`z0` and `z_goal` are
`[1, *state]`, on the same device) and returns a `CEMResult` holding the planned actions `[T, A]`
and the lowest sampled cost in each iteration.

Each iteration works as follows.

1. It samples `num_samples` action sequences from a Gaussian with a per-step, per-dimension mean
   and standard deviation. The mean starts at zero and the standard deviation at `var_scale`.
2. If `max_norm` is set, it rescales every action to a norm of at most `max_norm`.
3. It scores every sequence with `plan_cost`, using the configured `objective`.
4. It takes the `num_elites` lowest-cost sequences and sets the new mean and standard deviation
   to their mean and their unbiased standard deviation.

After the last iteration, the planned sequence is the elite mean, not the best sample seen. The
horizon is `plan_length`, or the steps left in the episode when `steps_left` is given and smaller.
Every call starts afresh from a zero mean. A caller that plans again after each executed action
(model-predictive control) gets eb_jepa's behaviour by executing the first action of each plan.

**What it supports.** Continuous action vectors with an optional bound on each action's norm,
and nothing else. Actions are sampled and returned in float32 on `z0`'s device, whatever the
planning state's dtype. Because the elite mean of actions inside a norm bound is also inside it,
the returned plan respects the bound. Per-coordinate bounds, discrete parts such as a gripper, or
any constraint whose average is not itself valid would need the planner to project its samples
and its final plan; this planner does not. Settings must be finite, counts must be integers, and
a model that scores a candidate as NaN or infinite makes `cem_plan` raise an error rather than
return a plan.

`CEMConfig`'s defaults are eb_jepa's settings for Two Rooms (`planning_cem.yaml`): a horizon of 90
actions, 20 iterations, 200 samples, 20 elites, an initial standard deviation of 1.5, an action
norm limit of 2.45 and the summed objective. The planner mirrors `CEMPlanner.plan` in eb_jepa's
`eb_jepa/planning.py` at commit `966e61e` step by step, including its norm-rescaling formula, so
that the same model can be planned natively and through the protocol and the two can be compared.
Pass a `torch.Generator` for a reproducible plan.

## Adding a model

To plan with a new model:

1. Write an adapter class with the five members in the table above, following the adapter rules.
   It usually wraps the model's own encoder and predictor and picks an energy, often a mean
   squared or absolute distance between latents. If the model needs more than one latent, or
   past actions or poses, to predict, the planning state packs all of them into one fixed shape.
   List every input the model's own predictor takes and show where each one lives in the state.
2. Document the model's actions as described under Actions, and the state's capacity if it has
   one.
3. Check the adapter before trusting a plan. First, `rollout` through the adapter must give the
   same states as the model's own multi-step prediction. Second, if the model has its own
   planner, the reference planner fed the same random noise should return the same plan. Third,
   planning through the adapter should succeed about as often as the model's own planner on the
   same episodes.
4. Plan with `cem_plan`, then convert each planned action into an environment action as
   documented.

## Adapters

### eb_jepa Two Rooms

`EbJepaTwoRooms` in `eb_jepa_two_rooms.py` wraps the action-conditioned JEPA in
`examples/ac_video_jepa` of `facebookresearch/eb_jepa` (checked at commit `966e61e`). In that
task, a dot must cross a wall through a door to reach a goal position, and the model sees
two-channel 65 × 65 images. Its planning state is its 512-dimensional latent. The three operations
map onto the model as follows.

- `encode` applies eb_jepa's Two Rooms `Normalizer` (a min-max scaling of each channel of each
  image, followed by fixed per-channel statistics) to a raw `[B, 2, 65, 65]` observation, as
  `DotWall` renders it. It then runs the Impala encoder on the result as a one-frame clip and
  flattens its `[B, 512, 1, 1, 1]` output to `[B, 512]`.
- `predict` runs one step of the GRU predictor, with the action as the input and the latent as
  the hidden state, and then applies the encoder's final LayerNorm, which the predictor shares.
  The normalised output is the next latent. eb_jepa feeds it back as the next hidden state, so
  the latent carries everything the next step needs and `predict` is stateless, as the protocol
  requires. This is exactly one step of eb_jepa's `JEPA.unroll` in autoregressive mode. The state
  has no capacity limit.
- `energy` is the mean squared difference over the 512 dimensions. It equals eb_jepa's
  `ReprTargetDistMPCObjective` for one state. eb_jepa's README writes this cost with an L2 norm,
  but its code takes the mean of the squares, and the adapter follows the code.

**Actions.** An action is the dot's displacement for one environment step, in the
environment's position coordinates (the 65 × 65 pixel grid), in the same coordinate order as
`DotWall`'s positions, with no repeat. eb_jepa's planner limits its norm to 2.45. `DotWall`
declares a box of ±2.45 per coordinate but does not clip, so the bound comes from the planner. A
planned action is executed by passing it, as a float32 NumPy array, to `DotWall.step` unchanged.

**Inference.** `encode` and `predict` run without gradients, with the modules in evaluation mode
on the device given when the adapter is built.

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
episodes of up to 200 steps) then succeeded in 18 of 20 episodes (90%). The evaluation ran on the
same GPU, at about 45 s per episode (921 s in all). With 20 episodes, a success rate near 90% has
a standard error of about 7 points. eb_jepa's README reports 96 ± 2% for CEM, where each figure
averages three training seeds and the last three checkpoints and ± 2 is their spread. One run
and one checkpoint is noisier than that, so 90% is consistent with it.

eb_jepa counts an episode as a success by the agent's distance to the goal after the last allowed
step, not when it first reaches the goal, and the numbers here use the same rule.

The run was trained and evaluated with these commands, from the eb_jepa clone, in the environment
described under Adapters (on a CUDA machine for training). Boolean overrides need a capital
`True` or `False`: eb_jepa merges command-line values into an untyped config, so a lowercase
`false` arrives as the string `"false"`, which counts as true.

```bash
export EBJEPA_CKPTS=<checkpoint folder> WANDB_DISABLED=true
python -m examples.ac_video_jepa.main --logging.log_wandb False --meta.enable_plan_eval False --meta.seed 1
# <run> is $EBJEPA_CKPTS/ac_video_jepa/dev_<date>/<experiment>_seed1, the folder holding latest.pth.tar
cp <run>/config.yaml <run>/config.train.yaml   # the evaluation rewrites config.yaml
python -m examples.ac_video_jepa.main --meta.model_folder <run> --meta.eval_only_mode True \
  --meta.enable_plan_eval True --eval.plan_cfg_path examples/ac_video_jepa/cfgs/planning_cem.yaml \
  --logging.log_wandb False
```

**Through the protocol.** `tests/test_eb_jepa_two_rooms.py` checks the adapter against that same
run on a laptop CPU (Apple M3 Max, float32, torch 2.6.0).

- **Same computation.** On 8 observations and 8 random actions, `rollout` through the adapter
  matches eb_jepa's own `JEPA.unroll` exactly (maximum difference 0.0), and `plan_cost` matches
  eb_jepa's `ReprTargetDistMPCObjective` to a relative difference of 9e-8 (from the order of
  addition). The reference is a separate `JEPA` loaded by eb_jepa's own checkpoint loader, so the
  check does not compare the adapter with itself.
- **Same planner.** Given the same Gaussian noise, the reference CEM returns the same plan as
  eb_jepa's `CEMPlanner` (maximum difference 6e-7 with a 25-step horizon, and 0.0 when the
  horizon is cut to the steps left). Changing the reference CEM's elite update to a biased
  standard deviation makes this check fail by 0.45, so it would catch the planner drifting from
  eb_jepa's.
- **Same starting frames.** The test rebuilds eb_jepa's evaluation environment and replays its
  20 episodes with eb_jepa's reset procedure. Each replayed start frame (wall, door and dot)
  matches the start frame eb_jepa saved for that episode, and shifting the pairing by one episode
  makes every frame differ. Goals are not compared directly: eb_jepa saves them only in a PDF.
  They come from the same generator calls as the start, but the test does not check them, and
  nothing here shows that the two runs' trajectories or wall-collision noise were the same.
- **Closed-loop planning, an engineering check.** Planning through the protocol solved 16 of the
  20 episodes, against 18 of 20 for eb_jepa's own planner, and passed a threshold fixed before
  the run: a protocol rate of at least 70%, and at most two more episodes solved only natively
  than only through the protocol. That second quantity equals the difference in total successes,
  so the rule is a net-count threshold, not a statistical test. The two runs also differed in
  device (CPU against GPU), random draws and possibly precision. The planners disagreed on two
  episodes, both solved only natively; under equal planners a split that favours the native
  planner this much happens about one time in four. So this small experiment does not show that
  the protocol loses no planning quality, nor does it measure a gap: 20 episodes cannot rule out
  a gap of 10 points or more. Replanned with two other seeds, both disputed episodes succeed,
  which suggests sampling luck, but it does not change the full comparison.
- **Baselines.** A policy that steps straight towards the goal solved 9 of 20 episodes (45%), and
  the same planner on an untrained model of the same architecture solved none of 10. These show
  the learned model and planner are doing useful work; they say nothing about equivalence with
  the native planner.

In short: on the tested CPU cases, the adapter reproduces native rollout and cost values, and the
reference CEM reproduces native plans given the same noise. In a separate closed-loop check on a
different device, it solved 16 of 20 tasks against 18 of 20 natively and passed the pre-set
threshold. That does not establish non-inferiority.

Provenance. The closed-loop run (82 minutes) used the code at commit `00e6919`. `plan_cost` has
since been changed to add energies one step at a time, which changes its result by about 1e-7,
and the same-noise check was added afterwards; both short checks were rerun on the later code. A
protocol episode took 166 s on average. eb_jepa's own CEM took 805 s per episode on the same
laptop in an earlier timing with an untrained 30-step checkpoint over two episodes; the
computation does not depend on the weights, but that timing is not a controlled benchmark.
eb_jepa's planner encodes the start observation once for each of its 200 samples, and the
protocol's once per re-plan, which likely accounts for much of the difference.

To reproduce the checks, train and evaluate a run with the commands above, then run this from the
eb_jepa clone, in the same environment:

```bash
JEPA_SCOPE_TWO_ROOMS_CHECKPOINT=<run folder> python -m pytest <path to jepa-scope>/tests/test_eb_jepa_two_rooms.py -s
```

The run folder is the one eb_jepa's training script writes, and it must also hold that run's
native CEM evaluation (`plan_eval/*_eval_only`), which the test compares against. The test uses
the folder's `latest.pth.tar`. Add `-k "match_eb_jepa or same_noise or replay"` for the short
checks, which take seconds; the full test takes over an hour on a laptop CPU.

**Not yet checked.** The closed-loop comparison on one device with shared candidate noise; the
replayed goals; a packed state's capacity on a real model; and a bounded-action planner whose
executed actions are checked against the ones it predicted with.

## Open points

- **Fitting V-JEPA 2-AC.** V-JEPA 2-AC is the next model this protocol should carry. Its own
  planner (`notebooks/utils/mpc_utils.py` in `facebookresearch/vjepa2`) starts from one image,
  grows the context with its own predicted frames during a rollout, and updates the robot arm's
  pose from each action. Its predictor is called with the token context, the actions so far and
  the poses so far. That is a Markov state, so it can fit `predict(z, a)` without a history
  argument, but only if the planning state holds all three. A sketch, not yet built: the state
  holds a token context, the actions and poses so far, and the number of valid frames, sized for
  a declared capacity; `energy` is the mean absolute difference over the latest valid frame's
  tokens, as in vjepa2's `l1`. The adapter must list the predictor's inputs against the pinned
  vjepa2 code and match its native rollout before this counts as compatible. If packing turns
  out costly, a structured state object is still an option; it would need an operation to
  repeat a state across candidates. Whether vjepa2 layer-normalises predicted tokens the same way
  as encoded ones (its `normalize_reps` setting) must also be checked then, because the energy
  depends on it.
- **A representation for drawing.** A tool that draws latents along a plan (latent-plan's
  playground, for example) needs the comparable part of a planning state, not the packed state,
  which may mix tokens with poses, counts and padding. An optional operation such as
  `latent(z) -> [B, *latent]`, with its own documented shape (`latent_shape`), would give it: the
  identity for Two Rooms, a slice for V-JEPA 2-AC. Planners do not need it, so it is not part of
  this draft. Decide it when latent-plan is planned, before its interface is built.
- **Memory.** `plan_cost` keeps only the current state, so planning memory is roughly the
  candidates times one planning state, plus the model's own working memory. A state that holds a
  growing context is still large: its size grows with the capacity, and in float32 it takes
  twice the bytes of bf16. Measure the peak on the target device when such an adapter is built,
  and score candidates in chunks if needed.
- **Action spaces.** The reference planner handles continuous actions with a norm bound only.
  V-JEPA 2-AC's planner instead clips each coordinate to a box, blends each new mean and
  standard deviation with the previous ones (momentum), keeps rotation at zero, and bounds and
  rounds the gripper. A planner matching it needs those settings and a projection of its final
  plan, not only the model's energy.
- **Stochastic predictors.** `predict` returns one next state. Models that predict a distribution
  would need samples or an expected energy.
- **Energy across models.** Energies are only comparable within one model. Comparing planners
  across models would need a calibrated scale. jepa-scope's surprise view meets the same problem
  when it compares prediction errors between clips.
- **Goals per sample.** `plan_cost` takes one goal per batch row. Planning towards several goals
  at once works by repeating the start state, which costs memory.
