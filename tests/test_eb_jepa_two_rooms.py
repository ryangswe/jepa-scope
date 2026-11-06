"""eb_jepa's Two Rooms model planned through the protocol, checked against eb_jepa's own planner.

Opt-in: set ``JEPA_SCOPE_TWO_ROOMS_CHECKPOINT`` to a run folder written by eb_jepa's
``examples/ac_video_jepa/main.py`` that also holds that run's native CEM evaluation
(``plan_eval/*_eval_only``), and run from an environment with eb_jepa installed:

    JEPA_SCOPE_TWO_ROOMS_CHECKPOINT=<run> python -m pytest tests/test_eb_jepa_two_rooms.py -s

The pass bars were fixed before anything ran. Planning 20 episodes on a laptop CPU takes hours.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

CHECKPOINT = os.environ.get("JEPA_SCOPE_TWO_ROOMS_CHECKPOINT")
pytestmark = pytest.mark.skipif(
    not CHECKPOINT, reason="set JEPA_SCOPE_TWO_ROOMS_CHECKPOINT to an eb_jepa Two Rooms run folder"
)

EPISODES = 20
CONTROL_EPISODES = 10
STEP_NORM = 2.45
LABEL_ROWS = 20  # of the 130-pixel GIF frame; the label ends at row 17


@dataclass
class Episode:
    success: bool
    distance: float
    wall: tuple[float, float]
    start: tuple[float, float]
    goal: tuple[float, float]
    first_obs: torch.Tensor
    seconds: float


def _cfg_dir() -> Path:
    import eb_jepa

    package = Path(next(iter(eb_jepa.__path__))).resolve()  # a namespace package: no __file__
    return package.parent / "examples" / "ac_video_jepa" / "cfgs"


def _yaml(path: Path) -> dict:
    import yaml

    with path.open() as f:
        return yaml.safe_load(f)


def _make_env():
    """The evaluation environment exactly as examples/ac_video_jepa/main.py builds it."""
    from eb_jepa.datasets.two_rooms.env import DotWall
    from eb_jepa.datasets.utils import init_data

    eval_cfg = _yaml(_cfg_dir() / "eval.yaml")
    _, _, env_config = init_data(env_name="two_rooms", cfg_data=dict(eval_cfg.get("data", {})))
    return DotWall(config=env_config, **eval_cfg["env"])


def _cem_config():
    from jepa_scope.planning import CEMConfig

    planner = _yaml(_cfg_dir() / "planning_cem.yaml")["planner"]
    assert planner["num_act_stepped"] == 1
    assert planner["planning_objective"]["sum_all_diffs"] is True
    (max_norm,) = planner["max_norms"]
    return CEMConfig(
        plan_length=planner["plan_length"],
        n_iters=planner["n_iters"],
        num_samples=planner["num_samples"],
        num_elites=planner["num_elites"],
        var_scale=planner["var_scale"],
        max_norm=max_norm,
    )


def _run_episodes(policy, count: int) -> list[Episode]:
    """Replay eb_jepa's evaluation episodes the way ``main_eval`` runs them (planning.py:176-296).

    One ``reset`` before the loop, then per episode a ``reset``, a zero-action step, all
    ``n_allowed_steps`` steps with no early stop, and success judged after the last step.
    """
    torch.manual_seed(0)  # wall-collision noise uses torch's global generator
    env = _make_env()
    env.reset()
    episodes = []
    for index in range(count):
        started = time.time()
        env.reset()
        obs, _, _, _, info = env.step(np.zeros(env.action_space.shape[0]))
        first_obs = obs.clone()
        start = tuple(float(v) for v in info["dot_position"])
        goal = tuple(float(v) for v in info["target_position"])
        policy.reset(info["target_obs"])
        for steps_left in range(env.n_allowed_steps, 0, -1):
            action = policy(obs, info, steps_left)
            obs, _, _, _, info = env.step(action)
        result = env.eval_state(info["target_position"], info["dot_position"])
        episode = Episode(
            success=bool(result["success"]),
            distance=float(result["state_dist"]),
            wall=(float(env.wall_x), float(env.hole_y)),
            start=start,
            goal=goal,
            first_obs=first_obs,
            seconds=time.time() - started,
        )
        print(
            f"  episode {index:2d}: wall x={episode.wall[0]:.0f} door y={episode.wall[1]:.0f} "
            f"start=({start[0]:.1f}, {start[1]:.1f}) goal=({goal[0]:.1f}, {goal[1]:.1f}) "
            f"{'success' if episode.success else 'fail   '} dist={episode.distance:.2f} "
            f"{episode.seconds:.0f}s",
            flush=True,
        )
        episodes.append(episode)
    return episodes


class ProtocolPolicy:
    """Plans with the reference CEM through the protocol and executes the first action."""

    def __init__(self, model, seed: int = 0):
        self.model = model
        self.config = _cem_config()
        self.generator = torch.Generator().manual_seed(seed)

    def reset(self, goal_obs):
        self.z_goal = self.model.encode(goal_obs[None].float())

    def __call__(self, obs, info, steps_left):
        from jepa_scope.planning import cem_plan

        z = self.model.encode(obs[None].float())
        plan = cem_plan(self.model, z, self.z_goal, self.config, steps_left=steps_left, generator=self.generator)
        return plan.actions[0].cpu().numpy().astype(np.float32)


class GreedyPolicy:
    """Steps straight towards the goal's true position, at most ``STEP_NORM`` per step."""

    def reset(self, goal_obs):
        pass

    def __call__(self, obs, info, steps_left):
        offset = (info["target_position"] - info["dot_position"]).double()
        norm = float(offset.norm())
        scale = min(1.0, STEP_NORM / norm) if norm > 0 else 0.0
        return (offset * scale).cpu().numpy().astype(np.float32)


@pytest.fixture(scope="module")
def run() -> Path:
    return Path(CHECKPOINT)


@pytest.fixture(scope="module")
def model(run):
    from jepa_scope.planning.eb_jepa_two_rooms import EbJepaTwoRooms

    return EbJepaTwoRooms.from_run(run)


@pytest.fixture(scope="module")
def native_episodes(run) -> list[Path]:
    """The ``ep_k`` folders of the run's own CEM evaluation, in episode order."""
    (eval_dir,) = sorted(run.glob("plan_eval/*_eval_only"))
    folders = [eval_dir / f"ep_{k}" for k in range(EPISODES)]
    assert all(folder.is_dir() for folder in folders)
    return folders


def _native_success(folder: Path) -> bool:
    (gif,) = list(folder.glob("agent_steps_*.gif"))
    return gif.stem.endswith("succ")


@pytest.fixture(scope="module")
def protocol_episodes(model):
    print(f"\nProtocol CEM, {EPISODES} episodes:")
    return _run_episodes(ProtocolPolicy(model), EPISODES)


def _reference_jepa(run: Path):
    """An eb_jepa ``JEPA`` built exactly as main.py:174-218 builds it, loaded by eb_jepa's own loader."""
    from eb_jepa.architectures import ImpalaEncoder, InverseDynamicsModel, RNNPredictor
    from eb_jepa.jepa import JEPA
    from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
    from eb_jepa.training_utils import load_checkpoint

    cfg = _yaml(run / "config.train.yaml" if (run / "config.train.yaml").exists() else run / "config.yaml")["model"]
    img_size = 65
    encoder = ImpalaEncoder(
        width=1,
        stack_sizes=(16, cfg["henc"], cfg["dstc"]),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=cfg["dobs"],
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(cfg["dobs"], img_size, img_size),
    )
    _, f, _, h, w = encoder(torch.rand(1, cfg["dobs"], 1, img_size, img_size)).shape
    predictor = RNNPredictor(hidden_size=encoder.mlp_output_dim, final_ln=encoder.final_ln)
    assert not cfg["regularizer"]["use_proj"]
    idm = InverseDynamicsModel(state_dim=h * w * f, hidden_dim=256, action_dim=2)
    reg = cfg["regularizer"]
    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=reg["cov_coeff"],
        std_coeff=reg["std_coeff"],
        sim_coeff_t=reg["sim_coeff_t"],
        idm_coeff=reg.get("idm_coeff", 0.1),
        idm=idm,
        first_t_only=reg.get("first_t_only"),
        projector=None,
        spatial_as_samples=reg["spatial_as_samples"],
        idm_after_proj=reg["idm_after_proj"],
        sim_t_after_proj=reg["sim_t_after_proj"],
    )
    jepa = JEPA(encoder, torch.nn.Identity(), predictor, regularizer, SquareLossSeq())
    load_checkpoint(run / "latest.pth.tar", jepa, strict=True)
    return jepa.eval()


def test_rollout_and_cost_match_eb_jepa(run, model):
    from eb_jepa.planning import ReprTargetDistMPCObjective

    from jepa_scope.planning import plan_cost, rollout

    reference = _reference_jepa(run)
    env = _make_env()
    obs = []
    for _ in range(9):
        env.reset()
        obs.append(env.step(np.zeros(2))[0])
    obs, goal_obs = torch.stack(obs[:8]).float(), obs[8][None].float()
    actions = 1.5 * torch.randn(8, 8, 2, generator=torch.Generator().manual_seed(0))

    with torch.no_grad():
        obs5d = env.normalizer.normalize_state(obs).unsqueeze(2)
        native, _ = reference.unroll(
            obs5d, actions.permute(0, 2, 1), nsteps=8, unroll_mode="autoregressive", compute_loss=False
        )  # [8, 512, 9, 1, 1]
        native_states = native.flatten(3).squeeze(-1).permute(0, 2, 1)  # [8, 9, 512]
        ours = rollout(model, model.encode(obs), actions)
        parity = (ours - native_states).abs().max().item()
        print(f"\nParity: max |protocol rollout - eb_jepa unroll| = {parity:.2e}")
        assert parity < 1e-5

        goal_5d = reference.encode(env.normalizer.normalize_state(goal_obs).unsqueeze(2))
        native_cost = ReprTargetDistMPCObjective(goal_5d, sum_all_diffs=True)(native)
        z_goal = model.encode(goal_obs).expand(8, -1)
        ours_cost = plan_cost(model, model.encode(obs), actions, z_goal)
        print(f"Cost: max relative difference = {((ours_cost - native_cost).abs() / native_cost).max().item():.2e}")
        torch.testing.assert_close(ours_cost, native_cost, rtol=1e-5, atol=1e-6)


class _SameNoise:
    """Stands in for the ``torch`` module inside a planner, so ``torch.randn`` returns given noise."""

    def __init__(self, noises, layout):
        self._noises = iter(noises)
        self._layout = layout

    def __getattr__(self, name):
        return getattr(torch, name)

    def randn(self, *shape, **kwargs):
        return self._layout(next(self._noises)).clone()


@pytest.mark.parametrize("steps_left", [None, 7])
def test_cem_matches_eb_jepa_planner_on_the_same_noise(run, model, monkeypatch, steps_left):
    """Fed the same Gaussian noise, the reference CEM returns eb_jepa's ``CEMPlanner`` plan."""
    import eb_jepa.planning as eb_planning
    from eb_jepa.planning import CEMPlanner, ReprTargetDistMPCObjective

    import jepa_scope.planning.cem as cem_module
    from jepa_scope.planning import CEMConfig, cem_plan

    reference = _reference_jepa(run)
    env = _make_env()
    env.reset()
    obs, _, _, _, info = env.step(np.zeros(2))
    goal = info["target_obs"].float()
    samples, length, iters, elites = 200, 25, 6, 20
    horizon = length if steps_left is None else min(length, steps_left)
    generator = torch.Generator().manual_seed(123)
    noises = [torch.randn(samples, horizon, 2, generator=generator) for _ in range(iters)]

    def unroll(obs_init, actions):
        states, _ = reference.unroll(
            obs_init.repeat(actions.shape[0], 1, 1, 1, 1),
            actions,
            nsteps=actions.shape[2],
            unroll_mode="autoregressive",
            compute_loss=False,
        )
        return states

    with torch.no_grad():
        obs5d = env.normalizer.normalize_state(obs.float()).unsqueeze(0).unsqueeze(2)
        goal5d = reference.encode(env.normalizer.normalize_state(goal).unsqueeze(0).unsqueeze(2))
        native_planner = CEMPlanner(
            unroll,
            n_iters=iters,
            num_samples=samples,
            plan_length=length,
            action_dim=2,
            var_scale=1.5,
            num_elites=elites,
            max_norms=[STEP_NORM],
            decode_each_iteration=False,
        )
        native_planner.set_objective(ReprTargetDistMPCObjective(goal5d, sum_all_diffs=True))
        # eb_jepa samples in a [T, N, A] layout; the reference CEM in [N, T, A].
        monkeypatch.setattr(eb_planning, "torch", _SameNoise(noises, lambda n: n.permute(1, 0, 2)))
        native = native_planner.plan(obs5d, steps_left=steps_left).actions
        monkeypatch.setattr(cem_module, "torch", _SameNoise(noises, lambda n: n))
        config = CEMConfig(
            plan_length=length, n_iters=iters, num_samples=samples, num_elites=elites, max_norm=STEP_NORM
        )
        ours = cem_plan(model, model.encode(obs[None].float()), model.encode(goal[None]), config, steps_left=steps_left)
    difference = (ours.actions - native).abs().max().item()
    print(f"\nSame-noise CEM, steps_left={steps_left}: max |protocol plan - eb_jepa plan| = {difference:.2e}")
    assert ours.actions.shape == native.shape == (horizon, 2)
    assert difference < 1e-5


def test_replay_matches_the_native_evaluation(tmp_path, native_episodes):
    """The replayed episodes are the ones eb_jepa evaluated: compare every first frame, drawn the
    same way eb_jepa drew its GIFs."""
    from eb_jepa.vis_utils import save_gif
    from PIL import Image

    class StandStill:
        def reset(self, goal_obs):
            pass

        def __call__(self, obs, info, steps_left):
            return np.zeros(2, dtype=np.float32)

    def frame(path):
        # The top rows hold the frame label ("Frame 1/201" there, "Frame 1/1" here); skip them.
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)[LABEL_ROWS:]

    def differing(a, b):
        return int((np.abs(a - b).max(axis=-1) > 16).sum())

    replayed = _run_episodes(StandStill(), EPISODES)
    natives, ours = [], []
    for index, (episode, folder) in enumerate(zip(replayed, native_episodes, strict=True)):
        (native_gif,) = list(folder.glob("agent_steps_*.gif"))
        ours_gif = tmp_path / f"ep_{index}.gif"
        save_gif(episode.first_obs[None], save_path=str(ours_gif), show_frame_numbers=True, fps=20)
        natives.append(frame(native_gif))
        ours.append(frame(ours_gif))
    for index in range(EPISODES):
        assert differing(natives[index], ours[index]) <= 4, f"episode {index}: first frames differ"
    # Negative control: pairing each replay with the next native episode must fail clearly.
    shifted = [differing(natives[index + 1], ours[index]) for index in range(EPISODES - 1)]
    print(f"\nReplay: shifted-by-one pairs differ in {min(shifted)} to {max(shifted)} pixels")
    assert min(shifted) > 100


def test_protocol_planning_matches_native_success(protocol_episodes, native_episodes):
    native = [_native_success(folder) for folder in native_episodes]
    ours = [episode.success for episode in protocol_episodes]
    native_only = sum(n and not o for n, o in zip(native, ours, strict=True))
    protocol_only = sum(o and not n for n, o in zip(native, ours, strict=True))
    print(f"\n{'episode':>8} {'native':>7} {'protocol':>9}")
    for index, (n, o) in enumerate(zip(native, ours, strict=True)):
        print(f"{index:>8} {'success' if n else 'fail':>7} {'success' if o else 'fail':>9}")
    print(
        f"Native {sum(native)}/{EPISODES}, protocol {sum(ours)}/{EPISODES}; "
        f"native only {native_only}, protocol only {protocol_only}; "
        f"mean episode time {np.mean([e.seconds for e in protocol_episodes]):.0f}s"
    )
    assert sum(ours) >= 0.7 * EPISODES
    margin = native_only - protocol_only
    if margin in (3, 4):
        pytest.fail(f"Marginal: native-only minus protocol-only is {margin}. Rerun both on 50 episodes.")
    assert margin <= 2


def test_greedy_baseline_is_well_below_planning(protocol_episodes):
    print("\nGreedy baseline:")
    greedy = _run_episodes(GreedyPolicy(), EPISODES)
    greedy_rate = sum(e.success for e in greedy) / EPISODES
    protocol_rate = sum(e.success for e in protocol_episodes) / EPISODES
    print(f"Greedy {greedy_rate:.0%}, protocol {protocol_rate:.0%}")
    assert greedy_rate < protocol_rate - 0.10, "the task needs less planning than assumed: escalate"


def test_untrained_model_fails(run):
    from jepa_scope.planning.eb_jepa_two_rooms import EbJepaTwoRooms

    print(f"\nUntrained control, {CONTROL_EPISODES} episodes:")
    episodes = _run_episodes(ProtocolPolicy(EbJepaTwoRooms.untrained(run, seed=0)), CONTROL_EPISODES)
    rate = sum(e.success for e in episodes) / CONTROL_EPISODES
    print(f"Untrained {rate:.0%}")
    assert rate <= 0.20
