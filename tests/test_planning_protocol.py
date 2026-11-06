"""The planning protocol and the reference CEM, checked on a point mass where the answer is known."""

import pytest
import torch

from jepa_scope.planning import CEMConfig, WorldModel, cem_plan, clip_action_norm, plan_cost, rollout

GOAL = (5.0, 5.0)
STEPS = 12
SPEED = 1.0  # largest distance the point moves per step


class PointMass:
    """The latent is the position in R², each action moves it by at most ``SPEED``, and the
    energy is the squared distance to the goal."""

    latent_shape = (2,)
    action_dim = 2

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def predict(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return z + clip_action_norm(a, SPEED)

    def energy(self, z: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        return ((z - z_goal) ** 2).sum(dim=-1)


SMALL = CEMConfig(plan_length=STEPS, n_iters=10, num_samples=100, num_elites=10, var_scale=1.5, max_norm=SPEED)


def _generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_point_mass_implements_the_protocol():
    assert isinstance(PointMass(), WorldModel)


def test_rollout_keeps_the_start_state_and_one_state_per_action():
    z0 = torch.zeros(3, 2)
    actions = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).expand(3, 2, 2)
    states = rollout(PointMass(), z0, actions)
    assert states.shape == (3, 3, 2)
    assert torch.equal(states[:, 0], z0)
    assert torch.allclose(states[0, 2], torch.tensor([1.0, 1.0]), atol=1e-5)


def test_plan_cost_sums_energy_over_every_state_including_the_start():
    model = PointMass()
    z0 = torch.zeros(1, 2)
    z_goal = torch.tensor([[3.0, 4.0]])  # energy 25 from the origin
    standing_still = torch.zeros(1, 4, 2)
    assert plan_cost(model, z0, standing_still, z_goal).item() == pytest.approx(25.0 * 5)


def test_cem_reaches_the_goal_when_replanning_every_step():
    model = PointMass()
    z = model.encode(torch.zeros(1, 2))
    z_goal = model.encode(torch.tensor([GOAL]))
    generator = _generator()
    for step in range(STEPS):
        plan = cem_plan(model, z, z_goal, SMALL, steps_left=STEPS - step, generator=generator)
        assert plan.actions.shape == (STEPS - step, 2)
        z = model.predict(z, plan.actions[:1])
    assert torch.dist(z, z_goal).item() < 0.5


def test_planned_sequence_beats_random_sequences():
    model = PointMass()
    z0, z_goal = torch.zeros(1, 2), torch.tensor([GOAL])
    plan = cem_plan(model, z0, z_goal, SMALL, generator=_generator())
    random_actions = 1.5 * torch.randn(64, STEPS, 2, generator=_generator(1))
    random_costs = plan_cost(model, z0.expand(64, 2), random_actions, z_goal.expand(64, 2))
    planned_cost = plan_cost(model, z0, plan.actions.unsqueeze(0), z_goal).item()
    assert planned_cost < random_costs.min().item()
    assert plan.best_costs.shape == (SMALL.n_iters,)
    assert plan.best_costs[-1] < plan.best_costs[0]


def test_same_seed_gives_the_same_plan():
    model = PointMass()
    z0, z_goal = torch.zeros(1, 2), torch.tensor([GOAL])
    first = cem_plan(model, z0, z_goal, SMALL, generator=_generator(7))
    second = cem_plan(model, z0, z_goal, SMALL, generator=_generator(7))
    assert torch.equal(first.actions, second.actions)


def test_horizon_shrinks_to_the_steps_left_and_actions_respect_the_norm_limit():
    plan = cem_plan(PointMass(), torch.zeros(1, 2), torch.tensor([GOAL]), SMALL, steps_left=3, generator=_generator())
    assert plan.actions.shape == (3, 2)
    assert (plan.actions.norm(dim=-1) <= SPEED).all()


def test_clip_action_norm_matches_the_eb_jepa_formula():
    actions = torch.tensor([[3.0, 4.0], [0.3, 0.4], [0.0, 0.0]])
    clipped = clip_action_norm(actions, 2.45)
    assert clipped[0].norm().item() == pytest.approx(2.45, rel=1e-5)
    assert torch.allclose(clipped[1], actions[1] * (0.5 / (0.5 + 1e-6)))
    assert torch.equal(clipped[2], actions[2])


@pytest.mark.parametrize(
    "settings",
    [{"num_elites": 1}, {"num_elites": 300}, {"plan_length": 0}, {"var_scale": 0.0}, {"max_norm": -1.0}],
)
def test_bad_settings_are_refused(settings):
    with pytest.raises(ValueError):
        CEMConfig(**settings)


def test_cem_refuses_a_batch_of_start_states():
    with pytest.raises(ValueError, match="batch size 1"):
        cem_plan(PointMass(), torch.zeros(2, 2), torch.zeros(2, 2), SMALL)
