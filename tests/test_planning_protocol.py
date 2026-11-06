"""The planning protocol and the reference CEM, checked on a point mass where the answer is known."""

import dataclasses

import pytest
import torch

from jepa_scope.planning import ACTION_DTYPE, CEMConfig, WorldModel, cem_plan, clip_action_norm, plan_cost, rollout

GOAL = (5.0, 5.0)
STEPS = 12
SPEED = 1.0  # largest distance the point moves per step


class PointMass:
    """The latent is the position in R², each action moves it by at most ``SPEED``, and the
    energy is the squared distance to the goal."""

    state_shape = (2,)
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


def test_last_objective_scores_only_the_final_state():
    model = PointMass()
    z0 = torch.zeros(1, 2)
    z_goal = torch.tensor([[3.0, 4.0]])
    actions = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])  # ends at (2, 0): energy 1 + 16
    assert plan_cost(model, z0, actions, z_goal, "last").item() == pytest.approx(17.0, rel=1e-5)
    with pytest.raises(ValueError, match="objective"):
        plan_cost(model, z0, actions, z_goal, "mean")


def test_cem_reaches_the_goal_with_either_objective():
    model = PointMass()
    z0, z_goal = torch.zeros(1, 2), torch.tensor([GOAL])
    for objective in ("sum", "last"):
        config = dataclasses.replace(SMALL, objective=objective)
        plan = cem_plan(model, z0, z_goal, config, generator=_generator())
        final = rollout(model, z0, plan.actions.unsqueeze(0))[0, -1]
        assert torch.dist(final, z_goal[0]).item() < 0.5, objective


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
    [
        {"num_elites": 1},
        {"num_elites": 300},
        {"plan_length": 0},
        {"var_scale": 0.0},
        {"max_norm": -1.0},
        {"objective": "mean"},
    ],
)
def test_bad_settings_are_refused(settings):
    with pytest.raises(ValueError):
        CEMConfig(**settings)


def test_cem_refuses_a_batch_of_start_states():
    with pytest.raises(ValueError, match="batch size 1"):
        cem_plan(PointMass(), torch.zeros(2, 2), torch.zeros(2, 2), SMALL)


class PackedContext:
    """A planning state that packs a history: rows 0..CAPACITY-1 hold positions, and the last row
    holds the number of valid positions. ``energy`` compares only the latest valid position, so
    it must read a different row of each state when states have different lengths."""

    CAPACITY = 6
    state_shape = (CAPACITY + 1, 2)
    action_dim = 2

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(obs.shape[0], *self.state_shape)
        z[:, 0] = obs
        z[:, -1, 0] = 1
        return z

    @staticmethod
    def _counts(z: torch.Tensor) -> torch.Tensor:
        return z[:, -1, 0].long()

    def _latest(self, z: torch.Tensor) -> torch.Tensor:
        return z[torch.arange(z.shape[0]), self._counts(z) - 1]

    def predict(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        counts = self._counts(z)
        if (counts >= self.CAPACITY).any():
            raise ValueError("context is full: the plan is longer than this model's capacity")
        out = z.clone()  # never write into the input, which may be an expanded view
        rows = torch.arange(z.shape[0])
        out[rows, counts] = self._latest(z) + clip_action_norm(a, SPEED)
        out[:, -1, 0] = counts + 1
        return out

    def energy(self, z: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        return ((self._latest(z) - self._latest(z_goal)) ** 2).sum(dim=-1)


def _cost_from_rollout(model, z0, actions, z_goal, objective):
    states = rollout(model, z0, actions)
    energies = torch.stack([model.energy(states[:, t], z_goal) for t in range(states.shape[1])], dim=1)
    return energies.sum(dim=1) if objective == "sum" else energies[:, -1]


@pytest.mark.parametrize("model", [PointMass(), PackedContext()], ids=["point_mass", "packed_context"])
@pytest.mark.parametrize("objective", ["sum", "last"])
def test_plan_cost_matches_scoring_the_whole_rollout(model, objective):
    z0 = model.encode(torch.tensor([[0.0, 0.0], [1.0, -2.0], [3.0, 3.0]]))
    z_goal = model.encode(torch.tensor([[GOAL], [GOAL], [GOAL]]).squeeze(1))
    actions = torch.randn(3, 5, 2, generator=_generator(3))
    expected = _cost_from_rollout(model, z0, actions, z_goal, objective)
    torch.testing.assert_close(plan_cost(model, z0, actions, z_goal, objective), expected)
    zero = torch.zeros(3, 0, 2)
    torch.testing.assert_close(plan_cost(model, z0, zero, z_goal, objective), model.energy(z0, z_goal))


def test_packed_context_keeps_inputs_and_earlier_states_unchanged():
    model = PackedContext()
    assert isinstance(model, WorldModel)
    z0 = model.encode(torch.tensor([[0.0, 0.0]]))
    z0_copy = z0.clone()
    actions = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    first = rollout(model, z0, actions)
    first_copy = first.clone()
    second = rollout(model, z0, actions)  # branch again from the same start
    rollout(model, first[:, -1], actions[:, :1])  # continue from a returned state
    assert torch.equal(z0, z0_copy)
    assert torch.equal(first, first_copy)
    assert torch.equal(first, second)


def test_packed_context_accepts_an_expanded_start_and_scores_rows_independently():
    model = PackedContext()
    z0 = model.encode(torch.tensor([[0.0, 0.0]]))
    z_goal = model.encode(torch.tensor([GOAL]))
    actions = torch.randn(4, 3, 2, generator=_generator(4))
    expanded = plan_cost(model, z0.expand(4, *model.state_shape), actions, z_goal.expand(4, *model.state_shape))
    rows = torch.cat([plan_cost(model, z0, actions[i : i + 1], z_goal) for i in range(4)])
    torch.testing.assert_close(expanded, rows)

    # States with different valid lengths, scored together and one at a time.
    mixed = torch.cat([rollout(model, z0, actions[:1])[:, t] for t in range(4)])
    goals = z_goal.expand(4, *model.state_shape)
    together = model.energy(mixed, goals)
    alone = torch.cat([model.energy(mixed[i : i + 1], z_goal) for i in range(4)])
    torch.testing.assert_close(together, alone)


def test_packed_context_refuses_to_overflow_and_plans_within_capacity():
    model = PackedContext()
    z0 = model.encode(torch.tensor([[0.0, 0.0]]))
    full = torch.zeros(1, model.CAPACITY - 1, 2)
    rollout(model, z0, full)  # exactly at capacity
    with pytest.raises(ValueError, match="capacity"):
        rollout(model, z0, torch.zeros(1, model.CAPACITY, 2))
    config = dataclasses.replace(SMALL, plan_length=model.CAPACITY - 1, max_norm=SPEED)
    plan = cem_plan(model, z0, model.encode(torch.tensor([[3.0, 3.0]])), config, generator=_generator())
    final = rollout(model, z0, plan.actions.unsqueeze(0))[:, -1]
    assert torch.dist(model._latest(final), torch.tensor([[3.0, 3.0]])).item() < 0.5


@pytest.mark.parametrize(
    "settings",
    [
        {"var_scale": float("nan")},
        {"var_scale": float("inf")},
        {"max_norm": float("nan")},
        {"plan_length": 1.5},
        {"num_elites": 2.5},
        {"n_iters": True},
    ],
)
def test_non_finite_or_non_integer_settings_are_refused(settings):
    with pytest.raises(ValueError):
        CEMConfig(**settings)


class _BadEnergy(PointMass):
    def __init__(self, energy):
        self._energy = energy

    def energy(self, z, z_goal):
        return self._energy(super().energy(z, z_goal))


@pytest.mark.parametrize(
    ("energy", "message"),
    [
        (lambda e: torch.full_like(e, float("nan")), "NaN or infinite"),
        (lambda e: torch.where(e > 30, torch.full_like(e, float("inf")), e), "NaN or infinite"),
        (lambda e: e[:1], "one cost per candidate"),
    ],
    ids=["all_nan", "some_infinite", "wrong_shape"],
)
def test_invalid_costs_fail_loudly(energy, message):
    with pytest.raises(ValueError, match=message):
        cem_plan(_BadEnergy(energy), torch.zeros(1, 2), torch.tensor([GOAL]), SMALL, generator=_generator())


def test_mismatched_start_and_goal_are_refused():
    with pytest.raises(ValueError, match="state_shape"):
        cem_plan(PointMass(), torch.zeros(1, 2), torch.zeros(1, 3), SMALL)


def test_actions_are_float32_whatever_the_state_dtype():
    plan = cem_plan(
        PointMass(), torch.zeros(1, 2, dtype=torch.float64), torch.tensor([GOAL], dtype=torch.float64), SMALL
    )
    assert plan.actions.dtype == ACTION_DTYPE == torch.float32
