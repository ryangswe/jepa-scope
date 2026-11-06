"""A reference Cross-Entropy Method planner written only against the planning protocol (draft).

It mirrors ``CEMPlanner.plan`` in ``eb_jepa/planning.py`` of ``facebookresearch/eb_jepa`` (commit
``966e61e``), so a model planned through the protocol can be compared with the same model planned
natively:

- The horizon is ``min(plan_length, steps_left)``, so it shrinks near the end of an episode.
- The proposal mean starts at zero and the standard deviation at ``var_scale`` on every call; there
  is no warm start from the previous plan.
- Each sampled action is rescaled to a norm of at most ``max_norm`` before it is scored, and the
  rescaled actions are the ones the elites are drawn from.
- The new mean and standard deviation are the mean and the unbiased standard deviation of the
  elites.
- The returned plan is the final elite mean, not the best sample seen.

eb_jepa's ``max_norm_dims`` setting is not used by its planner, and it is not reproduced here.
Executing the plan (for example, one action and then planning again) is the caller's job.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from jepa_scope.planning.protocol import Objective, WorldModel, plan_cost

_NORM_EPS = 1e-6


@dataclass(frozen=True)
class CEMConfig:
    """Settings for :func:`cem_plan`. The defaults are eb_jepa's ``planning_cem.yaml``."""

    plan_length: int = 90
    """Longest horizon, in actions."""

    n_iters: int = 20
    """Number of sample-and-refit iterations."""

    num_samples: int = 200
    """Action sequences sampled per iteration."""

    num_elites: int = 20
    """Lowest-cost samples the proposal is refitted to. At least 2, so their spread is defined."""

    var_scale: float = 1.5
    """Initial standard deviation of every action component. Despite the name, it is not a variance."""

    max_norm: float | None = 2.45
    """Largest allowed norm of one action, or ``None`` for no limit."""

    objective: Objective = "sum"
    """How :func:`~jepa_scope.planning.protocol.plan_cost` scores a sample: ``"sum"`` over every
    state of its rollout (eb_jepa's Two Rooms setting) or the ``"last"`` state only."""

    def __post_init__(self) -> None:
        if self.plan_length < 1 or self.n_iters < 1:
            raise ValueError("plan_length and n_iters must be at least 1")
        if not 2 <= self.num_elites <= self.num_samples:
            raise ValueError("num_elites must be at least 2 and at most num_samples")
        if self.var_scale <= 0:
            raise ValueError("var_scale must be positive")
        if self.max_norm is not None and self.max_norm <= 0:
            raise ValueError("max_norm must be positive or None")
        if self.objective not in ("sum", "last"):
            raise ValueError("objective must be 'sum' or 'last'")


@dataclass(frozen=True)
class CEMResult:
    """What :func:`cem_plan` returns."""

    actions: Tensor
    """The planned action sequence ``[T, A]``: the elite mean after the last iteration."""

    best_costs: Tensor
    """The lowest sampled ``plan_cost`` in each iteration, ``[n_iters]``, on the CPU."""


def clip_action_norm(actions: Tensor, max_norm: float) -> Tensor:
    """Rescale each action (the last dimension) to a norm of at most ``max_norm``.

    This is eb_jepa's formula, ``min(norm, max_norm) / (norm + 1e-6)``, so actions under the limit
    also shrink by a factor of ``norm / (norm + 1e-6)``.
    """
    norms = actions.norm(dim=-1, keepdim=True)
    return actions * (norms.clamp(max=max_norm) / (norms + _NORM_EPS))


@torch.no_grad()
def cem_plan(
    model: WorldModel,
    z0: Tensor,
    z_goal: Tensor,
    config: CEMConfig | None = None,
    *,
    steps_left: int | None = None,
    generator: torch.Generator | None = None,
) -> CEMResult:
    """Plan an action sequence from ``z0`` towards ``z_goal`` with the Cross-Entropy Method.

    ``z0`` and ``z_goal`` are single planning states, ``[1, *latent]``. Each sample is scored
    with :func:`~jepa_scope.planning.protocol.plan_cost` using ``config.objective``.
    ``steps_left``, when given, caps the horizon at the steps remaining in the episode. Pass a
    ``generator`` on ``z0``'s device for a reproducible plan.
    """
    config = config or CEMConfig()
    if z0.shape[0] != 1 or z_goal.shape[0] != 1:
        raise ValueError("cem_plan plans for one start state: z0 and z_goal must have batch size 1")
    if steps_left is not None and steps_left < 1:
        raise ValueError("steps_left must be at least 1")
    horizon = config.plan_length if steps_left is None else min(config.plan_length, steps_left)

    device = z0.device
    mean = torch.zeros(horizon, model.action_dim, device=device)
    std = torch.full((horizon, model.action_dim), config.var_scale, device=device)
    starts = z0.expand(config.num_samples, *z0.shape[1:])
    goals = z_goal.expand(config.num_samples, *z_goal.shape[1:])

    best_costs = []
    for _ in range(config.n_iters):
        noise = torch.randn(config.num_samples, horizon, model.action_dim, generator=generator, device=device)
        actions = mean + std * noise  # [num_samples, T, A]
        if config.max_norm is not None:
            actions = clip_action_norm(actions, config.max_norm)

        costs = plan_cost(model, starts, actions, goals, config.objective)
        best_costs.append(costs.min().item())

        elites = actions[torch.topk(costs, config.num_elites, largest=False).indices]
        mean = elites.mean(dim=0)
        std = elites.std(dim=0)

    return CEMResult(actions=mean, best_costs=torch.tensor(best_costs))
