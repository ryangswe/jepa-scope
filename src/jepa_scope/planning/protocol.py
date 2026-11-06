"""A model-agnostic protocol for planning with a JEPA world model (draft).

A world model that implements three operations can be planned with, whatever its architecture:

- ``encode(obs) -> z`` turns an observation into a planning state.
- ``predict(z, a) -> z'`` predicts the planning state after one action.
- ``energy(z, z_goal)`` scores how far a planning state is from a goal; lower is better.

The planning state ``z`` is a single tensor of fixed shape (``model.latent_shape``). For many
models it is simply the latent. For others it also carries what the next prediction needs, such
as earlier predicted frames or a robot's pose, and ``energy`` decides which part it compares.

Planners in this package are written only against these operations, so a new model needs an
adapter, not a new planner. This is a draft: the signatures may change without notice until the
protocol has been checked against more than one model. See ``docs/planning-protocol.md``.

Shapes use ``B`` for the batch, ``T`` for the number of actions, ``A`` for the action size, and
``*latent`` for the shape of one planning state.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

import torch
from torch import Tensor

Objective = Literal["sum", "last"]
"""How :func:`plan_cost` combines the energies along a rollout."""


@runtime_checkable
class WorldModel(Protocol):
    """What a planner needs from a world model."""

    latent_shape: tuple[int, ...]
    """Shape of one planning state, without the batch dimension."""

    action_dim: int
    """Size of one action vector."""

    def encode(self, obs: Any) -> Tensor:
        """Encode a batch of ``B`` observations into planning states ``[B, *latent]``.

        The observation type is the model's own (an image tensor, or an image with a robot pose,
        for example). Planners pass observations through without looking inside them.
        """
        ...

    def predict(self, z: Tensor, a: Tensor) -> Tensor:
        """Predict the planning states ``[B, *latent]`` one action ``[B, A]`` after ``z`` ``[B, *latent]``.

        One step and stateless: everything the model needs to predict must be in ``z``.
        """
        ...

    def energy(self, z: Tensor, z_goal: Tensor) -> Tensor:
        """Score planning states ``[B, *latent]`` against goals ``[B, *latent]``, giving ``[B]``.

        Non-negative, lower is better, and zero when the parts of ``z`` that the model compares
        equal those of ``z_goal``. The scale is the model's own: energies are comparable within
        one model, never across models.
        """
        ...


def rollout(model: WorldModel, z0: Tensor, actions: Tensor) -> Tensor:
    """Apply ``predict`` once per action.

    ``z0`` is ``[B, *latent]`` and ``actions`` is ``[B, T, A]``. Returns ``[B, T + 1, *latent]``,
    with ``z0`` at index 0 and the state after action ``t`` at index ``t + 1``.
    """
    states = [z0]
    z = z0
    for t in range(actions.shape[1]):
        z = model.predict(z, actions[:, t])
        states.append(z)
    return torch.stack(states, dim=1)


def plan_cost(model: WorldModel, z0: Tensor, actions: Tensor, z_goal: Tensor, objective: Objective = "sum") -> Tensor:
    """Cost of an action sequence, from the ``energy`` of its rollout's states to the goal.

    ``z0`` is ``[B, *latent]``, ``actions`` is ``[B, T, A]`` and ``z_goal`` is ``[B, *latent]``.
    Returns ``[B]``.

    With ``objective="sum"`` (the default) the cost is the sum of the energies of all ``T + 1``
    states, including ``z0``. That rewards reaching the goal early and is more robust to errors
    that compound over a long prediction. With ``objective="last"`` it is the energy of the final
    state only, which is what some models' own planners use.
    """
    if objective not in ("sum", "last"):
        raise ValueError(f"objective must be 'sum' or 'last', not {objective!r}")
    states = rollout(model, z0, actions)
    if objective == "last":
        return model.energy(states[:, -1], z_goal)
    batch, steps = states.shape[:2]
    flat = states.reshape(batch * steps, *states.shape[2:])
    goals = z_goal.unsqueeze(1).expand(batch, steps, *z_goal.shape[1:]).reshape(batch * steps, *z_goal.shape[1:])
    return model.energy(flat, goals).reshape(batch, steps).sum(dim=1)
