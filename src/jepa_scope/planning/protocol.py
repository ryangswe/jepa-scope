"""A model-agnostic protocol for planning with a JEPA world model (draft).

A world model that implements three operations can be planned with, whatever its architecture:

- ``encode(obs) -> z`` turns an observation into a latent state.
- ``predict(z, a) -> z'`` predicts the next latent state after one action.
- ``energy(z, z_goal)`` scores how far a latent state is from a goal; lower is better.

Planners in this package are written only against these operations, so a new model needs an
adapter, not a new planner. This is a draft: the signatures may change without notice until the
protocol has been checked against more than one model. See ``docs/planning-protocol.md``.

Shapes use ``B`` for the batch, ``T`` for the number of actions, ``A`` for the action size, and
``*latent`` for the model's latent shape (``model.latent_shape``).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class WorldModel(Protocol):
    """What a planner needs from a world model."""

    latent_shape: tuple[int, ...]
    """Shape of one latent state, without the batch dimension."""

    action_dim: int
    """Size of one action vector."""

    def encode(self, obs: Tensor) -> Tensor:
        """Encode observations ``[B, *obs]`` into latent states ``[B, *latent]``."""
        ...

    def predict(self, z: Tensor, a: Tensor) -> Tensor:
        """Predict the latent states ``[B, *latent]`` one action ``[B, A]`` after ``z`` ``[B, *latent]``.

        One step and stateless: everything the model needs to predict must be in ``z``.
        """
        ...

    def energy(self, z: Tensor, z_goal: Tensor) -> Tensor:
        """Score latent states ``[B, *latent]`` against goals ``[B, *latent]``, giving ``[B]``.

        Non-negative, zero when ``z`` equals ``z_goal``, and lower is better. The scale is the
        model's own: energies are comparable within one model, never across models.
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


def plan_cost(model: WorldModel, z0: Tensor, actions: Tensor, z_goal: Tensor) -> Tensor:
    """Cost of an action sequence: the sum of ``energy`` to the goal over every state of its rollout.

    ``z0`` is ``[B, *latent]``, ``actions`` is ``[B, T, A]`` and ``z_goal`` is ``[B, *latent]``.
    Returns ``[B]``. The sum runs over all ``T + 1`` states, including ``z0``, rather than only the
    last one, which rewards reaching the goal early and is more robust to errors that compound
    over a long prediction.
    """
    states = rollout(model, z0, actions)
    batch, steps = states.shape[:2]
    flat = states.reshape(batch * steps, *states.shape[2:])
    goals = z_goal.unsqueeze(1).expand(batch, steps, *z_goal.shape[1:]).reshape(batch * steps, *z_goal.shape[1:])
    return model.energy(flat, goals).reshape(batch, steps).sum(dim=1)
