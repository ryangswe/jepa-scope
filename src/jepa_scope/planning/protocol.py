"""A model-agnostic protocol for planning with a JEPA world model (draft).

A world model that implements three operations can be planned with, whatever its architecture:

- ``encode(obs) -> z`` turns an observation into a planning state.
- ``predict(z, a) -> z'`` predicts the planning state after one action.
- ``energy(z, z_goal)`` scores how far a planning state is from a goal; lower is better.

The planning state ``z`` is a single tensor of fixed shape (``model.state_shape``). For many
models it is simply the latent. For others it also carries what the next prediction needs, such
as earlier predicted frames or a robot's pose, and ``energy`` decides which part it compares.

Planners in this package are written only against these operations, so a new model needs an
adapter, not a new planner. This is a draft: the signatures may change without notice until the
protocol has been checked against more than one model. See ``docs/planning-protocol.md``, which
also states the rules an adapter must follow (read-only inputs, independent batch rows, and so
on).

Shapes use ``B`` for the batch, ``T`` for the number of actions, ``A`` for the action size, and
``*state`` for the shape of one planning state.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

import torch
from torch import Tensor

Objective = Literal["sum", "last"]
"""How :func:`plan_cost` combines the energies along a rollout."""


@runtime_checkable
class WorldModel(Protocol):
    """What a planner needs from a world model.

    The methods must treat their input tensors as read-only, keep each batch row independent of
    the others, and accept expanded or non-contiguous inputs. A state they returned earlier must
    not change later. They are called for inference only, without gradients.
    """

    state_shape: tuple[int, ...]
    """Shape of one planning state, without the batch dimension."""

    action_dim: int
    """Size of one action vector."""

    def encode(self, obs: Any) -> Tensor:
        """Encode a batch of ``B`` observations into planning states ``[B, *state]``.

        The observation type is the model's own (an image tensor, or an image with a robot pose,
        for example). Planners pass observations through without looking inside them.
        """
        ...

    def predict(self, z: Tensor, a: Tensor) -> Tensor:
        """Predict the planning states ``[B, *state]`` one action ``[B, A]`` after ``z`` ``[B, *state]``.

        One step and stateless: everything the model needs to predict must be in ``z``.
        """
        ...

    def energy(self, z: Tensor, z_goal: Tensor) -> Tensor:
        """Score planning states ``[B, *state]`` against goals ``[B, *state]``, giving ``[B]``.

        Non-negative, lower is better, and zero when the parts of ``z`` that the model compares
        equal those of ``z_goal``. Each row's score depends only on that row. The scale is the
        model's own: energies are comparable within one model, never across models.
        """
        ...


def rollout(model: WorldModel, z0: Tensor, actions: Tensor) -> Tensor:
    """Apply ``predict`` once per action and keep every state.

    ``z0`` is ``[B, *state]`` and ``actions`` is ``[B, T, A]``. Returns ``[B, T + 1, *state]``,
    with ``z0`` at index 0 and the state after action ``t`` at index ``t + 1``. Planners do not
    need the whole trajectory; :func:`plan_cost` does not keep it.
    """
    states = [z0]
    z = z0
    for t in range(actions.shape[1]):
        z = model.predict(z, actions[:, t])
        states.append(z)
    return torch.stack(states, dim=1)


def plan_cost(model: WorldModel, z0: Tensor, actions: Tensor, z_goal: Tensor, objective: Objective = "sum") -> Tensor:
    """Cost of an action sequence, from the ``energy`` of its rollout's states to the goal.

    ``z0`` is ``[B, *state]``, ``actions`` is ``[B, T, A]`` and ``z_goal`` is ``[B, *state]``.
    Returns ``[B]``.

    With ``objective="sum"`` (the default) the cost is the sum of the energies of all ``T + 1``
    states, including ``z0``, added one step at a time in the dtype ``energy`` returns. It gives
    every step of the plan a cost, and it is the objective of eb_jepa's Two Rooms planner. With
    ``objective="last"`` it is the energy of the final state only, which other native planners
    use. Only the current state is kept, so memory does not grow with ``T``.
    """
    if objective not in ("sum", "last"):
        raise ValueError(f"objective must be 'sum' or 'last', not {objective!r}")
    z = z0
    total = model.energy(z, z_goal) if objective == "sum" else None
    for t in range(actions.shape[1]):
        z = model.predict(z, actions[:, t])
        if total is not None:
            total = total + model.energy(z, z_goal)
    return total if total is not None else model.energy(z, z_goal)
