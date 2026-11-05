"""Token layout for V-JEPA style video encoders.

The encoder cuts a clip into tubelets (``tubelet`` frames x ``patch`` x ``patch`` pixels) with a
strided Conv3d and flattens the result, so the output is a flat ``[N, D]`` sequence whose order is
time first, then row, then column. Nothing in the tensor records that layout, so every tool that
wants to draw or slice tokens has to rebuild it. This module is the one place that does.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """How a clip of ``frames`` frames at ``crop`` x ``crop`` pixels maps to tokens."""

    frames: int
    crop: int
    patch: int
    tubelet: int

    def __post_init__(self) -> None:
        if self.frames % self.tubelet:
            raise ValueError(f"frames ({self.frames}) must be a multiple of tubelet ({self.tubelet})")
        if self.crop % self.patch:
            raise ValueError(f"crop ({self.crop}) must be a multiple of patch ({self.patch})")

    @property
    def t(self) -> int:
        """Number of tubelets along time."""
        return self.frames // self.tubelet

    @property
    def h(self) -> int:
        return self.crop // self.patch

    @property
    def w(self) -> int:
        return self.crop // self.patch

    @property
    def tokens_per_tubelet(self) -> int:
        return self.h * self.w

    @property
    def num_tokens(self) -> int:
        return self.t * self.h * self.w

    def with_frames(self, frames: int) -> GridSpec:
        return GridSpec(frames=frames, crop=self.crop, patch=self.patch, tubelet=self.tubelet)

    def token_ids(self, t_start: int, t_stop: int) -> np.ndarray:
        """Flat token ids covering tubelets ``t_start`` up to (not including) ``t_stop``."""
        if not 0 <= t_start <= t_stop <= self.t:
            raise ValueError(f"tubelet range {t_start}:{t_stop} is outside 0:{self.t}")
        n = self.tokens_per_tubelet
        return np.arange(t_start * n, t_stop * n, dtype=np.int64)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def to_grid(tokens, spec: GridSpec):
    """Reshape ``[..., N, D]`` tokens into ``[..., T, H, W, D]``.

    Works on numpy arrays and torch tensors alike, since both support ``reshape``.
    """
    *lead, n, d = tokens.shape
    if n != spec.num_tokens:
        raise ValueError(
            f"got {n} tokens but the grid {spec.t}x{spec.h}x{spec.w} needs {spec.num_tokens}; "
            "check frames per window, crop size and tubelet size"
        )
    return tokens.reshape(*lead, spec.t, spec.h, spec.w, d)


def from_grid(grid, spec: GridSpec):
    """Inverse of :func:`to_grid`."""
    *lead, t, h, w, d = grid.shape
    if (t, h, w) != (spec.t, spec.h, spec.w):
        raise ValueError(f"grid shape {(t, h, w)} does not match spec {(spec.t, spec.h, spec.w)}")
    return grid.reshape(*lead, t * h * w, d)
