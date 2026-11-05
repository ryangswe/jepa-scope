"""Project patch features to RGB with PCA, the trick popularised by DINO.

If the encoder and preprocessing are right, the same object keeps roughly the same colour across
frames. That makes this both a visualisation and a cheap correctness check.
"""

from __future__ import annotations

import numpy as np


def fit(features: np.ndarray, k: int = 3, max_samples: int = 50_000, seed: int = 0):
    """Fit a ``k`` component PCA on ``[..., D]`` features. Returns ``(mean, components[k, D])``."""
    x = features.reshape(-1, features.shape[-1]).astype(np.float32)
    if x.shape[0] > max_samples:
        idx = np.random.default_rng(seed).choice(x.shape[0], max_samples, replace=False)
        x = x[idx]
    mean = x.mean(axis=0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    components = vt[:k]
    # SVD signs are arbitrary. Pin them so the same input always gives the same colours.
    signs = np.sign(components[np.arange(k), np.abs(components).argmax(axis=1)])
    signs[signs == 0] = 1
    return mean, components * signs[:, None]


def to_rgb(features: np.ndarray, mean: np.ndarray, components: np.ndarray, lo_pct=1.0, hi_pct=99.0):
    """Project ``[..., D]`` features to ``[..., 3]`` uint8 colours.

    Each component is scaled using percentiles over the whole input, not per frame, so colours stay
    comparable across time.
    """
    flat = features.reshape(-1, features.shape[-1]).astype(np.float32)
    proj = (flat - mean) @ components.T
    lo = np.percentile(proj, lo_pct, axis=0)
    hi = np.percentile(proj, hi_pct, axis=0)
    scaled = np.clip((proj - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    rgb = (scaled * 255.0 + 0.5).astype(np.uint8)
    return rgb.reshape(*features.shape[:-1], components.shape[0])
