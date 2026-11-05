import numpy as np
import pytest
import torch

from jepa_scope.grid import GridSpec, from_grid, to_grid


def test_counts_match_vjepa2_defaults():
    spec = GridSpec(frames=16, crop=256, patch=16, tubelet=2)
    assert (spec.t, spec.h, spec.w) == (8, 16, 16)
    assert spec.num_tokens == 2048
    assert spec.with_frames(64).num_tokens == 8192


def test_roundtrip():
    spec = GridSpec(frames=4, crop=32, patch=16, tubelet=2)
    x = np.arange(2 * spec.num_tokens * 3).reshape(2, spec.num_tokens, 3)
    assert np.array_equal(from_grid(to_grid(x, spec), spec), x)


def test_wrong_token_count_is_a_clear_error():
    spec = GridSpec(frames=16, crop=256, patch=16, tubelet=2)
    with pytest.raises(ValueError, match="needs 2048"):
        to_grid(np.zeros((8192, 4)), spec)


def test_layout_matches_the_real_patch_embedding(tiny_encoder):
    """Light one pixel and check the token that changes lands where to_grid says it should.

    This pins the time-row-column order against the actual Conv3d in transformers, so a change
    upstream would fail here instead of silently scrambling every visualisation.
    """
    embed = tiny_encoder.model.encoder.embeddings
    spec = tiny_encoder.grid(8)  # 4 x 2 x 2 tokens
    frame, y, x = 5, 20, 3  # tubelet 2, row 1, column 0
    base = torch.zeros(1, 8, 3, 32, 32)
    lit = base.clone()
    lit[0, frame, :, y, x] = 1.0
    with torch.no_grad():
        diff = (embed(lit) - embed(base)).abs().sum(-1)[0]  # [N]
    changed = torch.nonzero(diff > 1e-6).flatten().tolist()
    assert len(changed) == 1
    grid_diff = to_grid(diff.reshape(-1, 1).numpy(), spec)[..., 0]
    assert np.argwhere(grid_diff > 1e-6).tolist() == [[frame // 2, y // 16, x // 16]]


def test_token_ids():
    spec = GridSpec(frames=8, crop=32, patch=16, tubelet=2)
    assert spec.token_ids(2, 4).tolist() == list(range(8, 16))
    with pytest.raises(ValueError):
        spec.token_ids(3, 5)
