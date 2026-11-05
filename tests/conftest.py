"""Shared fixtures. Nothing here downloads weights.

``tiny_encoder`` is a randomly initialised V-JEPA 2 with a 32 px crop and one layer each, so the
whole pipeline can run in a second on CPU. Random weights mean the numbers are meaningless; the
tests check shapes, plumbing and the token layout, not quality.
"""

from __future__ import annotations

import av
import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def tiny_encoder():
    from transformers import VJEPA2Config, VJEPA2Model
    from transformers.models.vjepa2.video_processing_vjepa2 import VJEPA2VideoProcessor

    from jepa_scope.model import Encoder

    torch.manual_seed(0)
    cfg = VJEPA2Config(
        crop_size=32,
        patch_size=16,
        tubelet_size=2,
        frames_per_clip=8,
        hidden_size=96,
        num_attention_heads=2,
        num_hidden_layers=1,
        pred_hidden_size=96,
        pred_num_attention_heads=2,
        pred_num_hidden_layers=1,
    )
    model = VJEPA2Model(cfg)
    processor = VJEPA2VideoProcessor(crop_size=32)
    return Encoder(model, processor, "test/tiny-vjepa2", torch.device("cpu"), torch.float32)


def write_video(path, frames: np.ndarray, fps: int = 24) -> None:
    """Write uint8 ``[T, H, W, 3]`` frames to an mp4 with PyAV."""
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        for f in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


@pytest.fixture
def moving_square_video(tmp_path):
    """Three seconds at 24 fps, 96x64, a white square moving left to right."""
    t, h, w = 72, 64, 96
    frames = np.zeros((t, h, w, 3), np.uint8)
    for i in range(t):
        x = int(i / t * (w - 16))
        frames[i, 24:40, x : x + 16] = 255
    path = tmp_path / "square.mp4"
    write_video(path, frames)
    return path
