"""A thin wrapper around the Hugging Face V-JEPA 2 model.

The wrapper exists for one reason beyond convenience. ``VJEPA2Model.forward`` runs the encoder on
the whole clip and applies the context mask afterwards, so "context" tokens have already attended
to the frames the predictor is asked to predict. That is fine for feature extraction but makes
prediction error useless as a surprise signal. :meth:`Encoder.surprise` instead encodes the context
frames on their own and then asks the predictor for the later tokens.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .grid import GridSpec

DEFAULT_MODEL = "facebook/vjepa2-vitl-fpc64-256"


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, requested: str | None = None) -> torch.dtype:
    if requested:
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[requested]
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


@dataclass
class Geometry:
    """Where the model's square crop sits inside the source frame, as fractions of width/height."""

    x: float
    y: float
    w: float
    h: float


class Encoder:
    def __init__(self, model, processor, model_id: str, device: torch.device, dtype: torch.dtype):
        self.model = model.eval()
        self.processor = processor
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        cfg = model.config
        self.crop = int(cfg.crop_size)
        self.patch = int(cfg.patch_size)
        self.tubelet = int(cfg.tubelet_size)
        self.hidden_size = int(cfg.hidden_size)

    @classmethod
    def from_pretrained(
        cls, model_id: str = DEFAULT_MODEL, device: str | None = None, dtype: str | None = None
    ) -> Encoder:
        from transformers import AutoModel, AutoVideoProcessor

        dev = pick_device(device)
        dt = pick_dtype(dev, dtype)
        model = AutoModel.from_pretrained(model_id, dtype=dt).to(dev)
        processor = AutoVideoProcessor.from_pretrained(model_id)
        return cls(model, processor, model_id, dev, dt)

    def grid(self, frames: int) -> GridSpec:
        return GridSpec(frames=frames, crop=self.crop, patch=self.patch, tubelet=self.tubelet)

    # -- preprocessing -------------------------------------------------------------------------

    def preprocess(self, clips: Sequence[np.ndarray]) -> torch.Tensor:
        """``clips`` is a list of uint8 ``[T, H, W, 3]`` arrays. Returns ``[B, T, C, crop, crop]``."""
        out = self.processor(videos=list(clips), return_tensors="pt", do_sample_frames=False)
        pixels = out["pixel_values_videos"]
        expected = (len(clips), clips[0].shape[0], 3, self.crop, self.crop)
        if tuple(pixels.shape) != expected:
            raise RuntimeError(f"preprocessing gave {tuple(pixels.shape)}, expected {expected}")
        return pixels

    def preprocess_info(self) -> dict:
        """The preprocessing recipe, plus a hash so cached latents can be checked against it."""
        p = self.processor
        recipe = {
            "processor": type(p).__name__,
            "shortest_edge": int(p.size["shortest_edge"]) if p.size and "shortest_edge" in p.size else None,
            "crop_size": self.crop,
            "image_mean": [float(v) for v in p.image_mean],
            "image_std": [float(v) for v in p.image_std],
            "resample": str(p.resample),
            # embed.py downscales at decode time (PyAV area averaging) to this shortest edge first.
            "decode_resize": "area-to-shortest-edge",
        }
        digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]
        return {**recipe, "hash": digest}

    def crop_geometry(self, width: int, height: int) -> Geometry:
        """Approximate crop rectangle of a resize-shortest-edge then center-crop pipeline."""
        short = min(width, height)
        edge = self.preprocess_info()["shortest_edge"] or short
        scale = edge / short
        side = self.crop / scale  # crop side measured in source pixels
        return Geometry(x=(width - side) / 2 / width, y=(height - side) / 2 / height, w=side / width, h=side / height)

    # -- model calls ---------------------------------------------------------------------------

    @torch.inference_mode()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """``[B, T, C, H, W]`` pixels to ``[B, N, D]`` encoder tokens (float32, on CPU)."""
        x = pixels.to(self.device, self.dtype)
        tokens = self.model.encoder(pixel_values_videos=x).last_hidden_state
        return tokens.float().cpu()

    @torch.inference_mode()
    def predict(self, pixels: torch.Tensor, context_frames: int) -> torch.Tensor:
        """Predict the tokens after ``context_frames`` from the context frames alone.

        Returns ``[B, N_target, D]`` (float32, on CPU). The encoder only ever sees
        ``pixels[:, :context_frames]``, so nothing about the future can leak into the prediction.
        """
        b, frames = pixels.shape[:2]
        if context_frames % self.tubelet or not 0 < context_frames < frames:
            raise ValueError(
                f"context_frames ({context_frames}) must be a multiple of tubelet ({self.tubelet}) "
                f"and less than the window ({frames})"
            )
        spec = self.grid(frames)
        split = context_frames // self.tubelet
        ctx_ids = torch.as_tensor(spec.token_ids(0, split), device=self.device)
        tgt_ids = torch.as_tensor(spec.token_ids(split, spec.t), device=self.device)

        ctx_pixels = pixels[:, :context_frames].to(self.device, self.dtype)
        ctx_tokens = self.model.encoder(pixel_values_videos=ctx_pixels).last_hidden_state
        assert ctx_tokens.shape[1] == len(ctx_ids), (ctx_tokens.shape, len(ctx_ids))

        # Token ids double as positions: the predictor derives (t, h, w) for RoPE from them.
        pred = self.model.predictor(
            encoder_hidden_states=ctx_tokens,
            context_mask=[ctx_ids.unsqueeze(0).expand(b, -1)],
            target_mask=[tgt_ids.unsqueeze(0).expand(b, -1)],
        ).last_hidden_state
        return pred.float().cpu()

    def surprise(self, pixels: torch.Tensor, full_tokens: torch.Tensor, context_frames: int) -> torch.Tensor:
        """Per-token prediction error for the frames after ``context_frames``.

        ``full_tokens`` must be :meth:`encode` of the same ``pixels``. Returns ``[B, N_target]``,
        the mean absolute error between :meth:`predict` and the layer-normalised actual tokens.
        That is the energy V-JEPA 2 is trained to minimise: ``forward_target`` in
        ``app/vjepa/train.py`` applies ``F.layer_norm`` to the targets and the loss is L1.
        """
        pred = self.predict(pixels, context_frames)
        target = full_tokens[:, full_tokens.shape[1] - pred.shape[1] :]
        target = F.layer_norm(target, (target.shape[-1],))
        return (pred - target).abs().mean(dim=-1)
