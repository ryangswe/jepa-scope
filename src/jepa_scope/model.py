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
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import OUTPUT_VERSION, latent_file
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


def device_available(device: str) -> bool:
    """Whether ``device`` (``cpu``, ``mps``, ``cuda`` or ``cuda:1`` and so on) can run here."""
    kind = torch.device(device).type
    if kind == "cpu":
        return True
    if kind == "mps":
        return torch.backends.mps.is_available()
    is_available = getattr(getattr(torch, kind, None), "is_available", None)
    return bool(is_available()) if callable(is_available) else False


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


WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
UNKNOWN_WEIGHTS = "unknown"
_DIGEST = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


def _content_name(f: Path) -> str:
    """A name for a weight file's content. A Hugging Face snapshot links each file to a blob named
    by its hash, which is used as is. Anything else (an ordinary file, or a link to an ordinary
    file such as a "latest" checkpoint) is described by the file it resolves to: path, size and
    modification time, so replacing it changes the name."""
    if f.is_symlink():
        target = Path(os.readlink(f))
        if target.parent.name == "blobs" and _DIGEST.fullmatch(target.name):
            return target.name
    st = f.stat()  # follows links
    return f"{f.resolve()}:{st.st_size}:{st.st_mtime_ns}"


def weights_identity(folder: Path) -> str:
    """An identity for the weights in a model folder, from their content rather than a version label.

    It hashes ``config.json``'s bytes and the content name of each weight file (see
    ``_content_name``). A folder with no weight file, such as a snapshot where only the config has
    been fetched so far, gives ``UNKNOWN_WEIGHTS``: the identity cannot be known without the
    weights, and callers must load the model to find it.
    """
    weights = sorted(p for p in folder.iterdir() if p.suffix in WEIGHT_SUFFIXES)
    if not weights:
        return UNKNOWN_WEIGHTS
    h = hashlib.sha256()
    cfg = folder / "config.json"
    h.update(cfg.read_bytes() if cfg.exists() else b"")
    for f in weights:
        h.update(f"{f.name}={_content_name(f)}\n".encode())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class ModelSource:
    """Where a model's files are, and what identifies its weights."""

    model_id: str  # the Hub id, or the absolute path of a local folder
    weights: str  # weights_identity of the folder
    revision: str | None  # the snapshot's commit, for information; None for a local folder


def resolve_model(model_id: str, revision: str | None = None, cache_dir: str | None = None) -> ModelSource:
    """Find a model's local files and identify its weights. Hub files must already be cached."""
    if Path(model_id).is_dir():
        folder = Path(model_id).resolve()
        return ModelSource(str(folder), weights_identity(folder), None)
    from huggingface_hub import snapshot_download

    folder = Path(snapshot_download(model_id, revision=revision, local_files_only=True, cache_dir=cache_dir))
    return ModelSource(model_id, weights_identity(folder), folder.name)


def _preprocess_info(processor, crop: int) -> dict:
    """The preprocessing recipe, plus a hash so cached latents can be checked against it."""
    p = processor
    recipe = {
        "processor": type(p).__name__,
        "shortest_edge": int(p.size["shortest_edge"]) if p.size and "shortest_edge" in p.size else None,
        "crop_size": crop,
        "image_mean": [float(v) for v in p.image_mean],
        "image_std": [float(v) for v in p.image_std],
        "resample": str(p.resample),
        # embed.py downscales at decode time (PyAV area averaging) to this shortest edge first.
        "decode_resize": "area-to-shortest-edge",
    }
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]
    return {**recipe, "hash": digest}


class Encoder:
    def __init__(
        self,
        model,
        processor,
        model_id: str,
        device: torch.device,
        dtype: torch.dtype,
        *,
        source: ModelSource,
    ):
        """Build with :meth:`from_pretrained`. ``source`` identifies the weights; it must describe the
        same files ``model`` was loaded from, or the cache check will disagree with the files."""
        self.model = model.eval()
        self.processor = processor
        self.model_id = model_id
        self.source = source
        self.device = device
        self.dtype = dtype
        cfg = model.config
        self.crop = int(cfg.crop_size)
        self.patch = int(cfg.patch_size)
        self.tubelet = int(cfg.tubelet_size)
        self.hidden_size = int(cfg.hidden_size)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        dtype: str | None = None,
        revision: str | None = None,
    ) -> Encoder:
        from transformers import AutoModel, AutoVideoProcessor

        dev = pick_device(device)
        dt = pick_dtype(dev, dtype)
        model = AutoModel.from_pretrained(model_id, dtype=dt, revision=revision).to(dev)
        processor = AutoVideoProcessor.from_pretrained(model_id, revision=revision)
        return cls(model, processor, model_id, dev, dt, source=resolve_model(model_id, revision))

    @classmethod
    def describe(cls, model_id: str, revision: str | None, window: int, sample_fps: float) -> dict:
        """The representation block :meth:`representation` would give, without loading weights.

        It reads the config, the processor and the names, sizes and modification times of the
        model's files, so the cache can be checked before deciding whether the model needs loading.
        Its ``weights`` is ``UNKNOWN_WEIGHTS`` when the weight files are not in the local snapshot
        yet; then only a loaded encoder's :meth:`representation` can say.
        """
        from transformers import AutoConfig, AutoVideoProcessor

        cfg = AutoConfig.from_pretrained(model_id, revision=revision)
        processor = AutoVideoProcessor.from_pretrained(model_id, revision=revision)
        source = resolve_model(model_id, revision)
        return latent_file.representation(
            model_id=source.model_id,
            weights=source.weights,
            hidden_size=int(cfg.hidden_size),
            preprocess_hash=_preprocess_info(processor, int(cfg.crop_size))["hash"],
            window=window,
            sample_fps=sample_fps,
            output_version=OUTPUT_VERSION,
        )

    def representation(self, window: int, sample_fps: float) -> dict:
        """What a window vector from this encoder is: the block ``embed_video`` writes to the header."""
        return latent_file.representation(
            model_id=self.source.model_id,
            weights=self.source.weights,
            hidden_size=self.hidden_size,
            preprocess_hash=self.preprocess_info()["hash"],
            window=window,
            sample_fps=sample_fps,
            output_version=OUTPUT_VERSION,
        )

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
        return _preprocess_info(self.processor, self.crop)

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
