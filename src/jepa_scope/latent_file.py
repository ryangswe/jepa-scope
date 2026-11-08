"""The ``.latent`` file: tensors plus the metadata needed to interpret them.

A ``.latent`` file is a plain safetensors file. Its metadata holds one JSON string under the key
``jepa_scope`` that records which model made the tensors, how the clip was preprocessed and how the
tokens are laid out. Any safetensors reader can open the tensors; this module adds validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import SafetensorError, safe_open
from safetensors.numpy import load_file, save_file

FORMAT = "jepa-scope.latent"
# Version 2 (2026-11-06) added the `timeline_pooled` tensor and the `representation` header block.
# Files of another version are refused; the cache re-embeds them.
FORMAT_VERSION = 2
META_KEY = "jepa_scope"

REQUIRED_HEADER_KEYS = (
    "format",
    "format_version",
    "model_id",
    "hidden_size",
    "grid",
    "window",
    "preprocess",
    "crop",
    "source",
    "run",
    "representation",
)
TIMELINE_KEYS = ("timeline_time", "timeline_window", "timeline_pca_rgb", "timeline_surprise", "timeline_pooled")
TOKEN_LAYER = "encoder.last_hidden_state"


class LatentFileError(ValueError):
    pass


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def representation_hash(block: dict[str, Any]) -> str:
    """Hash of a representation block's fields (everything except ``hash`` itself).

    The canonical form is JSON with keys sorted, no whitespace (``,`` and ``:`` separators), ASCII
    escapes, and floats written as Python writes them (``8.0``); the hash is the first 16 hex
    characters of its SHA-256.
    """
    fields = {k: v for k, v in block.items() if k != "hash"}
    text = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def representation(
    *,
    model_id: str,
    weights: str,
    hidden_size: int,
    preprocess_hash: str,
    window: int,
    sample_fps: float,
    output_version: int,
) -> dict[str, Any]:
    """The canonical description of what a window vector is, with a hash over it.

    Two files whose blocks have the same hash hold vectors made the same way: the same weights,
    preprocessing, token layer, normalisation, pooling and windowing. ``weights`` identifies the
    weights by content (see ``model.weights_identity``), not by a version label.
    """
    block = {
        "model_id": model_id,
        "weights": weights,
        "hidden_size": int(hidden_size),
        "preprocess_hash": preprocess_hash,
        "token_layer": TOKEN_LAYER,
        "token_norm": "none",
        "pooling": "mean_all",
        "window": {
            "frames": int(window),
            "stride": int(window) // 2,
            "context_frames": int(window) // 2,
            "sample_fps": float(sample_fps),
        },
        "output_version": int(output_version),
    }
    return {**block, "hash": representation_hash(block)}


def save(path: str | Path, tensors: dict[str, np.ndarray], header: dict[str, Any]) -> None:
    """Write a file in the current format. The current ``format`` and ``format_version`` always win
    over the caller's header, so saving never writes an old version number."""
    header = {**header, "format": FORMAT, "format_version": FORMAT_VERSION}
    _check_header(header)
    contiguous = {k: np.ascontiguousarray(v) for k, v in tensors.items()}
    # Write to a uniquely named file in the target's directory and rename it, so an interrupted
    # write never leaves a half file behind and two processes embedding the same video never share
    # a temporary name. The temporary file gets the permissions an ordinary new file would
    # (mkstemp creates it private).
    fd, tmp = tempfile.mkstemp(dir=Path(path).absolute().parent, prefix=Path(path).name + ".", suffix=".tmp")
    os.close(fd)
    try:
        save_file(contiguous, tmp, metadata={META_KEY: json.dumps(header, sort_keys=True)})
        os.chmod(tmp, 0o666 & ~_umask())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def read_header(path: str | Path) -> dict[str, Any]:
    try:
        with safe_open(str(path), framework="numpy") as f:
            meta = f.metadata() or {}
    except SafetensorError as exc:
        raise LatentFileError(f"{path} is not a readable safetensors file: {exc}") from exc
    if META_KEY not in meta:
        raise LatentFileError(f"{path} is a safetensors file but has no '{META_KEY}' header")
    header = json.loads(meta[META_KEY])
    _check_header(header)
    return header


def load(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    header = read_header(path)
    try:
        tensors = load_file(str(path))
    except SafetensorError as exc:
        raise LatentFileError(f"{path} is damaged: {exc}") from exc
    check_tensors(tensors, header)
    return tensors, header


def check_tensors(tensors: dict[str, np.ndarray], header: dict[str, Any]) -> None:
    """Check that the timeline tensors agree with each other and with the header's grid."""
    missing = [k for k in (*TIMELINE_KEYS, "pooled", "window_start_time") if k not in tensors]
    if missing:
        raise LatentFileError(f"missing tensors {missing}")
    grid = header["grid"]
    side = grid["crop"] // grid["patch"]
    count = len(tensors["timeline_time"])
    windows = len(tensors["window_start_time"])
    half = grid["frames"] // grid["tubelet"] // 2
    if count != (windows + 1) * half:
        raise LatentFileError(f"{count} timeline tubelets do not fit {windows} half-overlapping windows")
    owner = tensors["timeline_window"]
    if owner.size and (owner.min() < 0 or owner.max() >= windows):
        raise LatentFileError("timeline_window points outside the windows")
    expected = {
        "timeline_window": (count,),
        "timeline_pca_rgb": (count, side, side, 3),
        "timeline_surprise": (count, side, side),
        "pooled": (windows, header["hidden_size"]),
        "timeline_pooled": (count, header["hidden_size"]),
    }
    if "patches" in tensors:
        expected["patches"] = (
            windows,
            grid["frames"] // grid["tubelet"],
            side,
            side,
            header["hidden_size"],
        )
    for key, shape in expected.items():
        if tuple(tensors[key].shape) != shape:
            raise LatentFileError(f"{key} has shape {tuple(tensors[key].shape)}, expected {shape}")
    if tensors["timeline_pca_rgb"].dtype != np.uint8:
        raise LatentFileError("timeline_pca_rgb must be uint8")
    for key in ("pooled", "timeline_pooled"):
        if tensors[key].dtype != np.float32:
            raise LatentFileError(f"{key} must be float32, not {tensors[key].dtype}")
        if not np.isfinite(tensors[key]).all():
            raise LatentFileError(f"{key} holds NaN or infinite values")


def _check_header(header: dict[str, Any]) -> None:
    if header.get("format") != FORMAT:
        raise LatentFileError(f"not a {FORMAT} file (format={header.get('format')!r})")
    if header.get("format_version") != FORMAT_VERSION:
        raise LatentFileError(
            f"format_version {header.get('format_version')} is not supported (expected {FORMAT_VERSION}); "
            "re-embed the video with `jepa-scope embed`"
        )
    missing = [k for k in REQUIRED_HEADER_KEYS if k not in header]
    if missing:
        raise LatentFileError(f"header is missing {missing}")
    block = header["representation"]
    if not isinstance(block, dict) or block.get("hash") != representation_hash(block):
        raise LatentFileError("the representation block does not match its hash")
