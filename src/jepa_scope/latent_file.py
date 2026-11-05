"""The ``.latent`` file: tensors plus the metadata needed to interpret them.

A ``.latent`` file is a plain safetensors file. Its metadata holds one JSON string under the key
``jepa_scope`` that records which model made the tensors, how the clip was preprocessed and how the
tokens are laid out. Any safetensors reader can open the tensors; this module adds validation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import SafetensorError, safe_open
from safetensors.numpy import load_file, save_file

FORMAT = "jepa-scope.latent"
FORMAT_VERSION = 1
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
)
TIMELINE_KEYS = ("timeline_time", "timeline_window", "timeline_pca_rgb", "timeline_surprise")


class LatentFileError(ValueError):
    pass


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path: str | Path, tensors: dict[str, np.ndarray], header: dict[str, Any]) -> None:
    header = {"format": FORMAT, "format_version": FORMAT_VERSION, **header}
    _check_header(header)
    contiguous = {k: np.ascontiguousarray(v) for k, v in tensors.items()}
    # Write next to the target and rename, so an interrupted write never leaves a half file behind.
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    save_file(contiguous, str(tmp), metadata={META_KEY: json.dumps(header, sort_keys=True)})
    os.replace(tmp, path)


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


def _check_header(header: dict[str, Any]) -> None:
    if header.get("format") != FORMAT:
        raise LatentFileError(f"not a {FORMAT} file (format={header.get('format')!r})")
    if header.get("format_version") != FORMAT_VERSION:
        raise LatentFileError(
            f"format_version {header.get('format_version')} is not supported (expected {FORMAT_VERSION})"
        )
    missing = [k for k in REQUIRED_HEADER_KEYS if k not in header]
    if missing:
        raise LatentFileError(f"header is missing {missing}")
