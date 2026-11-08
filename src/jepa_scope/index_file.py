"""The index file: window vectors for a whole folder of videos in one safetensors file.

An index file is a plain safetensors file, built the same way as a ``.latent`` file. Its metadata
holds one JSON string under the key ``jepa_scope`` that records how the vectors were made (the
``representation`` block copied from the clips), which clips they came from, and how similarity is
to be computed. The tensors hold every clip's window vectors, in clip order. Any safetensors reader
can open the tensors; this module adds validation and a small typed view, ``Index``.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import SafetensorError, safe_open
from safetensors.numpy import load_file, save_file

from . import latent_file

INDEX_FORMAT = "jepa-scope.index"
INDEX_FORMAT_VERSION = 1
META_KEY = "jepa_scope"

REQUIRED_HEADER_KEYS = (
    "format",
    "format_version",
    "representation",
    "run",
    "embedding",
    "root",
    "recursive",
    "skipped",
    "clips",
)
RUN_KEYS = ("max_seconds", "device", "dtype")
SIMILARITIES = ("cosine", "centred-cosine")
_DRIVE = re.compile(r"[A-Za-z]:(/|$)")  # a Windows drive prefix, as in "C:/videos"
CLIP_KEYS = ("path", "sha256", "size", "mtime_ns", "duration", "width", "height", "windows")

# The exact dtype each tensor must have.
TENSOR_DTYPES = {
    "pooled": np.dtype(np.float32),
    "window_start_time": np.dtype(np.float64),
    "clip": np.dtype(np.int32),
    "mean": np.dtype(np.float32),
}


class IndexFileError(ValueError):
    """Raised when a file is not a valid index, or its tensors and header disagree."""


@dataclass
class Index:
    """A loaded index: the validated header and the four tensors.

    Row ``r`` of ``pooled`` is one window vector, ``window_start_time[r]`` is when that window
    starts in its clip, and ``clip[r]`` is the position of its clip in ``header["clips"]``.
    """

    header: dict[str, Any]
    pooled: np.ndarray
    window_start_time: np.ndarray
    clip: np.ndarray
    mean: np.ndarray

    def clip_rows(self, i: int) -> slice:
        """The rows of ``pooled`` that belong to clip ``i``, as a slice.

        Raises ``IndexError`` for a negative ``i`` or one past the last clip, instead of wrapping
        around the way a Python list would.
        """
        if not 0 <= i < len(self.header["clips"]):
            raise IndexError(f"clip {i} is out of range for an index of {len(self.header['clips'])} clips")
        start, stop = self.header["clips"][i]["windows"]
        return slice(start, stop)

    def clip_vector(self, i: int) -> np.ndarray:
        """The mean of clip ``i``'s raw window vectors, as float32 of shape ``[D]``.

        The rows are averaged exactly as stored. This function neither subtracts the index mean
        nor normalises the result; centring and normalisation belong to search.
        """
        return self.pooled[self.clip_rows(i)].mean(axis=0, dtype=np.float64).astype(np.float32)

    def resolve_root(self, index_path: str | Path, source_root: str | Path | None = None) -> Path:
        """The folder that the clips' relative paths are resolved against.

        ``source_root``, when given, wins over whatever the index recorded; this is for a folder
        that has moved. Otherwise a recorded root of ``"."`` means the directory that holds the
        index file itself, so a folder with its colocated index can be moved or copied. Any other
        recorded root is an absolute path and is used as it is.
        """
        if source_root is not None:
            return Path(source_root)
        root = self.header["root"]
        if root == ".":
            return Path(index_path).resolve().parent
        return Path(root)


def save(path: str | Path, tensors: dict[str, np.ndarray], header: dict[str, Any]) -> None:
    """Write an index file in the current format, after checking it.

    The current ``format`` and ``format_version`` always win over the caller's header, so saving
    never writes an old version number. The file is written under a temporary name and then
    renamed, so an interrupted write never leaves a half file behind.
    """
    header = {**header, "format": INDEX_FORMAT, "format_version": INDEX_FORMAT_VERSION}
    check(tensors, header)
    contiguous = {k: np.ascontiguousarray(v) for k, v in tensors.items()}
    try:
        text = json.dumps(header, sort_keys=True, allow_nan=False)
    except ValueError as exc:
        raise IndexFileError(f"the header cannot be written as JSON: {exc}") from exc
    # Write to a uniquely named file in the target's directory and rename it, so an interrupted
    # write never leaves a half file behind and two writers never share a temporary name.
    # The temporary file sits beside the name being replaced (not beside a symlink's target), and
    # gets the permissions an ordinary new file would (mkstemp creates it private).
    fd, tmp = tempfile.mkstemp(dir=Path(path).absolute().parent, prefix=Path(path).name + ".", suffix=".tmp")
    os.close(fd)
    try:
        save_file(contiguous, tmp, metadata={META_KEY: text})
        os.chmod(tmp, 0o666 & ~_umask())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_header(path: str | Path) -> dict[str, Any]:
    """Read and check only the header, without loading the tensors."""
    try:
        with safe_open(str(path), framework="numpy") as f:
            meta = f.metadata() or {}
    except SafetensorError as exc:
        raise IndexFileError(f"{path} is not a readable safetensors file: {exc}") from exc
    if META_KEY not in meta:
        raise IndexFileError(f"{path} is a safetensors file but has no '{META_KEY}' header")
    try:
        header = json.loads(meta[META_KEY], parse_constant=_refuse_constant)
    except ValueError as exc:
        raise IndexFileError(f"{path} has a header that is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise IndexFileError(f"{path} has a header that is not a JSON object")
    check_header(header)
    return header


def _refuse_constant(name: str) -> None:
    raise ValueError(f"the constant {name} is not allowed in the header")


def load(path: str | Path) -> Index:
    """Load and fully check an index file."""
    header = read_header(path)
    try:
        tensors = load_file(str(path))
    except SafetensorError as exc:
        raise IndexFileError(f"{path} is damaged: {exc}") from exc
    check(tensors, header)
    return Index(
        header=header,
        pooled=tensors["pooled"],
        window_start_time=tensors["window_start_time"],
        clip=tensors["clip"],
        mean=tensors["mean"],
    )


def check(tensors: dict[str, np.ndarray], header: dict[str, Any]) -> None:
    """Check a header and its tensors together, for callers that build both in memory.

    An index with no clips is valid: it has zero rows and a ``mean`` of zeros.
    """
    check_header(header)
    missing = [k for k in TENSOR_DTYPES if k not in tensors]
    if missing:
        raise IndexFileError(f"missing tensors {missing}")
    for key, dtype in TENSOR_DTYPES.items():
        if tensors[key].dtype != dtype:
            raise IndexFileError(f"{key} must be {dtype}, not {tensors[key].dtype}")
    pooled = tensors["pooled"]
    if pooled.ndim != 2:
        raise IndexFileError(f"pooled must be 2-dimensional, not shape {tuple(pooled.shape)}")
    rows, dim = pooled.shape
    hidden = header["representation"].get("hidden_size")
    if not _is_int(hidden) or dim != hidden:
        raise IndexFileError(f"pooled has width {dim} but the representation's hidden_size is {hidden}")
    expected = {"window_start_time": (rows,), "clip": (rows,), "mean": (dim,)}
    for key, shape in expected.items():
        if tuple(tensors[key].shape) != shape:
            raise IndexFileError(f"{key} has shape {tuple(tensors[key].shape)}, expected {shape}")
    for key in TENSOR_DTYPES:
        if not np.isfinite(tensors[key]).all():
            raise IndexFileError(f"{key} holds NaN or infinite values")
    _check_ranges(tensors, header["clips"], rows)
    _check_mean(tensors["mean"], pooled)


def _check_mean(mean: np.ndarray, pooled: np.ndarray) -> None:
    """``mean`` must be the window-weighted mean of ``pooled`` (zeros when there are no rows)."""
    if len(pooled) == 0:
        if mean.any():
            raise IndexFileError("mean must be all zeros for an index with no windows")
        return
    expected = pooled.mean(axis=0, dtype=np.float64)
    atol = 1e-5 * max(1.0, float(np.abs(pooled).max()))
    if not np.allclose(mean.astype(np.float64), expected, rtol=0.0, atol=atol):
        raise IndexFileError("mean is not the mean of pooled over all windows")


def check_header(header: dict[str, Any]) -> None:
    """Check the header on its own: its keys, the representation hash and every clip entry."""
    if header.get("format") != INDEX_FORMAT:
        raise IndexFileError(f"not a {INDEX_FORMAT} file (format={header.get('format')!r})")
    version = header.get("format_version")
    if not _is_int(version) or version != INDEX_FORMAT_VERSION:
        raise IndexFileError(
            f"format_version {header.get('format_version')} is not supported "
            f"(expected {INDEX_FORMAT_VERSION}); rebuild the index"
        )
    missing = [k for k in REQUIRED_HEADER_KEYS if k not in header]
    if missing:
        raise IndexFileError(f"header is missing {missing}")
    block = header["representation"]
    if not isinstance(block, dict) or block.get("hash") != latent_file.representation_hash(block):
        raise IndexFileError("the representation block does not match its hash")
    run = header["run"]
    if not isinstance(run, dict) or any(k not in run for k in RUN_KEYS):
        raise IndexFileError(f"run must be an object with {list(RUN_KEYS)}")
    max_seconds = run["max_seconds"]
    if max_seconds is not None and not (_is_number(max_seconds) and math.isfinite(max_seconds) and max_seconds > 0):
        raise IndexFileError("run.max_seconds must be null or a finite number above zero")
    embedding = header["embedding"]
    if not isinstance(embedding, dict) or embedding.get("kind") != "window_vector":
        raise IndexFileError("embedding.kind must be 'window_vector'")
    if embedding.get("similarity") not in SIMILARITIES:
        raise IndexFileError(f"embedding.similarity must be one of {list(SIMILARITIES)}")
    root = header["root"]
    if not isinstance(root, str) or not (root == "." or Path(root).is_absolute()):
        raise IndexFileError("root must be '.' or an absolute path")
    if not isinstance(header["recursive"], bool):
        raise IndexFileError("recursive must be true or false")
    skipped = header["skipped"]
    if not isinstance(skipped, list) or not all(
        isinstance(e, dict) and isinstance(e.get("path"), str) and isinstance(e.get("reason"), str) for e in skipped
    ):
        raise IndexFileError("skipped must be a list of objects with a string path and a string reason")
    if not isinstance(header["clips"], list):
        raise IndexFileError("clips must be a list")
    for i, entry in enumerate(header["clips"]):
        _check_clip_entry(i, entry)
    paths = [e["path"] for e in header["clips"]]
    if len(set(paths)) != len(paths):
        raise IndexFileError("two clips have the same path")


def _check_clip_entry(i: int, entry: Any) -> None:
    if not isinstance(entry, dict):
        raise IndexFileError(f"clips[{i}] must be an object")
    missing = [k for k in CLIP_KEYS if k not in entry]
    if missing:
        raise IndexFileError(f"clips[{i}] is missing {missing}")
    _check_path(i, entry["path"])
    if not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
        raise IndexFileError(f"clips[{i}].sha256 must be 64 lowercase hex characters")
    for key, floor in (("size", 0), ("mtime_ns", 0), ("width", 1), ("height", 1)):
        if not _is_int(entry[key]):
            raise IndexFileError(f"clips[{i}].{key} must be an integer")
        if entry[key] < floor:
            raise IndexFileError(f"clips[{i}].{key} must be at least {floor}")
    duration = entry["duration"]
    if not _is_number(duration) or not math.isfinite(duration) or duration <= 0:
        raise IndexFileError(f"clips[{i}].duration must be a finite number above zero")
    windows = entry["windows"]
    if not (isinstance(windows, list) and len(windows) == 2 and all(_is_int(v) for v in windows)):
        raise IndexFileError(f"clips[{i}].windows must be [start, stop) with two integers")


def is_valid_clip_path(path: Any) -> bool:
    """Whether ``path`` can be stored as a clip path: relative to the root, with forward slashes,
    and no empty, ``.`` or ``..`` part. A backslash, a NUL or a Windows drive prefix (``C:/``) is
    refused, because the path would mean something else on another system. A colon elsewhere, as
    in a timestamped name, is fine. The indexer skips files whose paths fail this, rather than
    failing the whole index."""
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        return False
    if _DRIVE.match(path):
        return False
    return not any(part in ("", ".", "..") for part in path.split("/"))


def _check_path(i: int, path: Any) -> None:
    if not is_valid_clip_path(path):
        raise IndexFileError(f"clips[{i}].path must be a plain relative path with forward slashes")


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_ranges(tensors: dict[str, np.ndarray], clips: list[dict[str, Any]], rows: int) -> None:
    """The clips' window ranges must tile ``[0, rows)`` in order and agree with the tensors."""
    expected_start = 0
    times = tensors["window_start_time"]
    owner = tensors["clip"]
    for i, entry in enumerate(clips):
        start, stop = entry["windows"]
        if stop <= start:
            raise IndexFileError(f"clips[{i}].windows [{start}, {stop}) is empty")
        if start != expected_start:
            raise IndexFileError(f"clips[{i}].windows starts at {start}, expected {expected_start}")
        if (owner[start:stop] != i).any():
            raise IndexFileError(f"clip does not say {i} for every row of clips[{i}]")
        if stop - start > 1 and (np.diff(times[start:stop]) <= 0).any():
            raise IndexFileError(f"window_start_time does not strictly increase within clips[{i}]")
        expected_start = stop
    if expected_start != rows:
        raise IndexFileError(f"the clips cover {expected_start} rows but pooled has {rows}")
