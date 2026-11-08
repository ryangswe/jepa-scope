"""The result cache: where a video's ``.latent`` file lives for a set of embedding settings, and
whether a file found there can be reused.

Every command that embeds goes through this rule, so a clip embedded once by ``jepa-scope clip.mp4``
is reused by ``jepa-scope index`` and ``jepa-scope search`` with the same settings, and the other
way round. The cache file name is a function of the video's SHA-256 and the settings that change
the output; it does not depend on where the video is or what it is called.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import OUTPUT_VERSION, latent_file

if TYPE_CHECKING:
    import torch

    from .model import Encoder

CACHE_DIR = Path.home() / ".cache" / "jepa-scope"


def _print(message: str) -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class EmbedSettings:
    """The embedding settings a command was given.

    ``device`` and ``dtype`` are kept as given (None means "the best available"); ``resolved``
    says what they resolve to on this machine. ``batch`` changes speed and memory, not the output,
    so it is not part of the cache key.
    """

    model: str
    fps: float
    window: int
    batch: int = 2
    max_seconds: float | None = None
    device: str | None = None
    dtype: str | None = None

    def __post_init__(self) -> None:
        # The cache key writes these with json.dumps, where 8 and 8.0 differ. The CLI always passes
        # floats; normalising keeps settings built elsewhere (from an index header) on the same key.
        object.__setattr__(self, "fps", float(self.fps))
        object.__setattr__(self, "window", int(self.window))
        if self.max_seconds is not None:
            object.__setattr__(self, "max_seconds", float(self.max_seconds))

    @classmethod
    def from_args(cls, args: Any) -> EmbedSettings:
        """Settings from parsed command line arguments (``--model``, ``--fps`` and so on).

        A ``--model`` that names an existing local folder becomes its absolute path, which is the
        ``model_id`` the representation block records, so settings rebuilt from an index header
        (``from_index``) give the same cache key. A Hugging Face id is kept as it is.
        """
        model = str(Path(args.model).resolve()) if Path(args.model).is_dir() else args.model
        return cls(
            model=model,
            fps=args.fps,
            window=args.window,
            batch=args.batch,
            max_seconds=args.max_seconds,
            device=args.device,
            dtype=args.dtype,
        )

    @classmethod
    def from_index(cls, header: dict[str, Any]) -> EmbedSettings:
        """The settings an index was built with, from its validated header: the model, window and
        sampling rate of its representation block and its run settings. ``batch`` is the default,
        since it does not change the output."""
        rep, run = header["representation"], header["run"]
        return cls(
            model=rep["model_id"],
            fps=rep["window"]["sample_fps"],
            window=rep["window"]["frames"],
            max_seconds=run["max_seconds"],
            device=run["device"],
            dtype=run["dtype"],
        )

    def _resolve(self) -> tuple[torch.device, torch.dtype]:
        from .model import pick_device, pick_dtype

        device = pick_device(self.device)
        return device, pick_dtype(device, self.dtype)

    def run(self) -> dict[str, Any]:
        """The run settings an index records (``index_file.RUN_KEYS``), with the device type and
        dtype resolved, so "best available" and naming the same device give the same block."""
        device, dtype = self._resolve()
        return {
            "max_seconds": self.max_seconds,
            "device": device.type,
            "dtype": str(dtype).removeprefix("torch."),
        }


def cache_path(video_sha256: str, settings: EmbedSettings, patches: bool, cache_dir: Path | None = None) -> Path:
    """Cache file for this video and these settings, with or without the ``patches`` tensor.

    The key covers everything that changes the output: the output version (bumped whenever the
    energy, preprocessing or windowing changes), the model, sampling, and the device and dtype
    actually used, so a CPU float32 run and an MPS float16 run never share a file. ``cache_dir``
    defaults to ``CACHE_DIR``, read when the function is called.
    """
    device, dtype = settings._resolve()
    key = {
        "output_version": OUTPUT_VERSION,
        "model": settings.model,
        "fps": settings.fps,
        "window": settings.window,
        "max_seconds": settings.max_seconds,
        "device": device.type,
        "dtype": str(dtype),
        "patches": patches,
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return (CACHE_DIR if cache_dir is None else cache_dir) / f"{video_sha256[:16]}-{digest}.latent"


class PretrainedModel:
    """Access to a pre-trained encoder that loads the weights only when they are needed.

    ``representation()`` describes what the settings would produce from the model's config,
    processor and file names, without loading weights, unless the weight files are not in the
    local snapshot yet; then it loads the model to find out. ``encoder()`` loads the model on
    first use and keeps it. Both remember their result.
    """

    def __init__(self, settings: EmbedSettings, log: Callable[[str], None] = _print):
        self.settings = settings
        self.log = log
        self._encoder: Encoder | None = None
        self._block: dict[str, Any] | None = None

    def representation(self) -> dict[str, Any]:
        if self._block is None:
            from .model import UNKNOWN_WEIGHTS, Encoder

            s = self.settings
            block = Encoder.describe(s.model, None, s.window, s.fps)
            if block["weights"] == UNKNOWN_WEIGHTS:
                block = self.encoder().representation(s.window, s.fps)
            self._block = block
        return self._block

    def encoder(self) -> Encoder:
        if self._encoder is None:
            from .model import Encoder

            s = self.settings
            self.log(f"loading {s.model} ...")
            self._encoder = Encoder.from_pretrained(s.model, device=s.device, dtype=s.dtype)
        return self._encoder


def expected_representation(settings: EmbedSettings) -> str:
    """Hash of the representation these settings would produce, read without loading weights when
    the weight files are already in the local snapshot, and by loading the model otherwise."""
    return PretrainedModel(settings, log=lambda _: None).representation()["hash"]


def cache_is_valid(path: Path, expected_representation: str, log: Callable[[str], None] = _print) -> bool:
    """Whether a cache file can be reused: readable in the current format, made by this output
    version, and holding the representation these settings would produce (same weights,
    preprocessing, pooling and windowing). Otherwise it is re-embedded over the same path, and
    ``log`` is told why (nothing is logged for a file that does not exist)."""
    if not path.exists():
        return False
    try:
        header = latent_file.read_header(path)
    except (latent_file.LatentFileError, OSError, ValueError) as exc:
        log(f"re-embedding: the cache file {path.name} cannot be used ({exc})")
        return False
    if header["run"].get("output_version") != OUTPUT_VERSION:
        log(f"re-embedding: the cache file {path.name} is from another output version")
        return False
    if header["representation"]["hash"] != expected_representation:
        log(f"re-embedding: the cache file {path.name} holds a different representation")
        return False
    return True


def find_cache_file(
    video_sha256: str,
    settings: EmbedSettings,
    expected_hash: str,
    cache_dir: Path | None = None,
    log: Callable[[str], None] = _print,
) -> Path | None:
    """A reusable cache file for this video and these settings, or None.

    The file with the ``patches`` tensor is tried first, then the one without. A file is reusable
    when ``cache_is_valid`` accepts it and its header's ``source.sha256`` is the full
    ``video_sha256`` (the cache name only carries a 16-character prefix of it).
    """
    for patches in (True, False):
        path = cache_path(video_sha256, settings, patches, cache_dir)
        if not cache_is_valid(path, expected_hash, log):
            continue
        try:
            source = latent_file.read_header(path)["source"]
        except (latent_file.LatentFileError, OSError, ValueError) as exc:
            log(f"re-embedding: the cache file {path.name} cannot be used ({exc})")
            continue
        if source.get("sha256") != video_sha256:
            log(f"re-embedding: the cache file {path.name} belongs to another video")
            continue
        return path
    return None
