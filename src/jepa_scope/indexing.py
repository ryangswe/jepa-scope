"""Build or refresh the index of a folder of videos.

``build_or_refresh`` walks a folder for videos, finds or makes each one's ``.latent`` file through
the result cache (``cache.py``), and writes one index file (``index_file.py``) holding every clip's
window vectors.

A refresh is incremental. A clip whose size and modification time match its entry in the existing
index, and whose cache file is still there and valid, is kept without reading the video again.
Any other clip is hashed, and it is embedded only when no valid cache file exists for that hash.
Byte-identical files share one cache file and are embedded at most once per run. An existing index
made with other settings, from another folder, or in a format this version cannot read is rebuilt
from scratch, which re-embeds only what the cache does not already hold. The similarity
(``centre``) and whether subfolders are included (``recursive``) are remembered from the existing
index unless the caller gives them.

One bad file never stops the run: it is reported and listed under ``skipped`` in the index with
the reason. A file whose content is the problem (too short, too few frames per second, no video
stream, not decodable) is remembered and not tried again until it changes; other failures (out of
memory, invalid model output, a read error) are tried again on the next run. The model is reached
through a provider object (``ModelProvider``) so that a refresh with nothing new to embed never
loads the weights.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import av
import numpy as np
from safetensors import SafetensorError, safe_open

from . import __version__, index_file, latent_file
from .cache import EmbedSettings, cache_path, find_cache_file
from .embed import embed_video
from .search import DEFAULT_SIMILARITY

if TYPE_CHECKING:
    from .model import Encoder

INDEX_NAME = "jepa-scope.index"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi")  # compared case-insensitively

# What embedding one video can raise when that video, or this run on it, is the problem. PyAV
# raises av.FFmpegError subclasses for files it cannot open or decode (InvalidDataError is also a
# ValueError, FileNotFoundError also an OSError, MemoryError also a MemoryError, but some are
# neither). decode raises ValueError for no video stream, no frames, or a source below the sampling
# rate; embed_video raises ValueError for a clip shorter than one window and RuntimeError for NaN or
# inf model output (Encoder.preprocess raises RuntimeError when preprocessing gives an unexpected
# shape, and torch raises RuntimeError subclasses when a device runs out of memory). OSError covers
# a file that cannot be read, MemoryError a clip too long for this machine.
CLIP_ERRORS = (av.FFmpegError, ValueError, RuntimeError, OSError, MemoryError)
# Of those, the ones that depend only on the file's content and so recur until it changes.
CONTENT_ERRORS = (ValueError, av.FFmpegError)
TRANSIENT_ERRORS = (OSError, MemoryError, RuntimeError)  # retried next run, even from PyAV


class ModelProvider(Protocol):
    """How the indexer reaches the model.

    ``representation()`` is the representation block the settings produce, ideally found without
    loading weights; ``encoder()`` returns the loaded encoder. Both should remember their result.
    ``cache.PretrainedModel`` is the implementation the command line uses.
    """

    def representation(self) -> dict[str, Any]: ...

    def encoder(self) -> Encoder: ...


@dataclass(frozen=True)
class ClipVectors:
    """What the index needs from one clip's ``.latent`` file (never its ``patches``)."""

    cache_file: Path
    pooled: np.ndarray  # float32 [W, D]
    window_start_time: np.ndarray  # float64 [W]
    duration: float
    width: int
    height: int


class ClipSkipped(Exception):
    """A clip cannot be indexed; the message is the reason recorded under ``skipped``.

    ``remember`` is True when the cause is the file's content, so the same bytes would fail the
    same way again; ``sha256`` is set once the file has been hashed. Only a skip with both is
    remembered across runs.
    """

    def __init__(self, reason: str, remember: bool = False):
        super().__init__(reason)
        self.remember = remember
        self.sha256: str | None = None


@dataclass
class Report:
    """What one ``build_or_refresh`` run did."""

    path: Path  # the index file written
    clips: int = 0
    windows: int = 0
    # The header's skipped entries: path and reason, plus sha256, size, mtime_ns and jepa_scope
    # (the version) for a skip that is remembered.
    skipped: list[dict[str, Any]] = field(default_factory=list)
    embedded: int = 0  # clips embedded in this run
    hashed: int = 0  # files whose SHA-256 was computed in this run
    rebuilt: str | None = None  # why an existing index was rebuilt from scratch, if it was
    centre: bool = False  # the choices this run used, given or remembered
    recursive: bool = False


def find_videos(folder: Path, recursive: bool = False) -> list[tuple[str, Path]]:
    """Videos under ``folder`` as ``(relative POSIX path, path)``, sorted by relative path.

    A video is a name ending in one of ``VIDEO_EXTENSIONS`` (any case) that is not a directory.
    Any name starting with ``.``, file or directory, is skipped. Subfolders are entered only when
    ``recursive`` is True, and symlinked directories never are, so a link cycle cannot occur.
    Symlinked files are included.
    """
    found: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(folder, followlinks=False):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")] if recursive else []
        for name in filenames:
            if name.startswith(".") or not name.lower().endswith(VIDEO_EXTENSIONS):
                continue
            path = Path(dirpath) / name
            found.append((path.relative_to(folder).as_posix(), path))
    # Emptying dirnames stops os.walk from descending. It lists a symlink to a directory under
    # dirnames, never filenames, so it is never mistaken for a video, and with followlinks=False
    # it is not entered either.
    return sorted(found, key=lambda item: item[0])


def read_clip_vectors(path: Path, hidden_size: int) -> ClipVectors:
    """Read the window vectors and the clip facts from a ``.latent`` file, without its patches.

    The caller has already matched the file to its video (``cache.find_cache_file``). Raises
    ``ClipSkipped`` when the vectors would not make a valid set of index rows.
    """
    try:
        source = latent_file.read_header(path)["source"]
        with safe_open(str(path), framework="numpy") as f:
            pooled = f.get_tensor("pooled")
            starts = f.get_tensor("window_start_time")
    except (latent_file.LatentFileError, SafetensorError, OSError, ValueError, KeyError) as exc:
        raise ClipSkipped(f"the cache file {path.name} cannot be read ({exc})") from exc
    if pooled.dtype != np.float32 or pooled.ndim != 2 or pooled.shape[1] != hidden_size or len(pooled) == 0:
        raise ClipSkipped(f"the cache file {path.name} has pooled of shape {pooled.shape} ({pooled.dtype})")
    if starts.dtype != np.float64 or starts.shape != (len(pooled),):
        raise ClipSkipped(f"the cache file {path.name} has window_start_time that does not match pooled")
    if not (np.isfinite(pooled).all() and np.isfinite(starts).all()):
        raise ClipSkipped(f"the cache file {path.name} holds NaN or infinite values")
    if len(starts) > 1 and (np.diff(starts) <= 0).any():
        raise ClipSkipped(f"the cache file {path.name} has window start times that do not increase")
    return ClipVectors(
        cache_file=path,
        pooled=pooled,
        window_start_time=starts,
        duration=float(source["duration"]),
        width=int(source["width"]),
        height=int(source["height"]),
    )


class ClipCache:
    """Finds or makes the ``.latent`` file for a video hash, at most once per hash per run."""

    def __init__(
        self,
        model: ModelProvider,
        settings: EmbedSettings,
        patches: bool,
        cache_dir: Path | None,
        log: Callable[[str], None],
    ):
        self.model = model
        self.settings = settings
        self.patches = patches
        self.cache_dir = cache_dir
        self.log = log
        self.embedded = 0
        self._done: dict[str, ClipVectors | ClipSkipped] = {}
        self._missing: set[str] = set()  # hashes already looked up with no valid cache file

    def find(self, video_sha256: str) -> ClipVectors | None:
        """The vectors from a valid cache file for this hash, or None if there is none."""
        done = self._done.get(video_sha256)
        if isinstance(done, ClipVectors):
            return done
        if done is not None or video_sha256 in self._missing:
            return None
        expected = self.model.representation()
        path = find_cache_file(video_sha256, self.settings, expected["hash"], self.cache_dir, self.log)
        if path is not None:
            try:
                vectors = read_clip_vectors(path, expected["hidden_size"])
            except ClipSkipped as exc:
                self.log(f"re-embedding: {exc}")
            else:
                self._done[video_sha256] = vectors
                return vectors
        self._missing.add(video_sha256)
        return None

    def get(self, video: Path, video_sha256: str) -> ClipVectors:
        """The vectors for ``video`` (whose hash is ``video_sha256``), embedding it when no valid
        cache file exists. Raises ``ClipSkipped`` when the video cannot be embedded; a second file
        with the same hash gets the same ``ClipSkipped`` without another attempt."""
        done = self._done.get(video_sha256)
        if isinstance(done, ClipSkipped):
            raise done
        found = self.find(video_sha256)
        if found is not None:
            return found
        try:
            vectors = self._embed(video, video_sha256)
        except ClipSkipped as exc:
            exc.sha256 = video_sha256
            self._done[video_sha256] = exc
            raise
        self._done[video_sha256] = vectors
        return vectors

    def _embed(self, video: Path, video_sha256: str) -> ClipVectors:
        s = self.settings
        expected = self.model.representation()
        encoder = self.model.encoder()  # outside the try: a model that cannot load stops the run
        try:
            tensors, header = embed_video(
                video,
                encoder,
                sample_fps=s.fps,
                window=s.window,
                batch_size=s.batch,
                max_seconds=s.max_seconds,
                keep_patches=self.patches,
                log=lambda _: None,
            )
        except CLIP_ERRORS as exc:
            raise _skipped(exc) from exc
        if header["representation"]["hash"] != expected["hash"]:
            # Every clip would disagree in the same way, so this is not one bad file.
            raise RuntimeError(
                "the loaded encoder's representation does not match the one its description gave "
                f"({header['representation']['hash']} against {expected['hash']})"
            )
        if header["source"]["sha256"] != video_sha256:
            raise ClipSkipped("the file changed while it was being indexed")
        out = cache_path(video_sha256, s, self.patches, self.cache_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        latent_file.save(out, tensors, header)  # a cache that cannot be written stops the run
        self.embedded += 1
        return read_clip_vectors(out, expected["hidden_size"])


def _skipped(exc: BaseException) -> ClipSkipped:
    """The skip for an error from embedding one clip, remembered only for a content cause."""
    remember = isinstance(exc, CONTENT_ERRORS) and not isinstance(exc, TRANSIENT_ERRORS)
    if isinstance(exc, av.FFmpegError):
        # PyAV's message repeats the absolute path; the index entry already names the file.
        return ClipSkipped(f"the file cannot be decoded ({exc.strerror or type(exc).__name__})", remember)
    return ClipSkipped(str(exc) or type(exc).__name__, remember)


def _index_dir(out: Path) -> Path:
    """The directory an index file at ``out`` sits in: the link's own directory when ``out`` is a
    symlink, with the directories above it resolved."""
    return Path(os.path.abspath(out)).parent.resolve()


def _root_for(folder: Path, out: Path) -> str:
    """``"."`` when the index sits directly in the folder, otherwise the folder's absolute path."""
    folder = folder.resolve()
    return "." if _index_dir(out) == folder else str(folder)


@dataclass
class _Previous:
    """What the existing index says: its entries by path (empty when it is rebuilt), why it is
    rebuilt (None when it is not, or there was none), and its remembered choices, if readable."""

    clips: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped: dict[str, dict[str, Any]] = field(default_factory=dict)
    rebuilt: str | None = None
    centre: bool | None = None
    recursive: bool | None = None


def _previous(
    out: Path, folder: Path, representation: dict[str, Any], run: dict[str, Any], log: Callable[[str], None]
) -> _Previous:
    if not out.exists():
        return _Previous()
    try:
        header = index_file.load(out).header
    except (index_file.IndexFileError, OSError) as exc:
        prev = _Previous(rebuilt=f"the existing index cannot be read ({exc})")
        try:  # the tensors may be damaged while the choices in the header are still readable
            header = index_file.read_header(out)
        except (index_file.IndexFileError, OSError):
            header = None
    else:
        prev = _Previous()
        old_root = Path(header["root"]) if header["root"] != "." else _index_dir(out)
        old_run = {k: header["run"][k] for k in index_file.RUN_KEYS}
        if header["representation"]["hash"] != representation["hash"]:
            prev.rebuilt = (
                f"the representation changed ({header['representation']['hash']} in the index, "
                f"{representation['hash']} now)"
            )
        elif old_run != run:
            changed = ", ".join(f"{k} {old_run[k]} -> {run[k]}" for k in index_file.RUN_KEYS if old_run[k] != run[k])
            prev.rebuilt = f"the run settings changed ({changed})"
        elif old_root.resolve() != folder.resolve():
            prev.rebuilt = f"the existing index was built from another folder ({old_root})"
        else:
            prev.clips = {e["path"]: e for e in header["clips"]}
            prev.skipped = {e["path"]: e for e in header["skipped"]}
    if header is not None:
        prev.centre = header["embedding"]["similarity"] == "centred-cosine"
        prev.recursive = header["recursive"]
    if prev.rebuilt is not None:
        log(f"rebuilding the index from scratch: {prev.rebuilt}")
    return prev


def build_or_refresh(
    folder: str | Path,
    model: ModelProvider,
    settings: EmbedSettings,
    out: str | Path | None = None,
    recursive: bool | None = None,
    patches: bool = False,
    centre: bool | None = None,
    verify: bool = False,
    log: Callable[[str], None] = print,
    cache_dir: Path | None = None,
) -> Report:
    """Index the videos in ``folder`` and write the index to ``out`` (default
    ``<folder>/jepa-scope.index``); return what was done.

    ``recursive`` includes subfolders, and ``centre`` records ``centred-cosine`` as the index's
    similarity (otherwise the default, plain cosine). Either one left as None keeps what the
    existing index recorded, even when it is rebuilt from scratch, and is False for a new index.
    Changing only ``centre`` rewrites the index without embedding. ``patches`` keeps the
    ``patches`` tensor in newly embedded cache files. ``verify`` hashes every file whatever its
    size and modification time say, and tries remembered skips again. ``cache_dir`` overrides the
    result cache directory.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a folder")
    out = Path(out) if out is not None else folder / INDEX_NAME
    t0 = time.perf_counter()
    representation = model.representation()
    run = settings.run()
    previous = _previous(out, folder, representation, run, log)
    centre = bool(previous.centre) if centre is None else centre
    recursive = bool(previous.recursive) if recursive is None else recursive
    similarity = "centred-cosine" if centre else DEFAULT_SIMILARITY
    log(f"similarity: {similarity}; subfolders: {'included' if recursive else 'not included'}")
    clips = ClipCache(model, settings, patches, cache_dir, log)
    report = Report(path=out, rebuilt=previous.rebuilt, centre=centre, recursive=recursive)

    videos = find_videos(folder, recursive)
    entries: list[dict[str, Any]] = []
    rows: list[ClipVectors] = []
    n_rows = 0
    for i, (rel, path) in enumerate(videos, 1):
        prefix = f"[{i}/{len(videos)}] {rel}"
        if not index_file.is_valid_clip_path(rel):
            reason = "the path cannot be stored portably (it holds a backslash, a NUL or a drive prefix)"
            report.skipped.append({"path": rel, "reason": reason})
            log(f"{prefix}: skipped, {reason}")
            continue
        started = time.perf_counter()
        embedded_before = clips.embedded
        try:
            st = _stat(path)
            was = previous.skipped.get(rel)
            if not verify and was is not None and _still_fails(was, st):
                # A file whose content failed before and has not changed since fails again; trying
                # would load the model for nothing. --verify tries every file again.
                report.skipped.append(was)
                log(f"{prefix}: skipped as before (unchanged), {was['reason']}")
                continue
            vectors, sha, how = _clip(path, st, previous.clips.get(rel), clips, verify, report)
        except ClipSkipped as exc:
            entry: dict[str, Any] = {"path": rel, "reason": str(exc)}
            if exc.remember and exc.sha256 is not None:
                entry.update(sha256=exc.sha256, size=st.st_size, mtime_ns=st.st_mtime_ns, jepa_scope=__version__)
            report.skipped.append(entry)
            log(f"{prefix}: skipped, {exc}")
            continue
        if clips.embedded > embedded_before:
            how = f"embedded in {time.perf_counter() - started:.1f}s"
        start, n_rows = n_rows, n_rows + len(vectors.pooled)
        rows.append(vectors)
        entries.append(
            {
                "path": rel,
                "sha256": sha,
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
                "duration": vectors.duration,
                "width": vectors.width,
                "height": vectors.height,
                "windows": [start, n_rows],
            }
        )
        log(f"{prefix}: {how}, {len(vectors.pooled)} windows")
    report.embedded = clips.embedded

    dim = int(representation["hidden_size"])
    if rows:
        pooled = np.concatenate([r.pooled for r in rows]).astype(np.float32)
        starts = np.concatenate([r.window_start_time for r in rows]).astype(np.float64)
        mean = pooled.mean(axis=0, dtype=np.float64).astype(np.float32)
    else:
        pooled, starts, mean = np.zeros((0, dim), np.float32), np.zeros(0, np.float64), np.zeros(dim, np.float32)
    owner = np.repeat(np.arange(len(rows), dtype=np.int32), [len(r.pooled) for r in rows])
    header = {
        "representation": representation,
        "run": run,
        "embedding": {"kind": "window_vector", "similarity": similarity},
        "root": _root_for(folder, out),
        "recursive": recursive,
        "skipped": report.skipped,
        "clips": entries,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    index_file.save(out, {"pooled": pooled, "window_start_time": starts, "clip": owner, "mean": mean}, header)
    report.clips, report.windows = len(entries), len(pooled)
    elapsed = time.perf_counter() - t0
    if entries:
        log(
            f"indexed {report.clips} clips ({report.windows} windows) into {out} in {elapsed:.1f}s; "
            f"embedded {report.embedded}, skipped {len(report.skipped)}"
        )
    else:
        what = f"all {len(report.skipped)} videos were skipped" if report.skipped else "no videos were found"
        log(f"nothing was indexed ({what}); wrote an empty index to {out}")
    return report


def _clip(
    path: Path,
    st: os.stat_result,
    prev: dict[str, Any] | None,
    clips: ClipCache,
    verify: bool,
    report: Report,
) -> tuple[ClipVectors, str, str]:
    """The vectors, hash and a word on how they were found, for one video."""
    if not verify and prev is not None and _unchanged(prev, st):
        found = clips.find(prev["sha256"])
        if found is not None:
            return found, prev["sha256"], "unchanged"
    sha = _sha256(path)
    report.hashed += 1
    return clips.get(path, sha), sha, "reused the cache"


def _unchanged(entry: dict[str, Any], st: os.stat_result) -> bool:
    """Whether a file's size and modification time are the ones ``entry`` recorded."""
    return entry.get("size") == st.st_size and entry.get("mtime_ns") == st.st_mtime_ns


def _still_fails(skipped: dict[str, Any], st: os.stat_result) -> bool:
    """Whether a skipped entry is a remembered one that still applies: made by this jepa-scope
    version, for a file whose size and modification time have not changed since."""
    return "sha256" in skipped and skipped.get("jepa_scope") == __version__ and _unchanged(skipped, st)


def _stat(path: Path) -> os.stat_result:
    try:
        return path.stat()
    except OSError as exc:
        raise ClipSkipped(f"the file cannot be read ({exc})") from exc


def _sha256(path: Path) -> str:
    try:
        return latent_file.file_sha256(path)
    except OSError as exc:
        raise ClipSkipped(f"the file cannot be read ({exc})") from exc


def find_query(
    video_sha256: str,
    model: ModelProvider,
    settings: EmbedSettings,
    log: Callable[[str], None] = print,
    cache_dir: Path | None = None,
) -> ClipVectors | None:
    """The window vectors of a video from the cache only, or None when it holds none."""
    return ClipCache(model, settings, False, cache_dir, log).find(video_sha256)


def load_query(
    video: Path,
    model: ModelProvider,
    settings: EmbedSettings,
    log: Callable[[str], None] = print,
    cache_dir: Path | None = None,
    video_sha256: str | None = None,
) -> ClipVectors:
    """The window vectors of a video that is not in an index, through the cache (embedding it,
    without patches, only when no valid cache file exists). Raises ``ClipSkipped`` with the reason
    when the video cannot be read or embedded."""
    sha = _sha256(video) if video_sha256 is None else video_sha256
    return ClipCache(model, settings, False, cache_dir, log).get(video, sha)
