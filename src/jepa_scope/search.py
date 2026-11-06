"""Brute-force search over an index: moments like a moment, and clips like a clip.

Every window vector in an index is L2-normalised once, in ``prepare``, and a query is one
matrix-vector product. When the index says ``centred-cosine`` (or the caller asks for it), the
index mean is subtracted from every vector first, so a direction shared by the whole folder does not
squeeze all the scores into a narrow band. Centring is an explicit, recorded choice: the plain
cosine over the stored window vectors is the default.

Two kinds of query share that primitive.

* Moments: the query is one window, and the hits are other windows. Within one clip, a hit that
  overlaps a higher-ranked hit is dropped, so a list of ten hits is ten distinct, non-overlapping
  moments.
* Clips: the query is a clip's vector, the mean of its raw (or centred) window vectors normalised
  after averaging, and the hits are other clips.

A vector whose norm is below ``MIN_NORM`` (after centring when centring is on), or that is not
finite, is "unavailable". It is never a candidate, and a query made from it returns an empty
result that says why. Equal scores are ordered by clip and then by window, so duplicates rank the
same way every time.

Two windows of one clip overlap when their window numbers are closer than the window length allows:
``abs(w1 - w2) * stride < frames``, with ``stride`` and ``frames`` from the index header. This is
decided by window number, never by start times, which are real frame timestamps and jitter.

Scores are computed in float32 to pick candidates and then recomputed in float64 for the
candidates, so rows that are exact duplicates get exactly equal scores and tie in
(clip, window) order whatever their position in the matrix.

Window numbers in this module are clip-local, counted from 0 within a clip. Every moment hit also
carries ``row``, the window's row in the index's tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from .index_file import Index

# The similarity a new index is built with unless the caller asks for centring. It is the choice
# recorded in the execution log of the folder-search plan (T0): plain cosine over the stored window
# means. "centred-cosine" is the other accepted value of the header's ``embedding.similarity``.
DEFAULT_SIMILARITY = "cosine"

MIN_NORM = 1e-6
# The most hits one search may ask for. Rescoring candidates in float64 copies them, so an
# unbounded k (one API request) could allocate hundreds of megabytes.
MAX_K = 1000
_SCORE_SLACK = 1e-6  # float32 products of duplicate rows can differ by an ulp or two; keep both
_CHUNK = 4096  # rows normalised at a time, so float64 working copies stay small

T = TypeVar("T")


@dataclass(frozen=True)
class Hit:
    """One moment: window ``window`` of clip ``clip``, at index-wide row ``row``.

    ``start_time`` is when the window starts in its clip, in seconds; ``score`` is the cosine
    similarity to the query (of centred vectors when the search is centred).
    """

    clip: int
    window: int
    row: int
    start_time: float
    score: float


@dataclass(frozen=True)
class ClipHit:
    """One clip in a "clips like this clip" result, with its cosine similarity to the query."""

    clip: int
    score: float


@dataclass(frozen=True)
class Result(Generic[T]):  # noqa: UP046 (kept as a plain Generic so the module reads simply)
    """What a search returns: ``hits`` best first, and ``reason``.

    ``reason`` is None unless the result is empty, in which case it says why in a sentence (the
    query is unavailable, the index has no windows, or no candidate is left after exclusion). A
    non-empty result with fewer than k hits means the index ran out of candidates.
    """

    hits: list[T]
    reason: str | None = None


@dataclass(frozen=True)
class Query:
    """A normalised query vector, or None with the reason it is unavailable."""

    vector: np.ndarray | None
    reason: str | None = None


@dataclass
class SearchSpace:
    """An index prepared for search. Build it with ``prepare``; the fields are read-only in use.

    ``matrix`` holds the normalised window vectors (float32, ``[N, D]``; unavailable rows are
    zero). ``available`` marks the rows that may be candidates. ``clip``, ``window``,
    ``start_time`` describe each row (``window`` is clip-local). ``clip_start`` and ``clip_stop``
    give each clip's row range. ``clip_vectors`` and ``clip_available`` are the per-clip vectors.
    ``window_seconds`` is the length of one window (used to find a window at a time). ``frames``
    and ``stride`` are the window length and step in frames; two windows of a clip overlap when
    ``abs(w1 - w2) * stride < frames``. ``mean`` is the vector subtracted before normalising, or None
    for plain cosine.
    """

    matrix: np.ndarray
    available: np.ndarray
    clip: np.ndarray
    window: np.ndarray
    start_time: np.ndarray
    clip_start: np.ndarray
    clip_stop: np.ndarray
    clip_vectors: np.ndarray
    clip_available: np.ndarray
    window_seconds: float
    frames: int
    stride: int
    mean: np.ndarray | None

    @property
    def centred(self) -> bool:
        """Whether the vectors were centred on the index mean before normalising."""
        return self.mean is not None

    @property
    def n_clips(self) -> int:
        return len(self.clip_start)

    def overlap(self, w1: int, w2: int) -> bool:
        """Whether clip-local windows ``w1`` and ``w2`` of one clip share frames."""
        return abs(int(w1) - int(w2)) * self.stride < self.frames


def _normalise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalise rows of float64 ``x``; return (float32 rows, available flags).

    A row that is not finite, or whose norm is below ``MIN_NORM``, becomes a zero row and is not
    available.
    """
    finite = np.isfinite(x).all(axis=1)
    safe = np.where(finite[:, None], x, 0.0)
    norms = np.sqrt((safe * safe).sum(axis=1))
    ok = finite & (norms >= MIN_NORM)
    out = np.zeros(x.shape, dtype=np.float32)
    out[ok] = (safe[ok] / norms[ok, None]).astype(np.float32)
    return out, ok


def prepare(index: Index, centre: bool | None = None) -> SearchSpace:
    """Prepare ``index`` for search.

    Vectors are centred (the index ``mean`` is subtracted, in float64) when ``centre`` is True, or
    when ``centre`` is None and the header's ``embedding.similarity`` is ``"centred-cosine"``;
    otherwise they are used as stored. The centring mean is the window-weighted mean of the rows,
    recomputed in float64. Rows are then L2-normalised. Clip vectors are the mean of
    the clip's raw or centred rows, normalised after averaging (never an average of normalised
    rows).
    """
    if centre is None:
        centre = index.header["embedding"]["similarity"] == "centred-cosine"
    pooled = index.pooled
    n, dim = pooled.shape
    # The centring mean is recomputed in float64 from the rows, not read from the file: the stored
    # mean may have been accumulated in float32, and ten identical rows must centre to exactly zero.
    mean = pooled.mean(axis=0, dtype=np.float64) if centre and n else None
    if centre and mean is None:
        mean = np.zeros(dim, dtype=np.float64)
    matrix = np.zeros((n, dim), dtype=np.float32)
    available = np.zeros(n, dtype=bool)
    for lo in range(0, n, _CHUNK):
        block = pooled[lo : lo + _CHUNK].astype(np.float64)
        if mean is not None:
            block -= mean
        matrix[lo : lo + _CHUNK], available[lo : lo + _CHUNK] = _normalise(block)

    clips = index.header["clips"]
    clip_start = np.array([c["windows"][0] for c in clips], dtype=np.int64)
    clip_stop = np.array([c["windows"][1] for c in clips], dtype=np.int64)
    raw_means = np.zeros((len(clips), dim), dtype=np.float64)
    for i in range(len(clips)):
        raw_means[i] = pooled[clip_start[i] : clip_stop[i]].mean(axis=0, dtype=np.float64)
    if mean is not None:
        raw_means -= mean  # the mean of centred rows is the centred mean of the rows
    clip_vectors, clip_available = _normalise(raw_means)

    window_info = index.header["representation"]["window"]
    window_seconds = float(window_info["frames"]) / float(window_info["sample_fps"])
    local = np.arange(n, dtype=np.int64) - clip_start[index.clip] if n else np.zeros(0, dtype=np.int64)
    return SearchSpace(
        matrix=matrix,
        available=available,
        clip=index.clip.astype(np.int64),
        window=local,
        start_time=index.window_start_time,
        clip_start=clip_start,
        clip_stop=clip_stop,
        clip_vectors=clip_vectors,
        clip_available=clip_available,
        window_seconds=window_seconds,
        frames=int(window_info["frames"]),
        stride=int(window_info["stride"]),
        mean=mean,
    )


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be at least 1, not {k}")
    if k > MAX_K:
        raise ValueError(f"k must be at most {MAX_K}, not {k}")


def _check_clip(space: SearchSpace, clip: int) -> None:
    if not 0 <= clip < space.n_clips:
        raise IndexError(f"clip {clip} is out of range for an index of {space.n_clips} clips")


def _row_of(space: SearchSpace, clip: int, window: int) -> int:
    _check_clip(space, clip)
    count = int(space.clip_stop[clip] - space.clip_start[clip])
    if not 0 <= window < count:
        raise IndexError(f"window {window} is out of range for clip {clip}, which has {count} windows")
    return int(space.clip_start[clip]) + window


def nearest_window(start_times: np.ndarray, window_seconds: float, t: float) -> int:
    """The position in ``start_times`` of the window whose centre is nearest to time ``t``.

    A window's centre is its start time plus half of ``window_seconds``. When ``t`` is exactly
    halfway between two centres the earlier window wins, and a time before the first centre or after
    the last one gives the first or last window. ``start_times`` must be non-empty and increasing.
    Raises ``ValueError`` when ``t`` is not finite or there are no windows.
    """
    starts = np.asarray(start_times, dtype=np.float64)
    if len(starts) == 0:
        raise ValueError("there are no windows to choose from")
    if not np.isfinite(t):
        raise ValueError(f"t must be finite, not {t}")
    centres = starts + window_seconds / 2
    return int(np.argmin(np.abs(centres - t)))  # argmin returns the first minimum: the earlier window


def window_at(space: SearchSpace, clip: int, t: float) -> int:
    """The clip-local window of ``clip`` whose centre is nearest to time ``t`` (seconds).

    See ``nearest_window`` for the tie and clamping rules. Raises ``IndexError`` for an unknown
    clip and ``ValueError`` when ``t`` is not finite.
    """
    _check_clip(space, clip)
    return nearest_window(space.start_time[space.clip_start[clip] : space.clip_stop[clip]], space.window_seconds, t)


def _rank(space: SearchSpace, scores: np.ndarray, vector: np.ndarray, eligible: np.ndarray, k: int):
    """The best ``k`` non-overlapping moments among the ``eligible`` rows, as (row, score) pairs.

    ``scores`` (float32) only chooses candidates; the candidates are then rescored in float64 and
    cast to float32, so duplicate rows tie exactly. Rows are taken best first, ties by
    (clip, window), and a row is dropped when it overlaps a kept row of the same clip. The candidate
    set starts at 2k rows and doubles until k rows are kept or every eligible row has been seen, so
    the result equals a full sort.
    """
    n = len(eligible)
    m = 2 * k
    v64 = vector.astype(np.float64)
    while True:
        if m >= n:
            cand = eligible
        else:
            top = scores[eligible]
            threshold = np.partition(top, n - m)[n - m]
            # Everything at or above the cutoff (less a little slack for rounding), so a tie never
            # depends on partition order or on a row's position in the matrix.
            cand = eligible[top >= threshold - _SCORE_SLACK]
        exact = np.clip((space.matrix[cand].astype(np.float64) @ v64).astype(np.float32), -1.0, 1.0)
        order = np.lexsort((space.window[cand], space.clip[cand], -exact))
        kept: list[tuple[int, float]] = []
        kept_windows: dict[int, list[int]] = {}
        for pos in order:
            row = int(cand[pos])
            c = int(space.clip[row])
            w = int(space.window[row])
            if any(space.overlap(w, other) for other in kept_windows.get(c, ())):
                continue
            kept.append((row, float(exact[pos])))
            kept_windows.setdefault(c, []).append(w)
            if len(kept) == k:
                return kept
        if m >= n:
            return kept
        m *= 2


def _moments(
    space: SearchSpace, vector: np.ndarray, k: int, query_clip: int | None, query_row: int | None, same_clip: bool
) -> Result[Hit]:
    if len(space.matrix) == 0:
        return Result([], "the index has no windows")
    scores = space.matrix @ vector
    mask = space.available.copy()
    if query_clip is not None:
        in_clip = space.clip == query_clip
        if not same_clip:
            mask &= ~in_clip
        else:
            assert query_row is not None
            near = np.abs(space.window - space.window[query_row]) * space.stride < space.frames
            mask &= ~(in_clip & near)  # the query window itself is in this set too
    eligible = np.flatnonzero(mask)
    if len(eligible) == 0:
        return Result([], "no other window is available to compare with")
    hits = [
        Hit(int(space.clip[r]), int(space.window[r]), r, float(space.start_time[r]), score)
        for r, score in _rank(space, scores, vector, eligible, k)
    ]
    return Result(hits)


def _unavailable(space: SearchSpace, what: str) -> str:
    """The reason given for an unavailable query, worded for plain or centred search."""
    if space.centred:
        return f"{what} is unavailable: its vector is zero (or not finite) after subtracting the index mean"
    return f"{what} is unavailable: its vector is zero or not finite"


def nearest_moments(space: SearchSpace, clip: int, window: int, k: int, same_clip: bool = False) -> Result[Hit]:
    """The ``k`` windows most like window ``window`` (clip-local) of clip ``clip``.

    The query's own clip is left out entirely unless ``same_clip`` is True; with ``same_clip``,
    the query window and every window of its clip that overlaps it are left out. Within one clip a
    hit that overlaps a higher-ranked hit is dropped. Fewer than ``k`` hits come back only when
    the index has no more candidates. Equal scores are ordered by (clip, window).

    An unavailable query window gives an empty result with a reason. Raises ``ValueError`` for
    ``k < 1`` and ``IndexError`` for an unknown clip or window.
    """
    _check_k(k)
    row = _row_of(space, clip, window)
    if not space.available[row]:
        return Result([], _unavailable(space, "the query window"))
    return _moments(space, space.matrix[row], k, clip, row, same_clip)


def nearest_clips(space: SearchSpace, clip: int, k: int) -> Result[ClipHit]:
    """The ``k`` clips most like clip ``clip``, other clips only, by their clip vectors.

    Clips whose vector is unavailable are never hits; an unavailable query clip gives an empty
    result with a reason. Equal scores are ordered by clip. Raises ``ValueError`` for ``k < 1`` and
    ``IndexError`` for an unknown clip.
    """
    _check_k(k)
    _check_clip(space, clip)
    if not space.clip_available[clip]:
        return Result([], _unavailable(space, "the query clip"))
    return _clips(space, space.clip_vectors[clip], k, clip)


def _clips(space: SearchSpace, vector: np.ndarray, k: int, query_clip: int | None) -> Result[ClipHit]:
    mask = space.clip_available.copy()
    if query_clip is not None:
        mask[query_clip] = False
    eligible = np.flatnonzero(mask)
    if len(eligible) == 0:
        return Result([], "no other clip is available to compare with")
    # Float64 products, cast to float32, so identical clip vectors tie exactly in any position.
    scores = np.clip(
        (space.clip_vectors[eligible].astype(np.float64) @ vector.astype(np.float64)).astype(np.float32), -1.0, 1.0
    )
    order = np.lexsort((eligible, -scores))[:k]
    return Result([ClipHit(int(eligible[i]), float(scores[i])) for i in order])


def query_vector(space: SearchSpace, pooled_rows: np.ndarray) -> Query:
    """A query vector for a clip that is not in the index, from its raw window vectors.

    ``pooled_rows`` is ``[W, D]`` (the clip's window vectors as ``.latent`` stores them). They are
    averaged, centred when the space is, and normalised, the same way ``prepare`` makes a clip
    vector. If the result is unavailable (no rows, not finite, or a zero vector) the returned
    ``Query`` has ``vector`` None and a reason. Raises ``ValueError`` when the width is not the
    index's.
    """
    rows = np.asarray(pooled_rows)
    dim = space.matrix.shape[1]
    if rows.ndim != 2 or rows.shape[1] != dim:
        raise ValueError(f"pooled_rows must have shape [W, {dim}], not {tuple(rows.shape)}")
    if len(rows) == 0:
        return Query(None, "the query has no windows")
    avg = rows.mean(axis=0, dtype=np.float64)
    if space.mean is not None:
        avg = avg - space.mean
    vec, ok = _normalise(avg[None, :])
    if not ok[0]:
        return Query(None, _unavailable(space, "the query vector"))
    return Query(vec[0])


def search_moments(space: SearchSpace, query: Query, k: int) -> Result[Hit]:
    """The ``k`` windows of the index most like an external ``query`` (from ``query_vector``).

    No clip is excluded, since the query is not in the index; overlapping hits within one clip are
    still suppressed. An unavailable query gives an empty result with its reason.
    """
    _check_k(k)
    if query.vector is None:
        return Result([], query.reason)
    return _moments(space, query.vector, k, None, None, True)


def search_clips(space: SearchSpace, query: Query, k: int) -> Result[ClipHit]:
    """The ``k`` clips of the index most like an external ``query`` (from ``query_vector``)."""
    _check_k(k)
    if query.vector is None:
        return Result([], query.reason)
    if space.n_clips == 0:
        return Result([], "the index has no windows")
    return _clips(space, query.vector, k, None)
