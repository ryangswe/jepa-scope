"""Retrieval evaluation for task T0 of the Weeks 3 and 4 plan (exploratory).

The plan, ``docs/plans/2026-11-06-weeks3-4-folder-search-trajectory.md``, fixes the data, the
candidates, the metrics, the controls, the bars and the decision rule before anything is measured.
This module implements them in plain NumPy. ``test_retrieval_candidates`` in
``test_real_model.py`` drives it with the real weights.

Every candidate turns one 16-frame window into one vector:

- (a) ``pooled`` as stored, the float32 mean of all final tokens, with plain cosine;
- (b) the same, centred on the window-weighted mean of the evaluated set's windows;
- (c) the mean of non-affine layer-normalised tokens (each token standardised over its
  dimensions, with no scale or bias), from the float16 ``patches``;
- (d) the mean over the predicted half's tokens only, from ``patches``.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

DATASET = "nateraw/kinetics-mini"
DATASET_REVISION = "9f4ed38128a355c352527209101be3e326471816"
MODEL_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
CLASSES = ("archery", "bowling", "flying_kite", "high_jump", "marching")
CLIPS_PER_CLASS = 10

CANDIDATES = ("a", "b", "c", "d")
SIMPLICITY = ("a", "b", "d", "c")  # how little changes in what is stored and how search works
BAR_RATE = 0.60
BAR_MARGIN = 0.15
TIE_CLIPS = 2
N_PERMUTATIONS = 200

SAMPLE_FPS = 8.0
WINDOW = 16
SEGMENT_SECONDS = 5
# The plan said 24 fps, but one source clip (val/bowling/-WH-lxmGJVY...) is 15 fps. The model
# samples at 8 fps, so the composite is built and written at 8 fps (logged as a deviation).
COMPOSITE_FPS = 8
QUERY_WINDOWS = 4  # windows starting at 0, 1, 2 and 3 s lie fully inside a 5 s segment
TOLERANCE_SECONDS = 1.0


@dataclass
class Clip:
    name: str  # path relative to the snapshot, for example "val/archery/<id>_000014_000024.mp4"
    label: str
    path: Path

    @property
    def source_id(self) -> str:
        """The YouTube id: the file stem without its two time fields."""
        return Path(self.name).stem.rsplit("_", 2)[0]


@dataclass
class Embedded:
    clip: Clip
    vectors: dict[str, np.ndarray]  # candidate -> [W, D] window vectors (b shares a's)
    starts: np.ndarray  # [W] window start times in seconds
    parity: float  # max relative L2 difference per row between pooled and (a) from patches
    width: int
    height: int
    kbps: float


@dataclass
class Report:
    split: str
    data: dict = field(default_factory=dict)

    def put(self, key, value):
        self.data[key] = value
        return value

    def write(self, out_dir: Path) -> Path:
        path = out_dir / f"retrieval-{self.split}.json"
        path.write_text(json.dumps(self.data, indent=2, sort_keys=True))
        return path


# ----------------------------------------------------------------------------------------------
# Data


def manifest(root: Path, split: str) -> list[Clip]:
    """The sorted clip list of ``split`` in a snapshot of the dataset. The label is the folder."""
    files = sorted((root / split).glob("*/*.mp4"))
    return [Clip(name=str(f.relative_to(root)), label=f.parent.name, path=f) for f in files]


def fit_frames(frames: np.ndarray, w: int = 480, h: int = 270) -> np.ndarray:
    """Resize to cover ``w`` x ``h`` and centre-crop, as the cut-clip test does."""
    out = []
    for f in frames:
        img = Image.fromarray(f)
        scale = max(w / img.width, h / img.height)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BILINEAR)
        left, top = (img.width - w) // 2, (img.height - h) // 2
        out.append(np.asarray(img.crop((left, top, left + w, top + h))))
    return np.stack(out)


def even(frame: np.ndarray) -> np.ndarray:
    """Drop one row or column when a side is odd, because libx264 refuses odd frame sizes."""
    h, w = frame.shape[:2]
    return frame[: h - h % 2, : w - w % 2]


# ----------------------------------------------------------------------------------------------
# Embedding through the CLI's cache rule


class CachedEmbedder:
    """Embeds through ``cli._cache_path`` and ``cli._cache_is_valid`` at the CLI defaults, so a rerun
    embeds nothing. The model loads only when a file is missing from the cache."""

    def __init__(self, log=print):
        from jepa_scope.model import DEFAULT_MODEL

        self.args = SimpleNamespace(
            model=DEFAULT_MODEL,
            fps=SAMPLE_FPS,
            window=WINDOW,
            batch=2,
            max_seconds=None,
            device=None,
            dtype=None,
            no_patches=False,
        )
        self._encoder = None
        self._expected = None
        self.embedded = 0
        self.log = log

    @property
    def encoder(self):
        if self._encoder is None:
            from jepa_scope.model import Encoder

            self._encoder = Encoder.from_pretrained(self.args.model)
        return self._encoder

    def shortest_edge(self) -> int:
        from transformers import AutoVideoProcessor

        processor = AutoVideoProcessor.from_pretrained(self.args.model)
        return int(processor.size["shortest_edge"])

    def load(self, path: Path) -> tuple[dict[str, np.ndarray], dict]:
        from jepa_scope import cli, latent_file
        from jepa_scope.embed import embed_video

        out = cli._cache_path(latent_file.file_sha256(path), self.args)
        if self._expected is None:
            self._expected = cli._expected_representation(self.args)
        if not cli._cache_is_valid(out, self._expected):
            self.log(f"embedding {path.name}")
            tensors, header = embed_video(
                path,
                self.encoder,
                sample_fps=self.args.fps,
                window=self.args.window,
                batch_size=self.args.batch,
                keep_patches=True,
                log=lambda m: None,
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            latent_file.save(out, tensors, header)
            self.embedded += 1
        return latent_file.load(out)


def window_vectors(tensors: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
    """Candidate vectors ``[W, D]`` for one clip, and the parity of (a) recomputed from patches."""
    pooled = tensors["pooled"].astype(np.float64)
    patches = tensors["patches"]  # float16 [W, T, H, W, D]
    n, t = patches.shape[:2]
    dim = patches.shape[-1]
    tokens = patches.astype(np.float32).reshape(n, -1, dim)
    a_from_patches = tokens.mean(axis=1, dtype=np.float64)
    parity = float((np.linalg.norm(a_from_patches - pooled, axis=1) / np.linalg.norm(pooled, axis=1)).max())
    mu = tokens.mean(axis=-1, keepdims=True)
    var = tokens.var(axis=-1, keepdims=True)
    c = ((tokens - mu) / np.sqrt(var + 1e-5)).mean(axis=1, dtype=np.float64)
    d = patches[:, t // 2 :].astype(np.float32).reshape(n, -1, dim).mean(axis=1, dtype=np.float64)
    return {"a": pooled, "b": pooled, "c": c, "d": d}, parity


def embed_clip(embedder: CachedEmbedder, clip: Clip) -> Embedded:
    tensors, header = embedder.load(clip.path)
    vectors, parity = window_vectors(tensors)
    src = header["source"]
    return Embedded(
        clip=clip,
        vectors=vectors,
        starts=tensors["window_start_time"].astype(np.float64),
        parity=parity,
        width=int(src["width"]),
        height=int(src["height"]),
        kbps=float(src["size"]) * 8 / 1000 / max(float(src["duration"]), 1e-9),
    )


# ----------------------------------------------------------------------------------------------
# Metrics. One function scores every candidate, baseline and control.


def normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.all(np.isfinite(x)) or np.any(norms < 1e-12):
        raise ValueError("a vector is zero or not finite; it cannot be compared by cosine")
    return x / norms


def set_centre(windows: list[np.ndarray]) -> np.ndarray:
    """The window-weighted mean of every window in the set."""
    return np.concatenate(windows).mean(axis=0)


def clip_matrix(windows: list[np.ndarray], centre: np.ndarray | None) -> np.ndarray:
    """One normalised vector per clip: the mean of its raw (or centred) rows, normalised after."""
    rows = [(w - centre if centre is not None else w).mean(axis=0) for w in windows]
    return normalise(np.stack(rows))


def top1(vectors: np.ndarray, groups: list[str]) -> np.ndarray:
    """Index of each row's most similar other row, never one from the same source group."""
    sims = vectors @ vectors.T
    same = np.asarray(groups)[:, None] == np.asarray(groups)[None, :]
    sims[same] = -np.inf
    return sims.argmax(axis=1)


def clip_hits(vectors: np.ndarray, labels: list[str], groups: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-clip-out top-1: whether each clip's nearest other clip has its label."""
    best = top1(vectors, groups)
    lab = np.asarray(labels)
    return lab[best] == lab, best


def window_rate(windows: list[np.ndarray], centre: np.ndarray | None, labels: list[str], groups: list[str]) -> float:
    """Window-level top-1 same-class rate, excluding every window of the query's own clip."""
    rows = np.concatenate([w - centre if centre is not None else w for w in windows])
    owner = np.concatenate([[i] * len(w) for i, w in enumerate(windows)])
    vec = normalise(rows)
    sims = vec @ vec.T
    grp = np.asarray(groups)[owner]
    sims[grp[:, None] == grp[None, :]] = -np.inf
    lab = np.asarray(labels)[owner]
    return float((lab[sims.argmax(axis=1)] == lab).mean())


def permutation_null(vectors: np.ndarray, labels: list[str], groups: list[str], n: int, seed: int = 0) -> np.ndarray:
    """Clip-level rates under random relabellings, each clip keeping all its windows."""
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        out[i] = clip_hits(vectors, list(rng.permutation(labels)), groups)[0].mean()
    return out


def chance(labels: list[str]) -> float:
    n = len(labels)
    counts = {c: labels.count(c) for c in set(labels)}
    return float(np.mean([(counts[c] - 1) / (n - 1) for c in labels]))


def paired(hits: dict[str, np.ndarray]) -> dict[str, dict[str, int]]:
    """For every pair of candidates, how many clips both, only one, or neither retrieved."""
    out = {}
    for x, y in itertools.combinations(sorted(hits), 2):
        hx, hy = hits[x], hits[y]
        out[f"{x}-{y}"] = {
            "both": int((hx & hy).sum()),
            f"only_{x}": int((hx & ~hy).sum()),
            f"only_{y}": int((~hx & hy).sum()),
            "neither": int((~hx & ~hy).sum()),
        }
    return out


# ----------------------------------------------------------------------------------------------
# Colour baselines


def colour_histogram(frames: np.ndarray, crop: int = 256) -> np.ndarray:
    """Joint RGB histogram, 8 x 8 x 8 = 512 bins, of the centre ``crop`` square of each frame,
    summed over frames and normalised to sum to one."""
    h, w = frames.shape[1:3]
    top, left = max((h - crop) // 2, 0), max((w - crop) // 2, 0)
    q = frames[:, top : top + crop, left : left + crop].astype(np.int64) // 32
    idx = (q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]).ravel()
    hist = np.bincount(idx, minlength=512).astype(np.float64)
    return hist / hist.sum()


# ----------------------------------------------------------------------------------------------
# Temporal localisation


def transition(start: float, boundaries: list[float], span: float) -> bool:
    """Whether a window starting at ``start`` and lasting ``span`` seconds crosses a boundary."""
    return any(start < b - 1e-6 and b + 1e-6 < start + span for b in boundaries)


def localise(
    queries: list[tuple[int, float, np.ndarray, int]],
    composite: np.ndarray,
    composite_starts: np.ndarray,
    others: list[np.ndarray],
    centre: np.ndarray | None,
    window_seconds: float,
) -> list[dict]:
    """Run the localisation queries for one candidate.

    ``queries`` holds ``(segment index i, query start t, raw vector, index of the query's own clip
    in others)``. A query passes when its top hit is a composite window starting within
    ``TOLERANCE_SECONDS`` of ``SEGMENT_SECONDS * i + t``. Transition windows are not candidates.
    """
    boundaries = [SEGMENT_SECONDS * k for k in range(1, len(composite_starts) + 1)]
    keep = np.asarray([not transition(s, boundaries, window_seconds) for s in composite_starts])
    comp = composite[keep]
    comp_starts = composite_starts[keep]
    rows = [comp] + others
    owner = np.concatenate([[-1] * len(comp)] + [[j] * len(o) for j, o in enumerate(others)])
    mat = np.concatenate(rows)
    mat = normalise(mat - centre if centre is not None else mat)
    results = []
    for i, t, vec, own in queries:
        q = normalise((vec - centre if centre is not None else vec)[None])[0]
        sims = mat @ q
        sims[owner == own] = -np.inf
        best = int(sims.argmax())
        target = SEGMENT_SECONDS * i + t
        hit_start = float(comp_starts[best]) if best < len(comp) else None
        ok = hit_start is not None and abs(hit_start - target) <= TOLERANCE_SECONDS + 1e-6
        results.append({"segment": i, "query_start": t, "target": target, "hit_composite_start": hit_start, "pass": ok})
    return results


def clip_average(windows: np.ndarray) -> np.ndarray:
    """The mutation: every window replaced by its clip's mean."""
    return np.repeat(windows.mean(axis=0, keepdims=True), len(windows), axis=0)


# ----------------------------------------------------------------------------------------------
# The decision rule


def at_least(count: int, n: int, fraction: float) -> bool:
    """``count / n >= fraction`` in integer arithmetic (fraction given to two decimals)."""
    return 100 * count >= round(100 * fraction) * n


def choose(
    counts: dict[str, int], n: int, baseline_count: int, localised: dict[str, bool]
) -> tuple[str | None, list[str]]:
    """Eligible: passes localisation, top-1 at least 0.60, and at least 0.15 above the better colour
    baseline. Among the eligible, the most correctly retrieved clips wins; a candidate within
    ``TIE_CLIPS`` clips of that count ties, and a tie goes to the simplest."""
    eligible = [
        c
        for c in CANDIDATES
        if localised[c] and at_least(counts[c], n, BAR_RATE) and at_least(counts[c] - baseline_count, n, BAR_MARGIN)
    ]
    if not eligible:
        return None, eligible
    best = max(counts[c] for c in eligible)
    tied = [c for c in eligible if best - counts[c] <= TIE_CLIPS]
    return next(c for c in SIMPLICITY if c in tied), eligible
