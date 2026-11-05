"""Video in, ``.latent`` file out.

Windows of ``window`` frames overlap by half. The first half of each window is the context and the
second half is what the predictor has to predict. With that overlap every tubelet after the first
half-window is predicted exactly once, which gives one continuous surprise timeline.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import OUTPUT_VERSION, __version__, pca
from .grid import to_grid
from .latent_file import file_sha256
from .model import Encoder
from .video import decode, window_starts


def timeline_owner(g: int, half: int) -> tuple[int, int]:
    """Which window and local tubelet index represent global tubelet ``g``.

    ``half`` is the number of tubelets in half a window. Tubelets in the first half of the first
    window come from window 0's context. Every later tubelet comes from the one window whose second
    (predicted) half contains it.
    """
    if g < half:
        return 0, g
    w = g // half - 1
    return w, g - w * half


def embed_video(
    video_path: str | Path,
    encoder: Encoder,
    *,
    sample_fps: float = 8.0,
    window: int = 16,
    batch_size: int = 2,
    max_seconds: float | None = None,
    keep_patches: bool = True,
    log: Callable[[str], None] = print,
) -> tuple[dict[str, np.ndarray], dict]:
    """Embed a video. ``keep_patches=False`` drops the full ``patches`` tensor (about 4 MB per
    window at the defaults), which the viewer does not need."""
    if window % (2 * encoder.tubelet):
        raise ValueError(f"window ({window}) must be a multiple of 2 x tubelet ({2 * encoder.tubelet})")
    t0 = time.perf_counter()
    video = decode(
        video_path,
        sample_fps=sample_fps,
        max_seconds=max_seconds,
        short_edge=encoder.preprocess_info()["shortest_edge"],
    )
    stride = window // 2
    starts = window_starts(len(video.frames), window, stride)
    if not starts:
        raise ValueError(
            f"the video gives {len(video.frames)} frames at {sample_fps} fps, but one window needs {window}; "
            f"use a clip longer than {window / sample_fps:.1f}s or lower --window"
        )
    spec = encoder.grid(window)
    half = spec.t // 2
    log(
        f"decoded {len(video.frames)} frames at {sample_fps} fps ({video.width}x{video.height}); "
        f"{len(starts)} windows of {window} frames on {encoder.device} ({encoder.dtype})"
    )

    patches, pooled, surprise = [], [], []
    for i in range(0, len(starts), batch_size):
        batch = starts[i : i + batch_size]
        pixels = encoder.preprocess([video.frames[s : s + window] for s in batch])
        tokens = encoder.encode(pixels)
        err = encoder.surprise(pixels, tokens, context_frames=window // 2)
        patches.append(to_grid(tokens.numpy(), spec).astype(np.float16))
        pooled.append(tokens.mean(dim=1).numpy())
        surprise.append(err.numpy().reshape(len(batch), spec.t - half, spec.h, spec.w))
        log(f"  windows {i + len(batch)}/{len(starts)}")
    patches_a = np.concatenate(patches)  # [W, T, H, W, D]
    pooled_a = np.concatenate(pooled)  # [W, D]
    surprise_a = np.concatenate(surprise)  # [W, T/2, H, W]
    if not np.isfinite(patches_a).all() or not np.isfinite(surprise_a).all():
        raise RuntimeError("the model produced NaN or inf values; try --dtype float32")

    # Flatten the overlapping windows into one tubelet timeline.
    n_tubelets = (starts[-1] + window) // encoder.tubelet
    owners = [timeline_owner(g, half) for g in range(n_tubelets)]
    tl_feats = np.stack([patches_a[w, j] for w, j in owners])  # [G, H, W, D]
    tl_surprise = np.stack(
        [surprise_a[w, j - half] if j >= half else np.full((spec.h, spec.w), np.nan, np.float32) for w, j in owners]
    ).astype(np.float32)
    tl_time = video.times[np.arange(n_tubelets) * encoder.tubelet].astype(np.float64)

    mean, comps = pca.fit(tl_feats)
    tl_rgb = pca.to_rgb(tl_feats, mean, comps)

    tensors = {
        "pooled": pooled_a.astype(np.float32),
        "window_start_time": video.times[starts].astype(np.float64),
        "timeline_time": tl_time,
        "timeline_window": np.asarray([w for w, _ in owners], dtype=np.int32),
        "timeline_pca_rgb": tl_rgb,
        "timeline_surprise": tl_surprise,
    }
    if keep_patches:
        tensors["patches"] = patches_a
    geom = encoder.crop_geometry(video.width, video.height)
    header = {
        "model_id": encoder.model_id,
        "hidden_size": encoder.hidden_size,
        "grid": spec.to_dict(),
        "window": {
            "frames": window,
            "stride": stride,
            "context_frames": window // 2,
            "sample_fps": sample_fps,
            "tubelet_seconds": encoder.tubelet / sample_fps,
        },
        "preprocess": encoder.preprocess_info(),
        "crop": {"x": geom.x, "y": geom.y, "w": geom.w, "h": geom.h},
        "source": {
            "path": str(Path(video_path).resolve()),
            "sha256": file_sha256(video_path),
            "width": video.width,
            "height": video.height,
            "rotation": video.rotation,
            "size": Path(video_path).stat().st_size,
            "source_fps": video.source_fps,
            "duration": video.duration,
            "max_seconds": max_seconds,
        },
        "run": {
            "jepa_scope": __version__,
            "output_version": OUTPUT_VERSION,
            "device": str(encoder.device),
            "dtype": str(encoder.dtype).removeprefix("torch."),
            "seconds": round(time.perf_counter() - t0, 2),
        },
    }
    log(f"done in {header['run']['seconds']}s")
    return tensors, header
