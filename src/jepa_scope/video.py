"""Video decoding and windowing.

PyAV ships wheels for macOS on Apple Silicon, Linux and Windows, which avoids the ``decord``
install problems that research repos tend to hit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np


@dataclass
class DecodedVideo:
    frames: np.ndarray  # uint8 [T, H, W, 3], upright, possibly downscaled (see ``short_edge``)
    times: np.ndarray  # float64 [T], seconds from the start of the stream
    width: int  # display width of the source, after rotation, before any downscaling
    height: int
    rotation: int  # degrees applied to make frames upright (0, 90, 180 or 270)
    source_fps: float
    duration: float


def _scaled_size(width: int, height: int, short_edge: int | None) -> tuple[int, int] | None:
    """Size that brings the shorter side down to ``short_edge``, or None to keep the frame as is."""
    short = min(width, height)
    if short_edge is None or short <= short_edge:
        return None
    scale = short_edge / short
    return max(1, round(width * scale)), max(1, round(height * scale))


def decode(
    path: str | Path,
    sample_fps: float,
    max_seconds: float | None = None,
    short_edge: int | None = None,
) -> DecodedVideo:
    """Decode ``path`` and keep frames at roughly ``sample_fps``.

    A frame is kept when its timestamp reaches the next sampling tick, so variable frame rate
    sources are handled without assuming a constant rate. Frames are rotated upright using the
    container's display matrix, which phone footage nearly always carries. When ``short_edge`` is
    set, frames are downscaled (area averaging) so the shorter side is at most that many pixels;
    this keeps memory flat for high-resolution sources since the model resizes to about 292 px
    anyway.

    Raises ``ValueError`` when the source has fewer frames per second than ``sample_fps``, because
    the time per window would silently stretch.
    """
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    step = 1.0 / sample_fps
    frames: list[np.ndarray] = []
    times: list[float] = []
    rotation = 0
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{path} has no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        source_fps = float(stream.average_rate or stream.guessed_rate or 0.0)
        coded_w, coded_h = stream.codec_context.width, stream.codec_context.height
        start = None
        next_tick = 0.0
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            if start is None:
                start = frame.time
                # np.rot90 by rotation // 90 matches ffmpeg's own autorotate (checked for 90, -90, 180).
                rotation = int(frame.rotation or 0) % 360
            t = frame.time - start
            if max_seconds is not None and t > max_seconds:
                break
            if t + 1e-6 < next_tick:
                continue
            size = _scaled_size(frame.width, frame.height, short_edge)
            if size is not None:
                frame = frame.reformat(width=size[0], height=size[1], interpolation="AREA")
            image = frame.to_ndarray(format="rgb24")
            if rotation:
                image = np.ascontiguousarray(np.rot90(image, rotation // 90))
            frames.append(image)
            times.append(t)
            next_tick += step
            # Catch up if the source skipped ahead (dropped frames, VFR gaps).
            while next_tick <= t:
                next_tick += step
    if not frames:
        raise ValueError(f"decoded no frames from {path}")
    if len(times) > 2:
        spacing = float(np.median(np.diff(times)))
        if spacing > 1.25 * step:
            raise ValueError(
                f"{path} only has about {1 / spacing:.1f} frames per second, fewer than the requested "
                f"{sample_fps}; pass --fps {max(1, int(1 / spacing))} or lower"
            )
    width, height = (coded_h, coded_w) if rotation in (90, 270) else (coded_w, coded_h)
    return DecodedVideo(
        frames=np.stack(frames),
        times=np.asarray(times),
        width=width,
        height=height,
        rotation=rotation,
        source_fps=source_fps,
        duration=times[-1] + step,
    )


def window_starts(num_frames: int, window: int, stride: int) -> list[int]:
    """Start indices of full windows of ``window`` frames, ``stride`` frames apart.

    Frames after the last full window are dropped. Returns an empty list when the video is shorter
    than one window.
    """
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    if num_frames < window:
        return []
    return list(range(0, num_frames - window + 1, stride))
