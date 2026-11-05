"""Local server for the viewer.

Large arrays go to the browser as raw bytes (``application/octet-stream``), not JSON, so the
viewer can upload them straight into WebGL textures.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from . import latent_file

STATIC_DIR = Path(__file__).parent / "static"

NOT_BUILT = """<!doctype html><meta charset="utf-8"><title>jepa-scope</title>
<body style="font:15px system-ui;max-width:40rem;margin:4rem auto;line-height:1.5">
<h1>The viewer is not built yet</h1>
<p>The API is running. Build the web viewer once with
<code>cd web &amp;&amp; npm install &amp;&amp; npm run build</code>, then reload this page.</p>
<p>For live frontend work, run <code>npm run dev</code> in <code>web/</code> instead.</p>
</body>"""


def surprise_to_u8(surprise: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Scale surprise to 0..255 between its 1st and 99th percentiles; NaN (no prediction) is 0.

    Energy has a large floor (about 0.55 for V-JEPA 2 ViT-L on natural video), so scaling from zero
    would paint every patch the same colour. Returns the bytes and the ``(lo, hi)`` range used.
    """
    finite = surprise[np.isfinite(surprise)]
    if not finite.size:
        return np.zeros(surprise.shape, np.uint8), (0.0, 1.0)
    lo, hi = (float(v) for v in np.percentile(finite, [1, 99]))
    span = hi - lo if hi > lo else 1.0
    scaled = np.clip((surprise - lo) / span, 0.0, 1.0) * 255.0 + 0.5
    return np.nan_to_num(scaled, nan=0.0).astype(np.uint8), (lo, hi)


def check_video_matches(video: Path, header: dict, video_sha256: str | None = None) -> None:
    """Refuse to pair latents with a different video, which would draw plausible but wrong overlays.

    Pass ``video_sha256`` when the caller has already hashed the file, to avoid hashing it twice.
    """
    source = header["source"]
    size = video.stat().st_size
    if size != source.get("size", size) or (video_sha256 or latent_file.file_sha256(video)) != source["sha256"]:
        raise ValueError(
            f"{video} is not the video these latents were made from "
            f"(expected sha256 {source['sha256'][:16]}..., recorded at {source['path']})"
        )


def create_app(
    latent_path: str | Path, video_path: str | Path | None = None, video_sha256: str | None = None
) -> FastAPI:
    tensors, header = latent_file.load(latent_path)
    video = Path(video_path or header["source"]["path"])
    if not video.exists():
        raise FileNotFoundError(f"source video not found at {video}; pass --video")
    check_video_matches(video, header, video_sha256)

    rgb = np.ascontiguousarray(tensors["timeline_pca_rgb"])
    surprise = tensors["timeline_surprise"]
    surprise_u8, surprise_range = surprise_to_u8(surprise)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows are expected (no prediction)
        means = np.nanmean(surprise.reshape(len(surprise), -1), axis=1)
    per_tubelet = [None if np.isnan(v) else float(v) for v in means]

    meta = {
        "header": header,
        "timeline": {
            "count": int(rgb.shape[0]),
            "grid_h": int(rgb.shape[1]),
            "grid_w": int(rgb.shape[2]),
            "time": tensors["timeline_time"].tolist(),
            "surprise_mean": per_tubelet,
            "surprise_range": list(surprise_range),
        },
    }

    app = FastAPI(title="jepa-scope")

    @app.middleware("http")
    async def no_cache_html(request, call_next):
        # Asset file names carry a content hash, but index.html does not; without this a browser
        # keeps showing an old viewer after an upgrade.
        response = await call_next(request)
        if not request.url.path.startswith(("/assets/", "/video")):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/api/meta")
    def get_meta():
        return meta

    @app.get("/api/pca")
    def get_pca():
        return Response(rgb.tobytes(), media_type="application/octet-stream")

    @app.get("/api/surprise")
    def get_surprise():
        return Response(surprise_u8.tobytes(), media_type="application/octet-stream")

    @app.get("/video")
    def get_video():
        return FileResponse(video)

    if (STATIC_DIR / "index.html").exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    else:

        @app.get("/")
        def not_built():
            return HTMLResponse(NOT_BUILT)

    return app
