"""End-to-end check against the real V-JEPA 2 weights. Opt-in, because it downloads about 1.3 GB.

    JEPA_SCOPE_MODEL_TESTS=1 uv run pytest tests/test_real_model.py -s

The test splices five seconds of an archery clip onto five seconds of a bowling clip (both from the
small Kinetics sample set the Hugging Face V-JEPA 2 docs use) and checks that surprise is highest
for the tubelets right after the cut. This is the property the viewer depends on. An earlier
version of the energy, without the layer norm on targets, ran without errors and drew a flat line.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from jepa_scope.embed import embed_video
from jepa_scope.video import decode

from .conftest import write_video

pytestmark = pytest.mark.skipif(
    os.environ.get("JEPA_SCOPE_MODEL_TESTS") != "1", reason="set JEPA_SCOPE_MODEL_TESTS=1 to run"
)

CLIPS = (
    "val/archery/-Qz25rXdMjE_000014_000024.mp4",
    "val/bowling/--dVV4_CSvw_000033_000043.mp4",
)


def _fit(frames: np.ndarray, w: int = 480, h: int = 270) -> np.ndarray:
    out = []
    for f in frames:
        img = Image.fromarray(f)
        scale = max(w / img.width, h / img.height)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BILINEAR)
        left, top = (img.width - w) // 2, (img.height - h) // 2
        out.append(np.asarray(img.crop((left, top, left + w, top + h))))
    return np.stack(out)


@pytest.fixture(scope="module")
def cut_video(tmp_path_factory):
    from huggingface_hub import hf_hub_download

    fps, seconds = 24, 5
    parts = []
    for name in CLIPS:
        path = hf_hub_download("nateraw/kinetics-mini", name, repo_type="dataset")
        parts.append(_fit(decode(path, sample_fps=fps, max_seconds=seconds).frames[: fps * seconds]))
    out = tmp_path_factory.mktemp("cut") / "cut.mp4"
    write_video(out, np.concatenate(parts), fps=fps)
    return out


def test_surprise_peaks_after_a_hard_cut(cut_video):
    from jepa_scope.model import Encoder

    encoder = Encoder.from_pretrained()
    tensors, header = embed_video(cut_video, encoder, sample_fps=8, window=16, log=print)
    times = tensors["timeline_time"]
    s = tensors["timeline_surprise"]
    has = ~np.isnan(s).all(axis=(1, 2))
    mean = np.where(has, np.nanmean(np.where(np.isnan(s), 0, s), axis=(1, 2)), np.nan)

    # Tubelets in the predicted half right after the cut at 5.0 s: their context is all archery.
    after_cut = (times >= 5.0) & (times < 5.0 + header["window"]["context_frames"] / 8)
    others = has & ~after_cut
    print("after cut:", np.round(mean[after_cut], 3), "max elsewhere:", round(float(np.nanmax(mean[others])), 3))
    assert after_cut.sum() == 4
    assert mean[after_cut].min() > np.nanmax(mean[others])
