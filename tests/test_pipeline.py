import dataclasses

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from jepa_scope import latent_file, pca
from jepa_scope.embed import embed_video, timeline_owner
from jepa_scope.server import create_app, surprise_to_u8
from jepa_scope.video import decode, window_starts

from .conftest import write_video


def test_decode_samples_requested_fps(moving_square_video):
    v = decode(moving_square_video, sample_fps=8)
    assert v.frames.shape == (24, 64, 96, 3)
    assert np.allclose(np.diff(v.times), 1 / 8, atol=1e-3)


def test_window_starts():
    assert window_starts(20, 8, 4) == [0, 4, 8, 12]
    assert window_starts(7, 8, 4) == []


def test_timeline_owner_covers_every_tubelet_once():
    half, n_windows = 4, 5
    n_tubelets = (n_windows - 1) * half + 2 * half
    owners = [timeline_owner(g, half) for g in range(n_tubelets)]
    assert all(w * half + j == g for g, (w, j) in enumerate(owners))
    assert all(0 <= w < n_windows for w, _ in owners)
    assert all(j >= half for g, (_, j) in enumerate(owners) if g >= half)


def test_prediction_ignores_the_future(tiny_encoder):
    """Two clips with the same past and different futures must get the same prediction.

    The negative control shows the test has teeth: the Hugging Face default path (encode the whole
    clip, then mask) does let the future leak into the prediction.
    """
    torch.manual_seed(1)
    a = torch.randn(1, 8, 3, 32, 32)
    b = a.clone()
    b[:, 4:] = torch.randn(1, 4, 3, 32, 32)

    assert torch.allclose(tiny_encoder.predict(a, 4), tiny_encoder.predict(b, 4))

    model = tiny_encoder.model
    ctx, tgt = [torch.arange(8).unsqueeze(0)], [torch.arange(8, 16).unsqueeze(0)]
    with torch.no_grad():
        leaky_a = model(pixel_values_videos=a, context_mask=ctx, target_mask=tgt).predictor_output.last_hidden_state
        leaky_b = model(pixel_values_videos=b, context_mask=ctx, target_mask=tgt).predictor_output.last_hidden_state
    assert not torch.allclose(leaky_a, leaky_b)


def test_surprise_is_l1_against_layer_normed_targets(tiny_encoder):
    torch.manual_seed(2)
    pixels = torch.randn(2, 8, 3, 32, 32)
    full = tiny_encoder.encode(pixels)
    got = tiny_encoder.surprise(pixels, full, context_frames=4)
    assert got.shape == (2, 8)  # batch of 2, 2 target tubelets x 2 x 2 patches
    target = torch.nn.functional.layer_norm(full[:, 8:], (full.shape[-1],))
    assert torch.allclose(got, (tiny_encoder.predict(pixels, 4) - target).abs().mean(-1), atol=1e-5)
    # Batching must not mix clips: each row equals the clip run on its own.
    single = tiny_encoder.surprise(pixels[1:], full[1:], context_frames=4)
    assert torch.allclose(got[1:], single, atol=1e-5)


def test_bad_context_frames(tiny_encoder):
    pixels = torch.zeros(1, 8, 3, 32, 32)
    full = tiny_encoder.encode(pixels)
    with pytest.raises(ValueError):
        tiny_encoder.surprise(pixels, full, context_frames=3)


@pytest.fixture
def latent(tmp_path, tiny_encoder, moving_square_video):
    tensors, header = embed_video(moving_square_video, tiny_encoder, sample_fps=8, window=8, log=lambda _: None)
    path = tmp_path / "square.latent"
    latent_file.save(path, tensors, header)
    return path


def test_embed_shapes(latent):
    tensors, header = latent_file.load(latent)
    # 24 frames, windows of 8 every 4 -> 5 windows, 12 tubelets on the timeline.
    assert tensors["patches"].shape == (5, 4, 2, 2, 96)
    assert tensors["patches"].dtype == np.float16
    assert tensors["pooled"].shape == (5, 96)
    assert tensors["timeline_pca_rgb"].shape == (12, 2, 2, 3)
    assert tensors["timeline_pca_rgb"].dtype == np.uint8
    s = tensors["timeline_surprise"]
    assert s.shape == (12, 2, 2)
    assert np.isnan(s[:2]).all() and np.isfinite(s[2:]).all()
    assert header["grid"] == {"frames": 8, "crop": 32, "patch": 16, "tubelet": 2}
    assert header["preprocess"]["shortest_edge"] == 36
    assert len(header["preprocess"]["hash"]) == 16
    tp = tensors["timeline_pooled"]
    assert tp.shape == (12, 96) and tp.dtype == np.float32
    # Each timeline tubelet's mean comes from the window that owns it (half = 2 tubelets here).
    for g in range(12):
        w, j = timeline_owner(g, 2)
        expected = tensors["patches"][w, j].astype(np.float32).reshape(-1, 96).mean(0)
        assert np.allclose(tp[g], expected, rtol=0, atol=2e-3), g
    # Window 0 owns all its tubelets, so their means average to its pooled vector.
    assert np.allclose(tp[:4].mean(0), tensors["pooled"][0], atol=1e-5)
    rep = header["representation"]
    assert rep["token_norm"] == "none" and rep["pooling"] == "mean_all"
    assert rep["window"] == {"frames": 8, "stride": 4, "context_frames": 4, "sample_fps": 8.0}
    assert rep["hash"] == latent_file.representation_hash(rep)


def test_latent_file_rejects_plain_safetensors(tmp_path):
    from safetensors.numpy import save_file

    p = tmp_path / "x.safetensors"
    save_file({"a": np.zeros(2, np.float32)}, str(p))
    with pytest.raises(latent_file.LatentFileError):
        latent_file.read_header(p)


def test_pca_is_deterministic_and_spans_range():
    x = np.random.default_rng(0).normal(size=(500, 32)).astype(np.float32)
    m1, c1 = pca.fit(x)
    m2, c2 = pca.fit(x)
    assert np.array_equal(c1, c2)
    rgb = pca.to_rgb(x, m1, c1)
    assert rgb.shape == (500, 3) and rgb.min() == 0 and rgb.max() == 255


def test_surprise_to_u8_handles_nan():
    s = np.array([np.nan, 0.5, 0.6, 0.7], np.float32)
    u8, (lo, hi) = surprise_to_u8(s)
    # The floor maps to 0 and the top to 255, so a narrow band of energies still uses the full range.
    assert u8[0] == 0 and u8[1] == 0 and u8[-1] == 255 and 0 < u8[2] < 255
    assert lo < hi


def test_server_endpoints(latent, moving_square_video):
    client = TestClient(create_app(latent, moving_square_video))
    meta = client.get("/api/meta").json()
    tl = meta["timeline"]
    assert tl["count"] == 12 and tl["grid_h"] == 2
    assert tl["surprise_mean"][0] is None and tl["surprise_mean"][-1] is not None
    assert len(client.get("/api/pca").content) == 12 * 2 * 2 * 3
    assert len(client.get("/api/surprise").content) == 12 * 2 * 2
    r = client.get("/video", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    assert client.get("/").status_code == 200


def test_rotated_video_is_decoded_upright(tmp_path):
    """Phone clips carry a display matrix; frames must come out the way a player shows them."""
    frames = np.zeros((16, 64, 96, 3), np.uint8)
    frames[:, :16, :16] = 255  # a white block in the top-left corner of the stored frame
    rotated = tmp_path / "rotated.mp4"
    write_video(rotated, frames, rotation=90)
    v = decode(rotated, sample_fps=8)
    assert v.rotation == 90
    assert (v.width, v.height) == (64, 96)
    assert v.frames.shape[1:] == (96, 64, 3)
    # ffmpeg's autorotate for this matrix puts the stored top-left corner at the bottom-left.
    assert v.frames[0, -16:, :16].mean() > 200 and v.frames[0, :16, :16].mean() < 50


def test_decode_downscales_to_short_edge(moving_square_video):
    v = decode(moving_square_video, sample_fps=8, short_edge=32)
    assert v.frames.shape[1:] == (32, 48, 3)
    assert (v.width, v.height) == (96, 64)  # header keeps the display size


def test_low_frame_rate_source_is_refused(tmp_path):
    frames = np.zeros((20, 32, 32, 3), np.uint8)
    path = tmp_path / "slow.mp4"
    write_video(path, frames, fps=4)
    with pytest.raises(ValueError, match="frames per second"):
        decode(path, sample_fps=8)
    assert len(decode(path, sample_fps=4).frames) == 20


def test_viewer_refuses_a_different_video(latent, tmp_path):
    other = tmp_path / "other.mp4"
    write_video(other, np.full((24, 64, 96, 3), 80, np.uint8))
    with pytest.raises(ValueError, match="not the video"):
        create_app(latent, other)


def test_load_rejects_inconsistent_shapes(latent, tmp_path):
    tensors, header = latent_file.load(latent)
    tensors["timeline_surprise"] = tensors["timeline_surprise"][:-1]
    bad = tmp_path / "bad.latent"
    latent_file.save(bad, tensors, header)
    with pytest.raises(latent_file.LatentFileError, match="timeline_surprise"):
        latent_file.load(bad)


def test_embed_without_patches(tiny_encoder, moving_square_video):
    tensors, header = embed_video(
        moving_square_video, tiny_encoder, sample_fps=8, window=8, keep_patches=False, log=lambda _: None
    )
    assert "patches" not in tensors
    latent_file.check_tensors(tensors, {**header, "format": latent_file.FORMAT})


def test_cache_key_depends_on_dtype_and_output_version(monkeypatch):
    from jepa_scope import cache

    base = cache.EmbedSettings(model="m", fps=8.0, window=16, device="cpu")
    k1 = cache.cache_path("ab" * 32, base, patches=True)
    k2 = cache.cache_path("ab" * 32, dataclasses.replace(base, dtype="float16"), patches=True)
    monkeypatch.setattr(cache, "OUTPUT_VERSION", 999)
    k3 = cache.cache_path("ab" * 32, base, patches=True)
    assert len({k1, k2, k3}) == 3


@pytest.mark.parametrize(
    "settings, patches, name",
    [
        (dict(model="m", fps=8.0, window=16, device="cpu"), True, "abababababababab-f685d2fd5882.latent"),
        (
            dict(model="facebook/vjepa2-vitl-fpc64-256", fps=8.0, window=16, device="mps"),
            False,
            "abababababababab-fae44bcf9031.latent",
        ),
        (
            dict(model="m", fps=4.0, window=8, max_seconds=12.5, device="cpu", dtype="bfloat16"),
            True,
            "abababababababab-5c2e8661c4a5.latent",
        ),
        # An int fps or window from elsewhere (an index header) must land on the same file.
        (dict(model="m", fps=8, window=16, device="cpu", batch=7), True, "abababababababab-f685d2fd5882.latent"),
    ],
)
def test_cache_key_is_unchanged(settings, patches, name, tmp_path):
    """Cache names recorded before the cache rule moved out of the CLI; existing cache files must
    keep hitting. A change here needs an OUTPUT_VERSION decision, not a new expected value."""
    from jepa_scope import cache

    assert cache.cache_path("ab" * 32, cache.EmbedSettings(**settings), patches).name == name
    assert cache.cache_path("ab" * 32, cache.EmbedSettings(**settings), patches, tmp_path).parent == tmp_path


def test_truncated_cache_file_is_treated_as_missing(latent, tmp_path):
    from jepa_scope import cache

    broken = tmp_path / "broken.latent"
    broken.write_bytes(latent.read_bytes()[:-100])
    with pytest.raises(latent_file.LatentFileError):
        latent_file.load(broken)
    expected = latent_file.read_header(latent)["representation"]["hash"]
    assert cache.cache_is_valid(latent, expected)
    assert not cache.cache_is_valid(broken, expected)
    assert not cache.cache_is_valid(tmp_path / "missing.latent", expected)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda t: t.update(pooled=t["pooled"][:-1]),
        lambda t: t.update(timeline_window=t["timeline_window"] + 50),
        lambda t: t.pop("window_start_time"),
        lambda t: t.update(timeline_pooled=t["timeline_pooled"][:-1]),
        lambda t: t["timeline_pooled"].__setitem__((0, 0), np.nan),
        lambda t: t.update(pooled=t["pooled"].astype(np.float16)),
    ],
    ids=["pooled-rows", "window-out-of-range", "no-window-times", "timeline-pooled-rows", "nan", "dtype"],
)
def test_load_rejects_other_inconsistencies(latent, tmp_path, corrupt):
    tensors, header = latent_file.load(latent)
    corrupt(tensors)
    bad = tmp_path / "bad.latent"
    latent_file.save(bad, tensors, header)
    with pytest.raises(latent_file.LatentFileError):
        latent_file.load(bad)


def _write_raw(path, tensors, header):
    """Write a .latent file bypassing ``save``, which would restamp the format version."""
    import json

    from safetensors.numpy import save_file

    save_file(
        {k: np.ascontiguousarray(v) for k, v in tensors.items()}, str(path), metadata={"jepa_scope": json.dumps(header)}
    )


def test_save_writes_the_current_format_version(latent, tmp_path):
    tensors, header = latent_file.load(latent)
    out = tmp_path / "restamped.latent"
    latent_file.save(out, tensors, {**header, "format_version": 1})
    assert latent_file.read_header(out)["format_version"] == latent_file.FORMAT_VERSION == 2


def test_an_older_format_is_refused_and_re_embedded(latent, tmp_path, capsys):
    from jepa_scope import cache

    tensors, header = latent_file.load(latent)
    old = tmp_path / "v1.latent"
    _write_raw(old, tensors, {**header, "format_version": 1})
    for read in (latent_file.read_header, latent_file.load):
        with pytest.raises(latent_file.LatentFileError, match="re-embed the video"):
            read(old)
    assert not cache.cache_is_valid(old, header["representation"]["hash"])
    assert "re-embedding" in capsys.readouterr().out


def test_cache_is_refused_for_another_output_version_or_representation(latent, capsys, monkeypatch):
    from jepa_scope import cache

    expected = latent_file.read_header(latent)["representation"]["hash"]
    assert not cache.cache_is_valid(latent, "0" * 16)
    assert "different representation" in capsys.readouterr().out
    monkeypatch.setattr(cache, "OUTPUT_VERSION", 999)
    assert not cache.cache_is_valid(latent, expected)
    assert "another output version" in capsys.readouterr().out


def test_an_edited_representation_block_is_refused(latent, tmp_path):
    tensors, header = latent_file.load(latent)
    edited = {**header, "representation": {**header["representation"], "pooling": "mean_predicted_half"}}
    bad = tmp_path / "edited.latent"
    _write_raw(bad, tensors, edited)
    with pytest.raises(latent_file.LatentFileError, match="does not match its hash"):
        latent_file.read_header(bad)


def test_representation_hash_changes_with_every_field():
    base = dict(
        model_id="m",
        weights="w",
        hidden_size=96,
        preprocess_hash="p",
        window=8,
        sample_fps=8.0,
        output_version=2,
    )
    ref = latent_file.representation(**base)["hash"]
    changes = dict(
        model_id="m2",
        weights="w2",
        hidden_size=97,
        preprocess_hash="p2",
        window=16,
        sample_fps=4.0,
        output_version=3,
    )
    for key, value in changes.items():
        assert latent_file.representation(**{**base, key: value})["hash"] != ref, key
    block = latent_file.representation(**base)
    for key in ("token_layer", "token_norm", "pooling"):
        assert latent_file.representation_hash({**block, key: "other"}) != ref, key


def test_describe_matches_the_loaded_encoder(tiny_encoder, tmp_path):
    """The cache check describes the representation without loading weights; it must agree with
    what embedding writes, or every run would re-embed (or reuse vectors from other weights)."""
    from jepa_scope.model import Encoder, resolve_model

    tiny_encoder.model.save_pretrained(tmp_path)
    tiny_encoder.processor.save_pretrained(tmp_path)
    source = resolve_model(str(tmp_path))
    assert source.revision is None and source.model_id == str(tmp_path.resolve())
    loaded = Encoder(
        tiny_encoder.model,
        tiny_encoder.processor,
        str(tmp_path),
        tiny_encoder.device,
        tiny_encoder.dtype,
        source=source,
    )
    assert Encoder.describe(str(tmp_path), None, 8, 8.0) == loaded.representation(8, 8.0)


def test_retraining_into_the_same_folder_changes_the_weights_identity(tiny_encoder, tmp_path):
    import os

    from jepa_scope.model import resolve_model

    tiny_encoder.model.save_pretrained(tmp_path)
    before = resolve_model(str(tmp_path)).weights
    weights = next(tmp_path.glob("*.safetensors"))
    st = weights.stat()
    os.utime(weights, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert resolve_model(str(tmp_path)).weights != before


def _fake_hub(cache, commit, weights_blob, config=b"{}"):
    """A Hugging Face cache entry for repo x/y, laid out as huggingface_hub writes one."""
    repo = cache / "models--x--y"
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "refs" / "main").write_text(commit)
    blobs = repo / "blobs"
    blobs.mkdir(exist_ok=True)
    (blobs / weights_blob).write_bytes(b"weights " + weights_blob.encode())
    (blobs / "cfg").write_bytes(config)
    snap = repo / "snapshots" / commit
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to(f"../../blobs/{weights_blob}")
    (snap / "config.json").symlink_to("../../blobs/cfg")


def test_hub_weights_are_identified_by_content_not_commit(tmp_path):
    from jepa_scope.model import resolve_model

    c1, c2 = "1" * 40, "2" * 40
    _fake_hub(tmp_path, c1, "a" * 64)
    first = resolve_model("x/y", cache_dir=str(tmp_path))
    assert first.revision == c1 and first.model_id == "x/y"
    assert resolve_model("x/y", "main", cache_dir=str(tmp_path)) == first
    assert resolve_model("x/y", c1, cache_dir=str(tmp_path)) == first
    # A new commit with the same weight blob (say, an edited model card) keeps the identity.
    _fake_hub(tmp_path, c2, "a" * 64)
    second = resolve_model("x/y", cache_dir=str(tmp_path))
    assert second.revision == c2 and second.weights == first.weights
    # New weights change it.
    _fake_hub(tmp_path, "3" * 40, "b" * 64)
    assert resolve_model("x/y", cache_dir=str(tmp_path)).weights != first.weights


def test_a_symlinked_local_weight_file_is_identified_by_what_it_points_to(tmp_path):
    from jepa_scope.model import resolve_model

    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "model.safetensors").write_bytes(b"first weights")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").symlink_to(runs / "model.safetensors")
    before = resolve_model(str(model)).weights
    (runs / "model.safetensors").write_bytes(b"retrained weights, longer")
    assert resolve_model(str(model)).weights != before


def test_a_snapshot_without_weight_files_has_unknown_weights(tmp_path):
    from jepa_scope.model import UNKNOWN_WEIGHTS, resolve_model

    _fake_hub(tmp_path, "1" * 40, "a" * 64)
    (tmp_path / "models--x--y" / "snapshots" / ("1" * 40) / "model.safetensors").unlink()
    assert resolve_model("x/y", cache_dir=str(tmp_path)).weights == UNKNOWN_WEIGHTS
