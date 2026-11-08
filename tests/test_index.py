"""The index file contract: round trips, rejection of malformed files, and root resolution.

Everything here uses small synthetic arrays; no model is involved.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from jepa_scope import index_file, latent_file
from jepa_scope.index_file import IndexFileError

DIM = 8
WINDOWS_PER_CLIP = (2, 3, 1)


def make_representation(hidden_size: int = DIM) -> dict:
    return latent_file.representation(
        model_id="test/model",
        weights="sha256:abc",
        hidden_size=hidden_size,
        preprocess_hash="pre123",
        window=16,
        sample_fps=8.0,
        output_version=2,
    )


def make_index(counts=WINDOWS_PER_CLIP, dim: int = DIM, similarity: str = "cosine"):
    """A valid synthetic index with one clip per entry of ``counts``, as (tensors, header)."""
    rng = np.random.default_rng(0)
    rows = sum(counts)
    clip_ids = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    times = np.concatenate([np.arange(c, dtype=np.float64) * 1.0 for c in counts]) if counts else np.zeros(0)
    pooled = rng.normal(size=(rows, dim)).astype(np.float32)
    mean = pooled.mean(axis=0) if rows else np.zeros(dim, dtype=np.float32)
    tensors = {
        "pooled": pooled,
        "window_start_time": times.astype(np.float64),
        "clip": clip_ids,
        "mean": mean.astype(np.float32),
    }
    clips = []
    start = 0
    for i, count in enumerate(counts):
        clips.append(
            {
                "path": f"sub/clip{i}.mp4",
                "sha256": f"{i:064x}",
                "size": 1000 + i,
                "mtime_ns": 1_700_000_000_000_000_000 + i,
                "duration": 4.0 + i,
                "width": 320,
                "height": 240,
                "windows": [start, start + count],
            }
        )
        start += count
    header = {
        "format": index_file.INDEX_FORMAT,
        "format_version": index_file.INDEX_FORMAT_VERSION,
        "representation": make_representation(dim),
        "run": {"max_seconds": 30.0, "device": "cpu", "dtype": "float32"},
        "embedding": {"kind": "window_vector", "similarity": similarity},
        "root": ".",
        "recursive": True,
        "skipped": [{"path": "bad.mp4", "reason": "could not decode"}],
        "clips": clips,
    }
    return tensors, header


def test_round_trip(tmp_path: Path):
    tensors, header = make_index()
    path = tmp_path / "jepa-scope.index"
    index_file.save(path, tensors, header)
    assert not (tmp_path / "jepa-scope.index.tmp").exists()

    index = index_file.load(path)
    assert index.header == header
    for key, value in tensors.items():
        assert getattr(index, key).dtype == value.dtype
        np.testing.assert_array_equal(getattr(index, key), value)
    assert index_file.read_header(path) == header


def test_empty_index_round_trips(tmp_path: Path):
    tensors, header = make_index(counts=())
    assert tensors["pooled"].shape == (0, DIM)
    assert not tensors["mean"].any()
    path = tmp_path / "empty.index"
    index_file.save(path, tensors, header)

    index = index_file.load(path)
    assert index.pooled.shape == (0, DIM)
    assert index.window_start_time.shape == (0,)
    assert index.clip.shape == (0,)
    assert index.mean.shape == (DIM,)
    assert index.header["clips"] == []


def test_save_stamps_current_format_and_version(tmp_path: Path):
    tensors, header = make_index()
    header["format_version"] = 0
    header["format"] = "something-else"
    path = tmp_path / "x.index"
    index_file.save(path, tensors, header)
    loaded = index_file.read_header(path)
    assert loaded["format"] == index_file.INDEX_FORMAT
    assert loaded["format_version"] == index_file.INDEX_FORMAT_VERSION


def test_clip_rows_and_clip_vector():
    tensors, header = make_index()
    index = index_file.Index(header=header, **tensors)
    assert index.clip_rows(0) == slice(0, 2)
    assert index.clip_rows(1) == slice(2, 5)
    assert index.clip_rows(2) == slice(5, 6)
    expected = tensors["pooled"][2:5].mean(axis=0)
    got = index.clip_vector(1)
    assert got.dtype == np.float32
    assert got.shape == (DIM,)
    np.testing.assert_allclose(got, expected, rtol=1e-6)
    # A one-window clip's vector is that row, unchanged and not normalised.
    np.testing.assert_array_equal(index.clip_vector(2), tensors["pooled"][5])


def test_resolve_root(tmp_path: Path):
    tensors, header = make_index()
    index = index_file.Index(header=header, **tensors)
    index_path = tmp_path / "folder" / "jepa-scope.index"
    (tmp_path / "folder").mkdir()

    # "." means the directory of the index file itself.
    assert index.resolve_root(index_path) == (tmp_path / "folder").resolve()

    # An absolute recorded root is used as it is, wherever the index file is.
    absolute = copy.deepcopy(header)
    absolute["root"] = "/data/videos"
    index_abs = index_file.Index(header=absolute, **tensors)
    assert index_abs.resolve_root(index_path) == Path("/data/videos")

    # source_root overrides either kind of root.
    override = tmp_path / "moved"
    assert index.resolve_root(index_path, source_root=override) == override
    assert index_abs.resolve_root(index_path, source_root=override) == override


def with_ranges(ranges):
    tensors, header = make_index()
    for entry, window in zip(header["clips"], ranges, strict=True):
        entry["windows"] = list(window)
    return tensors, header


@pytest.mark.parametrize(
    ("ranges", "message"),
    [
        pytest.param([(0, 2), (1, 5), (5, 6)], r"clips\[1\].windows starts at 1, expected 2", id="overlap"),
        pytest.param([(0, 2), (3, 5), (5, 6)], r"clips\[1\].windows starts at 3, expected 2", id="gap"),
        pytest.param([(0, 2), (2, 2), (2, 6)], r"clips\[1\].windows \[2, 2\) is empty", id="empty-range"),
        pytest.param([(0, 2), (2, 5), (5, 5)], r"clips\[2\].windows \[5, 5\) is empty", id="empty-last-range"),
        pytest.param([(1, 2), (2, 5), (5, 6)], r"clips\[0\].windows starts at 1, expected 0", id="not-from-zero"),
        pytest.param([(0, 2), (2, 5), (5, 7)], r"cover 7 rows but pooled has 6", id="runs-past-the-rows"),
        pytest.param([(0, 2), (2, 4), (4, 5)], r"clip does not say 2 for every row of clips\[2\]", id="wrong-owner"),
    ],
)
def test_rejects_bad_window_ranges(ranges, message):
    tensors, header = with_ranges(ranges)
    with pytest.raises(IndexFileError, match=message):
        index_file.check(tensors, header)


def test_rejects_ranges_that_stop_before_the_last_row():
    tensors, header = make_index()
    del header["clips"][2]
    tensors["clip"][5] = 1  # keep every remaining row owned by a listed clip
    with pytest.raises(IndexFileError, match="cover 5 rows but pooled has 6"):
        index_file.check(tensors, header)


def test_rejects_an_empty_clip_between_clips():
    # The rows and the clip tensor are consistent; only the empty range in the header is wrong.
    tensors, header = make_index(counts=(2, 4))
    tensors["clip"][2:] = 2
    empty = copy.deepcopy(header["clips"][1])
    empty["path"] = "sub/empty.mp4"
    empty["windows"] = [2, 2]
    header["clips"].insert(1, empty)
    header["clips"][2]["windows"] = [2, 6]
    with pytest.raises(IndexFileError, match="is empty"):
        index_file.check(tensors, header)


def test_rejects_clip_disagreeing_with_ranges():
    tensors, header = make_index()
    tensors["clip"][1] = 2
    with pytest.raises(IndexFileError, match="clip does not say"):
        index_file.check(tensors, header)


@pytest.mark.parametrize("bad_times", [[0.0, 1.0, 1.0], [0.0, 2.0, 1.0]], ids=["equal", "decreasing"])
def test_rejects_non_increasing_times_within_a_clip(bad_times):
    tensors, header = make_index()
    tensors["window_start_time"][2:5] = bad_times
    with pytest.raises(IndexFileError, match="strictly increase"):
        index_file.check(tensors, header)


def test_times_may_restart_between_clips():
    tensors, header = make_index()
    # The default fixture already restarts at 0 for every clip; this states the intent.
    assert tensors["window_start_time"][2] == 0.0
    index_file.check(tensors, header)


def test_rejects_wrong_hidden_size():
    tensors, header = make_index()
    header["representation"] = make_representation(hidden_size=DIM + 1)
    with pytest.raises(IndexFileError, match="hidden_size"):
        index_file.check(tensors, header)


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("pooled", np.float64),
        ("pooled", np.float16),
        ("window_start_time", np.float32),
        ("clip", np.int64),
        ("clip", np.uint8),
        ("mean", np.float64),
    ],
)
def test_rejects_wrong_dtype(key, bad):
    tensors, header = make_index()
    tensors[key] = tensors[key].astype(bad)
    with pytest.raises(IndexFileError, match="must be"):
        index_file.check(tensors, header)


@pytest.mark.parametrize("key", ["pooled", "window_start_time", "mean"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
def test_rejects_non_finite_values(key, bad):
    tensors, header = make_index()
    tensors[key][0 if key != "pooled" else (0, 0)] = bad
    with pytest.raises(IndexFileError, match="NaN or infinite"):
        index_file.check(tensors, header)


def test_rejects_missing_tensor():
    tensors, header = make_index()
    del tensors["mean"]
    with pytest.raises(IndexFileError, match="missing tensors"):
        index_file.check(tensors, header)


@pytest.mark.parametrize("key", ["window_start_time", "clip", "mean"])
def test_rejects_wrong_tensor_shape(key):
    tensors, header = make_index()
    tensors[key] = tensors[key][:-1]
    with pytest.raises(IndexFileError, match="shape"):
        index_file.check(tensors, header)


def test_rejects_non_matrix_pooled():
    tensors, header = make_index()
    tensors["pooled"] = tensors["pooled"].reshape(-1)
    with pytest.raises(IndexFileError, match="2-dimensional"):
        index_file.check(tensors, header)


def test_rejects_a_plain_safetensors_file(tmp_path: Path):
    path = tmp_path / "plain.safetensors"
    save_file({"x": np.zeros(3, dtype=np.float32)}, str(path))
    with pytest.raises(IndexFileError, match="no 'jepa_scope' header"):
        index_file.load(path)
    with pytest.raises(IndexFileError, match="no 'jepa_scope' header"):
        index_file.read_header(path)


def test_rejects_a_latent_file_header(tmp_path: Path):
    tensors, header = make_index()
    header_with_latent_format = {**header, "format": latent_file.FORMAT}
    # save() stamps the right format, so write the wrong one directly.
    path = tmp_path / "wrong.index"
    save_file(tensors, str(path), metadata={"jepa_scope": json.dumps(header_with_latent_format)})
    with pytest.raises(IndexFileError, match="not a jepa-scope.index file"):
        index_file.load(path)


def test_rejects_other_format_version(tmp_path: Path):
    tensors, header = make_index()
    header["format_version"] = index_file.INDEX_FORMAT_VERSION + 1
    with pytest.raises(IndexFileError, match="format_version"):
        index_file.check(tensors, header)


@pytest.mark.parametrize("key", index_file.REQUIRED_HEADER_KEYS)
def test_rejects_missing_header_key(key):
    tensors, header = make_index()
    del header[key]
    with pytest.raises(IndexFileError):
        index_file.check(tensors, header)


def test_rejects_a_header_without_recursive():
    """Named on its own, since the parametrised test above follows REQUIRED_HEADER_KEYS."""
    tensors, header = make_index()
    del header["recursive"]
    with pytest.raises(IndexFileError, match="missing"):
        index_file.check(tensors, header)


def test_rejects_representation_with_wrong_hash():
    tensors, header = make_index()
    header["representation"]["pooling"] = "max"
    with pytest.raises(IndexFileError, match="representation block"):
        index_file.check(tensors, header)


@pytest.mark.parametrize("similarity", ["dot", "euclidean", None])
def test_rejects_unknown_similarity(similarity):
    tensors, header = make_index()
    header["embedding"]["similarity"] = similarity
    with pytest.raises(IndexFileError, match="similarity"):
        index_file.check(tensors, header)


def test_accepts_centred_cosine():
    tensors, header = make_index(similarity="centred-cosine")
    index_file.check(tensors, header)


def test_rejects_unknown_embedding_kind():
    tensors, header = make_index()
    header["embedding"]["kind"] = "tubelet_vector"
    with pytest.raises(IndexFileError, match="embedding.kind"):
        index_file.check(tensors, header)


def test_rejects_run_without_settings():
    tensors, header = make_index()
    del header["run"]["dtype"]
    with pytest.raises(IndexFileError, match="run must be"):
        index_file.check(tensors, header)


@pytest.mark.parametrize("missing", index_file.CLIP_KEYS)
def test_rejects_clip_entry_missing_a_field(missing):
    tensors, header = make_index()
    del header["clips"][0][missing]
    with pytest.raises(IndexFileError, match="missing"):
        index_file.check(tensors, header)


@pytest.mark.parametrize("path", ["/abs/clip.mp4", "a\\b.mp4", "../up.mp4", "a/../b.mp4", ""])
def test_rejects_unportable_clip_paths(path):
    tensors, header = make_index()
    header["clips"][0]["path"] = path
    with pytest.raises(IndexFileError, match="path"):
        index_file.check(tensors, header)


def test_rejects_non_integer_clip_size():
    tensors, header = make_index()
    header["clips"][0]["size"] = 1.5
    with pytest.raises(IndexFileError, match="size"):
        index_file.check(tensors, header)


def test_save_refuses_an_invalid_index(tmp_path: Path):
    tensors, header = make_index()
    tensors["pooled"][0, 0] = np.nan
    path = tmp_path / "bad.index"
    with pytest.raises(IndexFileError):
        index_file.save(path, tensors, header)
    assert not path.exists()
    assert not (tmp_path / "bad.index.tmp").exists()


def test_rejects_a_mean_that_is_not_the_mean_of_pooled():
    tensors, header = make_index()
    tensors["mean"] = tensors["mean"] + 100
    with pytest.raises(IndexFileError, match="mean is not the mean"):
        index_file.check(tensors, header)


def test_rejects_a_nonzero_mean_for_an_empty_index():
    tensors, header = make_index(counts=())
    tensors["mean"] = np.ones(DIM, dtype=np.float32)
    with pytest.raises(IndexFileError, match="all zeros"):
        index_file.check(tensors, header)


def test_mean_tolerance_scales_with_the_data():
    tensors, header = make_index()
    tensors["pooled"] = tensors["pooled"] * np.float32(1e4)
    tensors["mean"] = tensors["pooled"].mean(axis=0, dtype=np.float64).astype(np.float32)
    index_file.check(tensors, header)


@pytest.mark.parametrize("i", [-1, 3, 100])
def test_clip_rows_and_vector_reject_out_of_range_clips(i):
    tensors, header = make_index()
    index = index_file.Index(header=header, **tensors)
    with pytest.raises(IndexError):
        index.clip_rows(i)
    with pytest.raises(IndexError):
        index.clip_vector(i)


def set_clip(key, value):
    def mutate(header):
        header["clips"][0][key] = value

    return mutate


def set_top(key, value):
    def mutate(header):
        header[key] = value

    return mutate


def set_run(key, value):
    def mutate(header):
        header["run"][key] = value

    return mutate


def duplicate_paths(header):
    header["clips"][1]["path"] = header["clips"][0]["path"]


BAD_HEADER_FIELDS = [
    pytest.param(set_top("root", "videos"), "root must be", id="root-relative"),
    pytest.param(set_top("root", "~/videos"), "root must be", id="root-tilde"),
    pytest.param(set_top("root", ""), "root must be", id="root-empty"),
    pytest.param(set_top("root", 5), "root must be", id="root-not-a-string"),
    pytest.param(set_top("format_version", True), "format_version", id="format-version-bool"),
    pytest.param(set_top("recursive", 1), "recursive must be", id="recursive-int"),
    pytest.param(set_top("recursive", None), "recursive must be", id="recursive-null"),
    pytest.param(set_top("skipped", "x"), "skipped must be", id="skipped-not-a-list"),
    pytest.param(set_top("skipped", {}), "skipped must be", id="skipped-empty-object"),
    pytest.param(set_top("skipped", [1, "x"]), "skipped must be", id="skipped-not-objects"),
    pytest.param(set_top("skipped", [{"path": "a.mp4"}]), "skipped must be", id="skipped-no-reason"),
    pytest.param(set_top("skipped", [{"path": 3, "reason": "r"}]), "skipped must be", id="skipped-path-type"),
    pytest.param(set_top("clips", {}), "clips must be a list", id="clips-not-a-list"),
    pytest.param(set_top("clips", [5]), r"clips\[0\] must be an object", id="clip-not-an-object"),
    pytest.param(set_top("run", []), "run must be", id="run-not-an-object"),
    pytest.param(set_run("max_seconds", float("nan")), "max_seconds", id="max-seconds-nan"),
    pytest.param(set_run("max_seconds", float("inf")), "max_seconds", id="max-seconds-inf"),
    pytest.param(set_run("max_seconds", 0), "max_seconds", id="max-seconds-zero"),
    pytest.param(set_run("max_seconds", "30"), "max_seconds", id="max-seconds-string"),
    pytest.param(set_run("max_seconds", True), "max_seconds", id="max-seconds-bool"),
    pytest.param(set_clip("sha256", 5), "sha256", id="sha256-not-a-string"),
    pytest.param(set_clip("sha256", "zz"), "sha256", id="sha256-short"),
    pytest.param(set_clip("sha256", "A" * 64), "sha256", id="sha256-uppercase"),
    pytest.param(set_clip("size", -1), "size", id="size-negative"),
    pytest.param(set_clip("size", True), "size", id="size-bool"),
    pytest.param(set_clip("mtime_ns", 1.5), "mtime_ns", id="mtime-float"),
    pytest.param(set_clip("mtime_ns", -1), "mtime_ns", id="mtime-negative"),
    pytest.param(set_clip("width", 0), "width", id="width-zero"),
    pytest.param(set_clip("width", 3.0), "width", id="width-float"),
    pytest.param(set_clip("height", -5), "height", id="height-negative"),
    pytest.param(set_clip("duration", float("nan")), "duration", id="duration-nan"),
    pytest.param(set_clip("duration", float("inf")), "duration", id="duration-inf"),
    pytest.param(set_clip("duration", -1.0), "duration", id="duration-negative"),
    pytest.param(set_clip("duration", 0), "duration", id="duration-zero"),
    pytest.param(set_clip("duration", "4"), "duration", id="duration-string"),
    pytest.param(set_clip("duration", True), "duration", id="duration-bool"),
    pytest.param(set_clip("windows", [0]), "windows must be", id="windows-one-value"),
    pytest.param(set_clip("windows", [0, 2.0]), "windows must be", id="windows-float"),
    pytest.param(set_clip("windows", [True, 2]), "windows must be", id="windows-bool"),
    pytest.param(set_clip("windows", "0:2"), "windows must be", id="windows-string"),
    pytest.param(set_clip("path", "./a.mp4"), "path", id="path-dot-segment"),
    pytest.param(set_clip("path", "a//b.mp4"), "path", id="path-empty-segment"),
    pytest.param(set_clip("path", "sub/"), "path", id="path-trailing-slash"),
    pytest.param(set_clip("path", "."), "path", id="path-dot"),
    pytest.param(set_clip("path", "C:/x.mp4"), "path", id="path-drive-letter"),
    pytest.param(set_clip("path", "a\x00b.mp4"), "path", id="path-nul"),
    pytest.param(set_clip("path", 7), "path", id="path-not-a-string"),
    pytest.param(duplicate_paths, "same path", id="paths-duplicate"),
]


@pytest.mark.parametrize(("mutate", "message"), BAD_HEADER_FIELDS)
def test_rejects_malformed_header_field(mutate, message):
    tensors, header = make_index()
    mutate(header)
    with pytest.raises(IndexFileError, match=message):
        index_file.check(tensors, header)


def test_rejects_non_integer_hidden_size():
    tensors, header = make_index()
    block = {**header["representation"], "hidden_size": float(DIM)}
    header["representation"] = {**block, "hash": latent_file.representation_hash(block)}
    with pytest.raises(IndexFileError, match="hidden_size"):
        index_file.check(tensors, header)


def test_paths_with_spaces_and_unicode_are_accepted():
    tensors, header = make_index()
    header["clips"][0]["path"] = "Ünï dir/clip 🎬.mp4"
    index_file.check(tensors, header)


def test_save_refuses_a_nan_in_the_header(tmp_path: Path):
    tensors, header = make_index()
    header["clips"][0]["duration"] = float("nan")
    with pytest.raises(IndexFileError, match="duration"):
        index_file.save(tmp_path / "x.index", tensors, header)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_read_header_refuses_json_constants(tmp_path: Path, constant):
    tensors, header = make_index()
    text = json.dumps(header, sort_keys=True).replace('"duration": 4.0', f'"duration": {constant}', 1)
    assert constant in text
    path = tmp_path / "nan.index"
    save_file(tensors, str(path), metadata={"jepa_scope": text})
    with pytest.raises(IndexFileError, match="not allowed|not valid JSON"):
        index_file.read_header(path)


@pytest.mark.parametrize("text", ["[1, 2]", "{broken"], ids=["not-an-object", "not-json"])
def test_read_header_refuses_a_non_object_or_broken_header(tmp_path: Path, text):
    tensors, _ = make_index()
    path = tmp_path / "odd.index"
    save_file(tensors, str(path), metadata={"jepa_scope": text})
    with pytest.raises(IndexFileError, match="JSON"):
        index_file.read_header(path)


def test_read_header_of_a_missing_file_raises_like_latent_file(tmp_path: Path):
    # latent_file.read_header lets the missing-file error through; the index does the same.
    with pytest.raises(FileNotFoundError):
        latent_file.read_header(tmp_path / "missing.latent")
    with pytest.raises(FileNotFoundError):
        index_file.read_header(tmp_path / "missing.index")


def test_failed_save_leaves_no_temporary_file(tmp_path: Path):
    tensors, header = make_index()
    target = tmp_path / "target.index"
    target.mkdir()  # os.replace of a file onto a directory fails
    with pytest.raises(OSError):
        index_file.save(target, tensors, header)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["target.index"]


def test_saves_use_distinct_temporary_names(tmp_path: Path, monkeypatch):
    tensors, header = make_index()
    seen = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append(Path(src).name)
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    index_file.save(tmp_path / "a.index", tensors, header)
    index_file.save(tmp_path / "a.index", tensors, header)
    assert len(set(seen)) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.index"]


def test_resolve_root_makes_the_colocated_path_absolute(tmp_path: Path, monkeypatch):
    tensors, header = make_index()
    index = index_file.Index(header=header, **tensors)
    (tmp_path / "folder").mkdir()
    monkeypatch.chdir(tmp_path)
    root = index.resolve_root(Path("folder") / "jepa-scope.index")
    assert root.is_absolute()
    assert root == (tmp_path / "folder").resolve()


def test_save_writes_strict_json_even_if_the_check_is_bypassed(tmp_path: Path, monkeypatch):
    # The check already rejects a NaN; this pins the second line of defence in the writer itself.
    monkeypatch.setattr(index_file, "check", lambda tensors, header: None)
    tensors, header = make_index()
    header["clips"][0]["duration"] = float("nan")
    with pytest.raises(IndexFileError, match="cannot be written as JSON"):
        index_file.save(tmp_path / "x.index", tensors, header)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("path", "ok"),
    [
        ("cam 2026-11-06 10:15:00.mp4", True),  # a colon in a top-level timestamped name
        ("day/10:15:00.mp4", True),
        ("a:b.mp4", True),
        ("C:/x.mp4", False),
        ("C:", False),
        ("sub\\x.mp4", False),
    ],
)
def test_is_valid_clip_path(path, ok):
    assert index_file.is_valid_clip_path(path) is ok


def test_saved_index_follows_the_umask(tmp_path):
    import os

    tensors, header = make_index()
    old = os.umask(0o077)
    try:
        index_file.save(tmp_path / "private.index", tensors, header)
    finally:
        os.umask(old)
    index_file.save(tmp_path / "shared.index", tensors, header)
    assert (tmp_path / "private.index").stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "shared.index").stat().st_mode & 0o777 == 0o666 & ~old


def test_saving_through_a_symlink_writes_beside_the_link(tmp_path, monkeypatch):
    import tempfile

    made_in = []
    real_mkstemp = tempfile.mkstemp

    def recording_mkstemp(*args, **kwargs):
        made_in.append(Path(kwargs["dir"]))
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(index_file.tempfile, "mkstemp", recording_mkstemp)
    tensors, header = make_index()
    target_dir = tmp_path / "elsewhere"
    target_dir.mkdir()
    (target_dir / "real.index").write_bytes(b"old")
    link = tmp_path / "jepa-scope.index"
    link.symlink_to(target_dir / "real.index")
    index_file.save(link, tensors, header)
    assert not link.is_symlink() and index_file.load(link).pooled.shape == tensors["pooled"].shape
    assert not list(tmp_path.glob("*.tmp")) and not list(target_dir.glob("*.tmp"))
    # Beside the link, so a link to another filesystem (an external drive) still renames in place.
    assert made_in == [tmp_path]
