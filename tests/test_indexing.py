"""The folder indexer and the ``index`` and ``search`` commands, on the tiny random encoder.

Every test indexes a fresh copy of one small folder: a moving square, an exact copy of it under
another name, a constant grey clip, and a half-second clip that is shorter than one window. They
are embedded with ``window=8`` at 8 fps, so a three-second clip gives 5 windows. The cache lives in
the test's own temporary folder. The random encoder supports no ranking claim beyond an exact copy
scoring 1.0; these tests check plumbing and the refresh rules.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from jepa_scope import cache, cli, index_file, indexing, latent_file, search
from jepa_scope.cache import EmbedSettings

from .conftest import write_video

SETTINGS = EmbedSettings(model="test/tiny-vjepa2", fps=8.0, window=8, device="cpu", dtype="float32")
SQUARE, COPY, GREY, SHORT = "square.mp4", "square-copy.mp4", "grey.mp4", "short.mp4"
EMBED_ARGS = ["--model", "test/tiny-vjepa2", "--fps", "8", "--window", "8", "--device", "cpu", "--dtype", "float32"]


class TinyModel:
    """A model provider around the tiny encoder. ``Encoder.describe`` cannot resolve its fake model
    id, so its representation comes from the encoder itself. ``loads`` counts ``encoder()`` calls."""

    def __init__(self, encoder, settings: EmbedSettings = SETTINGS):
        self.enc = encoder
        self.settings = settings
        self.loads = 0

    def representation(self):
        return self.enc.representation(self.settings.window, self.settings.fps)

    def encoder(self):
        self.loads += 1
        return self.enc


def _free_box(payload: bytes) -> bytes:
    """An MP4 ``free`` box: padding any MP4 reader skips."""
    return struct.pack(">I", 8 + len(payload)) + b"free" + payload


def _square_frames() -> np.ndarray:
    t, h, w = 72, 64, 96
    frames = np.zeros((t, h, w, 3), np.uint8)
    for i in range(t):
        x = int(i / t * (w - 16))
        frames[i, 24:40, x : x + 16] = 255
    return frames


@pytest.fixture(scope="session")
def template(tmp_path_factory) -> Path:
    """The four videos, written once per session; tests copy them."""
    folder = tmp_path_factory.mktemp("template")
    write_video(folder / SQUARE, _square_frames())
    shutil.copyfile(folder / SQUARE, folder / COPY)
    write_video(folder / GREY, np.full((72, 64, 96, 3), 128, np.uint8))
    # A trailing free box, so a test can change the file's bytes without changing its size or
    # what it decodes to.
    with open(folder / GREY, "ab") as f:
        f.write(_free_box(b"\0" * 8))
    write_video(folder / SHORT, _square_frames()[:12])
    return folder


@pytest.fixture
def folder(tmp_path, template) -> Path:
    out = tmp_path / "videos"
    shutil.copytree(template, out)
    return out


@pytest.fixture
def cache_dir(tmp_path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def embeds(monkeypatch) -> list[str]:
    """The names of the videos ``embed_video`` is called on, in order."""
    calls: list[str] = []
    real = indexing.embed_video

    def counting(video, *args, **kwargs):
        calls.append(Path(video).name)
        return real(video, *args, **kwargs)

    monkeypatch.setattr(indexing, "embed_video", counting)
    return calls


@pytest.fixture
def run(tiny_encoder, cache_dir):
    """``run(folder, **kwargs)`` builds or refreshes an index, logging into ``run.lines``."""
    lines: list[str] = []

    def go(folder, model=None, settings=SETTINGS, **kwargs):
        return indexing.build_or_refresh(
            folder,
            model or TinyModel(tiny_encoder, settings),
            settings,
            log=lines.append,
            cache_dir=cache_dir,
            **kwargs,
        )

    go.lines = lines
    return go


def _entry(index: index_file.Index, path: str) -> dict:
    return next(c for c in index.header["clips"] if c["path"] == path)


def _clip_id(index: index_file.Index, path: str) -> int:
    return [c["path"] for c in index.header["clips"]].index(path)


# ----------------------------------------------------------------------------------------------
# Building


def test_index_holds_three_clips_and_skips_the_short_one(folder, run, embeds):
    report = run(folder)
    index = index_file.load(folder / indexing.INDEX_NAME)
    assert [c["path"] for c in index.header["clips"]] == [GREY, COPY, SQUARE]  # sorted by path
    assert all(c["windows"][1] - c["windows"][0] == 5 for c in index.header["clips"])
    assert index.pooled.shape == (15, 96)
    [skipped] = index.header["skipped"]
    assert skipped["path"] == SHORT and "one window needs 8" in skipped["reason"]
    assert index.header["root"] == "."
    assert index.header["run"] == {"max_seconds": None, "device": "cpu", "dtype": "float32"}
    assert index.header["embedding"] == {"kind": "window_vector", "similarity": "cosine"}
    assert sorted(embeds) == sorted([GREY, SQUARE, SHORT]) or sorted(embeds) == sorted([GREY, COPY, SHORT])
    assert report.embedded == 2 and report.clips == 3 and report.windows == 15
    # The copy and the square share one cache file and its vectors.
    assert _entry(index, COPY)["sha256"] == _entry(index, SQUARE)["sha256"]
    assert np.array_equal(index.pooled[index.clip_rows(1)], index.pooled[index.clip_rows(2)])
    # The stored mean is the window-weighted mean of the rows the indexer wrote.
    assert np.allclose(index.mean, index.pooled.mean(axis=0, dtype=np.float64), atol=1e-6)
    # One progress line per clip.
    assert sum(line.startswith("[") for line in run.lines) == 4


def test_new_clips_are_embedded_without_patches_unless_asked(folder, run, cache_dir):
    run(folder)
    sha = _entry(index_file.load(folder / indexing.INDEX_NAME), SQUARE)["sha256"]
    plain = cache.cache_path(sha, SETTINGS, False, cache_dir)
    assert plain.exists() and "patches" not in latent_file.load(plain)[0]
    assert not cache.cache_path(sha, SETTINGS, True, cache_dir).exists()


def test_patches_keeps_them(folder, run, cache_dir):
    run(folder, patches=True)
    sha = _entry(index_file.load(folder / indexing.INDEX_NAME), SQUARE)["sha256"]
    assert "patches" in latent_file.load(cache.cache_path(sha, SETTINGS, True, cache_dir))[0]


def test_a_with_patches_cache_file_is_preferred(folder, run, cache_dir, embeds):
    """With both cache files present, the with-patches one is used."""
    run(folder)
    sha = _entry(index_file.load(folder / indexing.INDEX_NAME), SQUARE)["sha256"]
    tensors, header = latent_file.load(cache.cache_path(sha, SETTINGS, False, cache_dir))
    marked = {**tensors, "pooled": tensors["pooled"] * 2}
    latent_file.save(cache.cache_path(sha, SETTINGS, True, cache_dir), marked, header)
    (folder / indexing.INDEX_NAME).unlink()
    embeds.clear()
    run(folder)
    index = index_file.load(folder / indexing.INDEX_NAME)
    assert embeds == [SHORT]  # the index is gone, so only the clip with no cache file is tried
    assert np.array_equal(index.pooled[index.clip_rows(_clip_id(index, SQUARE))], marked["pooled"])


def test_a_without_patches_cache_file_is_reused_by_a_patches_run(folder, run, embeds):
    run(folder)
    embeds.clear()
    run(folder, patches=True, verify=True)
    assert embeds == [SHORT]  # only the clip that has no cache file at all is tried again


def test_the_indexer_never_loads_whole_latent_files(folder, run, monkeypatch):
    """It reads only the window vectors; loading every tensor would read the patches too."""
    run(folder, patches=True)

    def refuse(*args, **kwargs):
        raise AssertionError("latent_file.load was called")

    monkeypatch.setattr(latent_file, "load", refuse)
    run(folder, verify=True)


def test_a_rerun_embeds_nothing_and_never_loads_the_model(folder, run, embeds, tiny_encoder):
    run(folder)
    embeds.clear()
    model = TinyModel(tiny_encoder)
    report = run(folder, model=model)
    assert embeds == [] and model.loads == 0 and report.hashed == 0 and report.embedded == 0
    index = index_file.load(folder / indexing.INDEX_NAME)
    assert len(index.header["clips"]) == 3
    # The short clip is skipped again from its record, without trying (and loading the model).
    [skipped] = index.header["skipped"]
    assert skipped["path"] == SHORT and "one window needs 8" in skipped["reason"]
    assert any("skipped as before (unchanged)" in line for line in run.lines)


def test_a_skipped_file_is_tried_again_when_it_changes_or_with_verify(folder, run, embeds):
    run(folder)
    embeds.clear()
    run(folder, verify=True)
    assert embeds == [SHORT]
    embeds.clear()
    write_video(folder / SHORT, _square_frames()[:36])  # now 1.5 s: 12 frames at 8 fps, 2 windows
    run(folder)
    assert embeds == [SHORT]
    index = index_file.load(folder / indexing.INDEX_NAME)
    assert index.header["skipped"] == [] and len(index.header["clips"]) == 4


def test_touching_a_file_rehashes_but_does_not_re_embed(folder, run, embeds):
    (folder / SHORT).unlink()
    run(folder)
    embeds.clear()
    st = (folder / GREY).stat()
    os.utime(folder / GREY, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    report = run(folder)
    assert embeds == [] and report.hashed == 1
    assert _entry(index_file.load(folder / indexing.INDEX_NAME), GREY)["mtime_ns"] == st.st_mtime_ns + 5_000_000_000


def test_rewriting_a_file_re_embeds_only_it_and_deleting_one_drops_it(folder, run, embeds):
    (folder / SHORT).unlink()
    run(folder)
    before = index_file.load(folder / indexing.INDEX_NAME)
    embeds.clear()
    write_video(folder / GREY, np.full((72, 64, 96, 3), 40, np.uint8))
    run(folder)
    assert embeds == [GREY]
    after = index_file.load(folder / indexing.INDEX_NAME)
    assert _entry(after, GREY)["sha256"] != _entry(before, GREY)["sha256"]

    embeds.clear()
    (folder / COPY).unlink()
    run(folder)
    index = index_file.load(folder / indexing.INDEX_NAME)
    assert embeds == [] and [c["path"] for c in index.header["clips"]] == [GREY, SQUARE]
    assert index.pooled.shape == (10, 96)


def test_deleting_a_shared_cache_file_embeds_once_and_restores_both(folder, run, embeds, cache_dir):
    (folder / SHORT).unlink()
    run(folder)
    sha = _entry(index_file.load(folder / indexing.INDEX_NAME), SQUARE)["sha256"]
    cache.cache_path(sha, SETTINGS, False, cache_dir).unlink()
    embeds.clear()
    run(folder)
    assert len(embeds) == 1 and embeds[0] in (SQUARE, COPY)
    index = index_file.load(folder / indexing.INDEX_NAME)
    assert {c["path"] for c in index.header["clips"]} == {GREY, SQUARE, COPY}


def test_a_cache_file_of_another_video_is_not_reused(folder, run, embeds, cache_dir):
    """The cache name holds only 16 characters of the hash; the header's full hash must match."""
    (folder / SHORT).unlink()
    run(folder)
    index = index_file.load(folder / indexing.INDEX_NAME)
    square = cache.cache_path(_entry(index, SQUARE)["sha256"], SETTINGS, False, cache_dir)
    grey = cache.cache_path(_entry(index, GREY)["sha256"], SETTINGS, False, cache_dir)
    shutil.copyfile(square, grey)
    embeds.clear()
    run(folder)
    assert embeds == [GREY]
    assert any("belongs to another video" in line for line in run.lines)
    index = index_file.load(folder / indexing.INDEX_NAME)
    assert not np.array_equal(index.pooled[index.clip_rows(0)], index.pooled[index.clip_rows(2)])


def test_an_invalid_cache_file_is_re_embedded(folder, run, embeds, cache_dir):
    (folder / SHORT).unlink()
    run(folder)
    grey = cache.cache_path(
        _entry(index_file.load(folder / indexing.INDEX_NAME), GREY)["sha256"], SETTINGS, False, cache_dir
    )
    grey.write_bytes(grey.read_bytes()[:-100])
    embeds.clear()
    run(folder)
    assert embeds == [GREY]
    assert sum("re-embedding" in line for line in run.lines) == 1


def _other_weights(tiny_encoder):
    from jepa_scope.model import Encoder, ModelSource

    source = ModelSource("test/tiny-vjepa2", weights="random-seed-0-retrained", revision=None)
    return Encoder(
        tiny_encoder.model,
        tiny_encoder.processor,
        "test/tiny-vjepa2",
        tiny_encoder.device,
        tiny_encoder.dtype,
        source=source,
    )


@pytest.mark.parametrize(
    "change, reason",
    [
        ("representation", "the representation changed"),
        ("max_seconds", "max_seconds None -> 2.5"),
        ("device", "device cpu -> meta"),
        ("dtype", "dtype float32 -> float16"),
    ],
)
def test_a_settings_change_rebuilds_everything(folder, run, embeds, tiny_encoder, change, reason):
    (folder / SHORT).unlink()
    run(folder)
    embeds.clear()
    settings, model = SETTINGS, None
    if change == "representation":
        model = TinyModel(_other_weights(tiny_encoder))
    else:
        value = {"max_seconds": 2.5, "device": "meta", "dtype": "float16"}[change]
        settings = dataclasses.replace(SETTINGS, **{change: value})
    report = run(folder, model=model, settings=settings)
    assert report.rebuilt is not None and reason in report.rebuilt
    assert any(line.startswith("rebuilding the index from scratch") and reason in line for line in run.lines)
    assert sorted(embeds) in (sorted([GREY, SQUARE]), sorted([GREY, COPY]))
    index = index_file.load(folder / indexing.INDEX_NAME)
    assert len(index.header["clips"]) == 3
    assert index.header["run"] == settings.run()
    if change == "max_seconds":
        assert index.pooled.shape[0] == 3 * 4  # 2.5 s at 8 fps keeps 21 frames: 4 windows


def test_an_unreadable_or_older_index_is_rebuilt(folder, run, embeds):
    (folder / SHORT).unlink()
    out = folder / indexing.INDEX_NAME
    out.write_bytes(b"not an index")
    report = run(folder)
    assert report.rebuilt is not None and "cannot be read" in report.rebuilt
    assert report.hashed == 3

    import json

    from safetensors.numpy import save_file

    index = index_file.load(out)
    tensors = {
        "pooled": index.pooled,
        "window_start_time": index.window_start_time,
        "clip": index.clip,
        "mean": index.mean,
    }
    save_file(tensors, str(out), metadata={"jepa_scope": json.dumps({**index.header, "format_version": 99})})
    embeds.clear()
    report = run(folder)
    assert report.rebuilt is not None and "format_version 99" in report.rebuilt
    assert embeds == [] and report.hashed == 3  # rebuilt from scratch: hashed again, cache reused


def test_an_index_of_another_folder_is_rebuilt(folder, run, tmp_path):
    (folder / SHORT).unlink()
    out = tmp_path / "shared.index"
    run(folder, out=out)
    other = tmp_path / "other"
    shutil.copytree(folder, other)
    report = run(other, out=out)
    assert report.rebuilt is not None and "another folder" in report.rebuilt


def test_changing_only_centre_rewrites_the_similarity(folder, run, embeds):
    (folder / SHORT).unlink()
    run(folder)
    embeds.clear()
    report = run(folder, centre=True)
    index = index_file.load(folder / indexing.INDEX_NAME)
    assert index.header["embedding"]["similarity"] == "centred-cosine"
    assert embeds == [] and report.rebuilt is None and report.hashed == 0
    run(folder)
    assert index_file.load(folder / indexing.INDEX_NAME).header["embedding"]["similarity"] == "cosine"


def test_verify_finds_a_replacement_that_kept_size_and_mtime(folder, run, embeds):
    (folder / SHORT).unlink()
    run(folder)
    before = _entry(index_file.load(folder / indexing.INDEX_NAME), GREY)
    path = folder / GREY
    st = path.stat()
    data = bytearray(path.read_bytes())
    data[-1] = 1  # inside the trailing free box: same size, same frames, other bytes
    path.write_bytes(bytes(data))
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert path.stat().st_size == st.st_size

    embeds.clear()
    report = run(folder)
    assert embeds == [] and report.hashed == 0
    assert _entry(index_file.load(folder / indexing.INDEX_NAME), GREY)["sha256"] == before["sha256"]

    report = run(folder, verify=True)
    assert embeds == [GREY] and report.hashed == 3
    assert _entry(index_file.load(folder / indexing.INDEX_NAME), GREY)["sha256"] != before["sha256"]


# ----------------------------------------------------------------------------------------------
# Search over an indexed folder


def test_an_exact_copy_scores_one(folder, run):
    run(folder)
    index = index_file.load(folder / indexing.INDEX_NAME)
    space = search.prepare(index)
    square, copy = _clip_id(index, SQUARE), _clip_id(index, COPY)
    [best] = search.nearest_clips(space, square, 1).hits
    assert best.clip == copy and abs(best.score - 1.0) < 1e-6
    for w in range(5):
        [hit] = search.nearest_moments(space, square, w, 1).hits
        assert (hit.clip, hit.window) == (copy, w) and abs(hit.score - 1.0) < 1e-6


# ----------------------------------------------------------------------------------------------
# Walking the folder and bad files


def test_empty_folder_gives_a_valid_empty_index(tmp_path, run):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_text("not a video")
    report = run(empty)
    index = index_file.load(empty / indexing.INDEX_NAME)
    assert index.header["clips"] == [] and index.header["skipped"] == [] and index.pooled.shape == (0, 96)
    assert report.clips == 0
    assert any(line.startswith("nothing was indexed (no videos were found)") for line in run.lines)


def test_all_skipped_folder_gives_a_valid_empty_index(tmp_path, template, run):
    only = tmp_path / "only-short"
    only.mkdir()
    shutil.copyfile(template / SHORT, only / SHORT)
    run(only)
    index = index_file.load(only / indexing.INDEX_NAME)
    assert index.header["clips"] == [] and [s["path"] for s in index.header["skipped"]] == [SHORT]
    assert any(line.startswith("nothing was indexed (all 1 videos were skipped)") for line in run.lines)


def test_bad_files_are_skipped_and_the_run_continues(folder, run):
    (folder / "garbage.mov").write_bytes(b"\0" * 1000)
    (folder / "empty.MP4").write_bytes(b"")
    write_video(folder / "slow.mp4", _square_frames()[:24], fps=4)  # 4 fps, below the 8 fps asked for
    (folder / "back\\slash.mp4").write_bytes((folder / GREY).read_bytes())
    (folder / "dangling.mp4").symlink_to(folder / "missing.mp4")
    report = run(folder)
    reasons = {s["path"]: s["reason"] for s in report.skipped}
    assert set(reasons) == {SHORT, "garbage.mov", "empty.MP4", "slow.mp4", "back\\slash.mp4", "dangling.mp4"}
    assert "fewer than the requested 8" in reasons["slow.mp4"]
    assert reasons["garbage.mov"].startswith("the file cannot be decoded (")
    assert str(folder) not in reasons["garbage.mov"]
    assert "portably" in reasons["back\\slash.mp4"]
    assert "cannot be read" in reasons["dangling.mp4"]
    index = index_file.load(folder / indexing.INDEX_NAME)
    assert len(index.header["clips"]) == 3
    assert index.header["skipped"] == report.skipped


def test_the_walk_rules(tmp_path, template, run):
    top = tmp_path / "top"
    (top / "sub" / "deeper").mkdir(parents=True)
    (top / ".hidden").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    for path in (
        top / "a.MOV",
        top / "sub" / "b.mp4",
        top / "sub" / "deeper" / "c.webm",
        top / ".hidden" / "d.mp4",
        top / ".e.mp4",
        outside / "f.mp4",
    ):
        shutil.copyfile(template / GREY, path)
    (top / "notes.txt").write_text("x")
    (top / "link").symlink_to(outside, target_is_directory=True)
    (top / "g.mkv").symlink_to(template / SQUARE)  # a symlinked file is a video like any other

    assert [rel for rel, _ in indexing.find_videos(top)] == ["a.MOV", "g.mkv"]
    assert [rel for rel, _ in indexing.find_videos(top, recursive=True)] == [
        "a.MOV",
        "g.mkv",
        "sub/b.mp4",
        "sub/deeper/c.webm",
    ]
    run(top, recursive=True)
    paths = [c["path"] for c in index_file.load(top / indexing.INDEX_NAME).header["clips"]]
    assert paths == ["a.MOV", "g.mkv", "sub/b.mp4", "sub/deeper/c.webm"]


def test_out_outside_the_folder_records_an_absolute_root(folder, run, tmp_path):
    out = tmp_path / "elsewhere" / "videos.index"
    run(folder, out=out)
    assert index_file.load(out).header["root"] == str(folder.resolve())
    assert not (folder / indexing.INDEX_NAME).exists()


def test_out_in_a_subfolder_records_an_absolute_root(folder, run):
    out = folder / "indexes" / "x.index"
    run(folder, out=out)
    assert index_file.load(out).header["root"] == str(folder.resolve())


# ----------------------------------------------------------------------------------------------
# The command line


@pytest.fixture
def patched_cli(monkeypatch, tiny_encoder, cache_dir):
    """Points the CLI at the tiny encoder and the test's cache; returns the providers it made."""
    made: list[TinyModel] = []

    def provider(settings):
        made.append(TinyModel(tiny_encoder, settings))
        return made[-1]

    monkeypatch.setattr(cli, "_model_provider", provider)
    monkeypatch.setattr(cache, "CACHE_DIR", cache_dir)
    return made


def test_cli_index_then_search_a_clip(folder, patched_cli, capsys, monkeypatch, cache_dir):
    assert cli.main(["index", str(folder), *EMBED_ARGS]) == 0
    out = capsys.readouterr().out
    assert "[4/4] square.mp4" in out and "indexed 3 clips (15 windows)" in out
    assert any(cache_dir.iterdir())

    assert cli.main(["search", str(folder), SQUARE, "--at", "1.0", "-k", "3"]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].startswith("moments like square.mp4 at 1s (window")
    first = lines[2].split()
    assert first[0] == "1" and abs(float(first[1]) - 1.0) < 1e-4 and first[3] == COPY
    assert len(lines) == 2 + 3

    assert cli.main(["search", str(folder / indexing.INDEX_NAME), SQUARE, "--by", "clip"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "clips like square.mp4:" and lines[2].split()[3] == COPY and len(lines) == 2 + 2

    # A path to the file itself resolves to the indexed clip.
    monkeypatch.chdir(folder.parent)
    assert cli.main(["search", "videos", "videos/square.mp4", "--same-clip", "-k", "1"]) == 0
    assert capsys.readouterr().out.startswith("moments like square.mp4")


def test_cli_search_with_an_outside_video(folder, patched_cli, capsys, tmp_path, tiny_encoder):
    cli.main(["index", str(folder), *EMBED_ARGS])
    capsys.readouterr()
    query = tmp_path / "query.mp4"
    shutil.copyfile(folder / SQUARE, query)  # same bytes, so the cache already holds it
    assert cli.main(["search", str(folder), str(query), "--by", "clip", "-k", "2"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"clips like {query}:"
    assert {lines[2].split()[3], lines[3].split()[3]} == {SQUARE, COPY}
    assert abs(float(lines[2].split()[1]) - 1.0) < 1e-4
    # The query was embedded with the index's own settings.
    assert patched_cli[-1].settings == EmbedSettings(
        model="test/tiny-vjepa2", fps=8.0, window=8, device="cpu", dtype="float32"
    )

    fresh = tmp_path / "fresh.mp4"
    write_video(fresh, np.full((48, 64, 96, 3), 90, np.uint8))
    assert cli.main(["search", str(folder), str(fresh), "--at", "0.5"]) == 0
    assert capsys.readouterr().out.startswith(f"moments like {fresh} at 0.5s (window 0")


def test_cli_search_refuses_a_query_from_another_representation(
    folder, patched_cli, capsys, monkeypatch, tmp_path, tiny_encoder
):
    cli.main(["index", str(folder), *EMBED_ARGS])
    query = tmp_path / "query.mp4"
    shutil.copyfile(folder / GREY, query)
    monkeypatch.setattr(cli, "_model_provider", lambda settings: TinyModel(_other_weights(tiny_encoder), settings))
    with pytest.raises(SystemExit, match="cannot be embedded the way this index was"):
        cli.main(["search", str(folder), str(query)])


def test_cli_search_of_a_moved_external_index_needs_source_root(folder, patched_cli, capsys, tmp_path):
    out = tmp_path / "videos.index"
    cli.main(["index", str(folder), "--out", str(out), *EMBED_ARGS])
    moved = tmp_path / "moved"
    folder.rename(moved)
    with pytest.raises(SystemExit, match="--source-root"):
        cli.main(["search", str(out), SQUARE])
    capsys.readouterr()
    assert cli.main(["search", str(out), SQUARE, "--source-root", str(moved), "-k", "1"]) == 0
    assert capsys.readouterr().out.splitlines()[2].split()[3] == COPY
    assert cli.main(["search", str(out), str(moved / "square.mp4"), "--source-root", str(moved), "-k", "1"]) == 0
    assert capsys.readouterr().out.startswith("moments like square.mp4")


def test_cli_search_reports_an_unavailable_query(folder, patched_cli, capsys):
    """Ten identical clips centre to zero; here two identical clips alone do."""
    (folder / GREY).unlink()
    (folder / SHORT).unlink()
    cli.main(["index", str(folder), "--centre", *EMBED_ARGS])
    capsys.readouterr()
    assert cli.main(["search", str(folder), SQUARE, "--by", "clip"]) == 0
    assert "no results: the query clip is unavailable" in capsys.readouterr().out


def test_cli_search_errors(folder, patched_cli, capsys, tmp_path):
    with pytest.raises(SystemExit, match="jepa-scope index"):
        cli.main(["search", str(folder), SQUARE])
    cli.main(["index", str(folder), *EMBED_ARGS])
    with pytest.raises(SystemExit, match="neither a clip in the index nor a video file"):
        cli.main(["search", str(folder), "nope.mp4"])
    with pytest.raises(SystemExit, match="--by moment only"):
        cli.main(["search", str(folder), SQUARE, "--by", "clip", "--at", "1"])
    with pytest.raises(SystemExit, match="--at must be a finite"):
        cli.main(["search", str(folder), SQUARE, "--at", "nan"])
    with pytest.raises(SystemExit, match="-k must be between"):
        cli.main(["search", str(folder), SQUARE, "-k", "0"])
    with pytest.raises(SystemExit, match="cannot be embedded: the video gives"):
        cli.main(["search", str(folder), str(folder / SHORT)])


def test_cli_index_of_an_empty_folder(tmp_path, patched_cli, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.main(["index", str(empty), *EMBED_ARGS]) == 0
    assert "nothing was indexed" in capsys.readouterr().out
    assert cli.main(["search", str(empty), "x.mp4"]) == 0
    assert "nothing was indexed" in capsys.readouterr().out
