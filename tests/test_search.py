"""Search over hand-built vectors: ranking rules, exclusion, centring and error handling.

Every index here is built from vectors written in the test; no encoder is involved. A window is
2 seconds long (16 frames at 8 fps) and a window starts every second (stride 8), so only
neighbouring windows of a clip overlap. Each test says in its name or docstring whether it runs
centred or plain.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from jepa_scope import index_file, search
from jepa_scope.index_file import Index
from tests.test_index import make_representation

DIM = 8


def build(clips, similarity: str = "cosine", dim: int | None = None) -> Index:
    """An index from ``clips``, a list of ``(vectors, start_times)``; vectors are ``[W, D]``."""
    dim = dim or (len(clips[0][0][0]) if clips else DIM)
    if clips:
        pooled = np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1, dim) for v, _ in clips])
        times = np.concatenate([np.asarray(t, dtype=np.float64) for _, t in clips])
    else:
        pooled, times = np.zeros((0, dim), np.float32), np.zeros(0)
    counts = [len(t) for _, t in clips]
    owner = np.repeat(np.arange(len(clips), dtype=np.int32), counts)
    mean = pooled.mean(axis=0, dtype=np.float64).astype(np.float32) if len(pooled) else np.zeros(dim, np.float32)
    tensors = {"pooled": pooled, "window_start_time": times, "clip": owner, "mean": mean}
    entries, start = [], 0
    for i, c in enumerate(counts):
        entries.append(
            {
                "path": f"clip{i}.mp4",
                "sha256": f"{i:064x}",
                "size": 1,
                "mtime_ns": 1,
                "duration": 10.0,
                "width": 8,
                "height": 8,
                "windows": [start, start + c],
            }
        )
        start += c
    header = {
        "format": index_file.INDEX_FORMAT,
        "format_version": index_file.INDEX_FORMAT_VERSION,
        "representation": make_representation(dim),
        "run": {"max_seconds": None, "device": "cpu", "dtype": "float32"},
        "embedding": {"kind": "window_vector", "similarity": similarity},
        "root": ".",
        "recursive": False,
        "skipped": [],
        "clips": entries,
    }
    index_file.check(tensors, header)
    return Index(header=header, pooled=pooled, window_start_time=times, clip=owner, mean=mean)


def unit(i: int, dim: int = DIM) -> np.ndarray:
    v = np.zeros(dim)
    v[i] = 1.0
    return v


def random_clips(n_clips: int, windows: int, seed: int = 0, spacing: float = 1.0):
    rng = np.random.default_rng(seed)
    return [(rng.normal(size=(windows, DIM)), np.arange(windows) * spacing) for _ in range(n_clips)]


@pytest.mark.parametrize("centre", [False, True], ids=["plain", "centred"])
def test_duplicate_scores_one_and_ranks_first(centre):
    clips = random_clips(4, 3, seed=1)
    clips[2] = (np.concatenate([clips[2][0][:2], clips[0][0][1:2]]), clips[2][1])  # clip 2 window 2 = clip 0 window 1
    space = search.prepare(build(clips), centre=centre)
    res = search.nearest_moments(space, 0, 1, k=5)
    assert (res.hits[0].clip, res.hits[0].window) == (2, 2)
    assert res.hits[0].score == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("centre", [False, True], ids=["plain", "centred"])
def test_two_duplicates_tie_in_clip_window_order(centre):
    clips = random_clips(5, 2, seed=2)
    q = clips[0][0][0]
    for c in (4, 1, 3):  # plant the duplicate in clips 1, 3 and 4, window 1
        clips[c][0][1] = q
    space = search.prepare(build(clips), centre=centre)
    res = search.nearest_moments(space, 0, 0, k=3)
    assert [(h.clip, h.window) for h in res.hits] == [(1, 1), (3, 1), (4, 1)]
    assert len({h.score for h in res.hits}) == 1


def test_scaled_row_scores_one_plain():
    clips = random_clips(3, 2, seed=3)
    clips[1][0][0] = 2.0 * clips[0][0][0]
    space = search.prepare(build(clips), centre=False)
    hit = search.nearest_moments(space, 0, 0, k=1).hits[0]
    assert (hit.clip, hit.window) == (1, 0)
    assert hit.score == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("centre", [False, True], ids=["plain", "centred"])
def test_same_clip_excludes_query_and_overlaps(centre):
    # Clip 0 has windows every second, so each window overlaps its neighbours (2 s long).
    clips = [(np.random.default_rng(4).normal(size=(8, DIM)), np.arange(8) * 1.0)] + random_clips(2, 2, seed=5)
    space = search.prepare(build(clips), centre=centre)
    res = search.nearest_moments(space, 0, 3, k=20, same_clip=True)
    own = sorted(h.window for h in res.hits if h.clip == 0)
    assert all(abs(w - 3) >= 2 for w in own)  # starts differ by at least the 2 s window length
    assert 3 not in own and 2 not in own and 4 not in own
    assert any(h.clip != 0 for h in res.hits)


@pytest.mark.parametrize("centre", [False, True], ids=["plain", "centred"])
def test_not_same_clip_excludes_query_clip_entirely(centre):
    space = search.prepare(build(random_clips(4, 3, seed=6)), centre=centre)
    res = search.nearest_moments(space, 1, 2, k=50)
    assert res.hits and all(h.clip != 1 for h in res.hits)
    assert len(res.hits) >= 3  # at least one non-overlapping window from each of the other clips
    single = search.prepare(build(random_clips(4, 1, seed=6)), centre=centre)
    assert sorted(h.clip for h in search.nearest_moments(single, 1, 0, k=50).hits) == [0, 2, 3]


def test_overlapping_hits_in_one_clip_collapse_to_the_better_plain():
    q = unit(0)
    better, worse = unit(0) + 0.1 * unit(1), unit(0) + 0.5 * unit(1)
    other = unit(0) + 0.9 * unit(1)
    clips = [([q], [0.0]), ([worse, better], [0.0, 1.0]), ([other], [0.0])]  # clip 1 windows overlap
    space = search.prepare(build(clips), centre=False)
    res = search.nearest_moments(space, 0, 0, k=5)
    assert [(h.clip, h.window) for h in res.hits] == [(1, 1), (2, 0)]


def test_widening_still_returns_k_when_suppression_removes_most_plain():
    q = unit(0)
    # Clip 1 has twelve windows that all outrank the other clips, in an order (1, 0, 2, 4, 3, 5, ...)
    # where the best 2k=10 candidates keep only four non-overlapping hits, so k=5 must widen.
    order = [1, 0, 2, 4, 3, 5, 7, 6, 8, 10, 9, 11]
    heavy = np.zeros((12, DIM))
    for rank, w in enumerate(order):
        heavy[w] = unit(0) + (0.05 + 0.01 * rank) * unit(1)
    clips = [([q], [0.0]), (heavy, np.arange(12) * 1.0)]
    for i in range(4):  # four single-window clips ranked lower
        clips.append(([unit(0) + (1.0 + i) * unit(2)], [0.0]))
    space = search.prepare(build(clips), centre=False)
    res = search.nearest_moments(space, 0, 0, k=5)
    assert len(res.hits) == 5
    assert [h.window for h in res.hits if h.clip == 1] == [1, 4, 7, 10]
    assert res.hits[4].clip == 2
    full = search.nearest_moments(space, 0, 0, k=100)
    assert [h.row for h in res.hits] == [h.row for h in full.hits[:5]]


def test_k_larger_than_candidates_returns_all_that_exist():
    space = search.prepare(build(random_clips(3, 3, seed=8)), centre=False)
    res = search.nearest_moments(space, 0, 0, k=100)
    assert len(res.hits) == 4 and res.reason is None  # windows 0 and 2 of each of the two other clips


def test_common_offset_centred_recovers_structure_and_plain_does_not():
    """A large offset shared by every vector hides the planted structure from plain cosine."""
    rng = np.random.default_rng(0)
    offset = 10.0 * unit(7)
    clips = []
    for _ in range(5):  # clips 1 to 4 carry the query clip's structure, with noise
        clips.append((offset + unit(0) + rng.normal(scale=0.4, size=(1, DIM)), [0.0]))
    for _ in range(4):  # clips 5 to 8 are almost pure offset
        clips.append((offset + 0.3 * unit(1) + rng.normal(scale=0.05, size=(1, DIM)), [0.0]))
    index = build(clips)
    centred = search.nearest_moments(search.prepare(index, centre=True), 0, 0, k=4)
    plain = search.nearest_moments(search.prepare(index, centre=False), 0, 0, k=4)
    assert {h.clip for h in centred.hits} == {1, 2, 3, 4}
    assert any(h.clip >= 5 for h in plain.hits)


def test_header_decides_centring_when_not_given():
    index = build(random_clips(3, 2, seed=10), similarity="centred-cosine")
    assert search.prepare(index).centred
    assert not search.prepare(index, centre=False).centred
    assert not search.prepare(build(random_clips(3, 2, seed=10))).centred


def test_ten_identical_clips_centred_are_unavailable_plain_works():
    v = np.random.default_rng(11).normal(size=(1, DIM))
    index = build([(v, [0.0]) for _ in range(10)])
    centred = search.prepare(index, centre=True)
    assert not centred.available.any() and not centred.clip_available.any()
    res = search.nearest_moments(centred, 0, 0, k=3)
    assert res.hits == [] and res.reason
    clips = search.nearest_clips(centred, 0, k=3)
    assert clips.hits == [] and clips.reason
    plain = search.prepare(index, centre=False)
    res = search.nearest_moments(plain, 0, 0, k=3)
    assert len(res.hits) == 3 and res.hits[0].score == pytest.approx(1.0, abs=1e-6)


def test_unavailable_row_is_never_a_candidate_centred():
    # Rows x, -x and 0 average to zero, so the zero row is unavailable after centring.
    x = unit(0)
    clips = [([x], [0.0]), ([-x], [0.0]), ([0 * x], [0.0])]
    index = build(clips)
    space = search.prepare(index, centre=True)
    zero_row = 2
    assert not space.available[zero_row]
    res = search.nearest_moments(space, 0, 0, k=10)
    assert zero_row not in [h.row for h in res.hits]
    assert search.nearest_moments(space, 2, 0, k=3).reason
    assert [h.row for h in res.hits] == [1]


def test_near_identical_clips_centred_give_finite_deterministic_scores():
    rng = np.random.default_rng(12)
    v = rng.normal(size=(3, DIM))
    noisy = v + 1e-4 * rng.normal(size=v.shape)
    index = build([(v, np.arange(3) * 2.0), (noisy, np.arange(3) * 2.0)])
    space = search.prepare(index, centre=True)
    a = search.nearest_moments(space, 0, 0, k=3)
    b = search.nearest_moments(space, 0, 0, k=3)
    assert a == b
    assert all(np.isfinite(h.score) and -1.0 - 1e-6 <= h.score <= 1.0 + 1e-6 for h in a.hits)
    c = search.nearest_clips(space, 0, k=1)
    assert np.isfinite(c.hits[0].score)


def test_index_mean_is_window_weighted_and_long_clip_dominates():
    """Documents the weighting using this file's ``build`` helper, not product code.

    The real indexer's mean is checked against the same property in the indexer's own tests.
    """
    rng = np.random.default_rng(13)
    long_clip = unit(0) * 5 + rng.normal(scale=0.1, size=(10, DIM))
    shorts = [(unit(i) * 5 + rng.normal(scale=0.1, size=(1, DIM)), [0.0]) for i in (1, 2, 3)]
    index = build([(long_clip, np.arange(10) * 2.0)] + shorts)
    weighted = sum(len(index.pooled[index.clip_rows(i)]) * index.clip_vector(i) for i in range(4)) / 13
    np.testing.assert_allclose(index.mean, weighted, atol=1e-5)
    unweighted = np.mean([index.clip_vector(i) for i in range(4)], axis=0)
    assert not np.allclose(index.mean, unweighted, atol=1e-2)
    to_long = np.linalg.norm(index.mean - index.clip_vector(0))
    assert all(to_long < np.linalg.norm(index.mean - index.clip_vector(i)) for i in (1, 2, 3))


def test_clip_vector_averages_raw_rows_then_normalises():
    rows = np.array([10 * unit(0), unit(1)])
    clips = [(rows, [0.0, 2.0]), ([unit(2) + 0.0 * unit(3)], [0.0]), ([unit(1) + unit(2)], [0.0])]
    index = build(clips)
    plain = search.prepare(index, centre=False)
    raw_mean = rows.mean(axis=0)
    np.testing.assert_allclose(plain.clip_vectors[0], raw_mean / np.linalg.norm(raw_mean), atol=1e-6)
    wrong = (plain.matrix[:2]).mean(axis=0)  # the average of normalised rows would be [.5, .5, ...]
    assert not np.allclose(plain.clip_vectors[0], wrong / np.linalg.norm(wrong), atol=1e-3)
    centred = search.prepare(index, centre=True)
    cm = raw_mean - index.mean.astype(np.float64)
    np.testing.assert_allclose(centred.clip_vectors[0], cm / np.linalg.norm(cm), atol=1e-5)


@pytest.mark.parametrize("centre", [False, True], ids=["plain", "centred"])
def test_rows_are_unit_length(centre):
    space = search.prepare(build(random_clips(3, 4, seed=14)), centre=centre)
    norms = np.linalg.norm(space.matrix, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(space.clip_vectors, axis=1), 1.0, atol=1e-6)


def test_window_at_nearest_centre_tie_and_clamp():
    # Windows start at 0, 1, 2, 3 and last 2 s, so their centres are 1, 2, 3, 4.
    clips = [(np.eye(DIM)[:4], [0.0, 1.0, 2.0, 3.0])]
    space = search.prepare(build(clips), centre=False)
    assert search.window_at(space, 0, 1.4) == 0
    assert search.window_at(space, 0, 1.5) == 0  # exact midpoint: the earlier window
    assert search.window_at(space, 0, 1.6) == 1  # second half of the overlap: the later window
    assert search.window_at(space, 0, 99.0) == 3
    assert search.window_at(space, 0, -5.0) == 0
    with pytest.raises(ValueError):
        search.window_at(space, 0, float("nan"))
    with pytest.raises(IndexError):
        search.window_at(space, 5, 1.0)


def test_nearest_window_rule_directly():
    starts = np.array([0.0, 1.0, 2.0, 3.0])
    assert search.nearest_window(starts, 2.0, 1.5) == 0  # exact midpoint of centres 1 and 2: earlier
    assert search.nearest_window(starts, 2.0, 1.5000001) == 1
    assert search.nearest_window(starts, 2.0, 100.0) == 3
    assert search.nearest_window(starts, 2.0, -100.0) == 0
    assert search.nearest_window(np.array([5.0]), 2.0, 0.0) == 0
    with pytest.raises(ValueError):
        search.nearest_window(starts, 2.0, float("inf"))
    with pytest.raises(ValueError):
        search.nearest_window(np.array([]), 2.0, 1.0)


@pytest.mark.parametrize("centre", [False, True], ids=["plain", "centred"])
def test_nearest_clips_excludes_query_clip_and_orders_ties(centre):
    clips = random_clips(5, 2, seed=15)
    clips[3] = (clips[1][0], clips[1][1])  # clips 1 and 3 are identical
    clips[4] = (clips[1][0], clips[1][1])
    space = search.prepare(build(clips), centre=centre)
    res = search.nearest_clips(space, 0, k=10)
    assert [h.clip for h in res.hits if h.clip != 2] == sorted(h.clip for h in res.hits if h.clip != 2)
    assert 0 not in [h.clip for h in res.hits] and len(res.hits) == 4
    from_one = search.nearest_clips(space, 1, k=2)
    assert [h.clip for h in from_one.hits] == [3, 4]
    assert from_one.hits[0].score == pytest.approx(1.0, abs=1e-6)


def test_errors_for_k_and_unknown_clip_or_window():
    space = search.prepare(build(random_clips(2, 2, seed=16)), centre=False)
    for bad in (0, -1):
        with pytest.raises(ValueError):
            search.nearest_moments(space, 0, 0, k=bad)
        with pytest.raises(ValueError):
            search.nearest_clips(space, 0, k=bad)
    for clip, window in ((0, 2), (0, -1), (2, 0), (-1, 0)):
        with pytest.raises(IndexError):
            search.nearest_moments(space, clip, window, k=1)
    for clip in (2, -1):
        with pytest.raises(IndexError):
            search.nearest_clips(space, clip, k=1)


def test_empty_index_returns_empty_results_with_reason():
    space = search.prepare(build([], dim=DIM), centre=False)
    assert space.n_clips == 0 and space.matrix.shape == (0, DIM)
    query = search.query_vector(space, np.ones((1, DIM)))
    moments = search.search_moments(space, query, k=3)
    clips = search.search_clips(space, query, k=3)
    assert moments.hits == [] and moments.reason
    assert clips.hits == [] and clips.reason
    with pytest.raises(IndexError):
        search.nearest_moments(space, 0, 0, k=1)


@pytest.mark.parametrize("centre", [False, True], ids=["plain", "centred"])
def test_hits_carry_clip_local_window_and_index_wide_row(centre):
    index = build(random_clips(4, 3, seed=17, spacing=3.0))
    space = search.prepare(index, centre=centre)
    res = search.nearest_moments(space, 2, 1, k=8)
    assert res.hits
    for h in res.hits:
        assert h.row == index.clip_rows(h.clip).start + h.window
        assert index.clip[h.row] == h.clip
        assert h.start_time == index.window_start_time[h.row]
        assert h.score == pytest.approx(
            float(space.matrix[h.row] @ space.matrix[index.clip_rows(2).start + 1]), abs=1e-6
        )
    scores = [h.score for h in res.hits]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("centre", [False, True], ids=["plain", "centred"])
def test_query_vector_matches_in_index_clip_vector(centre):
    index = build(random_clips(4, 3, seed=18))
    space = search.prepare(index, centre=centre)
    for i in range(4):
        q = search.query_vector(space, index.pooled[index.clip_rows(i)])
        np.testing.assert_allclose(q.vector, space.clip_vectors[i], atol=1e-6)
    hits = search.search_clips(space, search.query_vector(space, index.pooled[index.clip_rows(1)]), k=4)
    assert hits.hits[0].clip == 1 and hits.hits[0].score == pytest.approx(1.0, abs=1e-6)
    moments = search.search_moments(space, search.query_vector(space, index.pooled[index.clip_rows(1)]), k=3)
    assert len(moments.hits) == 3


def test_query_vector_unavailable_and_shape():
    space = search.prepare(build(random_clips(2, 2, seed=19)), centre=False)
    zero = search.query_vector(space, np.zeros((2, DIM)))
    assert zero.vector is None and zero.reason
    assert search.search_moments(space, zero, k=2) == search.Result([], zero.reason)
    assert search.query_vector(space, np.zeros((0, DIM))).vector is None
    with pytest.raises(ValueError):
        search.query_vector(space, np.ones((2, DIM + 1)))
    with pytest.raises(ValueError):
        search.search_clips(space, zero, k=0)


JITTER = np.array([0.0, 0.003, 0.0, -0.003, 0.0, 0.0])


@pytest.mark.parametrize("centre", [False, True], ids=["plain", "centred"])
def test_same_clip_exclusion_ignores_start_time_jitter(centre):
    """Windows 1 and 3 share no frames, but jitter puts their start times 1.994 s apart."""
    rng = np.random.default_rng(20)
    vecs = rng.normal(size=(6, DIM)) * 0.3
    vecs[3] = unit(0)
    vecs[1] = unit(0) + 0.1 * unit(1)  # the best match for window 3 inside its own clip
    clips = [(vecs, np.arange(6) * 1.0 + JITTER)] + random_clips(1, 2, seed=21)
    index = build(clips)
    assert index.window_start_time[3] - index.window_start_time[1] < 2.0
    space = search.prepare(index, centre=centre)
    own = [h.window for h in search.nearest_moments(space, 0, 3, k=20, same_clip=True).hits if h.clip == 0]
    assert 1 in own and 2 not in own and 3 not in own and 4 not in own


@pytest.mark.parametrize("centre", [False, True], ids=["plain", "centred"])
def test_suppression_ignores_start_time_jitter(centre):
    """Windows 0 and 2 of a clip do not overlap even though jitter puts them 1.994 s apart."""
    q = unit(0)
    near = unit(0) + 0.05 * unit(1)
    far = unit(2)
    clip1 = np.array([near, far, near + 0.01 * unit(3)])
    clips = [([q], [0.0]), (clip1, np.array([0.003, 1.0, 1.997])), ([unit(1) * -1.0], [0.0])]
    index = build(clips)
    assert index.window_start_time[2] - index.window_start_time[0] < 2.0
    space = search.prepare(index, centre=centre)
    res = search.nearest_moments(space, 0, 0, k=2)
    assert sorted((h.clip, h.window) for h in res.hits) == [(1, 0), (1, 2)]


def _tie_index(seed: int, dim: int = 1024, n: int = 37):
    """37 single-window clips of width 1024 with exact copies of clip 0 in clips 7, 8 and 36."""
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    for c in (7, 8, n - 1):
        vecs[c] = vecs[0]
    return build([(vecs[i : i + 1], [0.0]) for i in range(n)], dim=dim)


@pytest.mark.parametrize("centre", [False, True], ids=["plain", "centred"])
@pytest.mark.parametrize("seed", range(40))
def test_exact_duplicates_tie_in_order_at_full_width(seed, centre):
    """Duplicate rows at matrix positions 7, 8 and the last must score equal and rank by clip."""
    space = search.prepare(_tie_index(seed), centre=centre)
    res = search.nearest_moments(space, 0, 0, k=3)
    assert [(h.clip, h.window) for h in res.hits] == [(7, 0), (8, 0), (36, 0)]
    assert len({h.score for h in res.hits}) == 1


@pytest.mark.parametrize("centre", [False, True], ids=["plain", "centred"])
@pytest.mark.parametrize("seed", range(40))
def test_exact_duplicate_clips_tie_in_order_at_full_width(seed, centre):
    space = search.prepare(_tie_index(seed), centre=centre)
    res = search.nearest_clips(space, 0, k=3)
    assert [h.clip for h in res.hits] == [7, 8, 36]
    assert len({h.score for h in res.hits}) == 1 and res.hits[0].score <= 1.0


def _many_duplicates(seed: int, copies: int, dim: int = 1024, n: int = 37):
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    vecs[1 : copies + 1] = vecs[0]
    return build([(vecs[i : i + 1], [0.0]) for i in range(n)], dim=dim)


@pytest.mark.parametrize("seed", range(40))
def test_cutoff_keeps_one_ulp_twins_of_the_best_score_plain(seed):
    """With twenty exact copies of the query and k=1, float32 scores differ by an ulp by position."""
    space = search.prepare(_many_duplicates(seed, 20), centre=False)
    assert [h.clip for h in search.nearest_moments(space, 0, 0, k=1).hits] == [1]


@pytest.mark.parametrize("seed", range(40))
def test_many_duplicate_clips_tie_in_clip_order_plain(seed):
    space = search.prepare(_many_duplicates(seed, 12), centre=False)
    assert [h.clip for h in search.nearest_clips(space, 0, k=3).hits] == [1, 2, 3]


def test_scores_are_clipped_to_unit_range():
    space = search.prepare(build(random_clips(3, 1, seed=22)), centre=False)
    for h in search.nearest_moments(space, 0, 0, k=3).hits:
        assert -1.0 <= h.score <= 1.0


def test_identical_rows_with_float32_accumulated_mean_are_unavailable_centred():
    """A stored mean off by a few float32 ulps must not make identical rows look distinct."""
    v = np.random.default_rng(23).normal(size=(1, DIM)).astype(np.float32) * 40
    index = build([(v, [0.0]) for _ in range(10)])
    sloppy = (v[0] + 3e-5).astype(np.float32)  # what a float32 accumulation might leave behind
    index = dataclasses.replace(index, mean=sloppy)
    index_file.check(
        {"pooled": index.pooled, "window_start_time": index.window_start_time, "clip": index.clip, "mean": index.mean},
        index.header,
    )
    space = search.prepare(index, centre=True)
    assert not space.available.any() and not space.clip_available.any()
    assert search.nearest_moments(space, 0, 0, k=2).reason


def test_min_norm_threshold_separates_tiny_residues():
    """Nine identical rows and one that differs by 4e-6: after centring the nine have norm 4e-7."""
    base = np.random.default_rng(24).normal(size=DIM).astype(np.float32)
    odd = base.copy()
    odd[0] += np.float32(4e-6)
    vecs = np.array([base] * 9 + [odd])
    space = search.prepare(build([(vecs[i : i + 1], [0.0]) for i in range(10)]), centre=True)
    assert space.available.tolist() == [False] * 9 + [True]


def test_unavailable_reason_names_centring_only_when_centred():
    x = unit(0)
    index = build([([x], [0.0]), ([-x], [0.0]), ([0 * x], [0.0])])
    centred = search.nearest_moments(search.prepare(index, centre=True), 2, 0, k=1).reason
    plain = search.nearest_moments(search.prepare(index, centre=False), 2, 0, k=1).reason
    assert "index mean" in centred and "index mean" not in plain


def test_default_similarity_is_plain_cosine():
    assert search.DEFAULT_SIMILARITY == "cosine"


def test_a_suppressed_hit_never_suppresses_another_plain():
    """Windows 0, 1 and 2 of one clip rank in that order; 1 overlaps both others and is dropped,
    but 2 does not overlap 0, so it stays."""
    q = unit(0)
    w0, w1, w2 = (unit(0) + d * unit(1) for d in (0.1, 0.2, 0.3))
    space = search.prepare(build([([q], [0.0]), ([w0, w1, w2], [0.0, 1.0, 2.0])]), centre=False)
    assert [(h.clip, h.window) for h in search.nearest_moments(space, 0, 0, k=5).hits] == [(1, 0), (1, 2)]


def test_scores_above_one_are_clipped_plain():
    """A stored row a little longer than unit length would score above 1 without the clip."""
    space = search.prepare(build([([unit(0)], [0.0]), ([unit(0)], [0.0])]), centre=False)
    matrix = space.matrix.copy()
    matrix[1] *= np.float32(1 + 1e-5)
    long_row = dataclasses.replace(space, matrix=matrix)
    assert float(matrix[1].astype(np.float64) @ matrix[0].astype(np.float64)) > 1.0
    (hit,) = search.nearest_moments(long_row, 0, 0, k=1).hits
    assert hit.score == 1.0


def test_k_is_capped():
    space = search.prepare(build(random_clips(3, 2, seed=3)), centre=False)
    assert len(search.nearest_moments(space, 0, 0, k=search.MAX_K).hits) == 2  # one per other clip
    with pytest.raises(ValueError, match="at most"):
        search.nearest_moments(space, 0, 0, k=search.MAX_K + 1)
    with pytest.raises(ValueError, match="at most"):
        search.nearest_clips(space, 0, k=search.MAX_K + 1)
