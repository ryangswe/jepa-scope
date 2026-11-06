"""Fast checks of the T0 retrieval-evaluation helpers on hand-built vectors (no model, no data)."""

from __future__ import annotations

import numpy as np
import pytest

from . import retrieval_eval as rv


def test_transition_windows_are_the_ones_crossing_a_segment_boundary():
    boundaries = [5, 10, 15, 20, 25, 30]
    assert [s for s in range(29) if rv.transition(float(s), boundaries, 2.0)] == [4, 9, 14, 19, 24]


def test_bars_use_integer_counts():
    assert rv.at_least(30, 50, 0.60) and not rv.at_least(29, 50, 0.60)
    assert rv.at_least(8, 50, 0.15) and not rv.at_least(7, 50, 0.15)


@pytest.mark.parametrize(
    ("counts", "baseline", "localised", "expected"),
    [
        # A candidate below the bar is not eligible, even if it is the simplest.
        ({"a": 29, "b": 20, "c": 31, "d": 10}, 10, "abcd", ("c", ["c"])),
        # Within two clips of the best eligible count, the simplest wins (order a, b, d, c).
        ({"a": 40, "b": 42, "c": 42, "d": 41}, 10, "abcd", ("a", ["a", "b", "c", "d"])),
        # The margin over the colour baseline and the localisation veto both remove candidates.
        ({"a": 40, "b": 42, "c": 42, "d": 41}, 33, "acd", ("d", ["c", "d"])),
        # A gap of three clips is not a tie.
        ({"a": 39, "b": 0, "c": 42, "d": 0}, 10, "abcd", ("c", ["a", "c"])),
        # Nothing eligible escalates.
        ({"a": 40, "b": 40, "c": 40, "d": 40}, 10, "", (None, [])),
    ],
)
def test_decision_rule(counts, baseline, localised, expected):
    assert rv.choose(counts, 50, baseline, {c: c in localised for c in rv.CANDIDATES}) == expected


def test_clip_vector_averages_raw_rows_and_normalises_after():
    big = np.array([[10.0, 0.0], [10.0, 0.0]])
    small = np.array([[0.0, 1.0], [0.0, 1.0]])
    clip = np.concatenate([big, small])
    got = rv.clip_matrix([clip], None)[0]
    # Averaging normalised rows would give the diagonal; averaging raw rows leans toward the big ones.
    assert got[0] > 0.99 and np.isclose(np.linalg.norm(got), 1.0)


def test_nearest_clip_never_comes_from_the_same_source_group():
    vecs = rv.normalise(np.array([[1.0, 0.0], [1.0, 0.01], [0.0, 1.0]]))
    assert rv.top1(vecs, ["x", "x", "y"]).tolist() == [2, 2, 1]
    hits, _ = rv.clip_hits(vecs, ["p", "p", "q"], ["x", "y", "z"])
    assert hits.tolist() == [True, True, False]


def test_window_rate_excludes_the_query_clip():
    # Each clip's windows are identical to each other, so counting the own clip would score 1.0.
    clips = [np.array([[1.0, 0.0], [1.0, 0.0]]), np.array([[0.0, 1.0], [0.0, 1.0]])]
    assert rv.window_rate(clips, None, ["p", "p"], ["x", "y"]) == 1.0
    assert rv.window_rate(clips, None, ["p", "q"], ["x", "y"]) == 0.0


def test_zero_vectors_are_refused():
    with pytest.raises(ValueError, match="zero or not finite"):
        rv.normalise(np.zeros((1, 3)))


def test_colour_histogram_is_a_512_bin_distribution_of_the_centre_crop():
    frames = np.zeros((2, 300, 400, 3), np.uint8)
    frames[:, 150, 200] = 255  # one white pixel inside the centre crop
    hist = rv.colour_histogram(frames)
    assert hist.shape == (512,) and np.isclose(hist.sum(), 1.0)
    assert hist[511] == pytest.approx(1 / (256 * 256)) and hist[0] == pytest.approx(1 - 1 / (256 * 256))


def test_even_crops_odd_sides():
    assert rv.even(np.zeros((292, 519, 3))).shape == (292, 518, 3)


def _composite(rng, dim=16):
    """29 windows, one per second, each in its own random direction (so position is recoverable)."""
    starts = np.arange(29, dtype=float)
    return starts, rng.normal(size=(29, dim))


def _query(rng, comp, i, t):
    """The window at ``5 i + t`` seconds of the composite, slightly perturbed."""
    return comp[5 * i + t] + 0.05 * rng.normal(size=comp.shape[1])


def test_localisation_passes_a_position_aware_signal_and_fails_the_clip_average_mutation():
    rng = np.random.default_rng(0)
    starts, comp = _composite(rng)
    others = [rng.normal(size=(9, 16)) for _ in range(5)]
    queries = [(i, float(t), _query(rng, comp, i, t), 0) for i in range(4) for t in range(4)]
    good = rv.localise(queries, comp, starts, others, None, 2.0)
    assert all(r["pass"] for r in good)
    mutated = rv.localise([(i, t, v, o) for i, t, v, o in queries], rv.clip_average(comp), starts, others, None, 2.0)
    assert not all(r["pass"] for r in mutated)


def test_localisation_masks_the_query_clip_and_rejects_hits_outside_the_composite():
    rng = np.random.default_rng(1)
    starts, comp = _composite(rng)
    # The query's own clip holds an exact copy of the query; it must be masked out.
    query = _query(rng, comp, 2, 1)
    own = np.stack([query] * 9)
    res = rv.localise([(2, 1.0, query, 0)], comp, starts, [own], None, 2.0)
    assert res[0]["pass"] and res[0]["hit_composite_start"] is not None
    # Another clip holding a closer copy wins, and that is a miss, not a pass.
    res = rv.localise([(2, 1.0, query, 0)], comp, starts, [own, own], None, 2.0)
    assert not res[0]["pass"] and res[0]["hit_composite_start"] is None
