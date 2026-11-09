"""End-to-end check against the real V-JEPA 2 weights. Opt-in, because it downloads about 1.3 GB.

    JEPA_SCOPE_MODEL_TESTS=1 uv run pytest tests/test_real_model.py -s

The test splices five seconds of an archery clip onto five seconds of a bowling clip (both from the
small Kinetics sample set the Hugging Face V-JEPA 2 docs use) and checks that surprise is highest
for the tubelets right after the cut. This is the property the viewer depends on. An earlier
version of the energy, without the layer norm on targets, ran without errors and drew a flat line.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from jepa_scope.embed import embed_video
from jepa_scope.video import decode

from .conftest import write_video
from .retrieval_eval import fit_frames as _fit

pytestmark = pytest.mark.skipif(
    os.environ.get("JEPA_SCOPE_MODEL_TESTS") != "1", reason="set JEPA_SCOPE_MODEL_TESTS=1 to run"
)

CLIPS = (
    "val/archery/-Qz25rXdMjE_000014_000024.mp4",
    "val/bowling/--dVV4_CSvw_000033_000043.mp4",
)


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


# ----------------------------------------------------------------------------------------------
# T0 of the Weeks 3 and 4 plan: the retrieval experiment on cached vectors (exploratory).
#
#   JEPA_SCOPE_MODEL_TESTS=1 uv run pytest tests/test_real_model.py -s -k "retrieval_candidates and val"
#   JEPA_SCOPE_MODEL_TESTS=1 JEPA_SCOPE_EVAL_DEFAULT=<a|b|c|d> \
#       uv run pytest tests/test_real_model.py -s -k "retrieval_candidates and train"
#
# The plan fixes the data, candidates, metrics, controls, bars and decision rule. The test asserts
# only sanity properties; everything else is printed and written to retrieval-<split>.json in
# JEPA_SCOPE_EVAL_DIR (default: a pytest temporary directory).


def _eval_dir(tmp_path_factory) -> Path:
    named = os.environ.get("JEPA_SCOPE_EVAL_DIR")
    out = Path(named) if named else tmp_path_factory.mktemp("retrieval")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_once(path: Path, frames: np.ndarray, fps: int) -> Path:
    # Written once and reused, so the cache (keyed by the file's bytes) is hit on a rerun.
    # Written to a temporary name first, so an interrupted write is never reused.
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".partial.mp4")
        write_video(tmp, frames, fps=fps)
        os.replace(tmp, path)
    return path


def _composite_sources(clips):
    """T0's composite sources, in order: the first clip of archery, bowling, flying kite and
    marching (the four queried segments), then the second archery and bowling clips."""
    from . import retrieval_eval as rv

    by_class = {k: [c for c in clips if c.label == k] for k in rv.CLASSES}
    return [by_class[k][0] for k in ("archery", "bowling", "flying_kite", "marching")] + [
        by_class["archery"][1],
        by_class["bowling"][1],
    ]


def _composite(sources, out_dir: Path) -> tuple[Path, list[int]]:
    """The 30-second localisation composite: five seconds of each source, written once (so its
    cache file is reused), and the frame count of each segment."""
    from . import retrieval_eval as rv

    seconds = rv.SEGMENT_SECONDS
    parts = [
        _fit(decode(c.path, sample_fps=rv.COMPOSITE_FPS, max_seconds=seconds).frames[: rv.COMPOSITE_FPS * seconds])
        for c in sources
    ]
    path = _write_once(out_dir / "composite.mp4", np.concatenate(parts), fps=rv.COMPOSITE_FPS)
    return path, [len(p) for p in parts]


def _rate_line(name: str, count: int, n: int, extra: str = "") -> str:
    return f"  {name:<28} {count:>3}/{n}  {count / n:.2f}{extra}"


@pytest.mark.parametrize("split", ["val", "train"])
def test_retrieval_candidates(split, tmp_path_factory):
    from huggingface_hub import HfApi, snapshot_download

    from jepa_scope.model import DEFAULT_MODEL
    from jepa_scope.video import window_starts

    from . import retrieval_eval as rv

    default = os.environ.get("JEPA_SCOPE_EVAL_DEFAULT")
    if split == "train" and default not in rv.CANDIDATES:
        pytest.skip("the confirmation split runs with JEPA_SCOPE_EVAL_DEFAULT set to the chosen default")
    selecting = split == "val"
    candidates = rv.CANDIDATES if selecting else (default,)
    t_start = time.perf_counter()
    out_dir = _eval_dir(tmp_path_factory)
    report = rv.Report(split)
    failures = []  # sanity checks, asserted only after the tables are printed and written

    # Frozen revisions, manifest and labels.
    model_dir = Path(snapshot_download(DEFAULT_MODEL, local_files_only=True))
    assert model_dir.name == rv.MODEL_REVISION, f"model snapshot is {model_dir.name}, not the frozen one"
    names = HfApi().list_repo_files(rv.DATASET, repo_type="dataset", revision=rv.DATASET_REVISION)
    videos = [n for n in names if n.endswith(".mp4") and n.split("/")[0] in ("train", "val")]
    ids = [Path(n).stem.rsplit("_", 2)[0] for n in videos]
    assert len(videos) == 100 and len(set(ids)) == 100, "two files share a YouTube id"
    root = Path(
        snapshot_download(rv.DATASET, repo_type="dataset", revision=rv.DATASET_REVISION, allow_patterns=f"{split}/*")
    )
    clips = rv.manifest(root, split)
    assert len(clips) == len(rv.CLASSES) * rv.CLIPS_PER_CLASS
    assert all(sum(c.label == k for c in clips) == rv.CLIPS_PER_CLASS for k in rv.CLASSES)
    report.put("dataset", {"repo": rv.DATASET, "revision": rv.DATASET_REVISION, "split": split})
    report.put("model", {"id": DEFAULT_MODEL, "revision": rv.MODEL_REVISION})
    report.put("download_mb", round(sum(c.path.stat().st_size for c in clips) / 1e6, 1))

    # Embed through the cache rule.
    emb = rv.CachedEmbedder()
    edge = emb.shortest_edge()
    kept, skipped = [], []
    for clip in clips:
        try:
            kept.append(rv.embed_clip(emb, clip))
        except (ValueError, RuntimeError) as exc:
            skipped.append({"clip": clip.name, "reason": str(exc)})
            print(f"skipped {clip.name}: {exc}")
    labels = [e.clip.label for e in kept]
    groups = [e.clip.source_id for e in kept]
    n = len(kept)
    per_class = {k: labels.count(k) for k in rv.CLASSES}
    report.put("skipped", skipped)
    if skipped and selecting:
        failures.append(f"{len(skipped)} selection clip(s) skipped; the manifest is frozen, so this escalates")
    report.put("per_class", per_class)
    report.put("chance", rv.chance(labels))
    parity = max(e.parity for e in kept)
    report.put("parity_max_relative_l2", parity)
    if not parity < 1e-2:
        failures.append(f"(a) recomputed from patches differs from pooled by {parity:.3g}")

    # Candidates: clip-level top-1 (primary), window-level top-1, permutation nulls.
    centre = rv.set_centre([e.vectors["a"] for e in kept])

    def wins(c, items=kept):
        return [e.vectors[c] for e in items]

    def cen(c, value=centre):
        return value if c == "b" else None

    hits, counts, best_idx, nulls = {}, {}, {}, {}
    cand_report = {}
    for c in candidates:
        mat = rv.clip_matrix(wins(c), cen(c))
        hits[c], best_idx[c] = rv.clip_hits(mat, labels, groups)
        counts[c] = int(hits[c].sum())
        nulls[c] = rv.permutation_null(mat, labels, groups, rv.N_PERMUTATIONS)
        cand_report[c] = {
            "clip_top1_count": counts[c],
            "clip_top1_rate": counts[c] / n,
            "window_top1_rate": rv.window_rate(wins(c), cen(c), labels, groups),
            "per_class_hits": {k: int(hits[c][np.asarray(labels) == k].sum()) for k in rv.CLASSES},
            "null_mean": float(nulls[c].mean()),
            "null_p95": float(np.percentile(nulls[c], 95)),
        }
    report.put("candidates", cand_report)
    report.put("paired", rv.paired(hits) if len(hits) > 1 else {})

    # Colour baselines and the repeated-frame control.
    mids, wholes, controls = [], [], []
    for e in kept:
        v = decode(e.clip.path, sample_fps=rv.SAMPLE_FPS, short_edge=edge)
        starts = window_starts(len(v.frames), rv.WINDOW, rv.WINDOW // 2)
        mid = v.frames[len(v.frames) // 2]
        mids.append(rv.colour_histogram(mid[None]))
        wholes.append(rv.colour_histogram(v.frames[: starts[-1] + rv.WINDOW]))
        ctrl = out_dir / "controls" / split / f"{Path(e.clip.name).stem}.mp4"
        frames = np.repeat(rv.even(mid)[None], rv.WINDOW, axis=0)
        _write_once(ctrl, frames, fps=int(rv.SAMPLE_FPS))
        controls.append(rv.embed_clip(emb, rv.Clip(name=ctrl.name, label=e.clip.label, path=ctrl)))
    if not all(len(c.starts) == 1 for c in controls):
        failures.append("a repeated-frame control is not exactly one window")
    baseline_counts = {}
    for name, hist in (("colour_middle_frame", mids), ("colour_whole_clip", wholes)):
        baseline_counts[name] = int(rv.clip_hits(rv.normalise(np.stack(hist)), labels, groups)[0].sum())
    baseline_count = max(baseline_counts.values())
    control_counts = {
        c: int(rv.clip_hits(rv.clip_matrix(wins(c, controls), cen(c)), labels, groups)[0].sum()) for c in candidates
    }
    report.put("baselines", baseline_counts)
    report.put("repeated_frame_control", control_counts)

    # Temporal localisation and its mutation (selection set only).
    localised = {}
    if selecting:
        order = _composite_sources([e.clip for e in kept])
        sources = [next(e for e in kept if e.clip is c) for c in order]
        comp_path, lengths = _composite(order, out_dir)
        if not all(n_ == rv.COMPOSITE_FPS * rv.SEGMENT_SECONDS for n_ in lengths):
            failures.append(f"a composite segment is short: {lengths} frames")
        comp = rv.embed_clip(emb, rv.Clip(name="composite.mp4", label="composite", path=comp_path))
        report.put("composite_window_starts", comp.starts.tolist())
        if not (len(comp.starts) == 29 and np.allclose(comp.starts, np.arange(29), atol=0.05)):
            failures.append(f"the composite does not have 29 windows at 0..28 s: {comp.starts.round(2).tolist()}")
        own = {id(s): kept.index(s) for s in sources[:4]}
        window_seconds = rv.WINDOW / rv.SAMPLE_FPS
        loc_report = {}
        for c in candidates:
            queries = [
                (i, float(s.starts[w]), s.vectors[c][w], own[id(s)])
                for i, s in enumerate(sources[:4])
                for w in range(rv.QUERY_WINDOWS)
            ]
            res = rv.localise(queries, comp.vectors[c], comp.starts, wins(c), cen(c), window_seconds)
            mutated_queries = [
                (i, float(s.starts[w]), rv.clip_average(s.vectors[c])[w], own[id(s)])
                for i, s in enumerate(sources[:4])
                for w in range(rv.QUERY_WINDOWS)
            ]
            mutated = rv.localise(
                mutated_queries,
                rv.clip_average(comp.vectors[c]),
                comp.starts,
                [rv.clip_average(w) for w in wins(c)],
                cen(c),
                window_seconds,
            )
            localised[c] = all(r["pass"] for r in res)
            loc_report[c] = {
                "passed": sum(r["pass"] for r in res),
                "of": len(res),
                "mutation_passed": sum(r["pass"] for r in mutated),
                "queries": res,
            }
            if all(r["pass"] for r in mutated):
                failures.append(f"the clip-average mutation passes localisation for ({c})")
        report.put("localisation", loc_report)

    # The decision rule (selection set), or the confirmation of the chosen default.
    if selecting:
        choice, eligible = rv.choose(counts, n, baseline_count, localised)
        report.put("eligible", eligible)
    else:
        choice = default
    report.put("chosen_default", choice)
    bar_ok = None
    if choice is not None:
        bar_ok = rv.at_least(counts[choice], n, rv.BAR_RATE) and rv.at_least(
            counts[choice] - baseline_count, n, rv.BAR_MARGIN
        )
        report.put("chosen_meets_bars", bar_ok)
        p95 = float(np.percentile(nulls[choice], 95))
        if not counts[choice] / n > p95:
            failures.append(
                f"({choice}) rate {counts[choice] / n:.2f} does not beat its null 95th percentile {p95:.2f}"
            )

    # Print the tables.
    print(f"\n=== retrieval, {split} split: {n} clips, chance {rv.chance(labels):.2f}, per class {per_class}")
    print(f"skipped: {skipped or 'none'}; parity (max relative L2) {parity:.2e}; embedded this run {emb.embedded}")
    print("clip-level top-1 (count, rate) | window-level top-1 | null mean / 95th")
    for c in candidates:
        r = cand_report[c]
        extra = f" | {r['window_top1_rate']:.2f} | {r['null_mean']:.2f} / {r['null_p95']:.2f} | {r['per_class_hits']}"
        print(_rate_line(f"({c})", counts[c], n, extra))
    for name, cnt in baseline_counts.items():
        print(_rate_line(name, cnt, n))
    for c, cnt in control_counts.items():
        print(_rate_line(f"repeated frame ({c})", cnt, n))
    if len(hits) > 1:
        print("paired:", report.data["paired"])
    if selecting:
        for c, r in report.data["localisation"].items():
            print(
                f"localisation ({c}): {r['passed']}/{r['of']} pass; "
                f"clip-average mutation {r['mutation_passed']}/{r['of']}"
            )
        print(f"eligible: {report.data['eligible']}; chosen default: {choice or 'none, escalate'}")
    else:
        print(f"confirmation of ({choice}): {'meets' if bar_ok else 'MISSES'} the bars")
    shown = choice or "a"
    print(f"per clip, top hit for ({shown}):")
    for i, e in enumerate(kept):
        hit = kept[int(best_idx[shown][i])]
        mark = "ok" if hits[shown][i] else "--"
        print(f"  {mark} {e.clip.name:<52} -> {hit.clip.label:<12} {e.width}x{e.height} {e.kbps:.0f} kbps")
    report.put(
        "per_clip",
        [
            {
                "clip": e.clip.name,
                "top_hit": kept[int(best_idx[shown][i])].clip.name,
                "width": e.width,
                "height": e.height,
                "kbps": e.kbps,
            }
            for i, e in enumerate(kept)
        ],
    )
    report.put("seconds", round(time.perf_counter() - t_start, 1))
    report.put("embedded_this_run", emb.embedded)
    report.put("failures", failures)
    print(f"wrote {report.write(out_dir)} in {report.data['seconds']} s")
    assert not failures, "; ".join(failures)


def test_search_reproduces_experiment(tmp_path_factory):
    """T0's result through the product code: the folder indexer, the index file and ``search``.

    The 50 val clips are copied out of the Hugging Face cache into a folder with one subfolder per
    class (so the index is never written beside the cache), indexed recursively with the CLI
    defaults, and scored with ``nearest_clips``. The vectors come from a fresh embedding run, so
    the clip-level count may differ from T0's by one clip. Localisation runs on a second index
    that also holds T0's composite, through ``nearest_moments`` as the viewer calls it. The
    clip-level count and the window-level rate must match T0's within one clip (0.02), and
    localisation may miss at most one of T0's 16 passes.
    """
    import json
    import shutil

    from huggingface_hub import snapshot_download

    from jepa_scope import index_file, search
    from jepa_scope.cache import EmbedSettings, PretrainedModel
    from jepa_scope.indexing import INDEX_NAME, build_or_refresh
    from jepa_scope.model import DEFAULT_MODEL

    from . import retrieval_eval as rv

    t_start = time.perf_counter()
    model_dir = Path(snapshot_download(DEFAULT_MODEL, local_files_only=True))
    assert model_dir.name == rv.MODEL_REVISION, f"model snapshot is {model_dir.name}, not the frozen one"
    root = Path(
        snapshot_download(rv.DATASET, repo_type="dataset", revision=rv.DATASET_REVISION, allow_patterns="val/*")
    )
    clips = rv.manifest(root, "val")
    assert len(clips) == len(rv.CLASSES) * rv.CLIPS_PER_CLASS
    out_dir = _eval_dir(tmp_path_factory)
    settings = EmbedSettings(model=DEFAULT_MODEL, fps=rv.SAMPLE_FPS, window=rv.WINDOW)
    model = PretrainedModel(settings)
    result = {"t0": rv.T0_VAL}

    def folder_of(name: str, extra: list[Path] = ()) -> Path:
        folder = tmp_path_factory.mktemp(name)
        for c in clips:
            (folder / c.label).mkdir(exist_ok=True)
            shutil.copyfile(c.path, folder / c.label / c.path.name)
        for p in extra:
            shutil.copyfile(p, folder / p.name)
        return folder

    def build(folder: Path):
        t0 = time.perf_counter()
        report = build_or_refresh(folder, model, settings, recursive=True, log=lambda m: print(m, flush=True))
        assert report.skipped == [], report.skipped
        index = index_file.load(folder / INDEX_NAME)
        return index, search.prepare(index), report, round(time.perf_counter() - t0, 1)

    # Clip-level and window-level retrieval.
    index, space, report, seconds = build(folder_of("val"))
    header = index.header
    assert header["embedding"]["similarity"] == search.DEFAULT_SIMILARITY == "cosine"
    assert header["representation"]["pooling"] == "mean_all" and header["representation"]["token_norm"] == "none"
    paths = [c["path"] for c in header["clips"]]
    labels = [p.split("/")[0] for p in paths]
    n = len(paths)
    assert n == len(clips)
    clip_hits = []
    # T0 excluded the query's YouTube source group; nearest_clips excludes only the query clip.
    # These coincide because every clip in the frozen manifest has its own id (T0 asserts it).
    for i in range(n):
        hits = search.nearest_clips(space, i, 1).hits
        clip_hits.append(bool(hits) and labels[hits[0].clip] == labels[i])
    count = sum(clip_hits)
    per_class = {k: sum(h for h, lab in zip(clip_hits, labels, strict=True) if lab == k) for k in rv.CLASSES}
    window_hits = []
    for i in range(n):
        rows = index.clip_rows(i)
        for w in range(rows.stop - rows.start):
            hits = search.nearest_moments(space, i, w, 1).hits
            window_hits.append(bool(hits) and labels[hits[0].clip] == labels[i])
    result.update(
        clips=n,
        windows=int(len(index.pooled)),
        embedded=report.embedded,
        index_seconds=seconds,
        clip_top1_count=count,
        clip_top1_rate=count / n,
        per_class_hits=per_class,
        window_top1_rate=sum(window_hits) / len(window_hits),
        per_clip=[{"clip": p, "hit": h} for p, h in zip(paths, clip_hits, strict=True)],
    )

    # Localisation on a second index that also holds the composite.
    comp_path, lengths = _composite(_composite_sources(clips), out_dir)
    assert all(n_ == rv.COMPOSITE_FPS * rv.SEGMENT_SECONDS for n_ in lengths), lengths
    index2, space2, report2, seconds2 = build(folder_of("val-composite", [comp_path]))
    paths2 = [c["path"] for c in index2.header["clips"]]
    comp = paths2.index("composite.mp4")
    comp_starts = index2.window_start_time[index2.clip_rows(comp)]
    assert len(comp_starts) == 29 and np.allclose(comp_starts, np.arange(29), atol=0.05), comp_starts
    window_seconds = rv.WINDOW / rv.SAMPLE_FPS
    boundaries = [rv.SEGMENT_SECONDS * k for k in range(1, len(comp_starts) + 1)]
    queries = []
    for i, source in enumerate(_composite_sources(clips)[:4]):
        own = paths2.index(f"{source.label}/{source.path.name}")
        starts = index2.window_start_time[index2.clip_rows(own)]
        for w in range(rv.QUERY_WINDOWS):
            target = rv.SEGMENT_SECONDS * i + float(starts[w])
            row = {"segment": i, "query_start": float(starts[w]), "target": target}
            for key, same in (("cross_clip", False), ("same_clip", True)):
                top = search.nearest_moments(space2, own, w, 1, same_clip=same).hits[0]
                on_comp = top.clip == comp
                crossing = on_comp and rv.transition(top.start_time, boundaries, window_seconds)
                row[key] = {
                    "hit": paths2[top.clip],
                    "hit_start": top.start_time,
                    "transition": crossing,
                    # A transition window was not a candidate in T0, so a hit on one counts as a miss.
                    "pass": on_comp and not crossing and abs(top.start_time - target) <= rv.TOLERANCE_SECONDS + 1e-6,
                    "own_clip": top.clip == own,
                }
            queries.append(row)
    loc = {
        key: {
            "passed": sum(q[key]["pass"] for q in queries),
            "transition_hits": sum(q[key]["transition"] for q in queries),
            "own_clip_hits": sum(q[key]["own_clip"] for q in queries),
            "of": len(queries),
        }
        for key in ("cross_clip", "same_clip")
    }
    result.update(localisation=loc, localisation_queries=queries, composite_index_seconds=seconds2)
    result.update(embedded_composite_index=report2.embedded, seconds=round(time.perf_counter() - t_start, 1))
    bars = rv.at_least(count, n, rv.BAR_RATE) and rv.at_least(count - rv.T0_VAL["colour_baseline"], n, rv.BAR_MARGIN)
    result["meets_bars"] = bars

    print(f"\n=== search reproduces T0 (val): {n} clips, {result['windows']} windows")
    print(f"index built in {seconds} s, embedded {report.embedded}; composite index {seconds2} s")
    print(_rate_line("clip-level top-1", count, n, f"  (T0 {rv.T0_VAL['clip_top1_count']}/{n}) {per_class}"))
    print(f"  window-level top-1 rate {result['window_top1_rate']:.4f} (T0 {rv.T0_VAL['window_top1_rate']:.4f})")
    for key, r in loc.items():
        print(
            f"localisation {key}: {r['passed']}/{r['of']} pass; {r['transition_hits']} on a transition window; "
            f"{r['own_clip_hits']} in the query's own clip"
        )
    path = out_dir / "search-reproduction-val.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {path} in {result['seconds']} s")

    assert abs(count - rv.T0_VAL["clip_top1_count"]) <= 1, f"{count}/{n} against T0's {rv.T0_VAL['clip_top1_count']}"
    assert bars, f"{count}/{n} misses the bars (rate {rv.BAR_RATE}, margin {rv.BAR_MARGIN} over the colour baseline)"
    # nearest_moments carries the exclusion, overlap and tie rules that nearest_clips does not, so
    # its results are held to T0's too (same_clip=True is descriptive only).
    rate, t0_rate = result["window_top1_rate"], rv.T0_VAL["window_top1_rate"]
    assert abs(rate - t0_rate) <= 0.02, f"window-level top-1 {rate:.4f} against T0's {t0_rate:.4f}"
    passed, t0_passed = loc["cross_clip"]["passed"], rv.T0_VAL["localisation_passed"]
    assert passed >= t0_passed - 1, f"localisation {passed}/{len(queries)} against T0's {t0_passed}"
