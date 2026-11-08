"""Command line entry point.

jepa-scope clip.mp4              embed (cached) and open the viewer
jepa-scope embed clip.mp4 -o x   write a .latent file
jepa-scope view x.latent         open the viewer for an existing .latent file
jepa-scope index folder          embed every video in a folder (cached) and write its index
jepa-scope search folder clip    list the moments or clips in the index most like a clip
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import webbrowser
from pathlib import Path

from . import __version__, cache, latent_file
from .cache import EmbedSettings

SUBCOMMANDS = {"embed", "view", "index", "search"}


def _add_embed_args(p: argparse.ArgumentParser, patches_flag: str = "--no-patches") -> None:
    from .model import DEFAULT_MODEL

    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id (default: %(default)s)")
    p.add_argument("--fps", type=float, default=8.0, help="frames sampled per second (default: %(default)s)")
    p.add_argument("--window", type=int, default=16, help="frames per window (default: %(default)s)")
    p.add_argument("--batch", type=int, default=2, help="windows per forward pass (default: %(default)s)")
    p.add_argument("--max-seconds", type=float, default=None, help="only embed the first N seconds")
    p.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    if patches_flag == "--no-patches":
        p.add_argument(
            "--no-patches",
            action="store_true",
            help="skip the full patch features (about 4 MB per second of video); the viewer does not need them",
        )
    else:
        p.add_argument(
            "--patches",
            action="store_true",
            help="keep the full patch features of newly embedded clips (about 4 MB per second of video)",
        )


def _add_view_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")


def _embed(video: Path, out: Path, args) -> None:
    from .embed import embed_video
    from .model import Encoder

    print(f"loading {args.model} ...", flush=True)
    encoder = Encoder.from_pretrained(args.model, device=args.device, dtype=args.dtype)
    tensors, header = embed_video(
        video,
        encoder,
        sample_fps=args.fps,
        window=args.window,
        batch_size=args.batch,
        max_seconds=args.max_seconds,
        keep_patches=not args.no_patches,
        log=lambda m: print(m, flush=True),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    latent_file.save(out, tensors, header)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def _serve(latent: Path, video: Path | None, port: int, open_browser: bool, video_sha256: str | None = None) -> None:
    import uvicorn

    from .server import create_app

    try:
        app = create_app(latent, video, video_sha256)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None
    url = f"http://127.0.0.1:{port}/"
    print(f"viewer at {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def _model_provider(settings: EmbedSettings) -> cache.PretrainedModel:
    """How the index and search commands reach the model (replaced in tests)."""
    return cache.PretrainedModel(settings)


def _index(args) -> None:
    from .indexing import build_or_refresh

    if not args.folder.is_dir():
        raise SystemExit(f"error: {args.folder} is not a folder")
    settings = EmbedSettings.from_args(args)
    build_or_refresh(
        args.folder,
        _model_provider(settings),
        settings,
        out=args.out,
        recursive=args.recursive,
        patches=args.patches,
        centre=args.centre,
        verify=args.verify,
        log=lambda m: print(m, flush=True),
    )


def _search(args) -> int:
    from . import index_file, search
    from .indexing import INDEX_NAME

    if args.k < 1 or args.k > search.MAX_K:
        raise SystemExit(f"error: -k must be between 1 and {search.MAX_K}")
    if args.by == "clip" and (args.at is not None or args.same_clip):
        raise SystemExit("error: --at and --same-clip apply to --by moment only")
    if args.at is not None and not math.isfinite(args.at):
        raise SystemExit("error: --at must be a finite number of seconds")
    index_path = args.index / INDEX_NAME if args.index.is_dir() else args.index
    if not index_path.exists():
        hint = f"; run `jepa-scope index {args.index}` first" if args.index.is_dir() else ""
        raise SystemExit(f"error: {index_path} does not exist{hint}")
    try:
        index = index_file.load(index_path)
    except (index_file.IndexFileError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from None
    root = index.resolve_root(index_path, args.source_root)
    if not root.is_dir():
        if args.source_root is not None:
            raise SystemExit(f"error: --source-root {root} is not a folder")
        raise SystemExit(
            f"error: the folder this index was built from, {root}, is not there; "
            "if it moved, pass its new location with --source-root DIR"
        )
    clips = index.header["clips"]
    if not clips:
        print("nothing was indexed, so there is nothing to search")
        return 0
    space = search.prepare(index)
    names = [c["path"] for c in clips]
    at = 0.0 if args.at is None else args.at

    clip, query_sha = _indexed_clip(args.query, names, [c["sha256"] for c in clips], root)
    if clip is None and query_sha is None:
        raise SystemExit(f"error: {args.query} is neither a clip in the index nor a video file")
    if clip is not None:
        if query_sha is not None:
            print(f"{args.query} has the same contents as the indexed clip {names[clip]}; searching as that clip")
        if args.by == "clip":
            print(f"clips like {names[clip]}:")
            result = search.nearest_clips(space, clip, args.k)
        else:
            window = search.window_at(space, clip, at)
            start = float(space.start_time[space.clip_start[clip] + window])
            print(f"moments like {names[clip]} at {at:g}s (window {window}, from {start:.2f}s):")
            result = search.nearest_moments(space, clip, window, args.k, same_clip=args.same_clip)
    else:
        assert query_sha is not None
        if args.same_clip:
            raise SystemExit("error: --same-clip applies to a clip in the index only")
        query = _outside_query(args.query, query_sha, index.header)
        if args.by == "clip":
            print(f"clips like {args.query}:")
            result = search.search_clips(space, search.query_vector(space, query.pooled), args.k)
        else:
            window = search.nearest_window(query.window_start_time, space.window_seconds, at)
            start = float(query.window_start_time[window])
            print(f"moments like {args.query} at {at:g}s (window {window}, from {start:.2f}s):")
            vector = search.query_vector(space, query.pooled[window : window + 1])
            result = search.search_moments(space, vector, args.k)

    if not result.hits:
        print(f"no results: {result.reason}")
        return 0
    print(f"{'rank':>4}  {'score':>7}  {'time':>8}  clip")
    for rank, hit in enumerate(result.hits, 1):
        time = f"{hit.start_time:.2f}s" if isinstance(hit, search.Hit) else "-"
        print(f"{rank:>4}  {hit.score:>7.4f}  {time:>8}  {names[hit.clip]}")
    return 0


def _indexed_clip(query: Path, names: list[str], shas: list[str], root: Path) -> tuple[int | None, str | None]:
    """Which indexed clip ``query`` names, as ``(clip, sha256)``.

    An existing file is taken as a file, never as a name. It is the clip whose path it is, when
    its absolute path (with the folders above it resolved, but not a symlink at the end) lies under
    the root; then the result is ``(clip, None)``. Otherwise it is hashed, and the result is
    ``(clip, sha256)`` for the first clip with the same contents, or ``(None, sha256)`` for a video
    that is not in the index. Only an argument that is not an existing file is matched against the
    clips' relative paths, as written (``sub/clip.mp4``); ``(None, None)`` means no match.
    """
    if query.is_file():
        path = Path(os.path.abspath(query))
        path = path.parent.resolve() / path.name
        try:
            rel = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = None
        if rel in names:
            return names.index(rel), None
        try:
            sha = latent_file.file_sha256(query)
        except OSError as exc:
            raise SystemExit(f"error: {query} cannot be read ({exc})") from None
        return (shas.index(sha) if sha in shas else None), sha
    text = query.as_posix()
    if text in names:
        return names.index(text), None
    return None, None


def _outside_query(video: Path, video_sha256: str, header: dict):
    """The window vectors of a video that is not in the index, embedded (through the cache) with
    the index's model and settings. Exits with a message when that cannot be done here."""
    from .indexing import ClipSkipped, find_query, load_query
    from .model import device_available

    settings = EmbedSettings.from_index(header)
    rep = header["representation"]
    if not device_available(settings.device):
        raise SystemExit(
            f"error: {video} is not in the index, and embedding it the way the index was built needs "
            f"the {settings.device} device ({settings.dtype}), which is not available here; "
            "a clip in the index can still be the query"
        )
    provider = _model_provider(settings)
    got = provider.representation()["hash"]
    if got != rep["hash"]:
        raise SystemExit(
            f"error: {video} cannot be embedded the way this index was: the model here gives "
            f"representation {got}, the index holds {rep['hash']} (other weights or preprocessing); "
            "rebuild the index with `jepa-scope index` or search with the model it was built with"
        )
    log = lambda m: print(m, flush=True)  # noqa: E731
    found = find_query(video_sha256, provider, settings, log=log)
    if found is not None:
        return found
    try:
        return load_query(video, provider, settings, log=log, video_sha256=video_sha256)
    except ClipSkipped as exc:
        raise SystemExit(f"error: {video} cannot be embedded: {exc}") from None


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        p = argparse.ArgumentParser(prog="jepa-scope", description="Embed a video (cached) and open the viewer.")
        p.add_argument("video", type=Path)
        p.add_argument("--refresh", action="store_true", help="ignore the cache and embed again")
        _add_embed_args(p)
        _add_view_args(p)
        args = p.parse_args(argv)
        if not args.video.exists():
            p.error(f"{args.video} does not exist")
        video_sha256 = latent_file.file_sha256(args.video)
        settings = EmbedSettings.from_args(args)
        out = cache.cache_path(video_sha256, settings, patches=not args.no_patches)
        # The representation is described (a few seconds of imports) only when a file exists to check.
        if args.refresh or not out.exists() or not cache.cache_is_valid(out, cache.expected_representation(settings)):
            _embed(args.video, out, args)
        else:
            print(f"using cached {out}", flush=True)
        _serve(out, args.video, args.port, not args.no_browser, video_sha256)
        return 0

    p = argparse.ArgumentParser(
        prog="jepa-scope", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--version", action="version", version=f"jepa-scope {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("embed", help="write a .latent file for a video")
    pe.add_argument("video", type=Path)
    pe.add_argument("-o", "--out", type=Path, default=None, help="output path (default: <video name>.latent here)")
    _add_embed_args(pe)

    pv = sub.add_parser("view", help="open the viewer for a .latent file")
    pv.add_argument("latent", type=Path)
    pv.add_argument("--video", type=Path, default=None, help="source video, if it moved since embedding")
    _add_view_args(pv)

    pi = sub.add_parser(
        "index",
        help="embed every video in a folder (cached) and write the folder's index",
        description="Embed every video in a folder through the cache and write jepa-scope.index beside "
        "them. A rerun only embeds what is new or changed.",
    )
    pi.add_argument("folder", type=Path)
    pi.add_argument("--out", type=Path, default=None, help="index path (default: <folder>/jepa-scope.index)")
    pi.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include subfolders, never entering symlinked ones (default: as the existing index, else no)",
    )
    pi.add_argument(
        "--centre",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="search with cosine similarity centred on the folder mean (default: as the existing index, else no)",
    )
    pi.add_argument("--verify", action="store_true", help="hash every file, even when its size and time are unchanged")
    _add_embed_args(pi, patches_flag="--patches")

    ps = sub.add_parser(
        "search",
        help="list the moments or clips in an index most like a clip",
        description="Search a folder's index with one of its clips, or with any video, which is then "
        "embedded the way the index was.",
    )
    ps.add_argument("index", type=Path, help="an indexed folder, or an index file")
    ps.add_argument("query", type=Path, help="a clip in the index (its path relative to the folder) or a video file")
    ps.add_argument("--at", type=float, default=None, help="time in the query clip, in seconds (default: 0)")
    ps.add_argument("-k", type=int, default=10, help="number of results (default: %(default)s)")
    ps.add_argument("--by", choices=["moment", "clip"], default="moment", help="what to compare (default: %(default)s)")
    ps.add_argument("--same-clip", action="store_true", help="also list moments from the query's own clip")
    ps.add_argument("--source-root", type=Path, default=None, help="where the indexed folder is now, if it moved")

    args = p.parse_args(argv)
    if args.cmd == "embed":
        if not args.video.exists():
            p.error(f"{args.video} does not exist")
        _embed(args.video, args.out or Path(args.video.stem + ".latent"), args)
    elif args.cmd == "view":
        _serve(args.latent, args.video, args.port, not args.no_browser)
    elif args.cmd == "index":
        _index(args)
    elif args.cmd == "search":
        return _search(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
