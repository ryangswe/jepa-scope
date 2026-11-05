"""Command line entry point.

jepa-scope clip.mp4              embed (cached) and open the viewer
jepa-scope embed clip.mp4 -o x   write a .latent file
jepa-scope view x.latent         open the viewer for an existing .latent file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import webbrowser
from pathlib import Path

from . import OUTPUT_VERSION, __version__, latent_file

SUBCOMMANDS = {"embed", "view"}
CACHE_DIR = Path.home() / ".cache" / "jepa-scope"


def _add_embed_args(p: argparse.ArgumentParser) -> None:
    from .model import DEFAULT_MODEL

    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id (default: %(default)s)")
    p.add_argument("--fps", type=float, default=8.0, help="frames sampled per second (default: %(default)s)")
    p.add_argument("--window", type=int, default=16, help="frames per window (default: %(default)s)")
    p.add_argument("--batch", type=int, default=2, help="windows per forward pass (default: %(default)s)")
    p.add_argument("--max-seconds", type=float, default=None, help="only embed the first N seconds")
    p.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    p.add_argument(
        "--no-patches",
        action="store_true",
        help="skip the full patch features (about 4 MB per second of video); the viewer does not need them",
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


def _cache_path(video_sha256: str, args) -> Path:
    """Cache file for this video and these settings.

    The key covers everything that changes the output: the output version (bumped whenever the
    energy, preprocessing or windowing changes), the model, sampling, and the device and dtype actually
    used, so a CPU float32 run and an MPS float16 run never share a file.
    """
    from .model import pick_device, pick_dtype

    device = pick_device(args.device)
    settings = {
        "output_version": OUTPUT_VERSION,
        "model": args.model,
        "fps": args.fps,
        "window": args.window,
        "max_seconds": args.max_seconds,
        "device": device.type,
        "dtype": str(pick_dtype(device, args.dtype)),
        "patches": not args.no_patches,
    }
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    return CACHE_DIR / f"{video_sha256[:16]}-{digest}.latent"


def _cache_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        header = latent_file.read_header(path)
    except (latent_file.LatentFileError, OSError, ValueError) as exc:
        print(f"ignoring unreadable cache file {path}: {exc}", flush=True)
        return False
    return header["run"].get("output_version") == OUTPUT_VERSION


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
        out = _cache_path(video_sha256, args)
        if args.refresh or not _cache_is_valid(out):
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

    args = p.parse_args(argv)
    if args.cmd == "embed":
        if not args.video.exists():
            p.error(f"{args.video} does not exist")
        _embed(args.video, args.out or Path(args.video.stem + ".latent"), args)
    elif args.cmd == "view":
        _serve(args.latent, args.video, args.port, not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
