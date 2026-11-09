# jepa-scope

[![CI](https://github.com/ryangswe/jepa-scope/actions/workflows/ci.yml/badge.svg)](https://github.com/ryangswe/jepa-scope/actions/workflows/ci.yml)

jepa-scope shows you what a JEPA video encoder sees. Point it at a video and it opens a local viewer
with the model's patch features drawn over the picture, plus a timeline of how surprised the model
was by each moment. It runs Meta's pre-trained V-JEPA 2 on a laptop, so you need no training run, no
cluster and no CUDA.

```bash
uv run jepa-scope clip.mp4
```

The first run downloads the V-JEPA 2 ViT-L weights from Hugging Face (about 1.3 GB). After that, a
10-second clip embeds in about 3.5 seconds on an M3 Max. Results are cached under `~/.cache/jepa-scope`, keyed by the
video's hash, the settings, the device and dtype, and the jepa-scope output version, so opening the
same clip again takes a few seconds (mostly loading libraries) instead of re-embedding. A cached
file is reused only when its format version, output version and representation (described below)
match what the current settings would produce; otherwise it is re-embedded in place, with a message
saying why.

## Why this exists

A language model's output is text, so you can tell whether it worked by reading it. A JEPA encoder
outputs a tensor such as `float16[2048, 1024]` with nothing in it to say whether it is right. If the
frame sampling, crop or normalisation is off, the model still returns plausible-looking numbers.
jepa-scope makes those numbers visible, which turns "I got a tensor, now what?" into something you
can look at and check.

## What the viewer shows

- **Features (PCA).** Each 16×16-pixel patch is coloured by the first three principal components of
  its feature vector. When the encoder and preprocessing are working, the same object keeps roughly
  the same colour from frame to frame. This is the visualisation DINO made popular, and it doubles
  as a quick correctness check.
- **Surprise.** The model is shown the first half of each window and asked to predict the second
  half in latent space. The overlay shows the prediction error per patch, and the timeline shows
  the average per tubelet. Grey on the timeline marks the opening context, which the model never
  had to predict.
- **Colour and bar scale.** Both are relative to the clip you are looking at. The overlay spans the
  clip's 1st to 99th percentile of surprise and the bars span its lowest to highest tubelet, so
  even a clip where nothing happens will show some hot patches and some tall bars. Compare clips
  using the raw values in the facts panel.
- **Facts panel.** Model, token grid, sampling, device, dtype and a hash of the preprocessing
  recipe, so you can tell which settings produced what you are looking at.

The overlay covers only the centre square, because that is the region the model sees after its
resize and centre crop.

Keyboard shortcuts: `Space` plays or pauses, `1`, `2` and `3` switch the overlay, and the arrow keys
step one tubelet at a time.

## How surprise is computed

Two details matter here, and getting either wrong still produces numbers without any error.

The first is about what the predictor sees. The Hugging Face `VJEPA2Model.forward` runs the encoder
on the whole clip and applies the context mask afterwards. The "context" tokens have therefore
already attended to the frames the predictor is asked to predict. That is fine for feature
extraction but makes prediction error meaningless as a surprise signal. jepa-scope encodes the
context frames on their own and then calls the predictor. `tests/test_pipeline.py` checks that
two clips with the same past and different futures get identical predictions, and it also checks
that the Hugging Face default path fails that test.

The second is about the target. V-JEPA 2 is trained to predict *layer-normalised* target features
with an L1 loss (`forward_target` in `app/vjepa/train.py` of `facebookresearch/vjepa2`). The raw
encoder output has a standard deviation around 3.3 while predictions sit around 0.65, so comparing
them directly gives a flat line. jepa-scope compares against `layer_norm(target)`, as training does.

Expect a floor of about 0.55 to 0.6 on natural video, since the model is never exactly right. What
carries information is the change relative to that floor. On an archery clip spliced onto a bowling
clip, surprise stays between about 0.57 and 0.62 within each scene and rises to 0.65 to 0.67 for
the four tubelets right after the cut. `tests/test_real_model.py` reproduces this. It is one
example, not a benchmark, so treat the size of that margin as anecdotal.

Expect a small sawtooth with a period of one second at the default settings. Within each
window, the tubelets furthest from the context are a little harder to predict, by about 0.01 on
the clip above. That is a property of the method, not of the video, so ignore steps of that size.

V-JEPA 2's predictor was trained mostly on masks that span the whole clip in time. Predicting
the future from the past alone is outside that training, so treat surprise as a useful signal, not
a calibrated probability. Screen recordings and static title cards are far from its training data
too, and give weak signals.

## The `.latent` file

`jepa-scope embed clip.mp4 -o clip.latent` writes a plain
[safetensors](https://github.com/huggingface/safetensors) file. Any safetensors reader can open the
tensors. The metadata holds one JSON header under the key `jepa_scope`, which records the model,
token grid, window settings, preprocessing recipe and hash, crop rectangle and source video hash.
`jepa_scope.latent_file.load` checks that header before handing the tensors over.

The header's `representation` block says exactly how a window vector was made: the model, an
identity of its weights, the preprocessing hash, the token layer, the normalisation before pooling,
the pooling, the window settings and the output version, with a hash over them. The weights are
identified by content, not by a version label: for a downloaded model, by the content hashes
Hugging Face stores for its files, so an upstream edit to the model card usually changes nothing
(at most one clip is re-embedded while the new snapshot is filled in, and on systems where Hugging
Face copies files instead of linking them, the copies' times stand in for the hashes); for a model
in a local folder, by its config and each weight file's size and modification time. The
snapshot's commit is recorded for information as `run.model_revision`. Two files with the same
representation hash hold vectors made the same way. The device, dtype and `max_seconds` are
recorded too (`run.device`, `run.dtype`, `source.max_seconds`); vectors from float16 and float32
runs differ slightly, so the cache, which is keyed by them, reuses a file only when they match as
well. The current format is version 2; a version-1 file is refused with a message to re-embed the
video with `jepa-scope embed`.

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `patches` | `[windows, T, H, W, D]` float16 | Encoder output for every window, reshaped to its grid. Left out with `--no-patches` |
| `pooled` | `[windows, D]` float32 | Mean over tokens, for search and simple classifiers |
| `timeline_pooled` | `[tubelets, D]` float32 | Mean over each timeline tubelet's patches, from the same tokens as `pooled` |
| `window_start_time` | `[windows]` | Start of each window in seconds |
| `timeline_time` | `[tubelets]` | Start of each tubelet on the flattened timeline |
| `timeline_window` | `[tubelets]` | Which window each timeline tubelet was taken from |
| `timeline_pca_rgb` | `[tubelets, H, W, 3]` uint8 | PCA colours used by the viewer |
| `timeline_surprise` | `[tubelets, H, W]` float32 | Prediction error per patch, NaN where there was no prediction |

Windows overlap by half. With the defaults (16 frames at 8 fps), each window covers 2 seconds,
becomes an 8×16×16 grid of 1,024-dimensional tokens, and starts 1 second after the previous one.
Every tubelet after the first second is predicted by exactly one window.

The token order is time first, then row, then column. `jepa_scope.grid` is the one place that
knows this, and `tests/test_grid.py` pins it against the real `Conv3d` in transformers by lighting a
single pixel and checking which token changes.

## Commands

```bash
jepa-scope clip.mp4                # embed (cached in ~/.cache/jepa-scope) and open the viewer
jepa-scope embed clip.mp4 -o x.latent --fps 8 --window 16
jepa-scope view x.latent           # reopen a saved file; --video if the clip moved
```

Useful options are `--max-seconds` for long videos, `--device cpu|mps|cuda` and
`--dtype float32|float16|bfloat16`. Run `jepa-scope embed --help` for the full list.

Videos with rotation metadata, which includes most phone footage, are turned upright before the
model sees them. Frames are downscaled while decoding, so decoded frames take about 3.6 MB per second
of video at the defaults whatever the source resolution, on top of about 2 GB for the model. The full `patches` tensor adds about
4 MB per second to the `.latent` file. Pass `--no-patches` to leave it out, since the viewer does
not use it. A few minutes of video is practical today. Longer videos work but take
proportionally more memory and disk.

`jepa-scope view` refuses to pair a `.latent` file with a video whose SHA-256 differs from the one
recorded when it was made. A video with fewer frames per second than `--fps` is refused too,
because each window would silently cover more time than the header says.

## Indexing and searching a folder

```bash
jepa-scope index ~/videos                  # embed every video (cached) and write ~/videos/jepa-scope.index
jepa-scope search ~/videos beach.mp4 --at 12
jepa-scope search ~/videos beach.mp4 --by clip
jepa-scope search ~/videos ~/elsewhere/query.mp4
```

`jepa-scope index` embeds every video in a folder through the same cache as the other commands and
writes one index file, `jepa-scope.index`, beside the videos. It looks for `.mp4`, `.mov`, `.m4v`,
`.mkv`, `.webm` and `.avi` files in any letter case, ignores names that start with a dot, and only
enters subfolders with `--recursive`, which never follows a symlinked folder. It takes the same
embedding options as `jepa-scope embed`. Unlike `embed`, it leaves the large `patches` tensor out of
the cache files it writes, because search does not use it, and `--patches` keeps it.

Running it again refreshes the index. A video whose size and modification time match the index, and
whose cache file is still valid, is kept without being read. Any other video is hashed again and
embedded only if the cache holds nothing for its contents, so renaming or touching a file costs one
hash, and a file that was deleted is dropped from the index. A rerun with nothing new to embed does
not load the model. `--verify` hashes every video whatever its size and time say, which catches a
file replaced by another of the same size and time. If the model, the embedding settings, the
device or the dtype change, the index is rebuilt from scratch, and the program says why.

The index remembers two choices. `--recursive` and `--centre` (see below) stay in effect on later
runs until `--no-recursive` or `--no-centre` turns them off, even when the index is rebuilt. Turning
`--recursive` off drops the clips in subfolders from the index, and turning it on again adds them
back from the cache without embedding them again, though each one is hashed again.

One bad file does not stop a run. A video that cannot be read, is too short for one window, has
fewer frames per second than `--fps`, has no video stream, cannot be decoded, runs out of memory or
makes the model produce invalid values is reported, skipped and listed in the index with the
reason. When the cause is the file's content (too short, too few frames per second, no video
stream, not decodable), the file is not tried again until it changes, `--verify` is given or
jepa-scope is upgraded. Any other failure is tried again on the next run.

`jepa-scope search` takes the folder (or the index file) and a query. A query that is a path to
one of the indexed clips is searched with the vectors already in the index, and so is a file with
the same contents as an indexed clip, which the output then says. If the query is not an existing
file, it is looked up as a clip's path inside the folder (`sub/clip.mp4`). Any other video is
embedded through the cache with the model and settings the index was built with. It is refused if
this machine's model files would produce a different representation, or if the device the index
was built on is not available here. By default the results are the moments in other clips most like the
window at `--at` seconds, with overlapping windows of one clip collapsed to the best one;
`--same-clip` also lists moments from the query's own clip, and `--by clip` compares whole clips. Each
window is two seconds long at the defaults and a new one starts every second. Similarity is plain
cosine between the mean tokens of two windows, and `jepa-scope index --centre` subtracts the
folder's mean vector first.

Only an index written beside the videos moves with the folder: it records the folder as "the
directory I am in", so the folder can be copied or moved and searched again as is. An index written
elsewhere with `--out` records the folder's absolute path. If that folder moves, pass its new
location to `jepa-scope search` with `--source-root`.

Each clip is embedded in one piece, so its decoded frames and the tokens of all its windows are in
memory at once. By our estimates, clips of up to about 10 minutes each fit on a laptop with 32 GB
of memory, and up to about 5 minutes with 16 GB. Use `--max-seconds` to index only the start of
longer clips.

## Development

```bash
uv sync
uv run pytest                      # fast tests on a tiny random model, no downloads
JEPA_SCOPE_MODEL_TESTS=1 uv run pytest tests/test_real_model.py -s   # real weights, about 1.3 GB
cd web && npm install && npm run build   # build the viewer into src/jepa_scope/static
./scripts/build.sh                 # viewer + wheel, and checks the wheel contains the viewer
```

For frontend work, run `uv run jepa-scope view x.latent --no-browser` in one terminal and
`npm run dev` in `web/` in another. Vite forwards `/api` and `/video` to the Python server.

The Python side lives in `src/jepa_scope/`. `video.py` decodes with PyAV, which has wheels for
Apple Silicon and avoids the usual `decord` install problems. `model.py` wraps the encoder and
predictor, `embed.py` turns a video into a `.latent` file, and `server.py` serves it to the
browser as raw bytes. The viewer in `web/src/` uploads the whole timeline into two WebGL2 texture
arrays once, so scrubbing only changes a uniform.

## Roadmap

The first week's milestones are done. The model runs on a laptop, the token grid is pinned by a
test, the PCA overlay and surprise timeline work, the file format is defined, and the viewer runs
in the browser. [docs/roadmap.md](docs/roadmap.md) has the full programme: the developer
bottleneck this addresses, the missing toolchain layers, the three planned tools of which
jepa-scope is the first, and what comes next. Settled design choices and their reasons are in
[docs/decisions.md](docs/decisions.md).

## License

Apache 2.0. jepa-scope downloads V-JEPA 2 weights from Hugging Face at run time and does not
redistribute them. Meta releases V-JEPA 2 under the MIT license, according to the model card and
the `facebookresearch/vjepa2` repository.
