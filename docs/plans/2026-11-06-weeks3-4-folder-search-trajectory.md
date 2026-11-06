# jepa-scope Weeks 3 and 4: folder indexing, nearest-neighbour search and the latent trajectory view

- **Date:** 2026-11-06
- **Status:** Draft
- **Planned by:** planner (Fable 5.1), briefed by the lead session (Opus 5.5)
- **Spec / source:** `docs/roadmap.md`, "Next" item 1 and the project section's MVP schedule; the
  2026-11-05 process decision that milestones follow the roadmap's weekly order
- **Branch:** `week3-folder-search` (T1–T7), then `week4-trajectory` (T8–T10)

## Goal

Make jepa-scope work on a folder of clips instead of one clip. `jepa-scope <folder>` embeds every
video in the folder once (through the existing result cache), writes one index file, and opens the
viewer with a clip list, "similar moments" and "similar clips" search, and, in Week 4, a 3D view of
the open clip's path through latent space drawn through a cloud of every window in the folder.
Search is brute-force cosine over mean-pooled window embeddings, and it is checked on natural video
against a bar set in this plan before anything is measured.

Week 3 (T1–T7) delivers indexing and search end to end, from the `.latent` change to the viewer
panels, with the real-model check. Week 4 (T8–T10) delivers the trajectory view and the closeout.
The owner can stop after Week 3 and the repository is consistent.

This is one plan for both weeks because the roadmap names "Weeks 3 and 4" as one milestone and the
three deliverables share one contract: the folder index and the multi-clip server feed both the
search panels and the trajectory view, and the `.latent` change that the trajectory needs is
cheapest before the owner's folders are indexed, not after.

This plan does not build patch-level ("where in the frame") search, an approximate-nearest-neighbour
library, thumbnails in the result lists, embedding health metrics (Weeks 5 and 6), or any change to
how embeddings are computed. Everything runs on the laptop; no task needs a GPU.

## Approach

**The `.latent` file gains one tensor (T1).** `timeline_pooled`, `float32[G, D]`, is the mean over
the H×W patch features of each timeline tubelet (the `tl_feats` array `embed.py` already builds and
then discards after PCA). It costs 4 KB per tubelet at ViT-L (about 160 KB for a 10-second clip)
and is what the trajectory view plots. Because the folder indexer embeds without patches by default
(below), it is the only per-tubelet vector a folder clip keeps, so it has to be in the file.
`FORMAT_VERSION` goes to 2 and `load` keeps refusing other versions with its existing message;
`OUTPUT_VERSION` goes to 3 so cached version-1 files are re-embedded rather than reported as
unreadable. No embedding, energy or PCA computation changes. Doing this first means every clip
indexed in Week 3 already carries the tensor Week 4 needs.

**The index is one self-contained safetensors file (T2),** `jepa-scope.index` in the indexed folder
by default, with the same shape of contract as `.latent`: a validated JSON header under the
metadata key `jepa_scope` and a few tensors.

- Header: `format: "jepa-scope.index"`, `format_version: 1` (`INDEX_FORMAT_VERSION` in a new
  `src/jepa_scope/index_file.py`), `model_id`, `hidden_size`,
  `embedding: {"kind": "window_mean", "normalise": "centred-cosine"}`, the `window` block and
  `preprocess` hash copied from the clips' headers, `output_version`, `root` (the absolute folder
  path when built), and `clips`, a list of `{path (relative to root, forward slashes), sha256,
  size, mtime, duration, width, height, latent (absolute path of the clip's .latent file), rows:
  [start, stop)}`.
- Tensors: `pooled float32[N, D]` (every clip's window vectors stacked in clip order),
  `window_start_time float64[N]`, `clip int32[N]`, and `mean float32[D]` (the mean of `pooled`,
  used for centring).
- Validation checks that the row ranges tile `[0, N)` in order, that `clip[r]` agrees with the
  ranges, that `hidden_size` matches `D`, and that every clip's `latent` header agrees with the
  index's `model_id`, `window` and preprocess hash when it is opened.

For 200 ten-second clips the file is about 7 MB. Answering a search never needs the `.latent`
files; viewing a hit does.

**Search is brute force in NumPy (T3),** in `src/jepa_scope/search.py`. Vectors are centred by the
index `mean` and L2-normalised once at load; a query is one matrix–vector product and an
`argpartition`. Two query types share that primitive.

- "Moments like this moment": the query is one window (the window owning the current tubelet, from
  `timeline_window`). The results are windows across the folder, with the query window and any
  window of the same clip overlapping it excluded. Within one clip, a hit that overlaps a
  higher-ranked hit is dropped, so a list of ten hits is ten distinct moments.
- "Clips like this clip": a clip's vector is the mean of its window vectors (centred and normalised
  after averaging); results are the other clips.

Centring is there because mean-pooled transformer features share a large common direction that
compresses raw cosine into a narrow band; T5 measures raw against centred on natural video and
records both. A query over 30,000 windows (about 3,000 ten-second clips) is a 120 MB matrix and
about 30 ms. Brute force stops being comfortable around half a million windows (a few hundred hours
of video, 2 GB resident, about half a second per query). At that point an approximate-nearest-
neighbour library replaces the body of `search.py` behind the same function signatures, and the
index file does not change.

**The indexer (T4)** in `src/jepa_scope/indexing.py` walks the folder (not recursively by default;
`--recursive` opts in) for `.mp4 .mov .m4v .mkv .webm .avi` and skips anything under `.`-prefixed
names. For each video it finds or makes its `.latent` through the existing cache rule: it reuses a
with-patches cache file if one exists, otherwise a without-patches one, otherwise it embeds with
`keep_patches=False` (saving 4 MB per window; `--patches` keeps them).

- Refresh is incremental. A file whose size and mtime match its index entry is kept without
  re-hashing; otherwise it is hashed, and a changed hash means re-embedding. Files that disappeared
  are dropped.
- An index whose `model_id`, window settings, preprocess hash or `output_version` differ from the
  current run is rebuilt from scratch with a printed reason.
- Progress prints per clip. An interrupted run loses only the index file, because every embedded
  clip is already in the cache, so rerunning resumes.
- Embedding dominates the cost: about 0.35 s per second of video on an M3 Max, so an hour of video
  takes about 20 minutes.

The CLI gains `jepa-scope index <folder> [--out PATH] [--recursive] [--patches] <embed args>` and
`jepa-scope search <folder|index> <clip-in-index | query.mp4> [--at SECONDS] [-k 10]
[--by moment|clip]` (a query video not in the index is embedded through the cache). The default
command accepts a folder (`jepa-scope <folder>` indexes, refreshing, then serves), and
`jepa-scope view` accepts a folder, an index file or a `.latent`.

**The server serves many clips (T6).** `create_app` takes an index path, a folder, or a single
`.latent`, which it wraps as a one-clip in-memory index so the viewer sees one shape in both modes.
Clip tensors load lazily on first request through an LRU of eight. Endpoints:

- `GET /api/meta` returns `{mode: "folder"|"single", index: {root, model_id, format_version,
  windows: N, embedding}, clips: [{id, name, path, duration, width, height, windows}]}`.
- `GET /api/clips/{id}/meta` returns today's per-clip `{header, timeline}` plus
  `timeline.window: int[]` and `windows.start_time: number[]`.
- `GET /api/clips/{id}/pca` and `/surprise` return bytes in the unchanged layout.
- `GET /api/clips/{id}/video` keeps the existing SHA-256 refusal, with the index's size-and-mtime
  shortcut so a 2 GB file is not re-hashed on every open. A mismatch returns 409 with "run
  `jepa-scope index` again".
- `GET /api/search?clip=&window=&k=` and `GET /api/search?clip=&k=&by=clip` return
  `{hits: [{clip, window, start_time, score}]}`.
- Week 4 adds `GET /api/clips/{id}/trajectory?fit=folder|clip`, returning `float32[G, 3]` bytes,
  and `GET /api/cloud`, returning `float32[N, 3]` bytes subsampled to at most 50,000 points, with
  the chosen rows listed in `/api/meta`.

The folder fit is a 3-component PCA (`pca.fit`, which pins signs) on the index's centred `pooled`.
`timeline_pooled` is projected with the same mean and components, so every clip and the cloud share
axes and a window's cloud point is exactly the centroid of its tubelets' path points. `fit=clip`
fits on the clip's own `timeline_pooled`; it is the only option in single mode. The viewer's
`web/src/types.ts` changes with every one of these.

**The viewer (T7, T9).** `App` loads `/api/meta`, picks the clip from the URL hash (`#clip=3`) or
the first, then loads that clip's meta, bytes and video. Switching clips swaps the video source and
rebuilds the `Overlay`, which is already keyed on the data. New components:

- `ClipList.tsx`: name and duration; click to open.
- `SearchPanel.tsx`: "Similar moments" at the current tubelet and "Similar clips" for the open
  clip. A row shows the clip name, time and score, and a click opens the clip and seeks.
- Week 4 adds `Trajectory.tsx`: a Three.js scene (a pinned exact version, plus `@types/three`, and
  `OrbitControls` from the addons). The cloud is drawn as grey `Points`, and the open clip's path
  as a line through its tubelet points, coloured by mean surprise with the same inferno ramp
  (context tubelets grey). A marker on the current tubelet follows playback, search hits are drawn
  in an accent colour, hovering shows the tubelet's time, and clicking seeks the video. The fit
  toggle (folder or clip) sits on the panel.

**Shared contracts that change together in this plan:** `.latent` (`latent_file.py`, `embed.py`,
`server.py`, and now `indexing.py`); the index file (`index_file.py`, `indexing.py`, `search.py`,
`server.py`); and `/api/*` with `web/src/types.ts`. Repository invariants kept: no test media or
weights in the repo, output-changing edits bump `OUTPUT_VERSION`, the viewer ships in the wheel
(`scripts/build.sh` still checks it), no new Python dependencies (NumPy only; `uv lock --check`
stays green), and one new npm dependency (`three`).

## Alternatives considered

- **Patch-level search ("regions like this region")** — rejected for now. It needs the `patches`
  tensor for every indexed clip (4 MB per window, about 7 GB for 200 clips) and is a different
  product question. The index format leaves room for it (a later `embedding.kind`).
- **Per-tubelet search instead of per-window** — rejected for search. Tubelet features are
  contextualised by their window, adjacent tubelets are near-duplicates that would crowd every
  result list, and the window is the unit the encoder saw as one clip. Kept for the trajectory,
  where resolution matters (0.25 s per point at the defaults against 1 s per window).
- **Raw cosine, no centring** — kept as the measured comparison in T5 rather than the default, for
  the reason in Approach. If T5 shows raw is better by five points or more at top-1, the default
  flips in a one-line change and the log says so.
- **An approximate-nearest-neighbour library (FAISS, hnswlib) now** — rejected. It is a dependency
  with native wheels for a scale no user has; the plan states where brute force breaks.
- **Index only under `~/.cache/jepa-scope`** — rejected as the default. It is invisible, not
  portable with the folder, and the owner cannot tell what was indexed. `--out` covers read-only
  folders. The index is self-contained for search, so moving the folder keeps search working;
  viewing then re-embeds what the cache lacks.
- **SQLite for the index** — rejected. Embeddings would be blobs, and the safetensors-plus-header
  contract already exists and is readable by any safetensors tool, which the roadmap's layer B
  wants.
- **Computing per-tubelet means at serve time from `patches`** — rejected, because folder clips have
  no patches by default.
- **The server embedding missing `.latent` files on demand** — rejected. It would put the model in
  the viewer process; `index` is the one place that embeds, and `jepa-scope <folder>` runs it
  before serving.
- **A 2D SVG projection instead of Three.js** — the cheaper option (no dependency, about a day
  less). Rejected because the roadmap names a Three.js 3D view and the folder cloud needs depth to
  stay legible. It remains the fallback if the owner declines the dependency.
- **Raw WebGL2 for the 3D view, reusing `overlay.ts`** — rejected. The overlay is a textured quad;
  orbit controls, picking and lines would be a few hundred new lines to maintain for no gain over
  Three.js.

Decision records to write in T10: the index file as a versioned contract beside the videos; search
as brute-force centred cosine over window means; `.latent` format version 2 with `timeline_pooled`
(version 1 files are re-embedded, not migrated); and Three.js in the viewer.

## Risks and open questions

The questions for the owner are collected at the end of this plan with recommendations; the risks
here stay with the implementer.

- **The fast near-duplicate test may be flaky on a random encoder** — the tiny random V-JEPA 2 is
  deterministic, so a byte-identical copy embeds identically (score 1.0, rank 1), and a
  constant-grey clip must rank last; those are the planted checks. A noisy near-duplicate is not
  used, because a one-layer random encoder gives no guarantee on it.
- **Mean-pooled features may group classes weakly** — T5's bar is set here, before measuring. It is
  either met or escalated with the numbers; it is never lowered in the same session.
- **The size-and-mtime shortcut can keep a stale entry** when a file is replaced with the same size
  and mtime. The viewer's SHA-256 check on open catches it and names the fix. Documented, not
  solved.
- **Browser memory for the cloud** — capped at 50,000 points by subsampling in the server.
- **Three.js adds about 0.5 MB to the wheel** (tree-shaken ESM). Acceptable; the build check is
  unchanged.
- **Relative paths on Windows** — stored with forward slashes (`PurePosixPath`), joined back with
  `Path`.
- **Interrupted indexing** loses only the index file (see Approach).
- **The real-model check downloads about 100 MB** of Kinetics-mini (50 validation clips) and takes
  about five minutes on an M3 Max. It is opt-in like the existing one and never runs in CI.
- **Port 8765 and MPS** are single-instance. A second worktree passes `--port`, and only one
  session embeds at a time.

## Dependency summary

```mermaid
flowchart LR
    T1 --> T2 --> T3 --> T4 --> T5
    T3 --> T6 --> T7
    T1 --> T8
    T6 --> T8 --> T9
    T7 --> T9 --> T10
    T5 --> T10
```

- **Week 3:** T1, T2, T3, T4, T5, T6, T7. Checkpoint after T7 with T5 recorded.
- **Week 4:** T8, T9, T10.
- **Parallel group A:** T4 and T6 (both need T3; different files).
- **Parallel group B:** T5 and T7.
- **Integration order:** T1, T2, T3, then T4 beside T6, then T5 beside T7, then T8, T9, T10.

## Task graph

### T1 — Week 3. Add `timeline_pooled` to the `.latent` file; FORMAT_VERSION 2, OUTPUT_VERSION 3 (the first small change)

- **Status:** Pending
- **Dependencies:** None
- **Parallel group:** Sequential
- **Complexity:** ROUTINE
- **Execution:** Lead session
- **Owner time:** 0.5 h (approve the architecture and this change; review the diff)
- **Expected files/workspaces:** `src/jepa_scope/__init__.py`, `src/jepa_scope/latent_file.py`,
  `src/jepa_scope/embed.py`, `tests/test_pipeline.py`, `README.md` (one sentence under the cache
  paragraph)
- **Objective:** In `embed_video`, after `tl_feats` is built, add
  `tensors["timeline_pooled"] = tl_feats.reshape(len(tl_feats), -1, tl_feats.shape[-1]).mean(axis=1).astype(np.float32)`
  (shape `[G, D]`). Add `"timeline_pooled"` to `TIMELINE_KEYS` and its expected shape
  `(count, hidden_size)` to `check_tensors`. Set `FORMAT_VERSION = 2` and `OUTPUT_VERSION = 3`,
  with a comment on each naming this change. Extend `test_embed_shapes` (shape `(12, 96)` and
  `np.allclose(timeline_pooled[g], patches[w, j].reshape(-1, 96).mean(0))` for every tubelet, using
  `timeline_owner`), add a `timeline_pooled` row-count case to
  `test_load_rejects_other_inconsistencies`, and add a test that a header with `format_version: 1`
  is refused.
- **Success criteria:**
  - `uv run pytest -q` passes; the new assertions fail if the tensor is dropped or mis-shaped
    (checked by reverting the `embed.py` line once).
  - `jepa-scope clip.mp4` on a clip with a version-1 cache file re-embeds and prints the cache
    reason.
- **Verification:** `uv run pytest -q tests/test_pipeline.py`; `uv run ruff check src tests`;
  `uv run jepa-scope <local clip> --no-browser` once.
- **Escalate if:** the owner prefers no format change (then the trajectory view would need patches
  for every indexed clip; re-plan T4 and T8).

### T2 — Week 3. The index file contract

- **Status:** Pending
- **Dependencies:** T1
- **Parallel group:** Sequential
- **Complexity:** ROUTINE
- **Execution:** Executor
- **Owner time:** None
- **Expected files/workspaces:** `src/jepa_scope/index_file.py` (new), `tests/test_index.py` (new)
- **Objective:** Implement `INDEX_FORMAT = "jepa-scope.index"`, `INDEX_FORMAT_VERSION = 1`,
  `IndexFileError`, `save(path, tensors, header)` (a temporary name then `os.replace`, like
  `latent_file.save`), `read_header`, `load`, and `check(tensors, header)` with the validation
  listed in Approach. Provide a small typed view, `Index` (a dataclass holding the header,
  `pooled`, `window_start_time`, `clip`, `mean`, and helpers `clip_rows(i) -> slice` and
  `clip_vector(i)`), so `search.py` and `server.py` share one object. Tests: a round trip on
  synthetic arrays; rejection of overlapping or gapped `rows`, of `clip` disagreeing with `rows`,
  of a wrong `hidden_size`, and of a plain safetensors file.
- **Success criteria:** tests pass, and each rejection test fails if its check is removed.
- **Verification:** `uv run pytest -q tests/test_index.py`; ruff.
- **Escalate if:** the header needs a field this plan did not list that changes what the viewer
  shows.

### T3 — Week 3. Search with planted checks

- **Status:** Pending
- **Dependencies:** T2
- **Parallel group:** Sequential
- **Complexity:** ROUTINE
- **Execution:** Executor
- **Owner time:** None
- **Expected files/workspaces:** `src/jepa_scope/search.py` (new), `tests/test_index.py`
- **Objective:** `prepare(index) -> SearchSpace` (the centred, normalised matrix, kept on the
  `Index`), `nearest_moments(space, clip, window, k)` and `nearest_clips(space, clip, k)` with the
  exclusion and suppression rules in Approach, plus `query_vector(space, pooled_rows)` for an
  external query video (T4's `search` command) and `centre: bool = True`. Tests on a synthetic
  index of a few clips:
  - an exact duplicate row scores 1.0 and ranks first;
  - a row scaled by 2 also scores 1.0 (normalisation has teeth);
  - the query window and its overlapping neighbours in the same clip never appear;
  - two overlapping hits in one clip collapse to the better one;
  - with a large common offset added to every vector plus a planted structure, centred search
    recovers the structure and raw search does not (the centring test has teeth);
  - `nearest_clips` excludes the query clip.
- **Success criteria:** every test fails when its rule is removed (checked once by hand, noted in
  the log).
- **Verification:** `uv run pytest -q tests/test_index.py`.
- **Escalate if:** the suppression rule makes result lists empty on realistic folders (report with
  numbers).

### T4 — Week 3. The indexer and the CLI

- **Status:** Pending
- **Dependencies:** T3
- **Parallel group:** A (with T6)
- **Complexity:** STRONG
- **Execution:** Lead session or an Opus executor
- **Owner time:** 0.5 h (review the diff and try `jepa-scope index` on one of their folders)
- **Expected files/workspaces:** `src/jepa_scope/indexing.py` (new), `src/jepa_scope/cache.py`
  (new), `src/jepa_scope/cli.py`, `tests/test_index.py`, `README.md`
- **Objective:** `build_or_refresh(folder, encoder_factory, settings, out, recursive, patches, log)`
  as in Approach. The cache lookup reuses `cli._cache_path`; move it and `_cache_is_valid` into a
  small `cache.py` so `indexing.py` does not import the CLI. The encoder is created lazily, only
  when a clip must be embedded, so a refresh with nothing new never loads the model. CLI additions
  are as in Approach; `jepa-scope <folder>` runs the refresh then serves. Tests use `tiny_encoder`
  on a tmp folder of three synthetic videos (the moving square, an exact copy under another name,
  and a constant-grey clip):
  - the index has three clips with 5 windows each;
  - a second run embeds nothing (count calls through a wrapper);
  - touching one file's mtime without changing bytes re-hashes but does not re-embed;
  - rewriting one file re-embeds only it; deleting one drops it;
  - a settings change rebuilds everything;
  - `nearest_clips` from the moving square ranks its copy first with a score within 1e-6 of 1.0
    and the grey clip last;
  - `nearest_moments` from any window of the square finds the copy's matching window with a score
    within 1e-6 of 1.0.
- **Success criteria:** the tests above; `jepa-scope index <folder>` on a real folder prints
  progress and writes `jepa-scope.index`, and a second run finishes in under two seconds with no
  model load; `jepa-scope search <folder> <clip> --at 3` prints a ranked table.
- **Verification:** `uv run pytest -q`; `uv run jepa-scope index <a local folder>` twice; ruff.
- **Escalate if:** cache reuse needs the cache key to change (an `OUTPUT_VERSION` question for the
  owner), or the tiny encoder's duplicate does not score 1.0 (non-determinism in the pipeline,
  which would be a finding in itself).

### T5 — Week 3. The real-model retrieval check, with its bar set here

- **Status:** Pending
- **Dependencies:** T4
- **Parallel group:** B (with T7)
- **Complexity:** STRONG (the bar is set in this plan; the implementation is routine)
- **Execution:** Lead session (runs locally with the weights)
- **Owner time:** 0.3 h (read the numbers)
- **Expected files/workspaces:** `tests/test_real_model.py`, the plan's log, `README.md` (the
  measured numbers, as an anecdote rather than a benchmark, as the surprise section does)
- **Objective:** Add `test_search_groups_kinetics_classes`, opt-in under the same environment
  variable. Download the 50 `val/` clips of `nateraw/kinetics-mini` (five classes: archery,
  bowling, flying_kite, high_jump and marching, ten clips each, about 10 s each) with
  `snapshot_download(allow_patterns="val/*")` into a tmp folder, build the index with the indexer
  at the defaults (8 fps, 16-frame windows, no patches, about 450 windows), and measure, leaving
  one clip out at a time:
  - (a) the clip-level top-1 same-class rate over the 50 clips (chance 9/49 = 18.4%);
  - (b) the window-level top-1 same-class rate over all windows, excluding the query's own clip;
  - (c) both of these with raw cosine as well as centred;
  - (d) a permutation control: the clip-level rate under 1,000 random relabellings, reporting the
    99th percentile.

  Print everything. **Bars, fixed now:** the test passes only if the centred clip-level top-1 rate
  is at least 0.60 and exceeds the permutation 99th percentile. Miss either and the task escalates
  with the numbers; the bar is not moved in the same session. If raw cosine beats centred by five
  points or more at clip level, the default flips (one line, logged); otherwise centred stays.
- **Success criteria:** the test passes at the bars above; the log records all four numbers, the
  download size and the run time.
- **Verification:** `JEPA_SCOPE_MODEL_TESTS=1 uv run pytest tests/test_real_model.py -s -k search`.
- **Escalate if:** either bar is missed. The options are then a different pooling (the mean of
  layer-normalised tokens, or dropping the context half of each window) or a different unit,
  chosen by the owner, and the fast checks stay as they are.

### T6 — Week 3. The server serves a folder

- **Status:** Pending
- **Dependencies:** T3
- **Parallel group:** A (with T4)
- **Complexity:** STRONG
- **Execution:** Lead session or an Opus executor
- **Owner time:** None
- **Expected files/workspaces:** `src/jepa_scope/server.py`, `src/jepa_scope/cli.py` (`_serve`),
  `web/src/types.ts` (types only, so T7 starts from a settled contract), `tests/test_pipeline.py`,
  `tests/test_index.py`
- **Objective:** Endpoints and behaviour as in Approach, without the Week 4 ones. Keep
  `create_app(latent_path, video_path, video_sha256)` working by wrapping the file as a one-clip
  index. Update `test_server_endpoints` to the new paths and add a folder-mode test on the T4
  fixture: `/api/meta` lists three clips; `/api/clips/1/pca` has the right length;
  `/api/clips/2/video` serves ranges; `/api/search?clip=0&window=2&k=3` returns the copy's
  matching window first; and a clip whose video was replaced returns 409 with the index message.
- **Success criteria:** both modes pass. `uv run jepa-scope view x.latent --no-browser` serves the
  data at the new paths; the built viewer reads them once T7 lands (until then the old viewer
  shows an error, which is expected and noted).
- **Verification:** `uv run pytest -q`; `curl -s localhost:8765/api/meta | head -c 400`.
- **Escalate if:** lazy loading needs more than an LRU (for example, memory pressure from
  patches-bearing cache files; the fix would be loading only the timeline tensors with
  `safe_open`, which is a small change but worth a note).

### T7 — Week 3. Viewer: clip list and search panels

- **Status:** Pending
- **Dependencies:** T6
- **Parallel group:** B (with T5)
- **Complexity:** STRONG
- **Execution:** Lead session (the lead has a browser for live checks)
- **Owner time:** 1 h (review; a live check on one of their folders; the Week 3 checkpoint)
- **Expected files/workspaces:** `web/src/App.tsx`, `web/src/ClipList.tsx` (new),
  `web/src/SearchPanel.tsx` (new), `web/src/types.ts`, `web/src/index.css`, `README.md`
- **Objective:** As in Approach. Clip switching keeps the mode, opacity and smoothing; the URL hash
  tracks the clip so reloads keep it; search rows open and seek; keyboard shortcuts are unchanged.
  Single mode hides the clip list and shows "similar moments" within the clip. The lead checks it
  in a browser on a small folder (the T4 fixture written to a temp folder, then a real folder of
  the owner's clips) and records what was seen in the log.
- **Success criteria:** `npx tsc -b`, `npx oxlint --type-aware src` and `npm run build` pass;
  `./scripts/build.sh` passes; in the browser, opening a moment from a search result shows that
  clip with the overlay at that time; the Week 3 checkpoint runs (`/checkpoint`) and a fresh
  `reviewer` pass on T1–T7 finds no material issue.
- **Verification:** the three web commands; `./scripts/build.sh`; the lead's browser check with
  notes in the log.
- **Escalate if:** the overlay cannot be rebuilt on clip switch without a WebGL context leak (then
  the `Overlay` gains a `dispose`, which is a small change but outside this task's files).

### T8 — Week 4. Trajectory and cloud endpoints

- **Status:** Pending
- **Dependencies:** T1, T6
- **Parallel group:** Sequential
- **Complexity:** ROUTINE
- **Execution:** Executor
- **Owner time:** None
- **Expected files/workspaces:** `src/jepa_scope/server.py`, `src/jepa_scope/pca.py` (a `project`
  helper if `to_rgb`'s scaling is not wanted; keep `fit`), `web/src/types.ts`,
  `tests/test_pipeline.py`
- **Objective:** `GET /api/clips/{id}/trajectory?fit=folder|clip` and `GET /api/cloud` as in
  Approach, with the 50,000-point cap and the chosen rows in `/api/meta`. Tests: the trajectory has
  `count` rows of 3 floats; with `fit=folder`, the mean of a window's tubelet points equals that
  window's cloud point within 1e-4 (the centroid property, which fails if the two use different
  fits); `fit=clip` works in single mode; `fit=folder` in single mode returns 400.
- **Success criteria:** tests pass; the centroid test fails when the cloud is projected with a
  per-clip fit (checked once).
- **Verification:** `uv run pytest -q`.
- **Escalate if:** none expected.

### T9 — Week 4. The Three.js trajectory view

- **Status:** Pending
- **Dependencies:** T7, T8
- **Parallel group:** Sequential
- **Complexity:** STRONG
- **Execution:** Lead session
- **Owner time:** 1 h (approve the dependency if not already decided; review; live check)
- **Expected files/workspaces:** `web/package.json`, `web/package-lock.json`,
  `web/src/Trajectory.tsx` (new), `web/src/App.tsx`, `web/src/index.css`, `README.md`
- **Objective:** As in Approach. Add `three` and `@types/three` pinned to exact versions current at
  execution time. The marker follows the current tubelet; hovering a path point shows its time;
  clicking seeks; search hits from the open result list are highlighted in the cloud; the fit
  toggle refetches the trajectory. Dispose the renderer and geometry on unmount and on clip switch.
  Check it on the archery-onto-bowling cut clip the existing real-model test builds (two clusters
  joined by one long segment, with the hot tubelets right after the cut) and on a folder.
- **Success criteria:** the three web commands and `./scripts/build.sh` pass; the wheel grows by
  less than 1 MB; the cut clip shows the two-cluster shape and the marker tracks playback without
  visible lag; a fresh `reviewer` pass on T8–T9.
- **Verification:** `cd web && npx tsc -b && npx oxlint --type-aware src && npm run build`;
  `./scripts/build.sh`; the lead's browser check with notes.
- **Escalate if:** the cut clip does not show the expected shape under either fit. That is a
  finding about the embedding, not the view; report it rather than adjust the view.

### T10 — Week 4. Documentation, decisions and closeout

- **Status:** Pending
- **Dependencies:** T5, T9
- **Parallel group:** Sequential
- **Complexity:** ROUTINE
- **Execution:** Lead session
- **Owner time:** 0.5 h (review; merge)
- **Expected files/workspaces:** `README.md`, `docs/roadmap.md` (progress table, "Next"),
  `docs/decisions.md`, `docs/decisions/product.md`, `docs/decisions/method.md`, `CLAUDE.md` (the
  layout list gains `index_file.py`, `indexing.py`, `search.py` and the index contract, staying
  under 14 KB), `docs/plans/README.md`, this plan's status and completion summary
- **Objective:** Write the four decision lines from Alternatives considered with their detail
  entries; document the CLI, the index file and the trajectory view for outside readers; record
  the T5 numbers; compress the log into the completion summary with its lesson or `NO_ENTRY`; run
  `python3 .claude/workflow/check.py`.
- **Success criteria:** the workflow check passes; the roadmap's "Next" names Weeks 5 and 6; the
  plan is `Done` and off the active list.
- **Verification:** `python3 .claude/workflow/check.py`; `rg -n 'jepa-scope.index' README.md docs`.
- **Escalate if:** none expected.

## Execution strategy

- **Lead/orchestrator owns:** the architecture, every shared-contract edit (T1, T6's `types.ts`),
  the viewer work (T7, T9), the real-model run (T5), integration, and every exchange with the
  owner.
- **Delegated tasks:** T2, T3 and T8 to `executor` (ROUTINE, self-contained, tests included); T4
  and T6 to an Opus `executor` if the lead wants them concurrent, otherwise the lead does them in
  order.
- **Model routing:** T1 ROUTINE (lead); T2, T3, T8 and T10 ROUTINE; T4, T6, T7 and T9 STRONG; T5
  STRONG with its bar fixed here by the planner. Every review runs on `reviewer` at its default.
- **Owner time, total:** about 4 hours over the two weeks: the approval gate (0.5), T4 (0.5), T5
  (0.3), T7 (1), T9 (1), T10 and merges (0.5), plus the questions below (0.3).
- **Compute:** laptop only. The heaviest step is T5 (about five minutes of embedding on an M3 Max
  and a 100 MB download). No GPU, no spend.
- **Approval gates:**
  - Owner approval of this plan's questions and of T1 as the first small change before any code
    file is edited.
  - Owner approval before dispatching code-writing executors (the task graph and routing above).
  - Owner approval of the Three.js dependency before T9 (question 6), and of the raw-versus-centred
    flip if T5 triggers it.
  - The Week 3 checkpoint after T7: the owner decides whether Week 4 starts.
- **Owner-only actions:** merging each pull request. Proposed grouping: PR 1 = T1; PR 2 = T2–T5;
  PR 3 = T6–T7; PR 4 = T8–T10. The owner may fold PR 1 into PR 2 for fewer merges. No release, no
  repository setting and no public post is in this plan; the launch demo is Weeks 5 and 6.
- **Review strategy:** `reviewer` with fresh context at the end of Week 3 (T1–T7, the plan and the
  diff, with T5's numbers) and at the end of Week 4 (T8–T10). Two strikes on any failing check go
  to `reviewer` or `architect` before a third attempt. `/checkpoint` after each integrated task.

### Concurrency and isolation

| Task | Actually concurrent with | Worker/session | Writes files? | Branch/base | Isolation | Shared mutable resources | Integration |
|---|---|---|---:|---|---|---|---|
| T1 | None | Lead | Yes | `week3-folder-search` from `main` | Main checkout | Result cache (re-embeds one file) | Direct |
| T2 | None | Executor | Yes | same branch | Main checkout | None | Direct |
| T3 | None | Executor | Yes | same branch | Main checkout | None | Direct |
| T4 | T6 (optional) | Lead or Opus executor | Yes | same branch | Worktree if concurrent | Result cache, MPS (embeds) | Cherry-pick |
| T5 | T7 (optional) | Lead | Tests only | same branch | Main checkout | Hugging Face cache, result cache, MPS | Direct |
| T6 | T4 (optional) | Lead or Opus executor | Yes | same branch | Worktree if concurrent, `--port 8766` | Port 8765 | Cherry-pick |
| T7 | T5 (optional) | Lead | Yes | same branch | Main checkout | Port 8765 | Direct |
| T8 | None | Executor | Yes | `week4-trajectory` from `main` after PR 3 | Main checkout | None | Direct |
| T9 | None | Lead | Yes | same branch | Main checkout | Port 8765 | Direct |
| T10 | None | Lead | Yes | same branch | Main checkout | None | Direct |

Default timing is sequential throughout. The two optional overlaps are worth taking only when two
workers are idle; then they load the `parallel-work` skill, use a worktree each (`uv sync`,
`cd web && npm ci && npm run build`), stage explicit paths, and never embed at the same time as
another session (MPS is single-instance).

## Execution log

Kept in `docs/plans/2026-11-06-weeks3-4-folder-search-trajectory-log.md` in the template's format.

## Integration and verification

### Integrated checkpoint
- [ ] `/checkpoint` after T7 (Week 3) and after T9 (Week 4)

### End-to-end acceptance
- [ ] `jepa-scope <folder>` on a folder of the owner's clips indexes once, reopens instantly, and a
  "similar moments" click lands on a plausible moment in another clip
- [ ] `jepa-scope clip.mp4` (single mode) still works, including the SHA-256 refusal
- [ ] The real-model retrieval check passes at its bars and its numbers are in the README and the
  log
- [ ] The trajectory view on the cut clip shows two clusters joined by one long segment, under both
  fits
- [ ] A replaced video is refused with the "run index again" message

### Plan review
- **Reviewers:** a fresh Fable agent and ChatGPT Astra, adversarially, before the owner approves
- **Verdict:** Pending

### Independent review
- **Reviewer:** `reviewer` (Opus, high), at the end of each week as above
- **Verdict:** Pending

## Questions for the owner, with recommendations

1. **One plan or two?** One plan, with Week 3 = T1–T7 and a stop point after T7. The index and the
   multi-clip server are shared by search and the trajectory, and the `.latent` change is cheapest
   before your folders are indexed.
2. **Search unit, embedding, distance, query types.** Windows for "moments like this moment" and the
   mean of windows for "clips like this clip", both over mean-pooled window vectors, with cosine
   after subtracting the index mean (raw cosine measured beside it in T5). Not patches, not
   tubelets.
3. **Where the index lives and whether it is a contract.** One self-contained `jepa-scope.index`
   safetensors file in the folder (`--out` for read-only folders), versioned by
   `INDEX_FORMAT_VERSION`, kept consistent by size-and-mtime then SHA-256, rebuilt when settings
   change, with a decision record. The cache directory stays for `.latent` files only.
4. **Brute force or an approximate-nearest-neighbour library.** NumPy brute force. It is
   comfortable to about half a million windows (a few hundred hours of video), and a library can
   replace `search.py`'s body later without changing the file.
5. **The `.latent` format change first.** Add `timeline_pooled` now as T1 (FORMAT_VERSION 2,
   OUTPUT_VERSION 3). It re-embeds your current cache (one 36 MB file today) and means Week 3's
   indexing already stores what Week 4 plots. The alternative is to defer it to Week 4 and
   re-embed every indexed folder then.
6. **Trajectory view: Three.js or no new dependency.** Three.js (pinned, about 0.5 MB in the wheel)
   for a 3D path with the folder cloud, PCA fitted on the folder by default with a per-clip toggle,
   per-tubelet points coloured by surprise, hover for the time and click to seek. The fallback is a
   2D SVG projection with no dependency, about a day cheaper and less legible with the cloud.
7. **CLI shape.** `jepa-scope <folder>` (index, then view), `jepa-scope index <folder>`,
   `jepa-scope search <folder|index> <clip|query.mp4> [--at S] [-k] [--by moment|clip]`, and
   `jepa-scope view` accepting a folder, an index or a `.latent`. The indexer embeds without
   patches by default (`--patches` keeps them) and reuses any cache file that already exists.
8. **The real-model bar.** On Kinetics-mini's 50 validation clips (5 classes × 10), the
   leave-one-clip-out clip-level top-1 same-class rate is at least 0.60 (chance 0.18) and above the
   99th percentile of 1,000 label permutations. A miss escalates with the numbers, and the bar is
   not lowered in the same session.
9. **Pull-request grouping.** Four PRs (T1; T2–T5; T6–T7; T8–T10), each merged by you; fold T1 into
   the second if you prefer three.

## Completion summary

To be written at closeout.
