# Execution log: jepa-scope Weeks 3 and 4

The plan is `2026-11-06-weeks3-4-folder-search-trajectory.md`. Entries are newest last.

- **2026-11-06 — Plan approved: every owner question as recommended.**
  - **Evidence:** the owner answered "Approve as recommended" after the adversarial review, the
    outside review and a consistency check.
  - **Impact:** T0 starts. The gates that remain are the chosen default after T0, the Week 3
    checkpoint after T7, and merging each pull request. Three.js is approved (question 7).
  - **Resolution:** Status set to In progress.
  - **Permanent decision record needed:** Yes, at T10, as the plan lists.

- **2026-11-06 — T0 deviation: the localisation composite is built at 8 fps, not 24.**
  - **Evidence:** the first `val` run stopped while building the composite, before any table was
    printed: `val/bowling/-WH-lxmGJVY_000005_000015.mp4`, the second bowling clip in sorted order
    and one of the two distractor sources, is 15 fps, and `decode` refuses to sample it at 24.
  - **Impact:** none on what is measured. The model samples every video at 8 fps, so a composite
    decoded and written at 8 fps gives the same 30 s and 29 windows, with the same transition
    windows (4, 9, 14, 19 and 24 s).
  - **Resolution:** `COMPOSITE_FPS = 8` in `tests/retrieval_eval.py`; the sources and order are
    unchanged. No result had been seen when this was changed.
  - **Permanent decision record needed:** No.

- **2026-11-06 — T0 result: the default is (a), the stored mean of all final tokens, with plain
  cosine. It awaits the owner's approval at the gate after T0.**
  - **Evidence (selection set, 50 `val` clips, chance 0.18, none skipped):**

    | Vector | Clip top-1 | Window top-1 | Null mean / 95th | Single-frame control |
    |---|---|---|---|---|
    | (a) mean of final tokens, plain cosine | 42/50 (0.84) | 0.74 | 0.19 / 0.28 | 35/50 (0.70) |
    | (b) the same, centred | 39/50 (0.78) | 0.76 | 0.19 / 0.30 | 38/50 (0.76) |
    | (c) mean of re-normalised tokens | 42/50 (0.84) | 0.74 | 0.19 / 0.30 | 35/50 (0.70) |
    | (d) mean of the predicted half | 42/50 (0.84) | 0.74 | 0.19 / 0.30 | 35/50 (0.70) |

    The colour baselines retrieved 10/50 (middle frame) and 15/50 (whole clip). Every candidate
    passed localisation 16/16, and the clip-average mutation passed 0/16 for each. The parity of
    (a) recomputed from patches was 7.2e-8. All four candidates were eligible; (a), (c) and (d) tie
    at 42 and (b) is three clips behind, so the rule chose (a), the simplest. The tie is not an
    artefact: (c) is almost the same vector as (a) (window cosine about 0.9997, because the
    encoder's final layer norm already standardises tokens), and (d) differs more (about 0.96)
    and picks a different nearest clip for 4 of 50, with the same count correct. Raw clip cosines
    for (a) span 0.42 to 0.94 (median 0.70), so plain cosine is not compressed into a narrow band.
  - **Evidence (confirmation set, 50 `train` clips, (a) only, nothing re-chosen):** 44/50 (0.88),
    window top-1 0.79, null 0.18 / 0.28; colour baselines 15/50 and 12/50; single-frame control
    35/50 (0.70). (a) meets both bars (0.60, and 0.58 above the better baseline).
  - **What it means:** search on these clips is far better than colour, and full windows beat a
    single repeated frame by 14 to 18 points, but the single frame already retrieves 70%. On these
    five classes, which differ by scene, search mostly reflects what is in the scene, with motion
    adding a measurable amount. The README should say so when the numbers go there (T10). The
    misses concentrate in archery, flying_kite and bowling, often across classes shot outdoors.
  - **Run:** downloads 95 MB (`val`) and about 68 MB (`train`); embedding the `train` set and its
    controls took 196 s; the `val` rerun took 24 s from cache. Logs and JSON are in
    `~/Workspace/jepa-scope-lab/retrieval/` (`t0_val_run2.log`, `t0_train_run1.log`,
    `retrieval-val.json`, `retrieval-train.json`). The result cache now holds 3.8 GB, to be
    deleted once T1 lands.
  - **Impact:** T1's descriptor records `pooling: mean of final tokens` and
    `similarity: cosine`; `pooled` is unchanged, so `OUTPUT_VERSION` stays. Centring stays an
    explicit `--centre` option. The result is exploratory, as the plan says.
  - **Resolution:** owner approval of (a) requested.
  - **Permanent decision record needed:** Yes, at T10 (search as plain cosine over window means).

- **2026-11-06 — Gate after T0: the owner approved (a) as the default.**
  - **Evidence:** the owner answered "Approve (a)".
  - **Impact:** T1 records `token_norm: none` and `pooling: mean_all`; `OUTPUT_VERSION` stays 2.
  - **Resolution:** T1 starts.
  - **Permanent decision record needed:** Yes, at T10, as already noted.

- **2026-11-06 — T1 implemented: format version 2, `timeline_pooled`, the representation block,
  `save` as a current-format writer.**
  - **Evidence:** the fast suite passes (86). Breaking the code three ways made the tests fail:
    taking `timeline_pooled` from the previous window (1 failure), dropping it (14), and restoring
    `save`'s old merge order (1). On the real model, the version-1 cache file for
    `val/archery/-Qz25rXdMjE…` printed the re-embed message, was re-embedded over the same path
    (202 cache files before and after), and the second run reused it, so `Encoder.describe` (no
    weights) agrees with what embedding writes. Its representation hash is `6cdbf94c50680ea2`
    (model revision `b3c1679…`, `mean_all`). The server serves the version-2 file.
  - **Impact:** every other version-1 cache file, including T0's, is now re-embedded on use.
  - **Resolution:** the T0 cache files (3.8 GB) are deleted once this change is merged.
  - **Permanent decision record needed:** Yes, at T10 (format version 2 and the descriptor).

- **2026-11-06 — T1 deviation after review: weights are identified by content, not by commit.**
  - **Evidence:** the fresh review found that for a model in a local folder the block held only
    the path, so retraining into the same folder reused old vectors; and that the snapshot commit
    changes when only a model card is edited upstream, which would force needless re-embedding.
  - **Impact:** the block's `model_revision` field is replaced by `weights`, an identity built from
    `config.json`'s bytes and each weight file's content name (the blob a Hugging Face snapshot
    links to, or size and modification time for an ordinary folder). The commit moves to
    `run.model_revision`, for information. A local model's `model_id` is its absolute path. The
    hash's canonical form is now written down (sorted keys, no whitespace). `describe` is skipped
    when no cache file exists, and the README states the reuse rule precisely. New fast tests use
    a fake Hugging Face cache (same identity across commits with the same blob, a new one for new
    weights) and a retrained local folder.
  - **Resolution:** on the real model the earlier version-2 file was refused (its block no longer
    matched), re-embedded over the same path, and reused on the next run; weights identity
    `550d387c4f4a1862`, representation hash `e8eb61d31255e782`. The fast suite passes (88).
  - **Permanent decision record needed:** Yes, at T10, with the format change.

- **2026-11-06 — T1 second review: two identity gaps closed.**
  - **Evidence:** a symlinked weight file in an ordinary folder was identified by its target's
    file name, so retraining a "latest" checkpoint kept the identity; and right after an upstream
    commit the new snapshot can hold only the config, so `describe` hashed the config alone.
  - **Resolution:** a link's name is trusted only when it points at a Hugging Face blob (a `blobs`
    folder and a 40- or 64-character hex name); anything else is described by the resolved path,
    size and modification time. A folder with no weight file has weights `unknown`, and the CLI's
    cache check then loads the model to get the identity. Two fast tests cover these cases (90
    pass). On the real model the identity is unchanged (`550d387c4f4a1862`, hash
    `e8eb61d31255e782`), so nothing re-embeds.
  - **Permanent decision record needed:** No beyond the T10 entry.

- **2026-11-06 — PR 1 merged (T0 and T1, ryangswe/jepa-scope#5); old cache deleted; T2 starts.**
  - **Evidence:** the owner merged #5 (merge commit `ec26b1a`). 201 version-1 cache files (4.03 GB),
    T0's set plus one earlier clip's, were deleted; the one version-2 file was kept.
  - **Impact:** T2–T5 go on branch `week3-index` from `main`.
  - **Resolution:** T2 delegated to an executor, as the plan routes it.
  - **Permanent decision record needed:** No.

- **2026-11-06 — T2 complete: the index file contract (`index_file.py`).**
  - **Evidence:** an executor (Sonnet) wrote the module and its tests; a fresh reviewer (Opus) did
    two rounds. Round 1 found eight gaps that would have been hard to tighten once users have
    index files: a relative `root`; non-finite or out-of-range clip numbers and bare `NaN` in the
    JSON (which would break `/api/meta` for a whole folder); negative clip ids; path holes and
    duplicates; loosely shaped `skipped` and `sha256`; an unchecked `mean`; and a fixed temporary
    name. Round 2 found that its own suggested ":" rule rejected legitimate timestamped names, plus
    a fixed file mode and a temporary file placed beside a symlink's target. All are fixed; the
    path rule is now public as `is_valid_clip_path`, and T4 skips failing files instead of
    crashing. The fast suite passes (235). Every check was break-tested by removing it.
  - **Impact:** T3 builds on `Index`, `clip_rows` and `clip_vector`.
  - **Resolution:** committed on `week3-index`.
  - **Permanent decision record needed:** Yes, at T10 (the index file as a versioned contract).
