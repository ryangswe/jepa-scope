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
