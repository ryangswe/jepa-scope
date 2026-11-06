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
