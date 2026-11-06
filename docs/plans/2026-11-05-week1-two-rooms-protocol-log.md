# Execution log: Week 1 closeout (eb_jepa Two Rooms and the planning protocol)

Plan: `2026-11-05-week1-two-rooms-protocol.md`. Entries are appended as work proceeds.

- **2026-11-05 — Plan approved: the owner approved the plan as written.**
  - **Evidence:** the owner's approval on 2026-11-05 covers running eb_jepa on the laptop and on RunPod pods, RunPod spend up to $25 in total, and the protocol draft in `src/jepa_scope/planning/`.
  - **Impact:** T1 starts; T2 may create pods within the cap.
  - **Resolution:** Status set to In progress.
  - **Permanent decision record needed:** No (the cloud-spend rule is recorded in `CLAUDE.md`; its decision line is written in T5).

- **2026-11-05 — T1: lab set up; CPU training is about 20 s per step, so training goes to RunPod.**
  - **Evidence:**
    - The machine is an Apple M3 Max with 16 cores and 48 GB of memory.
    - eb_jepa is at `966e61e` in `~/Workspace/eb_jepa-lab/eb_jepa`, with Python 3.12, `torch==2.6.0` and `torchvision==0.21.0`, installed with `--no-deps` and without `decord`. `import examples.ac_video_jepa.main` works.
    - The 30-step CPU run (AMP on, bf16, `num_workers` 8, `--data.size 11520 --optim.epochs 1`) finished in 597 s, about 19.9 s per step, and saved a checkpoint.
    - The run completed, so `log_wandb` was read as false. The log has no "Planning eval:" line.
  - **Impact:**
    - 3,120 steps project to about 17 hours, which is over the 10-hour laptop threshold, so the laptop is only a weak fallback.
    - T2 runs on RunPod as planned.
  - **Resolution:** pick the pod GPU from a live read (next entry).
  - **Permanent decision record needed:** No.

- **2026-11-05 — T2: pod 1 about to be created.**
  - **Evidence:** the live reads before creating were:
    - `runpodctl template list --type official` lists `runpod-torch-v240` = `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.
    - `runpodctl gpu list` shows NVIDIA L40S at $0.79/h community ($1.09 secure), low stock in US-TX-3 and OC-AU-1. A100 SXM 80 GB is $1.39/h community.
  - **Impact:** the plan is one L40S community pod with a 20 GB container disk and a 40 GB volume. Training is projected at about 3 h until the smoke test measures it, about $2.40 plus volume storage. The running total before this pod is $0.
  - **Resolution:** create it; start the watchdogs at 2 × the projection (6 h).
  - **Permanent decision record needed:** No.

- **2026-11-06 — T2: pod 1 created and the CUDA path smoke-tested; full training started.**
  - **Evidence:**
    - The pod is `511sta3g8l3yvk` (`jepa-scope-two-rooms`): NVIDIA L40S, community cloud, billed $0.79/h. It has 24 vCPUs, 251 GB of RAM, driver 550.163.01 and host location TW. The image is `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, with a 20 GB container disk and a 40 GB volume at `/workspace`. `--wait` reached SSH after 7 s.
    - The read-back confirmed the GPU, image, the `22/tcp` port, the vCPUs and `costPerHr` 0.79.
    - The pod-scoped `RUNPOD_API_KEY` and `RUNPOD_POD_ID` are present in PID 1's environment, and `runpodctl` is at `/usr/bin/runpodctl`.
    - Setup: tmux, uv 0.13.7, uv-managed Python 3.12 and the uv cache on the volume, and eb_jepa at `966e61e`. torch is `2.6.0+cu124` with `torchvision==0.21.0`, and CUDA sees the L40S at capability (8, 9).
    - The 30-step smoke test (16 workers, bf16 AMP, `torch.compile`) logged "Using device: cuda", "Using AMP with dtype=torch.bfloat16" and "Compiling model with torch.compile". It took 82 s of wall clock, including the compile.
  - **Deviations:**
    - `opencv-python-headless` replaced `opencv-python` on the pod (no display libraries on a server; the same `cv2` API).
    - The first attempt to reset the on-pod watchdog used `pkill -f`, which matched its own SSH command line and ended that command. Both watchdogs are now script files: the on-pod one is `watchdog.sh 14400` (4 h), and the laptop backstop is `caffeinate -i` with a 4.5 h sleep.
  - **Impact:** full training is running in tmux session `train` through `train.sh` (`tee -a train.log`). The projection is about 1 h of training plus the evaluation, about $1.60 for the pod.
  - **Resolution:** check epoch 0's time, then wait for `TRAIN_EXIT` in `train.log`.
  - **Permanent decision record needed:** No.

- **2026-11-06 — T1: CPU planning cost measured.**
  - **Evidence:**
    - eb_jepa's own CEM evaluation (`planning_cem.yaml`, 2 episodes, `eval_2.yaml`) used the 30-step untrained checkpoint on the laptop CPU.
    - The log has "Loaded model state from". `avg_episode_time` is 805 s (about 13.4 min per 200-step episode), the success rate is 0.0 as expected for untrained weights, and the run took 1,628 s in all.
    - The AMP-off training timing was not run. It only informs the laptop training fallback, which T2 did not need.
  - **Impact:**
    - T4's 30 laptop episodes (20 protocol, 10 untrained) project to about 6–7 h. That fits an unattended overnight run.
    - The 50+50-episode contingency projects to about 22 h, which is over the 12 h escalation line, so it would need the owner first.
    - Native CEM episode times on CPU are not compared with the protocol planner's.
  - **Resolution:** T1 complete.
  - **Permanent decision record needed:** No.

- **2026-11-06 — T2: training finished on the pod.**
  - **Evidence:**
    - `TRAIN_EXIT=0` after 12 epochs (`e-0` to `e-11` and `latest.pth.tar`, about 34 MB each), from 02:52 to 04:27 UTC.
    - Epochs took 447–542 s each, about 7.5–9 min. The run was loader-bound, as the smoke test suggested.
    - The prediction loss fell from 0.198 in epoch 0 to 0.018 in epoch 11, and the regulariser loss from 0.82 to 0.31. The probe loss is noisy (0.21–1.72).
  - **Impact:** the pod has run about 1.6 h so far, about $1.30 at $0.79/h.
  - **Resolution:** the 20-episode CEM evaluation runs on the pod in tmux session `eval` through `eval.sh`, which copied `config.yaml` to `config.train.yaml` first and moved `train.log` into the run folder. The log has "Loaded model state from: …/latest.pth.tar".
  - **Permanent decision record needed:** No.

- **2026-11-06 — T3: protocol drafted; the owner approved the signatures.**
  - **Evidence:**
    - The owner approved the `WorldModel`, `rollout` and `plan_cost` signatures as drafted.
    - `cem.py` mirrors eb_jepa's `CEMPlanner.plan` (read at `966e61e`), including the norm-rescaling formula `min(norm, max_norm) / (norm + 1e-6)`, which also shrinks actions under the limit very slightly.
    - The 14 tests in `tests/test_planning_protocol.py` pass in 0.06 s. Across 30 seeds outside the suite, the point mass always ended within 1e-5 of the goal and the planned sequence always beat the best of 64 random sequences, so the seeded tests do not rest on a lucky seed.
    - Lint, the full suite (41 passed, 1 skipped) and the workflow check pass.
  - **Deviations:**
    - The lead wrote T3 instead of an executor, as the plan allows when the lead is idle during training.
    - `clip_action_norm` is exported, because the toy model and T4's adapter both need eb_jepa's exact formula.
    - eb_jepa's README averages over 3 training seeds and the last 3 epochs; this plan's single run and checkpoint will be noisier than its table.
  - **Impact:** T4 can build on the module.
  - **Resolution:** T3 complete.
  - **Permanent decision record needed:** Yes (the protocol's three operations and the summed cost; written in T5).
