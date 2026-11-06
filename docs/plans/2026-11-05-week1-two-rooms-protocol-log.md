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

- **2026-11-06 — T2: eb_jepa's CEM reproduced at 18 of 20 (90%); pod 1 deleted.**
  - **Evidence:**
    - The 20-episode CEM evaluation (`planning_cem.yaml`, `latest.pth.tar` = epoch 11, step 3119) gave `success_rate` 0.90 and `mean_state_dist` 2.57, with an average episode time of 45 s on the L40S. It took 921 s in all.
    - Episodes 2 and 3 failed and the other 18 succeeded (from the `agent_steps_succ` and `agent_steps_fail` files in each `ep_N` folder).
    - The validation unroll MSE rises from 0.0044 one step ahead to about 0.066 at step 17 (`unroll_eval/step-3119_eval_only/eval.csv`).
    - MPPI was not run; it is optional in this plan and T4 compares CEM only.
    - The run folder (429 MB, 168 files) was copied to `~/Workspace/eb_jepa-lab/checkpoints/pod1/dev_2026-11-06_02-52/impala_cov8_std16_simt12_idm1_seed1`. The SHA-256 of `latest.pth.tar` (`9dc74658…`) and `e-11.pth.tar` match the pod's, and the file counts match.
    - `runpodctl pod delete 511sta3g8l3yvk` returned `"deleted": true`, and `runpodctl pod list` is empty. Both watchdogs are gone (the on-pod one with the pod; the laptop backstop was killed).
  - **Billing:** the pod ran 7,490 s (about 2.08 h) at $0.79/h, about $1.64, plus a negligible amount of volume storage. The running total is about $1.65 of the $25 cap.
  - **Impact:** the pass bar (at least 80%) is met. T4 uses this checkpoint. eb_jepa's README gives 96 ± 2% for CEM on random walls (its ablation table), where ± 2 is the spread of averages over 3 seeds and 3 checkpoints. 90% is 6 points below that, which is within the sampling error of one 20-episode evaluation (about 7 points), so it is not evidence of a broken setup.
  - **Resolution:** T2 complete.
  - **Permanent decision record needed:** No.

- **2026-11-06 — T4: adapter written; parity is exact and the replay matches all 20 episodes.**
  - **Evidence:**
    - `EbJepaTwoRooms.from_run` loads `latest.pth.tar` with `weights_only=True` (no allowlist needed) and `strict=True` after dropping the `_orig_mod.` prefix (all 46 keys carry it) and the `regularizer.*` keys. Importing it does not import `eb_jepa.planning`.
    - Parity against an independent `JEPA` loaded by eb_jepa's own `load_checkpoint(strict=True)`: the maximum absolute difference between `rollout` and `unroll` over 8 observations and 8 random actions is 0.0, and `plan_cost` equals `ReprTargetDistMPCObjective(sum_all_diffs=True)` with a maximum relative difference of 0.0. The bar was 1e-5.
    - Replay: each replayed episode's first frame, drawn with eb_jepa's own `save_gif`, matches the first frame of that episode's GIF from T2 (0 differing pixels outside the frame label, limit 4). Pairing each replayed episode with the next native episode instead gives 356 to 1,344 differing pixels, so the check discriminates. The replay covers all 20 episodes' start frames (wall, door and start position); goal frames exist only in T2's PDFs and were not compared, but the goal is drawn from the same generator calls as the start.
    - The environment's layouts come only from its own NumPy generator. Wall-collision noise uses torch's global generator, which the test seeds, and it does not change later layouts, so the pairing holds whatever the agent does.
  - **Deviations:**
    - The replay check compares GIF first frames with the 20-row label band masked, rather than `state.pdf`; the plan allowed the GIF frame.
    - The episode loop lives in the test file; the plan named only the adapter and the test file, and no shared runner was needed.
  - **Impact:** the paired success, greedy and untrained tests can be read as evidence once they finish.
  - **Resolution:** the full opt-in suite runs on the laptop under `caffeinate -i`, logging to `~/Workspace/eb_jepa-lab/t4_run.log`.
  - **Permanent decision record needed:** No.

- **2026-11-06 — Design review: Fable architect on the protocol; the owner approved its changes.**
  - **Evidence:**
    - The owner offered Fable and outside (ChatGPT) reviews for complex work. A Fable architect reviewed the protocol before T5 records it as a decision, asking whether the three operations and stateless `predict` fit V-JEPA 2-AC, the second model latent-plan targets.
    - Its answer: keep the design. V-JEPA 2-AC's planner starts from one image and grows its context from its own predictions while updating the arm's pose, which is a Markov state that fits `predict(z, a)` by packing. A separate fetch of `notebooks/utils/mpc_utils.py` and `world_model_wrapper.py` in `facebookresearch/vjepa2` confirmed this, along with final-frame-only L1 scoring, per-dimension box clipping, momentum, zero rotation, and layer-normed predictions under `normalize_reps`.
  - **Changes (owner-approved 2026-11-06, "Approve as recommended"):**
    - `encode(obs: Any)`: the observation type is the model's own (for example an image plus a pose).
    - The spec and docstrings call `z` the planning state: one fixed-shape tensor that may hold more than the compared latent, with `energy` choosing what it compares.
    - `plan_cost(..., objective="sum" | "last")` and `CEMConfig.objective`, defaulting to `"sum"`, so a model can be compared with a native planner that scores only the last state. The T4 run uses the default and is unaffected.
    - `latent(z) -> [B, *latent]` is recorded in the spec as a named possible fourth operation, to decide when latent-plan is planned.
    - New open points: how V-JEPA 2-AC would fit, memory growth of `rollout` with a growing context, keeping the pose in float32, and V-JEPA 2-AC's action handling.
    - 17 toy tests pass (3 new), lint and the workflow check pass.
  - **Impact:** T5's decision line should read as the architect worded it: a Markov planning state as one fixed-shape tensor, a one-step stateless `predict`, a model-owned `energy`, and a summed objective by default.
  - **Resolution:** applied; the outside ChatGPT review of the spec happens at the PR.
  - **Permanent decision record needed:** Yes (in T5, with the protocol decision).

- **2026-11-06 — T4: all five opt-in tests pass; the protocol planner succeeds in 16 of 20 paired episodes.**
  - **Evidence:** `t4_run.log` in the lab, 5 passed in 4,903 s (1 h 22 min) on the M3 Max CPU, float32.
    - Paired success: protocol 16/20 (80%), native 18/20 (90%). Native-only successes are episodes 7 and 12; protocol-only successes are none; both failed episodes 2 and 3. The margin is 2, the largest the pre-set rule allows. The protocol rate clears the 70% bar.
    - Mean protocol episode time 166 s (155–192 s), against 805 s for eb_jepa's native CEM on the same CPU in T1, because the protocol encodes the current observation once per re-plan rather than once per sample.
    - Greedy baseline 9/20 (45%), 35 points below the protocol, so it is not an escalation. It succeeds on episode 2, which both planners fail, and on episode 7.
    - Untrained control 0/10.
  - **Impact:** T4's success criteria are met. A 2–0 split is what equal planners produce about half the time (each discordant episode is a coin flip under equal planners), so it is not evidence of a gap; 20 episodes cannot rule out a gap of about 10 points either.
  - **Resolution:** T4 complete. An exploratory reseed of episodes 7 and 12, outside the pass rule, follows.
  - **Permanent decision record needed:** No.

- **2026-11-06 — T4 (exploratory, outside the pass rule): episodes 7 and 12 succeed with other planner seeds.**
  - **Evidence:** `~/Workspace/eb_jepa-lab/scripts/t4_reseed.py` replanned only episodes 7 and 12 (the other episodes stood still, so the layouts are the same) with CEM generator seeds 1 and 2. All four runs succeeded (distances 0.11–0.22).
  - **Impact:** the two protocol-only failures in the paired run came from the planner's random sampling on that seed, not from the adapter. The pass verdict does not change and rests on the pre-set rule alone.
  - **Resolution:** recorded; T4 stays complete.
  - **Permanent decision record needed:** No.

- **2026-11-06 — T5: checkpoint found a pre-existing random failure in the build's wheel check; fixed.**
  - **Evidence:** `scripts/build.sh` failed in 3 of 5 runs with "does not contain the built viewer" while the wheel did contain it. Under bash with `set -o pipefail`, `unzip -l | grep -q` failed in 142 of 300 trials (grep exits at the first match and unzip dies on the closed pipe), and 0 of 300 without `-q`. The check came in with the first commit.
  - **Resolution:** the script reads the listing first and then searches it (commit 115f87c); 6 of 6 builds pass, and the check still fails on a wheel built without the viewer.
  - **Permanent decision record needed:** No.
