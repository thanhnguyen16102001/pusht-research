# Implementation and evaluation report

Implemented and tested the full data → World Model → CEM-MPC → simulator loop. Reliable success from every initial state is **not established**; this release is a working research baseline.

## Verification

- 46 automated tests pass in the existing `pusht-sim` environment.
- Interactive Pygame + TkAgg live plotting smoke run completed.
- New environment manifest solved in a Conda dry run; a fresh pip/Conda installation was not performed.
- Original demonstration stores, original model `.pkl`, and reference `diffusion_policy` sources were preserved.

## Models

The original residual MLP remains supported. Recommended: `checkpoints/WorldModel_contact_best.pt` with `contact_local_v2` body-frame/geometry features. Both are trained from the two self-collected datasets, using the same initial-pose-group split (199 training / 25 validation episodes). Checkpoint selection used validation loss, so rollout measurements are validation estimates, not a new independent model test.

| Horizon | T position RMSE (simulation units) | T angle RMSE (degrees) |
|---|---:|---:|
| 1 | 0.361 | 0.260 |
| 5 | 1.705 | 0.842 |
| 10 | 3.004 | 1.585 |
| 20 | 6.832 | 4.172 |

Full model metrics and baselines: `world_model_contact/model_report.json`; residual comparison: `world_model_v1/model_report.json`.

## Closed-loop results

Configuration `configs/mpc_default.json` was selected on development seeds 1000–1002, where it succeeded on 3/3. It was then frozen and evaluated once on reserved seeds 1005–1009. The five-seed evaluation is small and is not an estimate of guaranteed or original Diffusion Policy performance.

| Seed | Success (>95% overlap) | Max coverage | Completion time (s) | Actuator path | Acceleration RMS | Planning p95 (ms) |
|---|---|---:|---:|---:|---:|---:|
| 1005 | False | 0.9308 | not reached in 30 s | 4794.0 | 1160.0 | 92.7 |
| 1006 | True | 0.9539 | 4.4 | 931.3 | 1042.7 | 93.9 |
| 1007 | False | 0.9376 | not reached in 30 s | 3774.1 | 1068.3 | 94.8 |
| 1008 | True | 0.9607 | 10.6 | 1483.4 | 1034.5 | 96.7 |
| 1009 | False | 0.9023 | not reached in 30 s | 3956.4 | 1073.9 | 97.2 |

Reserved-seed success: **2/5 (40%)**. Hold-position baseline on the identical seeds: **0/5**. Mean maximum target coverage: **93.71%**.
Mean completion time among successful episodes only: **7.5 simulated seconds**. Failed episodes are censored, not counted as completed at the timeout.

Planning latency excludes GUI rendering. The simulator always advances by 0.1 s per selected action; optional wall-clock pacing cannot guarantee 10 Hz if computation/rendering exceeds that budget.

Per-episode JSON/NPZ/plots and frozen configuration/checkpoint hash: `mpc_evaluation/`. Development trials, including unsuccessful configurations, remain under `mpc_tuning_*`.

## Remaining performance limitations

- Three reserved seeds reached 90–94% maximum overlap but did not cross the strict success threshold within 300 steps.
- Model error, finite-horizon local minima, and near-goal contact selection still limit final alignment.
- Positive smoothness/path/time penalties express tradeoffs; they do not prove a globally optimal action sequence or hardware-safe motion.
- Further performance improvement should use separate development data (for example, failed-controller contact transitions and multi-step training). The reserved seeds above must not be reused for tuning while still claiming them as a final test.

## Reproduce the recommended demonstration

```bash
cd ~/AI-application/pusht-research
conda activate pusht-sim
python -B src/run_mpc.py --seeds 1006 --render --pace --output results/my_mpc_demo
```
