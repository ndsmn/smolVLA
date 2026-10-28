# Completed SmolVLA experiment — 40,000 steps

Completed: **24 October 2026, 21:29 IST**. The runner exited after finishing all stages.

**Final task success in the reconstructed simulator: 0/50 (0%).** All 50 episodes ran to the 301-step limit; no episodes were marked technically invalid. The 95% Wilson confidence interval is 0–7.1%. This is not a successful task-performance result.

Training completed all 40,000 optimizer updates with batch 32 in approximately 16 hours 3 minutes. Eight full checkpoints were saved at 5,000-step intervals. The complete job, including benchmarks and evaluations, took approximately 24 hours 5 minutes.

All eight trained checkpoints and the base baseline scored 0/20 on the fixed simulator selection seeds. Step **40,000** was therefore selected by the predeclared held-out normalized action-MAE tie-breaker. This selection does not demonstrate superior closed-loop task success.

On 128 held-out trajectory anchors (50-step chunks with padded elements masked), action MAE decreased from 0.212416 to 0.045478 radians, a 78.6% reduction. Final held-out flow loss was 0.077250. Offline prediction improved, but successful grasp-and-place behavior was not demonstrated in these rollouts.

The simulator is a reconstruction. Its camera calibration, dynamics, object properties, actuator gains, and reset distribution were assumed rather than recovered from the original environment. The root cause of the rollout failures has not been diagnosed; environment mismatch and policy/control errors remain possibilities. These results must not be reported as the original simulator's benchmark or physical-robot performance. The original final seeds and outcomes should remain frozen if further development changes the environment or policy.

## Saved artifacts

- Selected model: `<ORIGINAL_PROJECT_ROOT>/outputs/smolvla_so101_40k/runs/so101_40k/training/checkpoints/040000/pretrained_model`.
- All eight checkpoints: `training/checkpoints/`.
- Training graphs: `graphs/training.png`.
- Checkpoint comparison graphs: `graphs/checkpoint_comparison.png`.
- Per-checkpoint values: `checkpoint_comparison.csv` and `checkpoint_selection.json`.
- Actual per-update logs: `training_metrics.jsonl` and `training_metrics.csv`.
- Resource samples: `gpu_telemetry.csv`.
- All 50 final videos, 50 traces, and per-episode JSON/CSV outcomes: `final_50_episodes/`.
- Final machine-readable report: `final_report.json`.
- Data repair, splits, model/source information, and per-attempt snapshots remain in the experiment folder and `reproducibility/`.

The lightweight report ZIP contains reports, plots, metric files, and final episode outcome tables. Large model checkpoints, videos, and NPZ traces remain intact in the run folder and are not duplicated in that report ZIP.
