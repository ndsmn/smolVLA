# Archived data preparation and fine-tuning

This folder records the fine-tuning run and saved-checkpoint evaluation. The local 6 GB GPU was used for inference. Fine-tuning ran on a Threadripper PRO workstation with an NVIDIA RTX 6000 Ada 48 GB GPU. Evidence verification and CI do not launch training.

## Recorded workflow

1. Audit `gpudad/so101_pick_cube`: global timestamps, dummy reset actions, wrist-stream offset and exact numeric-trajectory duplicates.
2. Quarantine source episodes 232 and 2672; drop the first dummy action in each retained episode; correct wrist offsets after 2672 by −301/30 s.
3. Group identical numeric trajectories and split with seed 42: 4,725 training / 525 validation groups, comprising 9,911 / 1,080 episodes.
4. Recompute numeric normalization on training frames only. Preserve original videos by reference rather than rewriting them.
5. Benchmark batches on the earlier GPU, select batch 32, then perform 40,000 updates with BF16 through Accelerate. Freeze the vision encoder and train the action expert.
6. Save eight full checkpoints at 5,000-update intervals. Evaluate the base model and all checkpoints on 128 held-out trajectory anchors, masking padded action elements.

The original data repair and preparation evidence is retained under `preflight/`, with the finalized split under `configs/`. `configs/experiment_plan.json` is an archived record, including the older reconstructed-simulator selection/final seeds and results.

## Source

| File | Role |
|---|---|
| `prepare_dataset.py` | Dataset repair, numeric normalization and grouped split |
| `experiment_utils.py` | Camera rename map, training arguments, logging and plots |
| `train_entry.py` | LeRobot training entry point instrumented with per-update telemetry |
| `evaluate_checkpoints.py` | Archived held-out metrics and legacy simulation evaluation |
| `pipeline.py` | Archived host-side benchmark/training/evaluation orchestration |
| `simulator/` | Earlier reconstruction used during checkpoint selection; distinct from the current Rovo replica |

Portability edits replace the original machine-specific dataset and base-model paths with `SO101_DATASET` and `SO101_BASE_CHECKPOINT`. Historical record paths are replaced with labelled placeholders in this public export. Numeric measurements are preserved. These scripts retain the original workflow and its prerequisite checks; they are not a one-command reproduction bundle. Full original videos, derived data, base/fine-tuned weights, the original robot XML inputs to regenerate the older reconstruction (its compiled scene and meshes are included), and the earlier compatible training runtime must be supplied separately. Some archived pipeline steps expect the complete original preflight directory.

Do not rerun the preliminary dataset audit over this folder's finalized split. Preserve the completed split and source files; create a separate experiment directory for new training work.

## Training recipe and results

Batch 32, AdamW at initial LR 1e-4, 1,000 warmup updates, decay over 40,000 updates, 50-step action chunks. Training elapsed time was approximately 16 h 3 min; the full earlier job including benchmarks/evaluation took about 24 h 5 min.

Actual per-update CSV logs, GPU telemetry and saved configurations are under [`docs/evidence/training/`](../docs/evidence/training/). The training configuration has `use_amp: false`, while the launcher explicitly sets `ACCELERATE_MIXED_PRECISION=bf16`; refer to both rather than inferring precision from one field alone.

Held-out action MAE: **0.212416 rad → 0.045478 rad**, a **78.6% relative reduction**. This metric averages valid action elements on 128 fixed trajectory anchors; it is not task accuracy or exhaustive validation of every held-out timestep.

The earlier reconstruction returned zero successes for every checkpoint. The predeclared offline-error tie-breaker selected step 40,000. That selected checkpoint is fixed in the later material/rendering study. Later 34%/42% rates belong to a changed replica and reset distribution, so they cannot establish a controlled improvement from the earlier 0%.

## Dataset availability

Source: [gpudad/so101_pick_cube](https://huggingface.co/datasets/gpudad/so101_pick_cube). The public repository contains metadata, split and repair evidence, not the full training corpus. Source video paths from the transferred earlier-machine data are unavailable locally. Full data is unnecessary to inspect the saved results or run inference with the supplied checkpoint.
