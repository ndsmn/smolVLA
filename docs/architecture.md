# Architecture

## Policy inputs and actions

SmolVLA observes front, wrist and overhead RGB images, six joint angles and the instruction `pick up the object and place it in the target location`. The saved preprocessor maps the views to `camera1`, `camera2` and `camera3` and applies the checkpoint's normalization.

The policy predicts 50 absolute joint targets. The runner executes 25, starting at index 1, before requesting a fresh observation. Joint channels are mapped by name in this order:

```text
elbow_flex, gripper, shoulder_lift, shoulder_pan, wrist_flex, wrist_roll
```

The policy receives no cube or bin coordinates. The evaluator reads those coordinates from MuJoCo to calculate task metrics.

## Simulation

`env_rebuild/rovo_env.py` wraps the pinned Rovo Labs scene as a Gymnasium environment. One model owns physics; its state is mirrored into the two other camera-specific appearance models before rendering. Actions are held for 17 physics substeps at 2 ms each, giving a 34 ms simulated control interval.

The wrapper retains the upstream scene's geometry and finger contact pads. Cube motion follows contact dynamics. The original dataset simulator was unavailable, and source-action replay validation remains incomplete.

Uniform resets sample cube and bin positions within the documented workspace, with a minimum separation and collision clearance. The empirical catalogue draw is retained before sampling so that recorded uniform seed sequences remain unchanged.

## Training and checkpoint selection

The repaired dataset contains 10,991 episodes and 1,445,747 frames. Exact state/action trajectory duplicates stay in the same split. Numeric normalization is calculated from training frames only.

The fine-tuning run used 40,000 optimizer updates, batch 32, a frozen vision encoder and action-expert updates. Eight checkpoints were saved at 5,000-step intervals. Fine-tuning used a Threadripper PRO workstation with an RTX 6000 Ada 48 GB. The 6 GB RTX 3060 was used for local inference.

Offline evaluation uses 128 distinct held-out trajectory anchors and 50-step chunks with padding masked. Action MAE decreased from 0.2124 to 0.0455 rad. All checkpoints tied at zero successes in the earlier reconstruction, so the predeclared offline-error tie-breaker selected step 40,000.

## Evaluation

The material/rendering study fixes the 40k weights and reuses 50 starts across four configurations. Its 200 outcome records are under `evidence/`; all 100 main-comparison action/qpos traces are included.

The original-rendering comparison scored 17/50 placements with upstream materials and 21/50 with the source-fitted candidate. The exact paired McNemar test gives p = 0.4545. Candidate lift rate was 41/50, while bin XY was reached after a lift in 39/50. These intermediate events do not certify placement.

Strict placement requires prior lift, rotated-cube XY containment, floor-height compatibility, release from robot contact and low velocity for 15 consecutive control steps. XY containment permits 0.1 mm of numerical wall-contact overlap. A replay audit reproduced 100 old traces before applying that metric correction.

## Rendering and timing

With MSAA-4 enabled, repeated resets sometimes differed by one intensity level in a few pixels, affecting BF16 actions. MSAA-0 produced exact same-process action/state repetition in the tested rollout but lower observed completion in the study. Independent-process and cross-hardware exact repetition are untested.

Simulation waits while the policy computes each chunk. The simulated control frequency and video playback speed do not measure real-time hardware throughput. Videos rerender saved states rather than rerunning inference.
