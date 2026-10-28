# Components and attribution

| Component | Source | Role |
|---|---|---|
| SmolVLA and LeRobot | [Hugging Face LeRobot](https://github.com/huggingface/lerobot) | Policy architecture, processors and inference framework |
| Demonstrations | [gpudad/so101_pick_cube](https://huggingface.co/datasets/gpudad/so101_pick_cube) | Training/validation demonstration source |
| SO-101 scene replica | [Rovo Labs](https://github.com/Rovo-Labs/so-arm101-pick-place) | Robot meshes, camera-specific scene appearance and reference expert |
| Placement metadata | [rovolabs/so-arm101-pick-place](https://huggingface.co/datasets/rovolabs/so-arm101-pick-place) | Initial-condition catalogue for earlier selection probes |
| Physics | [MuJoCo](https://github.com/google-deepmind/mujoco) | Contact simulation and rendering |

The Rovo scene is pinned to commit `f27633c80eb6d852049fafb7c72a9584ec944ba7`. Its Apache-2.0 LICENSE and NOTICE are included under `evidence/licenses/`. The simulation videos render this scene.

This repository contains the dataset repair and training workflow, policy integration, named-joint/camera mapping, evaluation instrumentation and contact/rendering diagnostics.

The 40k policy used in the displayed paired experiments has SHA-256 `f914663e73f099d56074c0e4176b736ac56e889d1b4b41605bcaf2fd598f23a0`. Weights and the full training corpus are not bundled in this repository.

The earlier reconstruction in `training/simulator/` uses SO-101 assets from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100). Its Apache-2.0 license and asset hashes are retained in that folder. The article uses locally served DejaVu Sans; its font license is included beside the font.
