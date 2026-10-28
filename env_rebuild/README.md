# Frozen-policy inference and simulation

This directory contains the evaluated MuJoCo integration. `source_export.json` records hashes of the original local source and the portability changes. The wrapper supports the two profiles in the main study: `upstream` and `source_visual`. The older controller-fit and legacy reconstruction are excluded from the public inference entry point.

## What is included

- `rovo_env.py`: Gymnasium interface, named joint order, three-camera state mirroring, 500 Hz physics and placement metric.
- `run_frozen_policy.py`: saved processor loading, frozen SmolVLA action chunks, stage outcomes, inference latency and full action/qpos traces.
- `evaluate_material_pair.py`: fixed 50-seed material comparison, persistent compiled scene and paired statistics.
- `verify_environment.py`: reset, image, mapping, metric and material-only checks.
- `contact_materials.py`: isolated material changes.
- Pinned upstream scene/meshes, LICENSE, NOTICE and the empirical placement catalogue. The catalogue draw is retained even for uniform resets because removing it changes seeded starts.

## Runtime

The measured runtime used Python 3.12, PyTorch 2.12.0+cu130, MuJoCo 3.11.0, LeRobot 0.5.1 and Transformers 5.5.4. The archived training source identifies LeRobot 0.5.2; the imported local runtime is therefore not bit-identical to that training environment. See [source provenance](../docs/evidence/source_provenance.json).

Use an isolated environment. The recorded dependency list intentionally excludes PyTorch/torchvision, which must be installed as a compatible pair for your CUDA driver. It is a runtime record rather than a promise that all recorded packages support every other machine.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# Install the compatible PyTorch/torchvision pair for your CUDA driver first.
.venv/bin/python -m pip install --no-deps -r env_rebuild/requirements-inference.txt
.venv/bin/python env_rebuild/patch_inference_runtime.py
.venv/bin/python env_rebuild/download_runtime_assets.py
```

The import-only patch targets isolated LeRobot 0.5.1 and avoids eager imports of unrelated policies on Python 3.12. It does not alter SmolVLA's forward pass or weights. Read it before applying it to your environment.

## Supply the exact checkpoint

Weights are not included and currently have no public download in this repository. Supply the original `040000/pretrained_model/` directory, including its processor/normalization files. Do not substitute `smolvla_base` for the reported fine-tuned checkpoint.

```bash
export SO101_CHECKPOINT=/absolute/path/to/040000/pretrained_model
sha256sum "$SO101_CHECKPOINT/model.safetensors"
```

Expected model SHA-256:

```text
f914663e73f099d56074c0e4176b736ac56e889d1b4b41605bcaf2fd598f23a0
```

Processor assets are downloaded from `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` by the helper above. They supply configuration/tokenization files, not replacement policy weights. After those files are available, inference can run offline:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python env_rebuild/run_frozen_policy.py \
  --device cuda --environment rovo --reset-mode uniform \
  --contact-profile upstream --precision bfloat16 \
  --checkpoint-step 40000 --action-steps 25 --action-offset 1 \
  --render-samples 4 --placement-tolerance-m 0.0001 \
  --max-steps 301 --output results/new_seeds --seeds 36000 36001
```

Use new output directories. The runner refuses to overwrite an existing directory. The upstream profile remains the provisional default; the candidate's observed advantage was inconclusive.

Run the source integration checks with:

```bash
.venv/bin/python env_rebuild/verify_environment.py
```

The paired experiment is expensive relative to reading saved evidence. If deliberately rerunning it, inspect its seed list and plan first, then run `evaluate_material_pair.py --render-samples 4` or `--render-samples 0`. This starts policy inference, not training. It writes under `env_rebuild/release_audit/` and will not overwrite a completed experiment.

`audit_placement_tolerance.py` is retained as the historical replay audit implementation. It expects the older full-checkout rollout folders, which are not bundled here; its completed audit records are included under `docs/evidence/`.

## Interpretation

The simulator waits while chunks are computed. 500 Hz physics / 29.4 Hz simulated control does not mean inference runs at that wall-clock rate. MSAA-4 rendering is not exactly repeatable in the tested setup. A successful video is a selected example, while the complete outcome records define the reported rate.
