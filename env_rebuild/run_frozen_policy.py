"""Closed-loop inference only with the archived SmolVLA checkpoint.

No optimizer, training dataset, gradients, or policy updates are used.
The reconstructed scene is still a diagnostic until camera fidelity is validated.
"""
from pathlib import Path
import argparse
import collections
import contextlib
import hashlib
import json
import os
import sys
import time
import traceback
import math

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "outputs/smolvla_so101_40k"
CHECKPOINT = SOURCE / "runs/so101_40k/training/checkpoints/040000/pretrained_model"
CHECKPOINT = Path(os.environ.get("SO101_CHECKPOINT", HERE.parent / "checkpoints/040000/pretrained_model"))
TORCH_SITE = Path(os.environ.get("SO101_TORCH_SITE", HERE / ".optional_torch_site"))
if TORCH_SITE.exists():
    sys.path.append(str(TORCH_SITE))


import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import SmolVLANewLineProcessor  # noqa: F401 register saved step
from lerobot.policies.factory import make_pre_post_processors
from rovo_env import SO101PickCubeEnv

RENAME = {"front":"camera1", "wrist":"camera2", "overhead":"camera3"}
TASK = "pick up the object and place it in the target location"
EXPECTED_MODEL_SHA256 = "f914663e73f099d56074c0e4176b736ac56e889d1b4b41605bcaf2fd598f23a0"


def load(device,checkpoint=CHECKPOINT):
    checkpoint = Path(checkpoint)
    digest = hashlib.sha256()
    with (checkpoint / "model.safetensors").open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    if digest.hexdigest() != EXPECTED_MODEL_SHA256:
        raise ValueError("This public runner requires the reported 40k checkpoint. "
                         "Its model.safetensors SHA-256 does not match the published identity; "
                         "do not substitute the base model for the fine-tuned policy.")
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = device
    cfg.push_to_hub = False
    cfg.load_vlm_weights = False
    cfg.n_action_steps = 10
    local_vlm = HERE / "inputs/smolvlm2_processor"
    if (local_vlm / "config.json").exists():
        cfg.vlm_model_name = str(local_vlm)
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=cfg).to(device).eval()
    overrides={"device_processor":{"device":device}}
    if (local_vlm / "config.json").exists():
        overrides["tokenizer_processor"]={"tokenizer_name":str(local_vlm)}
    pre, post = make_pre_post_processors(cfg, str(checkpoint), preprocessor_overrides=overrides)
    return policy, pre, post


def batch_from_observation(obs, pre):
    raw = {"observation.state":torch.from_numpy(obs["observation.state"]), "task":TASK}
    for source, target in RENAME.items():
        raw[f"observation.images.{target}"] = torch.from_numpy(obs[f"observation.images.{source}"]).permute(2,0,1).float()/255.0
    return pre(raw)


def evaluate_one(seed, policy, pre, post, device, max_steps, out, env_class=SO101PickCubeEnv, env_kwargs=None, action_steps=10, action_offset=1, precision="bfloat16", existing_env=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = existing_env if existing_env is not None else env_class(image_size=512, max_steps=max_steps, **(env_kwargs or {}))
    queue = collections.deque()
    positions, states, actions, latencies, qposes = [], [], [], [], []
    valid, success, error, clips = True, False, None, 0
    started = time.perf_counter()
    try:
        obs, info = env.reset(seed=seed)
        initial = np.asarray(info["cube_position"], dtype=float)
        bin_xy = env.bin_xyz[:2] if hasattr(env,"bin_xyz") else env.model.body_pos[env.bin_body, :2].copy()
        for processor in (pre, post):
            if hasattr(processor, "reset"):
                processor.reset()
        policy.reset()
        with torch.inference_mode():
            for step in range(max_steps):
                if not queue:
                    t = time.perf_counter()
                    batch = batch_from_observation(obs, pre)
                    with torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" and precision=="bfloat16" else contextlib.nullcontext():
                        chunk = post(policy.predict_action_chunk(batch)).float().cpu().numpy()[0]
                    if device == "cuda":
                        torch.cuda.synchronize()
                    latencies.append(time.perf_counter()-t)
                    queue.extend(chunk[action_offset:action_offset+action_steps])
                action = np.asarray(queue.popleft(), dtype=float)
                states.append(obs["observation.state"].copy())
                qposes.append(env.data.qpos.copy())
                actions.append(action.copy())
                positions.append(np.asarray(info["cube_position"], dtype=float))
                obs, _, term, trunc, info = env.step(action)
                clips += int(info["action_clipped"])
                if term or trunc:
                    success = bool(term)
                    break
        positions.append(np.asarray(info["cube_position"], dtype=float))
        qposes.append(env.data.qpos.copy())
        pos = np.asarray(positions)
        distances = np.linalg.norm(pos[:, :2]-bin_xy, axis=1)
        result = {"seed":seed,"valid":valid,"strict_bin_success":success,"lifted_4cm":bool(info["lifted"]),
                  "max_cube_height_gain_m":float(pos[:,2].max()-initial[2]),
                  "max_cube_xy_displacement_m":float(np.linalg.norm(pos[:,:2]-initial[:2],axis=1).max()),
                  "min_cube_to_bin_xy_m":float(distances.min()),"final_cube_to_bin_xy_m":float(distances[-1]),
                  "steps":len(actions),"action_clipped_steps":clips,"mean_chunk_latency_s":float(np.mean(latencies)) if latencies else None,
                  "p95_chunk_latency_s":float(np.percentile(latencies,95)) if latencies else None,
                  "initial_cube_xyz":initial.tolist(),"final_cube_xyz":pos[-1].tolist(),
                  "reset_source_episode":info["reset_source_episode"]}
        for stage in ("reached","grasped","transported"):
            if stage in info:result[stage]=bool(info[stage])
        if "reset" in info:result["reset"]=info["reset"]
        result["bin_xy"]=bin_xy.tolist()
        np.savez_compressed(out / f"seed_{seed}_trace.npz",cube_xyz=pos,states=np.asarray(states),actions=np.asarray(actions),qpos=np.asarray(qposes),chunk_latency_s=np.asarray(latencies))
    except Exception:
        valid, error = False, traceback.format_exc()
        result = {"seed":seed,"valid":False,"strict_bin_success":False,"error":error,"steps":len(actions)}
    finally:
        if existing_env is None:env.close()
    result["wall_seconds"] = time.perf_counter()-started
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",nargs="+",type=int,default=[30000])
    parser.add_argument("--device",choices=["auto","cuda","cpu"],default="auto")
    parser.add_argument("--max-steps",type=int,default=301)
    parser.add_argument("--environment",choices=["rovo"],default="rovo")
    parser.add_argument("--reset-mode",choices=["empirical","uniform"],default="uniform")
    parser.add_argument("--output",default=None)
    parser.add_argument("--action-steps",type=int,default=25)
    parser.add_argument("--action-offset",type=int,choices=[0,1],default=1)
    parser.add_argument("--contact-profile",choices=["upstream","source_visual"],default="upstream")
    parser.add_argument("--checkpoint-step",type=int,default=40000)
    parser.add_argument("--precision",choices=["float32","bfloat16"],default="bfloat16")
    parser.add_argument("--placement-tolerance-m",type=float,default=.0001)
    parser.add_argument("--render-samples",type=int,choices=[0,4],default=4)
    args=parser.parse_args()
    if not 1<=args.action_steps<=49:raise ValueError("Action steps must be in [1,49]")
    checkpoint=Path(os.environ.get("SO101_CHECKPOINT", HERE.parent/f"checkpoints/{args.checkpoint_step:06d}/pretrained_model"))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(8)
    out=Path(args.output) if args.output else HERE / ("policy_diagnostic" if args.environment=="calibrated" else f"policy_rovo_{args.reset_mode}")
    out.mkdir(parents=True,exist_ok=False)
    digest = hashlib.sha256()
    with (checkpoint / "model.safetensors").open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    checkpoint_sha = digest.hexdigest()
    scene = HERE / "calibrated_env/scene.xml" if args.environment=="calibrated" else HERE / "so-arm101-pick-place-main/assets/scene_front.xml"
    env_class=SO101PickCubeEnv;env_kwargs={}
    if args.environment=="rovo":
        from rovo_env import SO101PickCubeEnv as RovoEnv
        env_class=RovoEnv;env_kwargs={"reset_mode":args.reset_mode,"contact_profile":args.contact_profile,
                                   "placement_tolerance_m":args.placement_tolerance_m,"render_samples":args.render_samples}
    import mujoco
    meta={"kind":"frozen-policy closed-loop diagnostic in reconstructed scene", "device":device,
          "checkpoint_sha256":checkpoint_sha,"scene_sha256":hashlib.sha256(scene.read_bytes()).hexdigest(),
          "checkpoint_step":args.checkpoint_step,"checkpoint_path":str(checkpoint),"contact_profile":args.contact_profile,
          "precision":args.precision,
          "environment":args.environment,"reset_mode":args.reset_mode,"mujoco_version":mujoco.__version__,"torch_version":torch.__version__,
          "inference_runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "max_steps":args.max_steps,"seeds":args.seeds,
          "action_execution":f"Predict 50 actions, execute {args.action_steps} starting at index {args.action_offset}",
          "action_steps":args.action_steps,"action_offset":args.action_offset,
          "placement_tolerance_m":args.placement_tolerance_m if args.environment=="rovo" else None,
          "success_metric_version":"v2_contact_tolerance" if args.environment=="rovo" and args.placement_tolerance_m>0 else "v1_zero_tolerance",
          "render_samples":args.render_samples if args.environment=="rovo" else None,
          "runtime_note":"LeRobot 0.5.1 packaged runtime may differ from checkpoint's recorded 0.5.2 source commit; verify loading and processor compatibility.",
          "benchmark_validated":False,"reason":"Replica evaluation; original simulator unavailable. Visual/source-action validation is reported separately."}
    if device=="cuda":
        meta["gpu_name"]=torch.cuda.get_device_name(0)
        meta["gpu_total_MiB"]=torch.cuda.get_device_properties(0).total_memory/1024**2
        meta["allow_tf32_matmul"]=torch.backends.cuda.matmul.allow_tf32
    if args.environment=="rovo":
        meta["environment_wrapper_sha256"]=hashlib.sha256((HERE/"rovo_env.py").read_bytes()).hexdigest()
        meta["upstream_commit"]="f27633c80eb6d852049fafb7c72a9584ec944ba7"
        meta["scene_files_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/"so-arm101-pick-place-main/assets").glob("*.xml")}
        if args.contact_profile=="source_calibrated":
            calibration=json.loads((HERE/"rovo_dynamics_calibration.json").read_text())
            meta["source_calibration_parameters"]=calibration["selected"]["parameters"]
        if args.contact_profile=="source_visual":
            path=HERE/"release_audit/material_calibration/selection.json"
            meta["source_material_parameters"]=json.loads(path.read_text())["selected"]["parameters"]
            meta["source_material_selection_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
            description="Two unchanged upstream 2.5 mm pads with source-image-selected separate cube and bin materials"
        elif args.contact_profile=="upstream":
            description="Two upstream 2.5 mm box pads"
        else:
            p=meta["source_calibration_parameters"]
            description=f"Two {p['pad_half_size_m']*2000:g} mm box pads, {p['controller']} controller, gripper {p['gripper_force_Nm']:g} Nm, cube friction {p['cube_friction']:g}"
        meta["contact_model"]=description+" plus robot mesh collisions; no attachment or teleport grasp"
        meta["reset_distribution"]=("100 Rovo source-replay placements, sampled with replacement" if args.reset_mode=="empirical" else "Cube and bin x uniform [0.075,0.18] m, y uniform [-0.08,0.08] m, separation >0.075 m and >3 mm axis-aligned object clearance, source home pose")
    (out / "run_meta.json").write_text(json.dumps(meta,indent=2))
    policy,pre,post=load(device,checkpoint)
    rows=[]
    for seed in args.seeds:
        result=evaluate_one(seed,policy,pre,post,device,args.max_steps,out,env_class,env_kwargs,args.action_steps,args.action_offset,args.precision)
        rows.append(result)
        (out / "episodes.json").write_text(json.dumps(rows,indent=2))
        print(f"seed {seed}: valid={result['valid']} success={result['strict_bin_success']} steps={result['steps']}",flush=True)
    summary={**meta,"episodes":len(rows),"valid_episodes":sum(r["valid"] for r in rows),
             "successes":sum(r["valid"] and r["strict_bin_success"] for r in rows),
             "lifted_episodes":sum(r.get("lifted_4cm",False) for r in rows),
             "denominator":"All requested seeds, including invalid episodes"}
    summary["success_rate"] = summary["successes"]/len(rows)
    n=len(rows); k=summary["successes"]; z=1.959963984540054
    center=(k/n+z*z/(2*n))/(1+z*z/n)
    radius=z*math.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    summary["wilson_95_interval"]=[center-radius,center+radius]
    if device=="cuda":summary["peak_gpu_allocated_MiB"]=torch.cuda.max_memory_allocated()/1024**2
    (out / "summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
