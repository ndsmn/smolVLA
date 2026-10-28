"""Predeclared paired inference comparison with one frozen 40k checkpoint."""
from pathlib import Path
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import json
import hashlib
import argparse
import numpy as np
import mujoco
from run_frozen_policy import load,evaluate_one,torch,CHECKPOINT
from rovo_env import SO101PickCubeEnv
from audit_placement_tolerance import wilson
from contact_materials import apply_materials

HERE=Path(__file__).resolve().parent;OUT=HERE/"release_audit/paired_material_evaluation"
SEEDS=list(range(35000,35050))


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument("--render-samples",type=int,choices=[0,4],default=0)
    args=parser.parse_args()
    if args.render_samples==4:OUT=HERE/"release_audit/paired_material_evaluation_msaa4"
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    material_path=HERE/"release_audit/material_calibration/selection.json"
    selected=json.loads(material_path.read_text())["selected"]["parameters"]
    meta={"kind":"Paired material comparison, same frozen checkpoint and 50 initial conditions",
        "seeds":SEEDS,"checkpoint_step":40000,"checkpoint_path":str(CHECKPOINT),
        "checkpoint_sha256":hashlib.sha256((CHECKPOINT/"model.safetensors").read_bytes()).hexdigest(),
        "reset_mode":"uniform","precision":"bfloat16","action_steps":25,"action_offset":1,"max_steps":301,
        "placement_tolerance_m":.0001,"success_metric_version":"v2_contact_tolerance",
        "contact_model":"Unchanged 2.5 mm finger pads and scene geometry; only the material profile differs",
        "material_selection":selected,"material_selection_sha256":hashlib.sha256(material_path.read_bytes()).hexdigest(),
        "torch_version":torch.__version__,"mujoco_version":mujoco.__version__,"device":"cuda",
        "deterministic_algorithms":True,"cublas_workspace_config":os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "scene_lifecycle":"One shared compiled scene and persistent renderers; only material arrays restored/changed between attempts",
        "render_samples":args.render_samples,
        "profile_order":"fixed upstream then candidate" if args.render_samples==0 else "balanced deterministic seed-based random order",
        "training_performed":False,"benchmark_validated":False,
        "source_success_labels_available":False,
        "selection_basis":"Material fixed by source episode 1 image similarity before these outcomes; 40k retained as fixed reference, no checkpoint selection"}
    if (OUT/"summary.json").exists():raise SystemExit("Paired experiment already completed; do not overwrite")
    existing=OUT/"plan.json"
    if existing.exists() and json.loads(existing.read_text())!=meta:raise RuntimeError("Experiment plan changed")
    existing.write_text(json.dumps(meta,indent=2))
    folders={p:OUT/p for p in ("upstream","source_visual")};rows={}
    for profile,folder in folders.items():
        folder.mkdir(exist_ok=True)
        (folder/"run_meta.json").write_text(json.dumps({**meta,"contact_profile":profile},indent=2))
        rows[profile]=json.loads((folder/"episodes.json").read_text()) if (folder/"episodes.json").exists() else []
    policy,pre,post=load("cuda")
    env=SO101PickCubeEnv(image_size=512,max_steps=301,reset_mode="uniform",placement_tolerance_m=.0001,render_samples=args.render_samples)
    names=("geom_friction","geom_priority","geom_condim","geom_solref","geom_solimp")
    original={cam:{k:getattr(m,k).copy() for k in names} for cam,m in env.models.items()}
    for seed in SEEDS:
        order=list(folders)
        if args.render_samples==4 and np.random.default_rng(seed+90000).integers(2):order.reverse()
        for profile in order:
            folder=folders[profile]
            if any(r["seed"]==seed for r in rows[profile]):continue
            for cam,m in env.models.items():
                for k in names:getattr(m,k)[:]=original[cam][k]
            if profile=="source_visual":apply_materials(env,selected)
            env.contact_profile=profile
            row=evaluate_one(seed,policy,pre,post,"cuda",301,folder,SO101PickCubeEnv,
                {"reset_mode":"uniform","contact_profile":profile,"placement_tolerance_m":.0001},25,1,"bfloat16",existing_env=env)
            if not row["valid"]:raise RuntimeError(row["error"])
            rows[profile].append(row);(folder/"episodes.json").write_text(json.dumps(rows[profile],indent=2))
            print(profile,seed,"success",row["strict_bin_success"],flush=True)
            if seed==SEEDS[0] and profile=="upstream":
                repeat=OUT/"repeatability_check";repeat.mkdir(exist_ok=True)
                r=evaluate_one(seed,policy,pre,post,"cuda",301,repeat,SO101PickCubeEnv,
                    {"reset_mode":"uniform","contact_profile":"upstream","placement_tolerance_m":.0001},25,1,"bfloat16",existing_env=env)
                a=np.load(folder/f"seed_{seed}_trace.npz");b=np.load(repeat/f"seed_{seed}_trace.npz")
                same=r["valid"] and all(a[k].shape==b[k].shape and np.array_equal(a[k],b[k]) for k in ("actions","qpos"))
                (repeat/"result.json").write_text(json.dumps({"same_process_exact_repeat":bool(same),"seed":seed,"independent_process_repeat_tested":False},indent=2))
                if not same and args.render_samples==0:raise RuntimeError("Identical baseline repeat was not exact; resolve before material comparison")
        (OUT/"progress.json").write_text(json.dumps({p:{"completed":len(rs),"successes":sum(r["strict_bin_success"] for r in rs)} for p,rs in rows.items()},indent=2))
    env.close()
    for profile,folder in folders.items():
        rs=rows[profile];k=sum(r["strict_bin_success"] for r in rs)
        summary={**meta,"contact_profile":profile,"episodes":len(rs),"valid_episodes":len(rs),"successes":k,
            "success_rate":k/len(rs),"wilson_95_interval":wilson(k,len(rs)),"lifted_episodes":sum(r["lifted_4cm"] for r in rs),
            "peak_gpu_allocated_MiB":torch.cuda.max_memory_allocated()/1024**2}
        (folder/"summary.json").write_text(json.dumps(summary,indent=2))
    base={r["seed"]:r for r in rows["upstream"]};candidate={r["seed"]:r for r in rows["source_visual"]}
    cells={"both_success":0,"only_upstream":0,"only_source_visual":0,"both_failure":0}
    for seed in SEEDS:
        a=base[seed]["strict_bin_success"];b=candidate[seed]["strict_bin_success"]
        cells["both_success" if a and b else "only_upstream" if a else "only_source_visual" if b else "both_failure"]+=1
    from math import comb
    d=cells["only_upstream"]+cells["only_source_visual"];small=min(cells["only_upstream"],cells["only_source_visual"])
    p=min(1.,2*sum(comb(d,k) for k in range(small+1))/2**d) if d else 1.
    (OUT/"summary.json").write_text(json.dumps({"plan":meta,"paired_counts":cells,"mcnemar_exact_two_sided_p":p,
        "upstream_successes":sum(r["strict_bin_success"] for r in rows["upstream"]),
        "source_visual_successes":sum(r["strict_bin_success"] for r in rows["source_visual"]),
        "episodes_per_profile":len(SEEDS)},indent=2))
    print("COMPLETED",cells,"exact p",p,flush=True)


if __name__=="__main__":main()
