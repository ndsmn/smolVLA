"""Reproduce saved actions and audit a disclosed numerical contact tolerance.

No inference, scene change, new policy selection, or training. The existing
online metric is reproduced at zero tolerance; a parallel counter evaluates
the same states with 0.1 mm horizontal wall-contact tolerance.
"""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
from rovo_env import SO101PickCubeEnv

HERE=Path(__file__).resolve().parent;OUT=HERE/"release_audit"


def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-r,c+r]


def main():
    OUT.mkdir(exist_ok=True);reports={}
    for name in ("final_evaluation","final_checkpoint_evaluation"):
        folder=HERE/name;source=json.loads((folder/"episodes.json").read_text())
        env=SO101PickCubeEnv(observations_with_images=False,max_steps=301,reset_mode="uniform",placement_tolerance_m=0)
        records=[]
        for row in source:
            if not row["valid"]:raise RuntimeError("Invalid original episode requires explicit handling")
            env.reset(seed=row["seed"],options=row["reset"])
            trace=np.load(folder/f"seed_{row['seed']}_trace.npz")
            max_error=float(np.abs(env.data.qpos-trace["qpos"][0]).max())
            streak=0;first_success=None;legacy_success=False
            for t,action in enumerate(trace["actions"]):
                _,_,term,trunc,info=env.step(action)
                max_error=max(max_error,float(np.abs(env.data.qpos-trace["qpos"][t+1]).max()))
                legacy_success|=term
                env.placement_tolerance_m=.0001
                streak=streak+1 if env._success_candidate() else 0
                if streak>=15 and first_success is None:first_success=t+1
                env.placement_tolerance_m=0
                if term or trunc:
                    if t!=len(trace["actions"])-1:raise RuntimeError("Replay terminated earlier than original trace")
            matches=max_error<1e-7 and legacy_success==row["strict_bin_success"]
            if not matches:raise RuntimeError(f"Cannot audit unmatched replay {name} seed={row['seed']} error={max_error}")
            rec={"seed":row["seed"],"legacy_success":legacy_success,"contact_tolerance_success":first_success is not None,
                "first_success_step_v2":first_success,"max_qpos_error_vs_saved":max_error,"legacy_reproduced":matches,
                "lifted":env.lifted,"transported":env.transported}
            records.append(rec)
            print(name,row["seed"],legacy_success,"->",rec["contact_tolerance_success"],"max error",max_error,flush=True)
        env.close()
        corrected=sum(r["contact_tolerance_success"] for r in records);old=sum(r["legacy_success"] for r in records)
        reports[name]={"checkpoint_step":40000 if name=="final_evaluation" else 30000,
            "episodes":len(records),"legacy_successes":old,"contact_tolerance_successes":corrected,
            "wilson_95_interval_v2":wilson(corrected,len(records)),"records":records}
        (OUT/f"{name}_metric_audit.json").write_text(json.dumps(reports[name],indent=2))
    (OUT/"placement_tolerance_audit.json").write_text(json.dumps({
        "method":"Exact dynamics replay of all saved commands; legacy state agreement and outcomes checked before parallel metric evaluation",
        "metric_change":"Horizontal all-corner containment allows 0.0001 m numerical contact overlap; all other stage/support/release/velocity/15-step requirements unchanged",
        "physics_changed":False,"model_weights_changed":False,"new_inference":False,
        "wrapper_sha256":hashlib.sha256((HERE/"rovo_env.py").read_bytes()).hexdigest(),
        "runs":reports},indent=2))


if __name__=="__main__":main()
