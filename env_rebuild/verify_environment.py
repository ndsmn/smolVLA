"""Verify the geometry-bounds repair, seeded reset and camera state mirroring."""
from pathlib import Path
import json
import os
os.environ.setdefault("MUJOCO_GL","egl")
import numpy as np
import mujoco
from rovo_env import SO101PickCubeEnv,HOME

HERE=Path(__file__).resolve().parent
checks=[]
for profile in ("upstream","source_visual"):
    env=SO101PickCubeEnv(observations_with_images=False,contact_profile=profile)
    try:
        env.reset(seed=12345);q=env.data.qpos.copy();target=env.data.mocap_pos.copy()
        env.reset(seed=12345)
        deterministic=bool(np.array_equal(q,env.data.qpos) and np.array_equal(target,env.data.mocap_pos))
        mapped=bool(np.array_equal(env.data.qpos[env.qadr],HOME) and np.array_equal(env.data.ctrl[env.aidx],HOME))
        bounds=[]
        for model in env.models.values():
            for geom in env.pad_geoms:
                size=model.geom_size[geom]
                bounds.append(bool(np.allclose(model.geom_aabb[geom,3:],size) and np.isclose(model.geom_rbound[geom],np.linalg.norm(size))))
        _,_,_,_,info=env.step(HOME)
        shapes=[]
        for cam in ("front","wrist","overhead"):
            image=env.render_camera(cam)
            shapes.append(bool(image.shape==(512,512,3) and image.dtype==np.uint8))
        mirrored=all(np.array_equal(data.qpos,env.data.qpos) and np.array_equal(data.mocap_pos,env.data.mocap_pos) for data in env.datas.values())
        row={"profile":profile,"deterministic_reset":deterministic,"named_joint_mapping":mapped,
             "compiled_pad_collision_bounds":all(bounds),"valid_rgb_outputs":all(shapes),"camera_states_mirrored":mirrored}
        if not all(value for key,value in row.items() if key!="profile"):raise AssertionError(row)
        checks.append(row)
    finally:env.close()
env=SO101PickCubeEnv(observations_with_images=False,reset_mode="uniform")
try:
    valid=True
    outer=env.model.geom_size[env.model.geom("bin_0").id,:2]
    for seed in range(100):
        env.reset(seed=seed)
        cube=env.data.qpos[env.cube_adr:env.cube_adr+3]
        target=env.bin_xyz;delta=np.abs(cube[:2]-target[:2])
        valid &= bool(np.linalg.norm(delta)>.075 and np.any(delta-outer-env.half_cube[:2]>.003-1e-12))
        valid &= bool(.075<=cube[0]<=.18 and .075<=target[0]<=.18 and abs(cube[1])<=.08 and abs(target[1])<=.08)
    if not valid:raise AssertionError("Uniform reset bounds or collision clearance failed")
    checks.append({"profile":"uniform_reset_sampler","100_seeds_within_bounds_and_nonoverlapping":True})
finally:env.close()
env=SO101PickCubeEnv(observations_with_images=False)
try:
    env.reset(seed=1,options={"cube_xyz":[.169,-.07,.025],"bin_xyz":[.16,-.07,0.]})
    env.lifted=True
    fixture=[]
    for overlap in (.000005,.001):
        env.data.qpos[env.cube_adr]=.16+env.bin_inner-env.half_cube[0]+overlap
        env.data.qvel[:]=0;mujoco.mj_forward(env.model,env.data)
        env.placement_tolerance_m=0;old=env._success_candidate()
        env.placement_tolerance_m=.0001;new=env._success_candidate()
        assert not old and new==(overlap<.0001)
        fixture.append({"wall_overlap_m":overlap,"legacy_candidate":old,"v2_candidate":new})
    checks.append({"profile":"placement_metric_regression","passed":True,"fixtures":fixture})
finally:env.close()
base=SO101PickCubeEnv(observations_with_images=False)
candidate=SO101PickCubeEnv(observations_with_images=False,contact_profile="source_visual")
try:
    attributes=("geom_size","geom_pos","geom_quat","body_pos","body_quat","body_mass","body_inertia","actuator_gainprm","actuator_biasprm","actuator_forcerange","actuator_ctrlrange")
    equal=all(np.array_equal(getattr(base.model,k),getattr(candidate.model,k)) for k in attributes)
    assert equal
    checks.append({"profile":"material_only_change","geometry_mass_and_actuators_unchanged":equal})
finally:base.close();candidate.close()
(HERE/"environment_verification.json").write_text(json.dumps({"passed":True,"checks":checks},indent=2))
print(json.dumps(checks,indent=2))
