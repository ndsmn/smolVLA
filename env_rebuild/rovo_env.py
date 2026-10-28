"""Gymnasium wrapper for the Apache-2.0 Rovo Labs GPU-DAD replica.

Physics is stepped in one model; its state is mirrored to the other two models
for camera-specific paint and lighting. Actions are absolute joint radians in
GPU-DAD dataset order. The two small upstream contact pads remain disclosed.
"""
from pathlib import Path
from itertools import product
import os
import json
os.environ.setdefault("MUJOCO_GL", "egl")
import gymnasium as gym
import mujoco
import numpy as np
import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parent
ASSETS=HERE/"so-arm101-pick-place-main/assets"
NAMES=["elbow_flex","gripper","shoulder_lift","shoulder_pan","wrist_flex","wrist_roll"]
HOME=np.array([-.8,.6,.5,0,.3,0])
SCENES={"front":("scene_front.xml","front"),"wrist":("scene_wrist.xml","wrist_cam"),"overhead":("scene_overhead.xml","overhead")}


class SO101PickCubeEnv(gym.Env):
    metadata={"render_modes":["rgb_array"],"render_fps":30}

    def __init__(self,image_size=512,max_steps=301,observations_with_images=True,reset_mode="empirical",contact_profile="upstream",pad_half_size_m=None,placement_tolerance_m=0.0001,render_samples=4,**kwargs):
        super().__init__()
        self.image_size=image_size;self.max_steps=max_steps
        self.observations_with_images=observations_with_images;self.reset_mode=reset_mode
        if not np.isfinite(placement_tolerance_m) or not 0<=placement_tolerance_m<=.0001:
            raise ValueError("Placement contact tolerance must be in [0, 0.0001] m")
        self.placement_tolerance_m=float(placement_tolerance_m)
        params=None
        material_params=None
        if contact_profile=="source_visual":
            material_params=json.loads((HERE/"release_audit/material_calibration/selection.json").read_text())["selected"]["parameters"]
        elif contact_profile!="upstream":raise ValueError("Unknown contact profile")
        def compile_scene(filename):
            if pad_half_size_m is None:return mujoco.MjModel.from_xml_path(str(ASSETS/filename))
            spec=mujoco.MjSpec.from_file(str(ASSETS/filename))
            for name in ("static_finger_pad","moving_finger_pad"):spec.geom(name).size=[pad_half_size_m]*3
            return spec.compile()
        self.models={k:compile_scene(v[0]) for k,v in SCENES.items()}
        if render_samples not in (0,4):raise ValueError("Use 0 for reproducible RGB, or 4 for archived rendering")
        self.render_samples=render_samples
        for model in self.models.values():model.vis.quality.offsamples=render_samples
        self.datas={k:mujoco.MjData(m) for k,m in self.models.items()}
        self.model=self.models["front"];self.data=self.datas["front"]
        self.qadr=np.array([self.model.joint(n).qposadr[0] for n in NAMES])
        self.aidx=np.array([self.model.actuator(n).id for n in NAMES])
        self.cube_geom=self.model.geom("cube").id
        body=self.model.body("cube").id
        joint=self.model.body_jntadr[body]
        self.cube_adr=int(self.model.jnt_qposadr[joint]);self.cube_dof=int(self.model.jnt_dofadr[joint])
        self.half_cube=self.model.geom_size[self.cube_geom].copy()
        self.bin_body=self.model.body("container").id
        self.bin_mocap=int(self.model.body_mocapid[self.bin_body])
        self.pad_geoms=np.array([self.model.geom(n).id for n in ("static_finger_pad","moving_finger_pad")])
        self.contact_profile=contact_profile
        if material_params is not None:
            from contact_materials import apply_materials
            apply_materials(self,material_params)
        self.fixed_body=self.model.body("gripper").id
        self.moving_body=self.model.body("moving_jaw_so101_v1").id
        root=self.model.body("base").id
        robot={root}
        for b in range(self.model.nbody):
            if int(self.model.body_parentid[b]) in robot:robot.add(b)
        self.robot_geoms={g for g in range(self.model.ngeom) if int(self.model.geom_bodyid[g]) in robot}
        floor=self.model.geom("bin_0").id;wall=self.model.geom("bin_1").id
        self.bin_inner=float(abs(self.model.geom_pos[wall,0])-self.model.geom_size[wall,0])
        self.bin_floor_top=float(self.model.geom_pos[floor,2]+self.model.geom_size[floor,2])
        ranges=self.model.actuator_ctrlrange[self.aidx]
        self.action_space=gym.spaces.Box(ranges[:,0].astype(np.float32),ranges[:,1].astype(np.float32))
        spaces={"observation.state":gym.spaces.Box(-np.inf,np.inf,(6,),np.float32)}
        if observations_with_images:
            for cam in SCENES:spaces[f"observation.images.{cam}"]=gym.spaces.Box(0,255,(image_size,image_size,3),np.uint8)
        self.observation_space=gym.spaces.Dict(spaces)
        self.renderers={}
        columns=["episode_index","placement_label","cube_starting_position_xyz_m","bin_starting_position_xyz_m"]
        table=pq.read_table(HERE/"inputs/rovolabs_dataset/meta/episodes/chunk-000/file-000.parquet",columns=columns)
        self.catalogue=table.slice(0,100).to_pylist()
        # All camera models must expose identical state layouts.
        for m in self.models.values():
            if (m.nq,m.nv,m.nu,m.nmocap)!=(self.model.nq,self.model.nv,self.model.nu,self.model.nmocap):
                raise RuntimeError("Camera scene state layouts differ")

    @property
    def bin_xyz(self):return self.data.mocap_pos[self.bin_mocap].copy()

    def _obs(self):
        obs={"observation.state":self.data.qpos[self.qadr].astype(np.float32).copy()}
        if self.observations_with_images:
            for cam in SCENES:obs[f"observation.images.{cam}"]=self.render_camera(cam)
        return obs

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        for k,m in self.models.items():mujoco.mj_resetData(m,self.datas[k])
        rec=self.catalogue[int(self.np_random.integers(len(self.catalogue)))]
        cube=np.array(rec["cube_starting_position_xyz_m"],dtype=float)
        target=np.array(rec["bin_starting_position_xyz_m"],dtype=float)
        label=rec["placement_label"]
        if self.reset_mode=="uniform":
            cube=np.array([self.np_random.uniform(.075,.18),self.np_random.uniform(-.08,.08),.02])
            for _ in range(100):
                target=np.array([self.np_random.uniform(.075,.18),self.np_random.uniform(-.08,.08),0.])
                delta=np.abs(cube[:2]-target[:2])
                if np.linalg.norm(delta)>.075 and np.any(delta>.058):break
            else:raise RuntimeError("Could not sample separated objects")
            label=f"uniform_seed_{seed}"
        if options:
            cube=np.asarray(options.get("cube_xyz",cube),dtype=float)
            target=np.asarray(options.get("bin_xyz",target),dtype=float)
            label=options.get("placement_label",label)
        if cube.shape!=(3,) or target.shape!=(3,) or not np.isfinite(np.r_[cube,target]).all():
            raise ValueError("Reset positions must each contain three finite coordinates")
        self.reset_record={"placement_label":label,"cube_xyz":cube.tolist(),"bin_xyz":target.tolist()}
        self.data.qpos[self.qadr]=HOME;self.data.ctrl[self.aidx]=HOME
        self.data.qpos[self.cube_adr:self.cube_adr+7]=[*cube,1,0,0,0]
        self.data.mocap_pos[self.bin_mocap]=target
        self.data.mocap_quat[self.bin_mocap]=[1,0,0,0]
        mujoco.mj_forward(self.model,self.data)
        self.steps=0;self.stable_steps=0;self._done=False
        self.initial_cube_z=float(cube[2])
        self.lifted=False;self.reached=False;self.grasped=False;self.transported=False
        return self._obs(),self._info(False)

    def _check_stages(self):
        cube=self.data.geom_xpos[self.cube_geom]
        midpoint=self.data.geom_xpos[self.pad_geoms].mean(axis=0)
        self.reached|=bool(np.linalg.norm(midpoint-cube)<.035)
        fixed=moving=False
        for c in self.data.contact:
            if self.cube_geom not in (c.geom1,c.geom2):continue
            other=c.geom2 if c.geom1==self.cube_geom else c.geom1
            body=int(self.model.geom_bodyid[other])
            fixed|=body==self.fixed_body;moving|=body==self.moving_body
        self.grasped|=bool(fixed and moving)
        self.lifted|=bool(cube[2]>self.initial_cube_z+.04)
        self.transported|=bool(self.lifted and np.all(np.abs(cube[:2]-self.bin_xyz[:2])<self.bin_inner))

    def _success_candidate(self):
        center=self.data.geom_xpos[self.cube_geom]
        rotation=self.data.geom_xmat[self.cube_geom].reshape(3,3)
        corners=(np.array(list(product([-1,1],repeat=3)))*self.half_cube)@rotation.T+center
        # Soft wall contacts can penetrate by micrometres at rest. Preserve the
        # bin geometry and allow at most 0.1 mm of numerical contact overlap.
        contained=bool(np.all(np.abs(corners[:,:2]-self.bin_xyz[:2])<self.bin_inner+self.placement_tolerance_m))
        bottom=float(corners[:,2].min());floor=float(self.bin_xyz[2]+self.bin_floor_top)
        supported=bool(floor-.003<bottom<floor+.008)
        released=True
        for c in self.data.contact:
            if (c.geom1==self.cube_geom and c.geom2 in self.robot_geoms) or (c.geom2==self.cube_geom and c.geom1 in self.robot_geoms):released=False
        v=self.data.qvel[self.cube_dof:self.cube_dof+6]
        still=bool(np.linalg.norm(v[:3])<.03 and np.linalg.norm(v[3:])<.4)
        return bool(self.lifted and contained and supported and released and still)

    def _info(self,success):
        return {"is_success":bool(success),"lifted":bool(self.lifted),"reached":bool(self.reached),"grasped":bool(self.grasped),
                "transported":bool(self.transported),"stable_steps":self.stable_steps,
                "cube_position":self.data.geom_xpos[self.cube_geom].tolist(),
                "reset_source_episode":self.reset_record["placement_label"],"reset":self.reset_record,
                "contact_profile":self.contact_profile,
                "placement_tolerance_m":self.placement_tolerance_m,
                "scene_fidelity":"Rovo Labs replica with disclosed contact pad/controller profile; measured source visual comparison"}

    def step(self,action):
        if self._done:raise RuntimeError("Call reset after episode end")
        action=np.asarray(action,dtype=float)
        if action.shape!=(6,) or not np.isfinite(action).all():raise ValueError("Expected six finite absolute joint targets")
        clipped=np.clip(action,self.action_space.low,self.action_space.high)
        self.data.ctrl[self.aidx]=clipped
        warn_ids=[int(getattr(mujoco.mjtWarning,name)) for name in ("mjWARN_BADQPOS","mjWARN_BADQVEL","mjWARN_BADQACC","mjWARN_BADCTRL")]
        counts=[int(self.data.warning[i].number) for i in warn_ids]
        for _ in range(17):
            mujoco.mj_step(self.model,self.data);self._check_stages()
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():raise RuntimeError("Nonfinite MuJoCo state")
        if any(int(self.data.warning[i].number)>n for i,n in zip(warn_ids,counts)):raise RuntimeError("MuJoCo reported an invalid numerical state")
        self.steps+=1
        self.stable_steps=self.stable_steps+1 if self._success_candidate() else 0
        success=self.stable_steps>=15;truncated=self.steps>=self.max_steps and not success
        self._done=bool(success or truncated)
        info=self._info(success);info["action_clipped"]=bool(np.any(action!=clipped))
        return self._obs(),float(success),bool(success),bool(truncated),info

    def render_camera(self,camera):
        m=self.models[camera];d=self.datas[camera]
        if camera!="front":
            d.qpos[:]=self.data.qpos;d.qvel[:]=self.data.qvel;d.ctrl[:]=self.data.ctrl
            d.mocap_pos[:]=self.data.mocap_pos;d.mocap_quat[:]=self.data.mocap_quat;d.time=self.data.time
            mujoco.mj_forward(m,d)
        if camera not in self.renderers:self.renderers[camera]=mujoco.Renderer(m,self.image_size,self.image_size)
        renderer=self.renderers[camera];renderer.update_scene(d,camera=SCENES[camera][1])
        return renderer.render().copy()

    def render(self):return self.render_camera("front")

    def close(self):
        for r in self.renderers.values():r.close()
        self.renderers.clear()
