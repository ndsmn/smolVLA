"""MuJoCo reconstruction; success here does not establish original-task success.

State and action ordering follows meta/info.json, NOT the robot actuator order.
Absolute joint targets in radians are a hypothesis supported by the traces.
No object attachment, teleport grasp, or proximity-based success is used.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'osmesa')
from pathlib import Path
from itertools import product
import gymnasium as gym
import mujoco
import numpy as np

JOINT_NAMES = ['elbow_flex', 'gripper', 'shoulder_lift', 'shoulder_pan', 'wrist_flex', 'wrist_roll']
HOME = np.array([-.8, .6, .5, 0, .3, 0])


class SO101PickCubeEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None, image_size=512, max_steps=301, observations_with_images=True):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name('scene.xml')))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.image_size = image_size
        self.max_steps = max_steps
        self.observations_with_images = observations_with_images
        self.renderer = None
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in JOINT_NAMES])
        self.aidx = np.array([self.model.actuator(n).id for n in JOINT_NAMES])
        ranges = self.model.actuator_ctrlrange[self.aidx]
        self.action_space = gym.spaces.Box(ranges[:, 0].astype('float32'), ranges[:, 1].astype('float32'))
        spaces = {'observation.state': gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)}
        if observations_with_images:
            for cam in ['front', 'wrist', 'overhead']:
                spaces['observation.images.'+cam] = gym.spaces.Box(0, 255, shape=(image_size, image_size, 3), dtype=np.uint8)
        self.observation_space = gym.spaces.Dict(spaces)
        self.cube_adr = self.model.joint('cube_free').qposadr[0]
        self.cube_dof = self.model.joint('cube_free').dofadr[0]
        self.cube_geom = self.model.geom('cube').id
        robot_bodies = {self.model.body('base').id}
        for i in range(self.model.nbody):
            if int(self.model.body_parentid[i]) in robot_bodies:
                robot_bodies.add(i)
        self.robot_geoms = {i for i in range(self.model.ngeom) if int(self.model.geom_bodyid[i]) in robot_bodies}
        self.steps = 0
        self.lifted = False
        self.stable_steps = 0
        self.initial_cube_z = .620
        self._done = False

    def _obs(self):
        obs = {'observation.state': self.data.qpos[self.qadr].astype(np.float32).copy()}
        if self.observations_with_images:
            for cam in ['front', 'wrist', 'overhead']:
                obs['observation.images.'+cam] = self.render_camera(cam)
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = HOME
        self.data.ctrl[self.aidx] = HOME
        xy = np.array([self.np_random.uniform(.18, .28), self.np_random.uniform(-.12, -.04)])
        if options and 'cube_xy' in options:
            xy = np.asarray(options['cube_xy'], dtype=float)
            if xy.shape != (2,) or not np.isfinite(xy).all():
                raise ValueError('cube_xy must contain two finite coordinates')
        self.data.qpos[self.cube_adr:self.cube_adr+3] = [*xy, .621]
        self.data.qpos[self.cube_adr+3:self.cube_adr+7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.lifted = False
        self.stable_steps = 0
        self.initial_cube_z = .620
        self._done = False
        return self._obs(), self._info(False)

    def _success_candidate(self):
        center = self.data.geom_xpos[self.cube_geom]
        rotation = self.data.geom_xmat[self.cube_geom].reshape(3, 3)
        corners = np.array(list(product([-.020, .020], repeat=3))) @ rotation.T + center
        # The entire rotated cube must lie inside the inner bin walls.
        contained = bool(np.all(np.abs(corners[:, :2] - [.27, .06]) < .033))
        supported = bool(.602 < corners[:, 2].min() < .616)
        released = True
        for c in self.data.contact:
            if (c.geom1 == self.cube_geom and c.geom2 in self.robot_geoms) or (c.geom2 == self.cube_geom and c.geom1 in self.robot_geoms):
                released = False
        velocity = self.data.qvel[self.cube_dof:self.cube_dof+6]
        still = np.linalg.norm(velocity[:3]) < .03 and np.linalg.norm(velocity[3:]) < .4
        return bool(self.lifted and contained and supported and released and still)

    def _info(self, success):
        return {'is_success': bool(success), 'lifted': bool(self.lifted), 'stable_steps': self.stable_steps, 'cube_position': self.data.geom_xpos[self.cube_geom].tolist(), 'scene_fidelity': 'uncalibrated reconstruction'}

    def step(self, action):
        if self._done:
            raise RuntimeError('Call reset() after episode termination or truncation')
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError('Expected six finite joint targets in dataset order')
        clipped = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[self.aidx] = clipped
        warning_ids=[int(getattr(mujoco.mjtWarning,name)) for name in ['mjWARN_BADQPOS','mjWARN_BADQVEL','mjWARN_BADQACC','mjWARN_BADCTRL']]
        before=[int(self.data.warning[i].number) for i in warning_ids]
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
            if self.data.geom_xpos[self.cube_geom, 2] > self.initial_cube_z + .04:
                self.lifted = True
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError('Nonfinite simulation state')
        if any(int(self.data.warning[i].number)>count for i,count in zip(warning_ids,before)):
            raise RuntimeError('MuJoCo reported an invalid numerical state or control')
        self.steps += 1
        self.stable_steps = self.stable_steps + 1 if self._success_candidate() else 0
        success = self.stable_steps >= 15
        truncated = self.steps >= self.max_steps and not success
        self._done = bool(success or truncated)
        info = self._info(success)
        info['action_clipped'] = bool(np.any(action != clipped))
        return self._obs(), float(success), bool(success), bool(truncated), info

    def render_camera(self, camera):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.image_size, width=self.image_size)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def render(self):
        return self.render_camera('front')

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
