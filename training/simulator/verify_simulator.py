"""Physics/API checks only. This script does NOT evaluate a trained policy."""
from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
import mujoco
from gymnasium.utils.env_checker import check_env
from so101_env import SO101PickCubeEnv, HOME

HERE = Path(__file__).resolve().parent
report = {'policy_evaluation_performed': False, 'scene_fidelity': 'uncalibrated reconstruction', 'checks': {}}
env = SO101PickCubeEnv(observations_with_images=False)
try:
    check_env(env, skip_render_check=True)
    report['checks']['gymnasium_api'] = 'pass'
    obs, _ = env.reset(seed=42)
    first = env.data.qpos.copy()
    env.reset(seed=42)
    np.testing.assert_array_equal(first, env.data.qpos)
    report['checks']['seeded_reset'] = 'pass'
    trials = []
    for seed in range(5):
        env.reset(seed=seed)
        for step in range(env.max_steps):
            obs, reward, terminated, truncated, info = env.step(HOME)
            assert np.isfinite(obs['observation.state']).all()
            assert reward == 0 and not info['is_success'], 'Idle arm must not count as success'
            if terminated or truncated:
                break
        assert truncated
        trials.append({'seed': seed, 'steps': step+1, 'finite': True, 'idle_false_success': False})
    report['checks']['five_idle_physics_trials'] = trials
    # Explicitly injected states exercise the metric; they are not rollouts.
    env.reset(seed=0)
    env.data.qpos[env.cube_adr:env.cube_adr+3] = [.27, .06, .626]
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    assert not env._success_candidate(), 'Cube in bin without prior lift must fail'
    env.lifted = True
    assert env._success_candidate(), 'Released, supported cube in bin after lift must pass'
    env.data.qpos[env.cube_adr] += .06
    mujoco.mj_forward(env.model, env.data)
    assert not env._success_candidate(), 'Partly outside cube must fail'
    report['checks']['success_predicate_injected_fixtures'] = 'pass (not policy episodes)'
    # Measure how well the provisional controller follows recorded actions.
    dataset = Path('<SOURCE_DATASET>')
    rows = pq.read_table(dataset/'data', columns=['observation.state', 'action']).slice(0, 301).to_pylist()
    env.reset(seed=0, options={'cube_xy': [.223, -.081]})
    errors = []
    for i in range(1, len(rows)):
        obs, _, term, trunc, _ = env.step(rows[i]['action'])
        if term or trunc:
            break
        errors.append(np.abs(obs['observation.state'] - np.array(rows[min(i+1,len(rows)-1)]['observation.state'])))
    report['recorded_action_replay'] = {'episode': 0, 'joint_mae_radians_in_dataset_order': np.mean(errors,axis=0).tolist(), 'note': 'Diagnostic only; cube pose is an assumption and controller parameters were not fitted. This is not a success-rate evaluation.'}
finally:
    env.close()

env = SO101PickCubeEnv(image_size=512)
try:
    obs, _ = env.reset(seed=42, options={'cube_xy': [.223, -.081]})
    sheet = Image.new('RGB', (1536, 548), '#171c25')
    draw = ImageDraw.Draw(sheet)
    for i, cam in enumerate(['front', 'wrist', 'overhead']):
        image = Image.fromarray(obs['observation.images.'+cam])
        image.save(HERE/(cam+'_preview.png'))
        sheet.paste(image, (i*512,36))
        draw.text((i*512+12,10), 'RECONSTRUCTION / '+cam, fill='white')
    sheet.save(HERE/'camera_previews.png')
    assert env.observation_space.contains(obs)
    report['checks']['three_camera_render_and_observation_schema'] = 'pass'
finally:
    env.close()
report['mujoco_version'] = mujoco.__version__
(HERE/'verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
