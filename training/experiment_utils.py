from pathlib import Path
import json
import os
import numpy as np

HERE = Path(__file__).resolve().parent
DATASET = HERE/'derived_dataset'
BASE = Path(os.environ.get('SO101_BASE_CHECKPOINT', HERE.parent/'checkpoints/smolvla_base'))
RENAME = {'observation.images.front':'observation.images.camera1','observation.images.wrist':'observation.images.camera2','observation.images.overhead':'observation.images.camera3'}
TASK = 'pick up the object and place it in the target location'


def configure_environment():
    scratch=HERE.parent.parent/'work'
    scratch.mkdir(exist_ok=True)
    for key,value in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_CACHE':str(scratch/'hf_datasets'),'MPLCONFIGDIR':str(scratch/'matplotlib'),'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'1','MUJOCO_GL':'osmesa','PYTHONUNBUFFERED':'1'}.items():
        os.environ.setdefault(key,value)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)


def training_args(batch,steps,output,save=True,device='cuda',workers=4):
    split=read_json(HERE/'configs/episode_split.json')
    return [f'--policy.path={BASE}',f'--dataset.root={DATASET}', '--dataset.repo_id=local/so101_pick_cube_repaired',f'--dataset.episodes={json.dumps(split["train"],separators=(",",":"))}', '--dataset.video_backend=pyav','--dataset.image_transforms.enable=false',f'--output_dir={output}',f'--policy.device={device}','--policy.push_to_hub=false','--policy.load_vlm_weights=false','--policy.n_action_steps=10','--policy.scheduler_decay_steps=40000','--policy.scheduler_warmup_steps=1000',f'--batch_size={batch}',f'--steps={steps}',f'--num_workers={workers}',f'--persistent_workers={str(workers>0).lower()}','--prefetch_factor=2','--log_freq=100','--save_freq=5000',f'--save_checkpoint={str(save).lower()}','--eval_freq=0','--seed=42','--wandb.enable=false',f'--rename_map={json.dumps(RENAME,separators=(",",":"))}']


def plot_training(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run)
    path=run/'training_metrics.jsonl'
    if not path.exists():return
    by_step={}
    for line in path.read_text().splitlines():
        try:
            row=json.loads(line);by_step[row['step']]=row
        except (ValueError,KeyError):continue
    rows=[by_step[k] for k in sorted(by_step)]
    if not rows:return
    x=np.array([r['step'] for r in rows])
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    y=np.array([r.get('loss',np.nan) for r in rows])
    axes[0,0].plot(x,y,alpha=.2,linewidth=.5)
    window=min(200,len(y))
    axes[0,0].plot(x[window-1:],np.convolve(y,np.ones(window)/window,mode='valid'))
    axes[0,0].set(title='Training flow-matching loss',ylabel='Loss')
    for ax,key,title in [(axes[0,1],'lr','Learning rate'),(axes[1,0],'wall_step_seconds','Wall time per optimizer update'),(axes[1,1],'peak_gpu_memory_gib','Peak allocated GPU memory (GiB)')]:
        ax.plot(x,[r.get(key,np.nan) for r in rows]);ax.set(title=title,xlabel='Optimizer step')
    fig.tight_layout()
    (run/'graphs').mkdir(exist_ok=True)
    fig.savefig(run/'graphs/training.png',dpi=160)
    plt.close(fig)


def wilson(successes,total):
    if not total:return [0.,1.]
    z=1.959963984540054
    p=successes/total
    d=1+z*z/total
    middle=(p+z*z/(2*total))/d
    half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [max(0.,float(middle-half)),min(1.,float(middle+half))]
