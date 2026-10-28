"""Offline validation, independent simulator selection, and 50 final rollouts."""
from pathlib import Path
import contextlib
import csv
import gc
import json
import sys
import time
import traceback
import numpy as np
from experiment_utils import HERE,DATASET,BASE,RENAME,TASK,configure_environment,read_json,write_json,wilson

configure_environment()
import torch
import av
from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
sys.path.insert(0,str(HERE/'simulator'))
from so101_env import SO101PickCubeEnv


def amp(device):
    return torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda' else contextlib.nullcontext()


def load_policy(checkpoint,device='cuda',baseline=False):
    checkpoint=Path(checkpoint)
    cfg=PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device=device
    cfg.push_to_hub=False
    cfg.load_vlm_weights=False
    cfg.n_action_steps=10
    policy=SmolVLAPolicy.from_pretrained(checkpoint,config=cfg).to(device).eval()
    if baseline:
        # Use target-task TRAINING normalization for a fair zero-update baseline;
        # generic base-model processor statistics do not describe these radians.
        stats=read_json(DATASET/'meta/stats.json')
        stats={RENAME.get(k,k):{name:torch.tensor(v) for name,v in values.items()} for k,values in stats.items()}
        pre,post=make_pre_post_processors(cfg,dataset_stats=stats)
    else:
        pre,post=make_pre_post_processors(cfg,str(checkpoint),preprocessor_overrides={'device_processor':{'device':device}})
    return policy,pre,post


def prepare_batch(raw,pre):
    batch={RENAME.get(k,k):v for k,v in raw.items()}
    for k,value in list(batch.items()):
        if k.startswith('observation.images') and isinstance(value,torch.Tensor) and value.dtype==torch.uint8:
            batch[k]=value.to(torch.float32)/255.
    return pre(batch)


@torch.inference_mode()
def offline_validation(policy,pre,post,device='cuda',batch_size=8,max_anchors=None):
    anchors=read_json(HERE/'configs/episode_split.json')['validation_anchor_global_indices']
    if max_anchors is not None:anchors=anchors[:max_anchors]
    dataset=LeRobotDataset('local/so101_pick_cube_repaired',root=DATASET,video_backend='pyav',delta_timestamps={'action':[i/30 for i in range(50)]},return_uint8=True)
    normalized_abs=0.;physical_abs=0.;physical_sq=0.;valid_count=0;loss_sum=0.;samples=0
    device_ids=[torch.cuda.current_device()] if device=='cuda' else []
    with torch.random.fork_rng(devices=device_ids):
        torch.manual_seed(12345)
        for start in range(0,len(anchors),batch_size):
            raw=lerobot_collate_fn([dataset[i] for i in anchors[start:start+batch_size]])
            physical_target=raw['action'].cpu().clone()
            batch=prepare_batch(raw,pre)
            policy.reset()
            with amp(device):
                loss,_=policy.forward(batch)
                pred=policy.predict_action_chunk(batch)
            mask=(~batch['action_is_pad']).unsqueeze(-1).expand_as(pred)
            normalized_abs+=float(torch.abs(pred-batch['action'])[mask].sum().item())
            physical_pred=post(pred).float().cpu()
            delta=physical_pred-physical_target
            cpu_mask=mask.cpu()
            physical_abs+=float(delta.abs()[cpu_mask].sum().item())
            physical_sq+=float(delta.square()[cpu_mask].sum().item())
            valid_count+=int(cpu_mask.sum().item())
            n=len(physical_target)
            loss_sum+=float(loss.item())*n
            samples+=n
    policy.reset()
    return {'anchors':samples,'distinct_trajectory_groups':samples,'flow_matching_loss':loss_sum/samples,'normalized_action_mae':normalized_abs/valid_count,'action_mae_radians':physical_abs/valid_count,'action_mse_radians_squared':physical_sq/valid_count,'valid_action_elements':valid_count,'method':'Fixed held-out anchor per trajectory group, 50-step chunks, padding masked; target normalization from training only.'}


def open_video(path):
    container=av.open(str(path),mode='w')
    stream=container.add_stream('libx264',rate=30)
    stream.width=1536;stream.height=512;stream.pix_fmt='yuv420p'
    stream.options={'crf':'23','preset':'fast'}
    return container,stream


@torch.inference_mode()
def rollouts(policy,pre,post,seeds,output,device='cuda',save_all_videos=False,max_steps=301):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    records=[]
    for index,seed in enumerate(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        env=SO101PickCubeEnv(image_size=512,max_steps=max_steps)
        policy.reset()
        for processor in [pre,post]:
            if hasattr(processor,'reset'):processor.reset()
        video=None;stream=None
        positions=[];actions=[];cubes=[];latencies=[];clipped=0
        started=time.perf_counter()
        item={'seed':seed,'success':False,'valid':True,'steps':0,'scene_fidelity':'reconstructed environment; physical parameters assumed'}
        try:
            obs,info=env.reset(seed=seed)
            item['initial_cube_position']=info['cube_position']
            if save_all_videos or index==0:
                video,stream=open_video(output/f'episode_{index:03d}_seed_{seed}.mp4')
            for step in range(max_steps):
                if video is not None:
                    picture=np.concatenate([obs['observation.images.'+cam] for cam in ['front','wrist','overhead']],axis=1)
                    for packet in stream.encode(av.VideoFrame.from_ndarray(picture,format='rgb24')):video.mux(packet)
                raw={'observation.state':torch.from_numpy(obs['observation.state']),'task':TASK}
                for cam in ['front','wrist','overhead']:
                    raw['observation.images.'+cam]=torch.from_numpy(obs['observation.images.'+cam]).permute(2,0,1)
                t=time.perf_counter()
                batch=prepare_batch(raw,pre)
                with amp(device):action=post(policy.select_action(batch)).squeeze(0).float().cpu().numpy()
                if device=='cuda':torch.cuda.synchronize()
                latencies.append(time.perf_counter()-t)
                positions.append(obs['observation.state'].copy())
                actions.append(action.copy())
                cubes.append(env.data.geom_xpos[env.cube_geom].copy())
                obs,reward,terminated,truncated,info=env.step(action)
                item['steps']=step+1
                clipped+=int(info['action_clipped'])
                if terminated or truncated:
                    item['success']=bool(info['is_success'])
                    break
            item['final_cube_position']=info['cube_position']
        except Exception:
            item['valid']=False
            item['error']=traceback.format_exc()
        finally:
            if video is not None:
                for packet in stream.encode():video.mux(packet)
                video.close()
            env.close()
        item['wall_seconds']=time.perf_counter()-started
        item['mean_policy_call_seconds']=float(np.mean(latencies)) if latencies else None
        item['p95_policy_call_seconds']=float(np.percentile(latencies,95)) if latencies else None
        item['action_clipped_steps']=clipped
        np.savez_compressed(output/f'episode_{index:03d}_seed_{seed}_trace.npz',states=np.asarray(positions),actions=np.asarray(actions),cube_positions=np.asarray(cubes),policy_call_seconds=np.asarray(latencies))
        records.append(item)
        write_json(output/'episodes.json',records)
        print(f'rollout {index+1}/{len(seeds)} seed={seed} success={item["success"]} valid={item["valid"]}',flush=True)
    successes=sum(r['success'] and r['valid'] for r in records)
    report={'episodes':len(records),'successes':successes,'success_rate_percent':100*successes/len(records),'wilson_95_percent':[100*x for x in wilson(successes,len(records))],'invalid_episodes':sum(not r['valid'] for r in records),'seeds':list(seeds),'scene_fidelity':'reconstructed environment; physical parameters assumed','n_action_steps':10,'denominator_policy':'All requested episodes; invalid episodes cannot count as successes. No silent replacements.'}
    write_json(output/'summary.json',report)
    with (output/'episodes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['seed','success','valid','steps','wall_seconds','mean_policy_call_seconds','p95_policy_call_seconds','action_clipped_steps'],extrasaction='ignore');writer.writeheader();writer.writerows(records)
    return report


def evaluate_all(run,device='cuda'):
    run=Path(run)
    directories=sorted(p for p in (run/'training/checkpoints').iterdir() if p.name.isdigit() and (p/'pretrained_model/model.safetensors').exists())
    if [int(p.name) for p in directories]!=list(range(5000,40001,5000)):
        raise RuntimeError('All eight scheduled checkpoints are required for complete selection')
    results=[]
    for name,checkpoint in [('base',BASE)]+[(p.name,p/'pretrained_model') for p in directories]:
        output=run/'evaluation'/name
        if (output/'result.json').exists():
            results.append(read_json(output/'result.json'));continue
        policy,pre,post=load_policy(checkpoint,device,baseline=name=='base')
        offline=offline_validation(policy,pre,post,device)
        simulation=rollouts(policy,pre,post,list(range(1000,1020)),output/'selection_rollouts',device)
        item={'checkpoint':str(checkpoint),'step':0 if name=='base' else int(name),'offline':offline,'simulation':simulation}
        write_json(output/'result.json',item);results.append(item)
        del policy,pre,post;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    eligible=[r for r in results if r['step']>0 and r['simulation']['invalid_episodes']==0]
    if not eligible:raise RuntimeError('No checkpoint has a technically valid simulator evaluation')
    best=min(eligible,key=lambda r:(-r['simulation']['successes'],r['offline']['normalized_action_mae'],r['step']))
    write_json(run/'checkpoint_selection.json',{'selected':best,'all_results':results,'criterion':'Maximum success count on seeds 1000–1019; ties use held-out normalized action MAE, then earlier step. Final seeds excluded from selection.'})
    policy,pre,post=load_policy(best['checkpoint'],device)
    final_dir=run/'final_50_episodes'
    if (final_dir/'summary.json').exists():
        final=read_json(final_dir/'summary.json')
    else:
        final=rollouts(policy,pre,post,list(range(10000,10050)),final_dir,device,save_all_videos=True)
    write_json(run/'final_report.json',{'training_steps':40000,'selected_checkpoint':best['checkpoint'],'selection':best,'final_evaluation':final,'scene_fidelity':'Reconstructed MuJoCo scene; not the recovered original simulator. Camera/physics differences limit external validity.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    steps=[r['step'] for r in results]
    for ax,key,title in [(axes[0],'flow_matching_loss','Held-out flow loss'),(axes[1],'normalized_action_mae','Held-out normalized action MAE')]:
        ax.plot(steps,[r['offline'][key] for r in results],marker='o');ax.set(title=title,xlabel='Training step')
    axes[2].plot(steps,[r['simulation']['success_rate_percent'] for r in results],marker='o');axes[2].set(title='Selection success in reconstructed simulator',xlabel='Training step',ylabel='Success (%)',ylim=(-2,102))
    fig.tight_layout();fig.savefig(run/'graphs/checkpoint_comparison.png',dpi=160);plt.close(fig)
    text=f'# SmolVLA final run report\n\nSelected checkpoint: `{best["checkpoint"]}`.\n\nTraining: 40,000 optimizer steps. Final evaluation: {final["successes"]}/{final["episodes"]} successes ({final["success_rate_percent"]:.1f}%). 95% Wilson interval: {final["wilson_95_percent"][0]:.1f}–{final["wilson_95_percent"][1]:.1f}%. Invalid episodes: {final["invalid_episodes"]}.\n\nThese results apply to the reconstructed MuJoCo environment. The original simulator and its physical parameters were not recovered. See the dataset-repair evidence and simulator assumptions before interpreting this as a benchmark.\n\nCheckpoint selection used 20 fixed seeds per checkpoint. The 50 final seeds were separate, and final outcomes were not used for tuning. All final videos and traces are in `final_50_episodes/`.\n'
    (run/'FINAL_REPORT.md').write_text(text)
    return final


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args()
    print(json.dumps(evaluate_all(Path(args.run)),indent=2))
