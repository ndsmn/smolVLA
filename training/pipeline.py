"""Fixed host-side experiment: benchmark, 40k training, selection, final 50 episodes.

This runner executes only the declared experiment. No remote command bridge,
driver changes, package installs, external uploads, or process termination.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import csv
import fcntl
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import hashlib
import numpy as np
from experiment_utils import HERE,DATASET,BASE,configure_environment,training_args,read_json,write_json,plot_training

configure_environment()


def run_training(args,metrics,log,stage):
    env=dict(os.environ,SMOLVLA_METRICS_PATH=str(metrics),SMOLVLA_STAGE=stage,ACCELERATE_MIXED_PRECISION='bf16')
    log.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(HERE/'train_entry.py'),*args]
    write_json(log.with_suffix('.command.json'),{'argv':command,'mixed_precision':'bf16'})
    with log.open('a') as handle:
        result=subprocess.run(command,env=env,stdout=handle,stderr=subprocess.STDOUT)
    return result.returncode


def pipeline(run,resume=False):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available here. Launch this script from the verified normal host terminal.')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('This prepared run requires CUDA bfloat16 support; no precision fallback is applied silently.')
    if shutil.disk_usage(HERE).free<30*2**30:
        raise RuntimeError('At least 30 GiB free disk is required for the planned checkpoints and artifacts.')
    repair=read_json(HERE/'preflight/dataset_repair.json')
    verification=read_json(HERE/'preflight/derived_loader_verification.json')
    if repair['duplicate_group_overlap']!=0 or not verification['all_three_cameras_decode']:
        raise RuntimeError('Dataset preflight has not passed')
    actual=hashlib.sha256((DATASET/'data/chunk-000/file-000.parquet').read_bytes()).hexdigest()
    if actual!=repair['data_sha256']:raise RuntimeError('Derived data changed after preflight verification.')
    if run.exists() and not resume:
        raise FileExistsError(f'{run} already exists; use --resume to continue its saved state.')
    run.mkdir(parents=True,exist_ok=True)
    snapshot=run/'reproducibility'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    snapshot.mkdir(parents=True,exist_ok=True)
    for name in ['episode_split.json','experiment_plan.json']:
        shutil.copy2(HERE/'configs'/name,snapshot/name)
    for name in ['dataset_repair.json','host_cuda_check.json','environment.json','source_integrity.json']:
        shutil.copy2(HERE/'preflight'/name,snapshot/name)
    for path in HERE.glob('*.py'):shutil.copy2(path,snapshot/path.name)
    shutil.copytree(HERE/'simulator',snapshot/'simulator',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(DATASET/'meta',snapshot/'derived_dataset_meta')
    write_json(HERE/'latest_run.json',{'run_dir':str(run)})
    def status(phase,**extra):
        value={'phase':phase,'updated_utc':datetime.now(timezone.utc).isoformat(),'run_dir':str(run),**extra}
        write_json(run/'pipeline_status.json',value)
        print(json.dumps(value),flush=True)
    status('host_preflight',gpu=torch.cuda.get_device_name(0),free_vram_gib=torch.cuda.mem_get_info()[0]/2**30)
    smi=subprocess.run(['nvidia-smi'],capture_output=True,text=True)
    (run/'host_nvidia_smi_start.txt').write_text(smi.stdout+smi.stderr)
    # Validate policy preprocessing, forward/inference, simulation, and video codec
    # before spending time on any optimizer benchmark.
    if not (run/'policy_simulator_smoke/summary.json').exists():
        from evaluate_checkpoints import load_policy,rollouts
        policy,pre,post=load_policy(BASE,baseline=True)
        smoke=rollouts(policy,pre,post,[999],run/'policy_simulator_smoke',max_steps=5)
        if smoke['invalid_episodes']:
            raise RuntimeError('Policy/simulator smoke test failed; inspect its saved episode error.')
        del policy,pre,post;gc.collect();torch.cuda.empty_cache()
    benchmark_path=run/'benchmark_summary.json'
    if benchmark_path.exists():
        benchmark=read_json(benchmark_path)
    else:
        status('benchmarking',warmup_steps=50,timed_steps=200,batch_candidates=[8,16,32])
        measurements=[]
        total_gib=torch.cuda.get_device_properties(0).total_memory/2**30
        for batch in [8,16,32]:
            directory=run/'benchmarks'/f'batch_{batch}'
            metrics=directory/'metrics.jsonl'
            output=directory/'upstream_run'
            if output.exists():
                output.rename(directory/f'archived_attempt_{time.time_ns()}')
            code=run_training(training_args(batch,250,output,save=False),metrics,directory/'console.log','benchmark')
            item={'batch_size':batch,'exit_code':code,'eligible':False}
            if code==0:
                by_step={r['step']:r for r in [json.loads(x) for x in metrics.read_text().splitlines()]}
                rows=[by_step[i] for i in range(51,251)]
                seconds=np.array([r['wall_step_seconds'] for r in rows])
                item.update(mean_step_seconds=float(seconds.mean()),median_step_seconds=float(np.median(seconds)),p95_step_seconds=float(np.percentile(seconds,95)),peak_allocated_gib=max(r['peak_gpu_memory_gib'] for r in rows),peak_reserved_gib=max(r['peak_gpu_reserved_gib'] for r in rows),examples_per_second=float(batch/seconds.mean()),eta_40000_updates_hours=float(40000*seconds.mean()/3600))
                item['eligible']=item['peak_reserved_gib']<.9*total_gib
            else:
                tail=(directory/'console.log').read_text(errors='replace')[-12000:]
                item['failure_log_tail']=tail
                if 'out of memory' not in tail.lower():
                    write_json(run/'benchmark_failure.json',item)
                    raise RuntimeError(f'Batch {batch} benchmark failed for a reason other than CUDA OOM. See its log.')
            measurements.append(item)
            write_json(run/'benchmark_progress.json',measurements)
        choices=[r for r in measurements if r['eligible']]
        if not choices:raise RuntimeError('No benchmark batch passed the VRAM headroom requirement.')
        chosen=max(choices,key=lambda r:r['batch_size'])
        benchmark={'measurements':measurements,'selected':chosen,'criterion':'Largest tested stable batch using less than 90% of physical GPU memory as peak allocator reservation; increases coverage for the fixed 40k-update budget.','timing_limits':'Measured with the current host workload; excludes saving and evaluation. Benchmark updates do not count toward the production 40k.'}
        write_json(benchmark_path,benchmark)
    chosen=benchmark['selected']
    status('training',batch_size=chosen['batch_size'],target_steps=40000,measured_training_eta_hours=chosen['eta_40000_updates_hours'])
    training_dir=run/'training'
    final_checkpoint=training_dir/'checkpoints/040000/pretrained_model/model.safetensors'
    if not final_checkpoint.exists():
        last=training_dir/'checkpoints/last/pretrained_model/train_config.json'
        if resume and last.exists():
            args=[f'--config_path={last}','--resume=true']
        else:
            if training_dir.exists():training_dir.rename(run/f'archived_training_attempt_{time.time_ns()}')
            args=training_args(chosen['batch_size'],40000,training_dir)
        result=run_training(args,run/'training_metrics.jsonl',run/'training_console.log','training')
        if result:raise RuntimeError('Training failed; inspect training_console.log and training_metrics.status.json. Use --resume after addressing the cause.')
    if not final_checkpoint.exists():raise RuntimeError('Training returned without the required 40,000-step checkpoint.')
    plot_training(run)
    status('evaluating_checkpoints_then_final_50',checkpoint_selection_seeds=[1000,1019],final_seeds=[10000,10049])
    from evaluate_checkpoints import evaluate_all
    final=evaluate_all(run)
    rows={}
    for line in (run/'training_metrics.jsonl').read_text().splitlines():
        row=json.loads(line);rows[row['step']]=row
    with (run/'training_metrics.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(next(iter(rows.values())).keys()))
        writer.writeheader();writer.writerows(rows[i] for i in sorted(rows))
    status('complete' if final['invalid_episodes']==0 else 'complete_with_invalid_episodes',successes=final['successes'],episodes=final['episodes'],success_rate_percent=final['success_rate_percent'],final_report=str(run/'FINAL_REPORT.md'))
    return final


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,default=HERE/'runs/so101_40k')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    lock=(HERE/'pipeline.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('An experiment runner already holds this folder lock.')
    try:
        pipeline(args.run_dir.resolve(),args.resume)
    except BaseException:
        error=traceback.format_exc()
        if args.run_dir.exists():write_json(args.run_dir/'pipeline_status.json',{'phase':'failed','error':error,'updated_utc':datetime.now(timezone.utc).isoformat()})
        raise


if __name__=='__main__':
    main()
