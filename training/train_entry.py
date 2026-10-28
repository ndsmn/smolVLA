"""Use upstream LeRobot training, adding local per-update telemetry only."""
import os
import time
import traceback
import json
import math
import subprocess
from pathlib import Path
from experiment_utils import configure_environment,write_json,plot_training

configure_environment()
import torch
from lerobot.scripts import lerobot_train as upstream


def main():
    metrics_path=Path(os.environ['SMOLVLA_METRICS_PATH'])
    metrics_path.parent.mkdir(parents=True,exist_ok=True)
    stage=os.environ.get('SMOLVLA_STAGE','training')
    status=metrics_path.with_suffix('.status.json')
    original=upstream.update_policy
    previous=time.perf_counter()
    started=previous
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()

    def instrumented(*args,**kwargs):
        nonlocal previous
        metrics,details=original(*args,**kwargs)
        if torch.cuda.is_available():torch.cuda.synchronize()
        now=time.perf_counter()
        row=metrics.to_dict(use_avg=False)
        row['step']=int(row.pop('steps'))+1
        row['samples']+=metrics._batch_size
        row['epochs']=row['samples']/metrics._num_frames
        row['wall_step_seconds']=now-previous
        row['wall_elapsed_seconds']=now-started
        row['peak_gpu_memory_gib']=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0.
        row['peak_gpu_reserved_gib']=torch.cuda.max_memory_reserved()/2**30 if torch.cuda.is_available() else 0.
        row['stage']=stage
        previous=now
        if not all(math.isfinite(float(row.get(k,float('nan')))) for k in ['loss','grad_norm']):
            raise FloatingPointError(f'Nonfinite loss or gradient norm at optimizer step {row["step"]}')
        for key,value in list(row.items()):
            if hasattr(value,'item'):row[key]=value.item()
        with metrics_path.open('a') as handle:
            handle.write(json.dumps(row,allow_nan=False)+'\n')
        if row['step']%100==0:
            write_json(status,{'phase':stage,'step':row['step'],'loss':float(row['loss']),'wall_elapsed_seconds':row['wall_elapsed_seconds'],'peak_gpu_memory_gib':row['peak_gpu_memory_gib']})
            if torch.cuda.is_available():
                probe=subprocess.run(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,memory.free,power.draw,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                telemetry=metrics_path.parent/'gpu_telemetry.csv'
                new=not telemetry.exists()
                with telemetry.open('a') as handle:
                    if new:handle.write('timestamp,gpu_utilization_percent,memory_used_mib,memory_free_mib,power_watts,temperature_celsius\n')
                    if probe.returncode==0:handle.write(probe.stdout)
        if stage=='training' and row['step']%1000==0:
            plot_training(metrics_path.parent)
        return metrics,details

    upstream.update_policy=instrumented
    try:
        write_json(status,{'phase':stage,'step':0})
        upstream.main()
        if stage=='training':plot_training(metrics_path.parent)
        write_json(status,{'phase':stage+'_complete','wall_elapsed_seconds':time.perf_counter()-started})
    except BaseException:
        write_json(status,{'phase':stage+'_failed','error':traceback.format_exc()})
        raise


if __name__=='__main__':
    main()
