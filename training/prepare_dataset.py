"""Produce a local corrected copy; original Parquet/video objects stay unchanged."""
from pathlib import Path
import hashlib
import json
import shutil
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ['SO101_DATASET'])
DEST = HERE/'derived_dataset'
EXCLUDE = {232, 2672}


def prepare():
    if DEST.exists():
        raise FileExistsError(f'Refusing to overwrite {DEST}')
    evidence = json.loads((HERE/'preflight/wrist_interior_verification.json').read_text())
    reset = json.loads((HERE/'preflight/full_reset_alignment.json').read_text())
    assert evidence['interior_comparisons'] == 120 and evidence['corrected_max_pixel_mae'] < .1
    assert all(p['reset_pixel_mae'] is not None and p['reset_pixel_mae'] < 3 for p in reset['cameras']['wrist']['probes'])
    assert all(p['reset_pixel_mae'] is not None and p['reset_pixel_mae'] < 8 for p in reset['cameras']['front']['probes'])
    assert all(p['reset_pixel_mae'] is not None and p['reset_pixel_mae'] < 15 for p in reset['cameras']['overhead']['probes'])
    source_info = json.loads((SOURCE/'meta/info.json').read_text())
    eps = pq.read_table(SOURCE/'meta/episodes').to_pandas().sort_values('episode_index')
    table = pq.read_table(SOURCE/'data')
    states = np.asarray(table['observation.state'].to_pylist(),dtype=np.float32)
    actions = np.asarray(table['action'].to_pylist(),dtype=np.float32)
    indices=[]
    metas=[]
    source_to_new={}
    frame_indices=[]
    episode_indices=[]
    position=0
    for _, row in eps.iterrows():
        source_ep=int(row.episode_index)
        if source_ep in EXCLUDE:
            continue
        start=int(row.dataset_from_index)
        end=int(row.dataset_to_index)
        assert np.all(actions[start] == 0), 'Expected known dummy zero-action reset record'
        selected=np.arange(start+1,end,dtype=np.int64)
        length=len(selected)
        ep=len(metas)
        source_to_new[source_ep]=ep
        metadata={k:(int(v) if k not in [c for c in eps.columns if 'timestamp' in c] else float(v)) for k,v in row.items()}
        metadata.update(episode_index=ep, source_episode_index=source_ep, length=length, dataset_from_index=position, dataset_to_index=position+length, **{'data/chunk_index':0,'data/file_index':0})
        for cam in ['front','wrist','overhead']:
            key=f'videos/observation.images.{cam}/'
            shift=-301/30 if cam=='wrist' and source_ep>2672 else 0
            metadata[key+'from_timestamp']=float(row[key+'from_timestamp'])+1/30+shift
            metadata[key+'to_timestamp']=metadata[key+'from_timestamp']+length/30
        metas.append(metadata)
        indices.append(selected)
        frame_indices.append(np.arange(length,dtype=np.int64))
        episode_indices.append(np.full(length,ep,dtype=np.int64))
        position+=length
    source_indices=np.concatenate(indices)
    frame_index=np.concatenate(frame_indices)
    ep_index=np.concatenate(episode_indices)
    groups=json.loads((HERE/'preflight/trajectory_groups.json').read_text())['groups']
    groups=[[source_to_new[x] for x in group if x in source_to_new] for group in groups]
    groups=[g for g in groups if g]
    order=np.random.default_rng(42).permutation(len(groups))
    n_val=int(np.ceil(len(groups)*.1))
    val_groups=[groups[i] for i in order[:n_val]]
    train_groups=[groups[i] for i in order[n_val:]]
    train=sorted(x for g in train_groups for x in g)
    val=sorted(x for g in val_groups for x in g)
    assert set(train).isdisjoint(val) and len(train)+len(val)==len(metas)
    train_mask=np.isin(ep_index,train)
    numeric={'observation.state':states[source_indices],'action':actions[source_indices]}
    stats={}
    for key,array in numeric.items():
        a=array[train_mask].astype(np.float64)
        stats[key]={'min':a.min(0).tolist(),'max':a.max(0).tolist(),'mean':a.mean(0).tolist(),'std':a.std(0).tolist(),'count':[len(a)]}
    for cam in ['front','wrist','overhead']:
        # Visual normalization is IDENTITY. These are declared fixed constants,
        # not statistics estimated from validation images.
        stats['observation.images.'+cam]={'min':[[[0.]]]*3,'max':[[[1.]]]*3,'mean':[[[.5]]]*3,'std':[[[.25]]]*3,'count':[int(train_mask.sum())]}
    (DEST/'data/chunk-000').mkdir(parents=True)
    (DEST/'meta/episodes/chunk-000').mkdir(parents=True)
    new_table=pa.table({'index':pa.array(np.arange(position,dtype=np.int64)), 'episode_index':pa.array(ep_index), 'frame_index':pa.array(frame_index), 'timestamp':pa.array((frame_index/30).astype(np.float32)), 'task_index':pa.array(np.zeros(position,dtype=np.int64)), **{key:pa.FixedSizeListArray.from_arrays(pa.array(a.reshape(-1),type=pa.float32()),6) for key,a in numeric.items()}})
    pq.write_table(new_table,DEST/'data/chunk-000/file-000.parquet',compression='zstd',row_group_size=65536)
    pq.write_table(pa.Table.from_pylist(metas),DEST/'meta/episodes/chunk-000/file-000.parquet',compression='zstd')
    for name in ['tasks.parquet','tasks.json']:
        shutil.copy2(SOURCE/'meta'/name,DEST/'meta'/name)
    info=source_info.copy()
    info.update(total_episodes=len(metas),total_frames=position,splits={'all':f'0:{len(metas)}'})
    (DEST/'meta/info.json').write_text(json.dumps(info,indent=2))
    (DEST/'meta/stats.json').write_text(json.dumps(stats,indent=2))
    (DEST/'videos').symlink_to(SOURCE/'videos',target_is_directory=True)
    # One fixed anchor per distinct validation trajectory, up to 128 groups.
    representatives=[g[0] for g in val_groups]
    anchor_eps=[representatives[i] for i in np.linspace(0,len(representatives)-1,min(128,len(representatives)),dtype=int)]
    anchors=[metas[x]['dataset_from_index']+max(0,metas[x]['length']//2) for x in anchor_eps]
    split={'status':'curated; reset and duplicate-interior alignment verified','seed':42,'unit':'identical state/action trajectory group','train':train,'validation':val,'train_groups':len(train_groups),'validation_groups':len(val_groups),'source_to_derived_episode':source_to_new,'validation_anchor_global_indices':anchors,'normalization':'computed only on training frames; visual IDENTITY','note':'Videos are absolute symlinks to original source files. No source data was modified.'}
    (HERE/'configs/episode_split.json').write_text(json.dumps(split,indent=2))
    report={'source':str(SOURCE),'derived':str(DEST),'excluded_source_episodes':sorted(EXCLUDE),'excluded_reason':'Camera episode durations do not match state/action data; wrist episode 2672 video is absent.','dropped_dummy_first_action_records':len(metas),'timestamp':'episode-relative frame_index / 30, float32','wrist_correction':'Subtract 301/30 seconds from video offsets for source episode >2672','episodes':len(metas),'frames':position,'train_episodes':len(train),'validation_episodes':len(val),'train_frames':int(train_mask.sum()),'validation_frames':int((~train_mask).sum()),'unique_trajectory_groups':len(groups),'duplicate_group_overlap':0,'data_sha256':hashlib.sha256((DEST/'data/chunk-000/file-000.parquet').read_bytes()).hexdigest(),'validation_limits':'All source reset starts checked, plus 120 duplicated-trajectory interior frame comparisons. Not every video frame was inspected.'}
    (DEST/'provenance.json').write_text(json.dumps(report,indent=2))
    (HERE/'preflight/dataset_repair.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    prepare()
