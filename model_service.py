"""SSH JSON broker for opt-in meat predictions and manual supervised model updates.

No HTTP listener, no training on startup, and no persistent CUDA process.
Each request is frozen; only successful verified training promotes a new model.
"""
import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
from uv_visibility import POLICY, annotation_mask

ROOT = Path(__file__).resolve().parent
TERMINAL = {'complete', 'failed'}

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.' + str(os.getpid()) + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); os.replace(tmp, p)

@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield

def job_path(state, ident):
    if not re.fullmatch(r'[a-f0-9]{32}', ident or ''): raise ValueError('无效任务编号')
    return state / 'jobs' / ident

def proc_stamp(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, IndexError): return None

def job_status(d):
    s = read(d / 'status.json')
    launch = read(d / 'launch.json') if (d/'launch.json').exists() else {}
    if s['stage'] not in TERMINAL and time.time() - (d/'status.json').stat().st_mtime > 30:
        if not launch.get('stamp') or proc_stamp(launch['pid']) != launch['stamp']:
            s.update(stage='failed', error='任务进程已退出；旧预选模型保持可用。可重新点击训练或预选。', updated_at=now())
            write(d/'status.json', s)
    result = dict(s, pid=launch.get('pid'))
    nested = d/'face_cache/status.json' if s['stage']=='face_features' else d/'cache/status.json' if s['stage'] == 'features' else d/'run/status.json'
    if s['stage'] not in TERMINAL and nested.exists(): result['progress'] = read(nested)
    if s['kind'] == 'predict' and s['stage'] == 'complete': result['prediction'] = read(d/'prediction.json')
    return result

def verified_model(run, model_id, dataset_fingerprint=None):
    release = read(run/'release.json')
    assert sha(run/'manifest.json') == release['manifest_sha256'], '训练清单校验失败'
    for name, expected in release['files'].items():
        assert Path(name).name == name and sha(run/name) == expected, '发布文件校验失败'
    if release.get('schema')=='mc-meat-surface-v3':
        from service_surface_v3 import verified_surface
        return verified_surface(run,model_id,dataset_fingerprint,release)
    ck = run/'dino_local/model.pt'
    assert sha(ck) == release['models']['dino_local']['sha256'], '模型校验失败'
    audit = read(run/'inference_audit.json')
    assert audit['invalid_excluded'] and not audit['features_use_labels']
    assert all(a['save_reload_max_abs'] == 0 for a in audit['models'])
    manifest = read(run/'manifest.json')
    return dict(id=model_id, checkpoint=str(ck.resolve()), sha256=sha(ck), created_at=now(),
                visibility_policy=manifest.get('visibility_policy'),count=len(manifest['rows']), dataset_fingerprint=dataset_fingerprint,
                examples=[dict(id=r['id'], skin_sha256=r['skin_sha256'], mask_sha256=r['mask_sha256']) for r in manifest['rows']],
                metrics=read(run/'comparison.json')['results']['dino_local']['macro'],
                report=str((run/'review_report_portable.html').resolve()), threshold=0.5,
                evaluation='按整张皮肤分组留一验证，历史样本；不代表新数据准确率')

def snapshot(bundle, out):
    import numpy as np
    from PIL import Image
    from prepare_human import validate, groups_for
    rows = bundle['examples']; checked = []
    assert 3 <= len(rows) <= 5000, '至少需要 3 张有效人工标注'
    for r in rows:
        ident = r['id']; assert re.fullmatch(r'[A-Za-z0-9_-]{1,100}', ident)
        skin = base64.b64decode(r['skin'], validate=True); mask = base64.b64decode(r['mask'], validate=True)
        mb = base64.b64decode(r['metadata'], validate=True); meta = json.loads(mb)
        assert meta['id'] == ident
        rgba, valid, truth = validate(skin, mask, meta)
        rgba = rgba.copy(); rgba[~valid] = 0
        checked.append((r, skin, mask, mb, rgba, valid, truth))
    assert len(set(r['id'] for r in rows)) == len(rows), '存在重复 ID，需先消除歧义'
    groups, audit = groups_for([x[4] for x in checked], [x[0]['id'] for x in checked])
    assert len(set(groups)) >= 3, '独立皮肤分组不足 3 组'
    out.mkdir(); saved=[]
    for group, (r, skin, mask, mb, rgba, valid, truth) in zip(groups, checked):
        d=out/r['id'];d.mkdir()
        for name, content in [('skin.png',skin),('meat.png',mask),('annotation.json',mb)]: (d/name).write_bytes(content)
        fine={}
        if r.get('semantic_sha256'):
            from semantic_schema import validate_labels,encode_labels,SCHEMA,CLASSES
            sb=base64.b64decode(r['semantic'],validate=True);sm=base64.b64decode(r['semantic_metadata'],validate=True);metadata=json.loads(sm)
            assert hashlib.sha256(sb).hexdigest()==r['semantic_sha256'] and hashlib.sha256(sm).hexdigest()==r['semantic_metadata_sha256']
            assert metadata['schema']==SCHEMA and metadata['classes']==CLASSES and metadata['id']==r['id'] and metadata['skin_sha256']==sha(d/'skin.png')
            from fine_supervision import weights as validate_fine_weights
            validate_fine_weights(metadata)
            assert metadata['visibility_policy']==POLICY and metadata['valid_sha256']==hashlib.sha256(valid.flatten().astype(np.uint8).tobytes()).hexdigest()
            labels=validate_labels(metadata['labels'],valid.flatten(),np.flatnonzero(truth.flatten()>0).tolist())
            assert sb==encode_labels(labels,valid.flatten()) and metadata['mask_sha256']==r['semantic_sha256']
            assert sum(c>0 for c in labels)==r['semantic_count']==metadata['known_count']
            (d/'semantic.png').write_bytes(sb);(d/'semantic.json').write_bytes(sm)
            fine=dict(semantic_sha256=sha(d/'semantic.png'),semantic_metadata_sha256=sha(d/'semantic.json'),semantic_count=metadata['known_count'])
        Image.fromarray(valid.astype(np.uint8)*255).save(d/'valid.png')
        Image.fromarray(rgba).save(d/'visible_skin.png')
        saved.append(dict(id=r['id'],group=group,model='classic',source_skin=r['source_skin'],source_annotation=r['source_annotation'],
                          skin_sha256=sha(d/'skin.png'),mask_sha256=sha(d/'meat.png'),metadata_sha256=sha(d/'annotation.json'),
                          visible_skin_sha256=sha(d/'visible_skin.png'),valid_sha256=sha(d/'valid.png'),valid_cells=int(valid.sum()),meat_cells=int((truth>0).sum()),
                          non_meat_cells=int((valid & (truth==0)).sum()),**fine))
    write(out/'manifest.json',dict(schema='mc-meat-human-snapshot-v1',source='saved_local_working_tree',
          dataset_root=bundle['dataset_root'],dataset_commit=bundle.get('dataset_commit','working-tree'),
          snapshot_at=now(),visibility_policy=POLICY,rows=saved,duplicate_audit=audit,independent_groups=len(set(groups)),
          note='Explicitly saved complete human masks only. Not a new blind benchmark.'))

def seed_cache(data, cache, state):
    from meat_features import provenance
    fp=provenance();cache.mkdir();write(cache/'provenance.json',fp)
    candidates=[ROOT/'cache/human_v2_20261029'] + sorted((state/'jobs').glob('*/cache'))
    for r in read(data/'manifest.json')['rows']:
        for src in candidates:
            if src == cache or not (src/'provenance.json').exists() or read(src/'provenance.json') != fp: continue
            p=src/(r['id']+'.npz'); m=p.with_suffix('.json')
            if p.exists() and m.exists():
                a=read(m)
                if a['input_sha256']==r['skin_sha256'] and a['cache_sha256']==sha(p):
                    shutil.copy2(p,cache/p.name);shutil.copy2(m,cache/m.name);break

def predict(d):
    import numpy as np
    from PIL import Image
    import torch
    from meat_features import provenance, matrix, extract, make_dino, make_renderer, read_skin
    from train_supervised import MeatHead
    req=read(d/'request.json'); model=req['model']; ckpath=Path(model['checkpoint'])
    assert sha(ckpath)==model['sha256'], '当前模型校验失败'
    skin=base64.b64decode(req['skin'],validate=True);(d/'skin.png').write_bytes(skin)
    assert sha(d/'skin.png')==req['skin_revision']
    read_skin(d/'skin.png')  # Reject unsupported geometry before loading a GPU model.
    torch.set_num_threads(4);ck=torch.load(ckpath,map_location='cpu',weights_only=True)
    if ck.get('schema')=='mc-meat-surface-v3':
        from service_surface_v3 import predict_surface_job
        return predict_surface_job(d,req,ckpath)
    fp=provenance()
    if ck['feature_provenance'] != fp:
        # Keep the old active model usable while the new visibility model trains.
        # Its immutable implementation must match the checkpoint's recorded code.
        frozen=ckpath.parent.parent/'source'
        for name,expected in ck['feature_provenance']['code'].items():
            assert Path(name).name==name and sha(frozen/name)==expected
        with lock(ROOT/'service_runtime/gpu.lock'):
            subprocess.run([sys.executable,str(frozen/'predict_meat.py'),'--checkpoint',str(ckpath),
                '--input',str(d/'skin.png'),'--output',str(d/'legacy_prediction.png')],check=True)
        _,_,valid=read_skin(d/'skin.png')
        old=np.asarray(Image.open(d/'legacy_prediction.png').convert('L'))>0
        selected=np.flatnonzero(old & valid).tolist()
        (d/'prediction.png').write_bytes(annotation_mask(selected,valid.astype(np.uint8).tobytes()))
        write(d/'prediction.json',dict(selected=selected,skin_revision=req['skin_revision'],model_id=model['id'],
              checkpoint_sha256=model['sha256'],human_label=False,threshold=.5,visibility_policy=POLICY))
        return
    # Reuse only features, never labels. Hash-match the input, including for training-set skins.
    values=None
    candidates=[ROOT/'cache/human_v2_20261029',ckpath.parent.parent.parent/'cache']
    for cache in candidates:
        if not (cache/'provenance.json').exists() or read(cache/'provenance.json')!=fp:continue
        for m in cache.glob('*.json'):
            a=read(m);p=m.with_suffix('.npz')
            if a.get('input_sha256')==req['skin_revision'] and p.exists() and a['cache_sha256']==sha(p):
                values=dict(np.load(p,allow_pickle=False));break
        if values is not None: break
    if values is None:
        set_stage(d,'waiting_gpu')
        with lock(ROOT/'service_runtime/gpu.lock'):
            set_stage(d,'predicting')
            backbone=make_dino();renderer=make_renderer('cuda')
            values,audit=extract(d/'skin.png',backbone,renderer)
            del backbone,renderer;torch.cuda.empty_cache()
    net=MeatHead(ck['input_dim']);net.load_state_dict(ck['state_dict']);net.eval()
    with torch.no_grad():prob=net(matrix(values,ck['variant'])).sigmoid().numpy()
    assert np.isfinite(prob).all()
    valid=values['valid'].astype(bool);selected=np.flatnonzero((prob>=.5)&valid).tolist()
    (d/'prediction.png').write_bytes(annotation_mask(selected,valid.astype(np.uint8).tobytes()))
    write(d/'prediction.json',dict(selected=selected,skin_revision=req['skin_revision'],model_id=model['id'],
          checkpoint_sha256=model['sha256'],human_label=False,threshold=.5))

def set_stage(d, stage, **kwargs):
    s=read(d/'status.json');s.update(stage=stage,updated_at=now(),**kwargs);write(d/'status.json',s)

def worker(state, ident):
    d=job_path(state,ident)
    try:
        req=read(d/'request.json')
        if req['kind']=='predict': predict(d)
        else:
            from service_surface_v3 import train_manual
            train_manual(state,d,req,ident)
        set_stage(d,'complete')
    except BaseException as e:
        traceback.print_exc()
        set_stage(d,'failed',error=(str(e) or type(e).__name__)[:500])
        return 1
    return 0

def dispatch(state, command, req):
    state.mkdir(parents=True,exist_ok=True)
    with lock(state/'dispatch.lock'):
        if command=='init':
            if not (state/'active.json').exists():
                write(state/'active.json',verified_model(ROOT/'runs/human_v2_20261029','human-v2-13'))
            return {'active':read(state/'active.json')}
        if command=='status':
            last=read(state/'last_train.json')['id'] if (state/'last_train.json').exists() else None
            return dict(active=read(state/'active.json') if (state/'active.json').exists() else None,
                        training=job_status(job_path(state,last)) if last else None)
        if command=='job':return job_status(job_path(state,req['id']))
        assert command in ('start_predict','start_train')
        kind=command.split('_')[1];ident=req['request_id'];d=job_path(state,ident)
        request_hash=hashlib.sha256(json.dumps(req,sort_keys=True).encode()).hexdigest()
        if d.exists():
            assert read(d/'request.json')['request_hash']==request_hash,'重复任务编号的请求内容不同'
            return job_status(d)
        if kind=='train' and (state/'last_train.json').exists():
            old=job_status(job_path(state,read(state/'last_train.json')['id']))
            if old['stage'] not in TERMINAL: return old
        if kind=='predict':
            # Prevent unbounded queued GPU jobs from repeated clicks/tabs.
            active=[p for p in (state/'jobs').glob('*/status.json') if read(p).get('stage') not in TERMINAL]
            if sum(job_status(p.parent)['kind']=='predict' and read(p).get('stage') not in TERMINAL for p in active)>=4:
                raise ValueError('预选队列已满，请稍后重试')
            req['model']=read(state/'active.json')
        else: assert isinstance(req.get('bundle',{}).get('examples'),list)
        req.update(kind=kind,request_hash=request_hash)
        d.mkdir(parents=True);write(d/'request.json',req)
        write(d/'status.json',dict(id=ident,kind=kind,stage='queued',created_at=now(),updated_at=now()))
        with (d/'worker.log').open('ab') as log:
            env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--state',str(state),'--id',ident],
                               stdin=subprocess.DEVNULL,stdout=log,stderr=log,cwd=ROOT,env=env,start_new_session=True)
        write(d/'launch.json',dict(pid=p.pid,stamp=proc_stamp(p.pid),created_at=now()))
        if kind=='train':write(state/'last_train.json',dict(id=ident))
        return job_status(d)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['init','status','job','start_predict','start_train','worker'])
    p.add_argument('--state',type=Path,default=ROOT/'service_runtime/production');p.add_argument('--id')
    a=p.parse_args();state=a.state.resolve()
    if not state.is_relative_to(ROOT/'service_runtime'):raise ValueError('State must be inside service_runtime')
    if a.command=='worker':return worker(state,a.id)
    try:
        raw=sys.stdin.buffer.read(64*1024*1024+1)
        if len(raw)>64*1024*1024:raise ValueError('训练快照过大')
        req=json.loads(raw or '{}');result=dispatch(state,a.command,req)
        print(json.dumps(result,ensure_ascii=False));return 0
    except Exception as e:
        print(json.dumps({'error':str(e) or type(e).__name__},ensure_ascii=False));return 1

if __name__=='__main__':sys.exit(main())
