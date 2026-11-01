"""Manual v3 training and prediction bridge; binary legacy inference is retained."""
from pathlib import Path
import numpy as np
import torch
from model_service import ROOT,read,write,sha,lock,now,set_stage,snapshot,seed_cache
from uv_visibility import POLICY,annotation_mask

def verified_surface(run,model_id,fingerprint,release):
 ck=run/'surface/model.pt';assert sha(ck)==release['models']['surface']['sha256']
 a=read(run/'inference_audit.json');assert not a['labels_used_as_input'] and a['face_coverage_complete'];assert all(x['save_reload_max_abs']==0 for x in a['models'])
 manifest=read(run/'manifest.json')
 return dict(id=model_id,architecture='surface_v3',checkpoint=str(ck.resolve()),sha256=sha(ck),created_at=now(),visibility_policy=POLICY,count=len(manifest['rows']),dataset_fingerprint=fingerprint,examples=[{k:r[k] for k in ['id','skin_sha256','mask_sha256','semantic_sha256'] if k in r} for r in manifest['rows']],metrics=read(run/'comparison.json')['results']['surface']['macro'],report=str((run/'review_report_portable.html').resolve()),threshold=.5,fine_labeled_cells=sum(r.get('semantic_count',0) for r in manifest['rows']),evaluation='历史皮肤整张分组留一验证，非新数据准确率')

def predict_surface_job(d,req,ckpath):
 from predict_surface_v3 import probabilities
 model=req['model'];cache=ckpath.parent.parent.parent/'face_cache'
 set_stage(d,'waiting_gpu')
 with lock(ROOT/'service_runtime/gpu.lock'):
  set_stage(d,'predicting');p,valid=probabilities(ckpath,d/'skin.png',cache)
 selected=np.flatnonzero((p>=.5)&valid).tolist();(d/'prediction.png').write_bytes(annotation_mask(selected,valid.astype(np.uint8).tobytes()))
 write(d/'prediction.json',dict(selected=selected,skin_revision=req['skin_revision'],model_id=model['id'],checkpoint_sha256=model['sha256'],human_label=False,threshold=.5,visibility_policy=POLICY))

def train_manual(state,d,req,ident):
 from train_supervised import prepare_cache
 from face_features_v3 import prepare
 from train_surface_v4 import train
 from report_surface_v3 import report
 from predict_surface_v3 import probabilities
 from model_service import verified_model
 torch.set_num_threads(4);torch.set_float32_matmul_precision('high');set_stage(d,'validating');snapshot(req['bundle'],d/'data')
 set_stage(d,'waiting_gpu')
 with lock(ROOT/'service_runtime/gpu.lock'):
  set_stage(d,'features');seed_cache(d/'data',d/'cache',state);prepare_cache(d/'data',d/'cache');set_stage(d,'face_features');prepare(d/'data',d/'cache',d/'face_cache')
  set_stage(d,'training');train(d/'data',d/'face_cache',d/'run')
 set_stage(d,'verifying');report(d/'run',d/'face_cache')
 model=verified_model(d/'run','manual-v3-'+now()[:10]+'-'+ident[:8],req['bundle']['fingerprint'])
 first=read(d/'data/manifest.json')['rows'][0]['id'];p,valid=probabilities(Path(model['checkpoint']),d/'data'/first/'skin.png',d/'face_cache')
 assert p.shape==(4096,) and np.isfinite(p).all() and not p[~valid].any()
 write(d/'promotion_audit.json',dict(checkpoint_sha256=model['sha256'],finite=True,invalid_excluded=True,input='skin.png only; hash-verified feature cache; no label read',output_cells=len(p),dataset_fingerprint=req['bundle']['fingerprint'],time=now()))
 with lock(state/'dispatch.lock'):
  write(d/'previous_model.json',read(state/'active.json'));write(state/'active.json',model)
