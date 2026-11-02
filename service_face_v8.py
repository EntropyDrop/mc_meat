"""Manual five-category training, verified candidate promotion, and inference."""
import shutil
from pathlib import Path
import torch
from model_service import ROOT,read,write,sha,lock,now,set_stage,snapshot,seed_cache
from uv_visibility import POLICY,annotation_mask

def verify_inference(run,data,cache):
 import numpy as np
 from predict_face_v8 import prediction_details
 from train_face_v8 import load,predict
 from face_decoder_v8 import FaceDecoder
 from calibrate_face_v8 import apply
 from rejection_face_v8 import apply as reject_unsupported
 checkpoint=run/'face_v8/model.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
 rows,_=load(data,cache);net=FaceDecoder(ck['input_dim']);net.load_state_dict(ck['state_dict']);net.eval();results=[]
 for row in rows[:2]:
  prob,q,balanced=predict(net,row,'face_v8',device='cpu')
  for policy in ['precision','balanced']:
   reference=apply(q,row['valid'].numpy(),row['cache']['region'],ck['body_calibration']) if policy=='precision' and ck.get('body_calibration') else balanced
   if ck.get('fine_support'):reference=reject_unsupported(reference,row['cache'],ck['fine_support'])
   actual,valid,details=prediction_details(checkpoint,data/row['row']['id']/'skin.png',cache,policy)
   difference=float(np.abs(actual-prob).max());assert difference<1e-5 and reference['labels']==details['labels']
   assert not any(details['labels'][i] for i in np.flatnonzero(~valid))
   assert details['selected']==[i for i,c in enumerate(details['labels']) if c in [1,2,3]]
   results.append(dict(id=row['row']['id'],policy=policy,probability_max_abs=difference,label_disagreements=0))
 audit=dict(checkpoint_sha256=sha(checkpoint),input='skin plus hash-verified frozen features only; no annotation input',policies=results)
 write(run/'production_inference_audit.json',audit)
 if (run/'release.json').exists():
  release=read(run/'release.json');release['files']['production_inference_audit.json']=sha(run/'production_inference_audit.json');write(run/'release.json',release)
 return audit

def verified_face(run,model_id,fingerprint,release):
 ck=run/'face_v8/model.pt'
 assert sha(ck)==release['models']['face_v8']['sha256']
 audit=read(run/'inference_audit.json')
 assert audit['dino_frozen'] and not audit['features_use_labels'] and audit['fine_output_validated']
 assert read(run/'training_complete.json')['checkpoint_sha256']==sha(ck)
 m=read(run/'manifest.json');c=read(run/'comparison.json');key=next(k for k in ['face_v8_guarded','face_v8_calibrated','face_v8'] if k in c['results'])
 return dict(id=model_id,architecture='face_v8',checkpoint=str(ck.resolve()),sha256=sha(ck),created_at=now(),visibility_policy=POLICY,count=len(m['rows']),dataset_fingerprint=fingerprint,
  examples=[{k:r[k] for k in ['id','skin_sha256','mask_sha256','semantic_sha256','semantic_metadata_sha256'] if k in r} for r in m['rows']],metrics=c['results'][key]['macro'],fine_metrics=c['results'][key]['fine'],
  report=str((run/'review_report_portable.html').resolve()),threshold=.5,fine_threshold=.5,fine_labeled_cells=sum(r.get('semantic_count',0) for r in m['rows']),evaluation='历史皮肤整张分组五折；细标含人工/助手；非新盲测')

def predict_job(d,req,checkpoint):
 from predict_face_v8 import prediction_details
 from semantic_schema import encode_labels
 run=checkpoint.parent.parent;cache=Path(read(run/'data_source.json')['cache'])
 set_stage(d,'waiting_gpu')
 with lock(ROOT/'service_runtime/gpu.lock'):
  set_stage(d,'predicting');prob,valid,result=prediction_details(checkpoint,d/'skin.png',cache,req.get('preselection_policy','precision'))
 (d/'prediction.png').write_bytes(annotation_mask(result['selected'],valid.astype('uint8').tobytes()))
 (d/'prediction_semantic.png').write_bytes(encode_labels(result['labels'],valid))
 write(d/'prediction.json',dict(**result,skin_revision=req['skin_revision'],model_id=req['model']['id'],checkpoint_sha256=req['model']['sha256'],visibility_policy=POLICY))

def train_manual(state,d,req,ident):
 from train_supervised import prepare_cache
 from face_features_v3 import prepare,provenance
 from train_face_v8 import train
 from calibrate_face_v8 import run as calibrate
 from rejection_face_v8 import run as guard
 from report_face_v8 import report
 from model_service import verified_model
 torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 set_stage(d,'validating');snapshot(req['bundle'],d/'data');previous=read(state/'active.json');write(d/'previous_model.json',previous)
 set_stage(d,'waiting_gpu')
 with lock(ROOT/'service_runtime/gpu.lock'):
  set_stage(d,'features');seed_cache(d/'data',d/'cache',state);prepare_cache(d/'data',d/'cache')
  faces=d/'face_cache';faces.mkdir();fp=provenance();write(faces/'provenance.json',fp)
  candidates=list((state/'jobs').glob('*/face_cache'))+list((ROOT/'cache').glob('*'))
  for r in read(d/'data/manifest.json')['rows']:
   for source in candidates:
    meta=source/(r['id']+'.json');p=meta.with_suffix('.npz')
    if source==faces or not meta.exists() or not p.exists() or not (source/'provenance.json').exists():continue
    m=read(meta)
    if m.get('skin_sha256')==r['skin_sha256'] and m.get('sha256')==sha(p) and read(source/'provenance.json')==fp:
     shutil.copy2(p,faces/p.name);shutil.copy2(meta,faces/meta.name);break
  set_stage(d,'face_features');prepare(d/'data',d/'cache',faces)
  set_stage(d,'training');train(d/'data',faces,d/'base_run',Path(previous['checkpoint']))
  calibrate(d/'base_run',d/'calibrated_run',d/'data',faces)
  guard(d/'calibrated_run',d/'run')
 set_stage(d,'verifying');report(d/'run',faces);verify_inference(d/'run',d/'data',faces)
 model=verified_model(d/'run','manual-face-v8-'+now()[:10]+'-'+ident[:8],req['bundle']['fingerprint'])
 write(d/'candidate.json',model);quality=read(d/'run/quality_gate.json')
 promoted=False
 with lock(state/'dispatch.lock'):
  assert read(state/'active.json')['sha256']==previous['sha256'],'Active model changed; candidate preserved without promotion'
  if quality['passed'] and not req.get('compare_only',False):write(state/'active.json',model);promoted=True
 write(d/'promotion_audit.json',dict(promoted=promoted,quality_gate=quality,checkpoint_sha256=model['sha256'],time=now()))
 set_stage(d,'verifying',promoted=promoted,quality_gate=quality)
