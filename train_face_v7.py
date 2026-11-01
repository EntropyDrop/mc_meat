"""Add reviewed weak labels; evaluate only unchanged human-labelled groups.

DINO features remain frozen. Every evaluation decoder starts from the same
random seed, excluding the entire held-out duplicate group. Assistant labels are
training-only, and are never presented as human evaluation ground truth.
"""
import argparse, json, shutil, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from meat_features import ROOT, sha, write
from face_decoder_v5b import FaceDecoder, pack
from semantic_schema import CLASS_IDS
from train_face_v6 import load_rows, predict, summarize, previous
from train_supervised import metric
from uv_visibility import annotation_mask

CONFIG=dict(model_module='face_decoder_v5b',variant='face_front_fine',epochs=200,
 lr=.002,eta_min=.0002,weight_decay=.01,seed=42,threshold=.5,dino_frozen=True,
 assistant_binary_weight=.5,assistant_evaluation=False,fine_loss_weight=.3,
 fine_scope='known inner/outer head-front cells only',hidden=48,graph_depth=2,
 initializer='random decoder per held-out human group; no held-out pretraining',
 weight_normalization='binary per skin then mean; absolute case confidence; fine absolute confidence times case confidence',
 validation='historical human-labelled groups only; assistant cases training-only; not a new blind test')

def weighted_loss(net,p,case_weights,details=False):
 logits,sem=net(p);y=p['y'];w=case_weights[p['owner']]
 pos=(w*y).sum();neg=(w*(1-y)).sum();pw=(neg/pos.clamp_min(1)).sqrt().clamp(1,5)
 per=F.binary_cross_entropy_with_logits(logits,y,pos_weight=pw,reduction='none')
 count=torch.bincount(p['owner']).to(y)
 binary=(per*w/count[p['owner']]).sum()/len(count)
 front=torch.zeros_like(p['semantic'],dtype=torch.bool);front[p['front_indices'][p['front_mask']]]=True
 known=(p['semantic']>0)&(p['fine_weights']>0)&front&(w>0);fine=sem.sum()*0
 if known.any():
  target=(p['semantic'][known,None]==p['semantic'].new_tensor(CLASS_IDS)[None,:]).long().argmax(1)
  n=torch.bincount(target,weights=w[known],minlength=5).to(y)
  balance=(n.sum()/n.clamp_min(1)).sqrt().clamp(max=8);balance=balance/balance[n>0].mean()
  weighted=F.cross_entropy(sem[known],target,reduction='none')*p['fine_weights'][known]*w[known]*balance[target]
  fine=weighted.sum()/balance[target].sum().clamp_min(1)
 total=binary+.3*fine
 return (total,binary,fine) if details else total

def fit(rows,path,status,stage,fold=None):
 torch.manual_seed(CONFIG['seed']);p=pack(rows,'cuda');net=FaceDecoder(p['x'].shape[1]).cuda()
 assert not p['x'].requires_grad and not any('dino' in k for k in net.state_dict())
 weights=torch.tensor([r['row']['binary_training_weight'] for r in rows],device='cuda')
 opt=torch.optim.AdamW(net.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,CONFIG['epochs'],eta_min=CONFIG['eta_min'])
 with path.open('x') as f:
  for epoch in range(1,CONFIG['epochs']+1):
   net.train();opt.zero_grad(set_to_none=True);total,binary,fine=weighted_loss(net,p,weights,True)
   assert torch.isfinite(total);total.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();scheduler.step()
   if epoch==1 or epoch%20==0:
    log=dict(epoch=epoch,loss=float(total.detach()),binary=float(binary.detach()),fine=float(fine.detach()))
    f.write(json.dumps(log)+'\n');f.flush();write(status,dict(stage=stage,fold=fold,total_folds=34,**log,updated_at=time.time()))
 return net.eval()

def save(path,net,rows,manifest,fp):
 torch.save(dict(schema='mc-meat-surface-v3',model_module='face_decoder_v5b',variant='face_front_fine',input_dim=rows[0]['x'].shape[1],state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},config=CONFIG,train_ids=[r['row']['id'] for r in rows],manifest_sha256=sha(manifest),feature_provenance=fp,case_weights={r['row']['id']:r['row']['binary_training_weight'] for r in rows}),path)

def train(data,cache,out,control):
 assert not out.exists(),'Use a new run';out.mkdir(parents=True);(out/'source').mkdir()
 for p in ROOT.glob('*.py'):shutil.copy2(p,out/'source'/p.name)
 shutil.copy2(ROOT/'surface_topology_v3.json',out/'source'/'surface_topology_v3.json')
 source_hashes={str(p.relative_to(out)):sha(p) for p in (out/'source').iterdir() if p.is_file()}
 write(out/'source_hashes.json',source_hashes);write(out/'configuration.json',CONFIG)
 write(out/'status.json',dict(stage='verifying_inputs',updated_at=time.time()))
 m=data/'manifest.json';rows,fp=load_rows(data,cache)
 human=[r for r in rows if r['row']['evaluation_eligible']];weak=[r for r in rows if not r['row']['evaluation_eligible']]
 assert len(human)==34 and len(weak)==20 and all(r['row']['binary_training_weight']==.5 for r in weak)
 assert all(r['row']['binary_training_weight']==1 for r in human)
 old={r['id']:r for r in json.loads((control/'manifest.json').read_text())['rows']}
 for r in human:
  for k in ['skin_sha256','mask_sha256','metadata_sha256','valid_sha256','semantic_sha256','semantic_metadata_sha256','group']:assert r['row'][k]==old[r['row']['id']][k],(r['row']['id'],k)
 assert json.loads((data/'feature_equivalence.json').read_text())['exactly_equal']
 for name in ['manifest.json','annotation_audit.json','feature_equivalence.json']:shutil.copy2(data/name,out/name)
 write(out/'feature_provenance.json',fp);write(out/'data_source.json',dict(data=str(data),cache=str(cache),control=str(control),manifest_sha256=sha(m)))
 active=ROOT/'service_runtime/production/active.json';shutil.copy2(active,out/'production_before.json')
 for r in rows:
  d=out/r['row']['id'];d.mkdir();shutil.copy2(data/r['row']['id']/'skin.png',d/'skin.png');shutil.copy2(data/r['row']['id']/'meat.png',d/'reference_meat.png')
 summaries={}
 for key,name in [('current_oof','online_scheme_oof'),('baseline_retrained','old_head_latest_labels'),('face_front_fine','previous_face_v6')]:summaries[name]=previous(human,out,control,key,name)
 groups=sorted({r['row']['group'] for r in human});folds=[]
 for i,g in enumerate(groups):
  train_ids=[r['row']['id'] for r in rows if r['row']['group']!=g];held_ids=[r['row']['id'] for r in human if r['row']['group']==g]
  assert set(train_ids).isdisjoint(held_ids)
  folds.append(dict(fold=i,held_group=g,held_ids=held_ids,train_ids=train_ids,excluded_weak_ids=[r['row']['id'] for r in weak if r['row']['group']==g]))
 write(out/'folds.json',folds)
 def comparison(per,complete=False):
  result=dict(summaries)
  if per:result['assistant20_face']=summarize(per)
  write(out/'comparison.json',dict(results=result,evaluation_cases=34,training_cases=54,completed_folds=len(per),total_folds=len(groups),complete=complete,evaluation=CONFIG['validation']))
 comparison([]);vd=out/'face_front_fine';vd.mkdir();per=[];started=time.time()
 for i,g in enumerate(groups):
  tr=[r for r in rows if r['row']['group']!=g];held=[r for r in human if r['row']['group']==g]
  net=fit(tr,vd/f'fold-{i:02d}.jsonl',out/'status.json','cross_validation',i+1)
  save(vd/f'fold-{i:02d}.pt',net,tr,m,fp)
  for r in held:
   prob,fine=predict(net,r,'face_front_fine');d=out/r['row']['id'];np.save(d/'assistant20_face_probability.npy',prob.reshape(64,64));np.save(d/'assistant20_face_fine_prediction.npy',fine.reshape(64,64))
   (d/'assistant20_face_meat.png').write_bytes(annotation_mask(np.flatnonzero((prob>=.5)&r['valid'].numpy()).tolist(),r['valid'].numpy().astype(np.uint8).tobytes()))
   met=metric(r['y'].numpy(),prob,r['valid'].numpy());met.update(id=r['row']['id'],fold=i,held_out_group=g,train_ids=[t['row']['id'] for t in tr]);per.append(met)
  del net;torch.cuda.empty_cache();comparison(per)
  print('FOLD',i+1,'of',len(groups),'IoU',round(per[-1]['iou'],4),'elapsed',round(time.time()-started),flush=True)
 net=fit(rows,vd/'final_train.jsonl',out/'status.json','final_fit');save(vd/'model.pt',net,rows,m,fp)
 ck=torch.load(vd/'model.pt',weights_only=True,map_location='cpu');loaded=FaceDecoder(ck['input_dim']).cuda();loaded.load_state_dict(ck['state_dict']);loaded.eval()
 maxdiff=0.
 for r in rows:
  prob,fine=predict(net,r,'face_front_fine');other,_=predict(loaded,r,'face_front_fine');maxdiff=max(maxdiff,float(np.abs(prob-other).max()))
  assert np.isfinite(prob).all() and not prob[~r['valid'].numpy()].any()
  if not r['row']['evaluation_eligible']:
   d=out/r['row']['id'];np.save(d/'training_fit_probability.npy',prob.reshape(64,64));np.save(d/'training_fit_fine.npy',fine.reshape(64,64))
 assert maxdiff==0
 for p,h in source_hashes.items():assert sha(out/p)==h
 assert sha(m)==ck['manifest_sha256']
 audit=dict(save_reload_max_abs=maxdiff,checkpoint_sha256=sha(vd/'model.pt'),features_detached=True,dino_frozen=True,labels_used_as_input=False,assistant_cases_training_only=True,human_labels_unchanged=True,face_coverage_complete=True,production_deployed=False,production_unchanged=sha(active)==sha(out/'production_before.json'),train_ids=ck['train_ids'])
 write(out/'inference_audit.json',audit);comparison(per,True)
 write(out/'training_complete.json',dict(epochs=CONFIG['epochs'],folds=len(groups),cases=len(rows),elapsed=time.time()-started,checkpoint_sha256=sha(vd/'model.pt')))
 write(out/'status.json',dict(stage='training_complete',updated_at=time.time(),elapsed=time.time()-started))
 del net,loaded;torch.cuda.empty_cache()

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['data','cache','output','control']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 from model_service import lock
 try:
  with lock(ROOT/'service_runtime/gpu.lock'):train(a.data,a.cache,a.output,a.control)
  from report_face_v7 import report
  report(a.output)
 except BaseException:
  if a.output.exists():write(a.output/'failure.json',dict(traceback=traceback.format_exc(),time=time.time()))
  raise
