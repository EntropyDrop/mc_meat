"""Latest-label OOF retraining; label-free frozen DINO cache, fixed protocols."""
import argparse, json, shutil, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from meat_features import ROOT, sha, write, matrix as old_matrix
from face_features_v3 import matrix, provenance
import train_supervised as baseline
from face_decoder_v5b import FaceDecoder, pack, forward, loss
from fine_supervision import weights as fine_weights
from semantic_schema import validate_labels, encode_labels, SCHEMA, CLASS_IDS
from uv_visibility import annotation_mask, POLICY

CONFIG=dict(semantic_schema=SCHEMA, semantic_class_ids=CLASS_IDS,
 model_module='face_decoder_v5b', epochs=200, lr=.002, eta_min=.0002,
 weight_decay=.01, seed=42, threshold=.5,
 variants=['baseline_retrained','face_binary','face_front_fine'],
 validation='leave-one-duplicate-group-out; fixed epochs and threshold',
 fine_loss_weight=.3, fine_scope='head front inner and outer',
 fine_provenance='saved latest mixed labels; preserve per-cell human/assistant weights',
 unknown_fine='binary only', hidden=48, graph_depth=2,
 visibility_policy=POLICY, dino_frozen=True,
 initializer='random decoder per fold; not pretrained on held-out skins')

def load_rows(data,cache):
 manifest=json.loads((data/'manifest.json').read_text());fp=json.loads((cache/'provenance.json').read_text());assert fp==provenance()
 rows=[]
 for r in manifest['rows']:
  d=data/r['id'];npz=cache/(r['id']+'.npz');cm=json.loads(npz.with_suffix('.json').read_text())
  assert cm['sha256']==sha(npz) and cm['skin_sha256']==r['skin_sha256'] and cm['labels_read'] is False
  for f,k in [('skin.png','skin_sha256'),('meat.png','mask_sha256'),('annotation.json','metadata_sha256'),('valid.png','valid_sha256'),('visible_skin.png','visible_skin_sha256')]:assert sha(d/f)==r[k]
  c=dict(np.load(npz));valid=c['valid'].astype(bool);assert np.array_equal(valid,np.array(Image.open(d/'valid.png')).flatten()>0)
  assert np.all(c['face_coverage'][valid]==1)
  y=(np.array(Image.open(d/'meat.png'))[...,0].flatten()>0).astype(np.float32);sem=np.zeros(4096,np.int64);fw=np.zeros(4096,np.float32)
  if r.get('semantic_sha256'):
   assert sha(d/'semantic.png')==r['semantic_sha256'] and sha(d/'semantic.json')==r['semantic_metadata_sha256']
   sm=json.loads((d/'semantic.json').read_text());sem=np.array(validate_labels(sm['labels'],valid,np.flatnonzero(y).tolist()),np.int64);fw=fine_weights(sm)
   assert (d/'semantic.png').read_bytes()==encode_labels(sem.tolist(),valid)
  rows.append(dict(row=r,cache=c,valid=torch.from_numpy(valid),y=torch.from_numpy(y),semantic=torch.from_numpy(sem),fine_weights=torch.from_numpy(fw),x=matrix(c),old_x=old_matrix(c,'dino_local')))
 return rows,fp

def fit(rows,variant,path):
 if variant=='baseline_retrained':return baseline.fit([dict(r,x=r['old_x']) for r in rows],path)
 torch.manual_seed(42);p=pack(rows,'cuda');net=FaceDecoder(p['x'].shape[1]).cuda()
 assert not p['x'].requires_grad and not any('dino' in k for k in net.state_dict())
 opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,200,eta_min=.0002)
 with path.open('w') as f:
  for epoch in range(1,201):
   net.train();opt.zero_grad(set_to_none=True);total,binary,fine=loss(net,p,variant,details=True);assert torch.isfinite(total)
   total.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();scheduler.step()
   if epoch==1 or epoch%20==0:f.write(json.dumps(dict(epoch=epoch,loss=float(total.detach()),binary=float(binary.detach()),fine=float(fine.detach())))+'\n');f.flush()
 return net.eval()

@torch.inference_mode()
def predict(net,row,variant):
 if variant=='baseline_retrained':prob=net(row['old_x']).sigmoid().numpy();prob[~row['valid'].numpy()]=0;return prob,None
 p=pack([row],'cuda');logits,sem=net(p);prob=np.zeros(4096,np.float32);prob[row['valid'].numpy()]=logits.sigmoid().cpu().numpy()
 fine=np.zeros(4096,np.uint8);fine[row['valid'].numpy()]=np.array(CLASS_IDS)[sem.argmax(-1).cpu().numpy()]
 return prob,fine

def save(path,net,variant,rows,manifest,fp):
 if variant=='baseline_retrained':
  baseline.checkpoint(path,net,'dino_local',manifest,fp['parent'],[r['row']['id'] for r in rows]);return
 torch.save(dict(schema='mc-meat-surface-v3',model_module='face_decoder_v5b',variant=variant,input_dim=rows[0]['x'].shape[1],state_dict={k:v.cpu() for k,v in net.state_dict().items()},config=CONFIG,train_ids=[r['row']['id'] for r in rows],manifest_sha256=sha(manifest),feature_provenance=fp,fine_labeled_cells=sum(int((r['semantic']>0).sum()) for r in rows)),path)

def summarize(per):
 return dict(per_case=per,macro={k:float(np.mean([p[k] for p in per])) for k in ['precision','recall','iou','dice','non_meat_iou']},pooled=baseline.counts_metric(*(sum(p[k] for p in per) for k in ['tp','fp','fn','tn'])))

def previous(rows,out,source,key,name):
 old={r['id']:r for r in json.loads((source/'manifest.json').read_text())['rows']}
 per_old={r['id']:r for r in json.loads((source/'comparison.json').read_text())['results'][key]['per_case']};per=[];audit=[]
 for r in rows:
  rid=r['row']['id'];prior=old[rid];info=per_old[rid]
  for k in ['skin_sha256','valid_sha256','group']:assert prior[k]==r['row'][k]
  held={t['row']['id'] for t in rows if t['row']['group']==r['row']['group']}
  assert held.isdisjoint(info['train_ids']) and info['held_out_group']==r['row']['group']
  probpath=source/rid/(key+'_probability.npy');prob=np.load(probpath).flatten();assert np.isfinite(prob).all()
  np.save(out/rid/(name+'_probability.npy'),prob.reshape(64,64))
  met=baseline.metric(r['y'].numpy(),prob,r['valid'].numpy());met.update({k:info[k] for k in ['id','fold','train_ids','held_out_group']});per.append(met)
  audit.append(dict(id=rid,prediction_path=str(probpath),sha256=sha(probpath),old_mask_sha256=prior['mask_sha256'],latest_mask_sha256=r['row']['mask_sha256']))
 write(out/(name+'_prediction_provenance.json'),dict(source=str(source),source_comparison_sha256=sha(source/'comparison.json'),evaluation='old-label trained OOF predictions rescored on latest labels; never full-model training-set metrics',rows=audit))
 return summarize(per)

def train(data,cache,out):
 assert not out.exists(),'New run required';out.mkdir(parents=True);(out/'source').mkdir();m=data/'manifest.json'
 rows,fp=load_rows(data,cache)
 for p in ROOT.glob('*.py'):shutil.copy2(p,out/'source'/p.name)
 shutil.copy2(ROOT/'surface_topology_v3.json',out/'source'/'surface_topology_v3.json');shutil.copy2(m,out/'manifest.json');shutil.copy2(data/'annotation_audit.json',out/'annotation_audit.json')
 write(out/'configuration.json',CONFIG);write(out/'feature_provenance.json',fp);write(out/'data_source.json',dict(data=str(data),cache=str(cache),manifest_sha256=sha(m)))
 active=ROOT/'service_runtime/production/active.json';shutil.copy2(active,out/'production_before.json')
 for r in rows:
  case=out/r['row']['id'];case.mkdir();shutil.copy2(data/r['row']['id']/'skin.png',case/'skin.png');shutil.copy2(data/r['row']['id']/'meat.png',case/'human_meat.png')
 summary={};audits=[];started=time.time()
 summary['current_oof']=previous(rows,out,ROOT/'service_runtime/production/jobs/f3eb607670b44982b97bfa92b493faec/run','dino_local','current_oof')
 summary['previous_face']=previous(rows,out,ROOT/'runs/face_v5b_20261101','face_front_fine','previous_face')
 groups=sorted(set(r['row']['group'] for r in rows))
 write(out/'folds.json',[dict(fold=i,held_group=g,held_ids=[r['row']['id'] for r in rows if r['row']['group']==g],train_ids=[r['row']['id'] for r in rows if r['row']['group']!=g]) for i,g in enumerate(groups)])
 for variant in CONFIG['variants']:
  vd=out/variant;vd.mkdir();per=[]
  for fold,g in enumerate(groups):
   tr=[r for r in rows if r['row']['group']!=g];held=[r for r in rows if r['row']['group']==g]
   write(out/'status.json',dict(stage='cross_validation',variant=variant,fold=fold+1,total=len(groups),elapsed=time.time()-started))
   net=fit(tr,variant,vd/f'fold-{fold}.jsonl');save(vd/f'fold-{fold}.pt',net,variant,tr,m,fp)
   for r in held:
    prob,fine=predict(net,r,variant);case=out/r['row']['id']
    if fine is not None:np.save(case/(variant+'_fine_prediction.npy'),fine.reshape(64,64))
    np.save(case/(variant+'_probability.npy'),prob.reshape(64,64));(case/(variant+'_meat.png')).write_bytes(annotation_mask(np.flatnonzero((prob>=.5)&r['valid'].numpy()).tolist(),r['valid'].numpy().astype(np.uint8).tobytes()))
    met=baseline.metric(r['y'].numpy(),prob,r['valid'].numpy());met.update(id=r['row']['id'],fold=fold,held_out_group=g,train_ids=[v['row']['id'] for v in tr]);per.append(met)
   del net;torch.cuda.empty_cache();print('FOLD',variant,fold+1,'iou',round(per[-1]['iou'],4),'elapsed',round(time.time()-started),flush=True)
  summary[variant]=summarize(per)
  write(out/'comparison.json',dict(cases=len(rows),groups=len(groups),results=summary,evaluation='latest saved labels; historical whole-skin grouped OOF; not new blind test'))
  write(out/'status.json',dict(stage='final_fit',variant=variant));net=fit(rows,variant,vd/'final_train.jsonl');save(vd/'model.pt',net,variant,rows,m,fp)
  ck=torch.load(vd/'model.pt',weights_only=True,map_location='cpu');reload=baseline.MeatHead(ck['input_dim']) if variant=='baseline_retrained' else FaceDecoder(ck['input_dim']).cuda();reload.load_state_dict(ck['state_dict']);reload.eval()
  diff=max(float(np.abs(predict(net,r,variant)[0]-predict(reload,r,variant)[0]).max()) for r in rows);assert diff==0
  del net,reload;torch.cuda.empty_cache();audits.append(dict(variant=variant,save_reload_max_abs=diff,checkpoint_sha256=sha(vd/'model.pt'),train_ids=ck['train_ids']))
 assert sha(active)==sha(out/'production_before.json'),'Concurrent production update: review before delivery'
 write(out/'inference_audit.json',dict(models=audits,labels_used_as_input=False,face_coverage_complete=True,threshold=.5,dino_frozen=True,features_detached=True,production_deployed=False,production_unchanged=True,fine_metrics_scope='known head-front fine labels only; provenance subsets separate'))
 write(out/'status.json',dict(stage='training_complete',cases=len(rows),groups=len(groups),elapsed=time.time()-started))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 from model_service import lock
 with lock(ROOT/'service_runtime/gpu.lock'):train(a.data,a.cache,a.output)
