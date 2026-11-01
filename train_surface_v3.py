"""Frozen 34-case comparison: original OOF, all-face MLP, surface graph."""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from meat_features import ROOT,sha,write
from face_features_v3 import matrix,provenance
from train_supervised import MeatHead,metric,counts_metric
from surface_model_v3 import SurfaceHead,pack,forward,loss
from semantic_schema import validate_labels,encode_labels
from uv_visibility import annotation_mask,POLICY
CONFIG=dict(epochs=200,lr=.002,eta_min=.0002,weight_decay=.01,seed=42,threshold=.5,variants=['all_faces','surface'],validation='leave-one-duplicate-group-out; fixed epochs and threshold',fine_loss_weight=.3,hidden=48,graph_depth=2,visibility_policy=POLICY,unknown_fine='ignored',no_pseudo_labels=True)

def fit(rows,variant,path):
 torch.manual_seed(42);p=pack(rows,'cuda');net=(MeatHead(p['x'].shape[1]) if variant=='all_faces' else SurfaceHead(p['x'].shape[1])).cuda()
 opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,200,eta_min=.0002)
 with path.open('w') as f:
  for epoch in range(1,201):
   net.train();opt.zero_grad(set_to_none=True);l=loss(net,p,variant);assert torch.isfinite(l)
   l.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();scheduler.step()
   if epoch==1 or epoch%20==0:f.write(json.dumps(dict(epoch=epoch,loss=float(l.detach())))+'\n');f.flush()
 return net.eval()

@torch.inference_mode()
def predict(net,row,variant):
 p=pack([row],'cuda');prob=np.zeros(4096,np.float32);prob[row['valid'].numpy()]=forward(net,p,variant).sigmoid().cpu().numpy();return prob

def save(path,net,variant,rows,manifest,fp):
 torch.save(dict(schema='mc-meat-surface-v3',variant=variant,input_dim=rows[0]['x'].shape[1],state_dict={k:v.cpu() for k,v in net.state_dict().items()},config=CONFIG,train_ids=[r['row']['id'] for r in rows],manifest_sha256=sha(manifest),feature_provenance=fp,fine_labeled_cells=sum(int((r['semantic']>0).sum()) for r in rows)),path)

def train(data,cache,out,baseline=None):
 assert not out.exists(),'New run required';out.mkdir(parents=True);(out/'source').mkdir();m=data/'manifest.json';manifest=json.loads(m.read_text());fp=json.loads((cache/'provenance.json').read_text());assert fp==provenance()
 for p in ROOT.glob('*.py'):shutil.copy2(p,out/'source'/p.name)
 shutil.copy2(ROOT/'surface_topology_v3.json',out/'source'/'surface_topology_v3.json');shutil.copy2(m,out/'manifest.json');write(out/'configuration.json',CONFIG);write(out/'feature_provenance.json',fp)
 rows=[]
 for r in manifest['rows']:
  d=data/r['id'];npz=cache/(r['id']+'.npz');cm=json.loads(npz.with_suffix('.json').read_text());assert cm['sha256']==sha(npz) and cm['skin_sha256']==r['skin_sha256']
  for f,k in [('skin.png','skin_sha256'),('meat.png','mask_sha256'),('annotation.json','metadata_sha256'),('valid.png','valid_sha256'),('visible_skin.png','visible_skin_sha256')]:assert sha(d/f)==r[k]
  c=dict(np.load(npz));valid=c['valid'].astype(bool);assert np.array_equal(valid,np.array(Image.open(d/'valid.png')).flatten()>0)
  assert np.all(c['face_coverage'][valid]==1)
  y=(np.array(Image.open(d/'meat.png'))[...,0].flatten()>0).astype(np.float32);sem=np.zeros(4096,np.int64)
  if r.get('semantic_sha256'):
   assert sha(d/'semantic.png')==r['semantic_sha256'] and sha(d/'semantic.json')==r['semantic_metadata_sha256']
   sm=json.loads((d/'semantic.json').read_text());sem=np.array(validate_labels(sm['labels'],valid,np.flatnonzero(y).tolist()),np.int64)
   assert (d/'semantic.png').read_bytes()==encode_labels(sem.tolist(),valid)
  rows.append(dict(row=r,cache=c,valid=torch.from_numpy(valid),y=torch.from_numpy(y),semantic=torch.from_numpy(sem),x=matrix(c)))
  case=out/r['id'];case.mkdir();shutil.copy2(d/'skin.png',case/'skin.png');shutil.copy2(d/'meat.png',case/'human_meat.png')
 summary={};audits=[]
 if baseline:
  assert sha(baseline/'manifest.json')==sha(m),'Baseline must use exactly same frozen dataset'
  summary['baseline']=json.loads((baseline/'comparison.json').read_text())['results']['dino_local']
  for r in rows:shutil.copy2(baseline/r['row']['id']/'dino_local_probability.npy',out/r['row']['id']/'baseline_probability.npy')
 groups=sorted(set(r['row']['group'] for r in rows));started=time.time()
 for variant in CONFIG['variants']:
  vd=out/variant;vd.mkdir();per=[]
  for fold,g in enumerate(groups):
   tr=[r for r in rows if r['row']['group']!=g];held=[r for r in rows if r['row']['group']==g];assert not {r['row']['id'] for r in tr}&{r['row']['id'] for r in held}
   write(out/'status.json',dict(stage='cross_validation',variant=variant,fold=fold+1,total=len(groups),started=started,elapsed=time.time()-started))
   net=fit(tr,variant,vd/f'fold-{fold}.jsonl');save(vd/f'fold-{fold}.pt',net,variant,tr,m,fp)
   for r in held:
    prob=predict(net,r,variant);case=out/r['row']['id'];np.save(case/(variant+'_probability.npy'),prob.reshape(64,64))
    (case/(variant+'_meat.png')).write_bytes(annotation_mask(np.flatnonzero((prob>=.5)&r['valid'].numpy()).tolist(),r['valid'].numpy().astype(np.uint8).tobytes()))
    met=metric(r['y'].numpy(),prob,r['valid'].numpy());met.update(id=r['row']['id'],fold=fold,held_out_group=g,train_ids=[v['row']['id'] for v in tr]);per.append(met)
   print('FOLD',variant,fold+1,'iou',round(per[-1]['iou'],4),'elapsed',round(time.time()-started),flush=True)
  summary[variant]=dict(per_case=per,macro={k:float(np.mean([p[k] for p in per])) for k in ['precision','recall','iou','dice','non_meat_iou']},pooled=counts_metric(*(sum(p[k] for p in per) for k in ['tp','fp','fn','tn'])))
  write(out/'comparison.json',dict(cases=len(rows),groups=len(groups),fine_labeled_cells=sum(int((r['semantic']>0).sum()) for r in rows),results=summary,evaluation='Historical DDJ whole-case OOF; not new blind test'))
  write(out/'status.json',dict(stage='final_fit',variant=variant));net=fit(rows,variant,vd/'final_train.jsonl');save(vd/'model.pt',net,variant,rows,m,fp)
  ck=torch.load(vd/'model.pt',weights_only=True,map_location='cpu');reload=(MeatHead(ck['input_dim']) if variant=='all_faces' else SurfaceHead(ck['input_dim'])).cuda();reload.load_state_dict(ck['state_dict']);reload.eval()
  diff=max(float(np.abs(predict(net,r,variant)-predict(reload,r,variant)).max()) for r in rows);assert diff==0
  audits.append(dict(variant=variant,save_reload_max_abs=diff,checkpoint_sha256=sha(vd/'model.pt'),train_ids=ck['train_ids']))
 write(out/'inference_audit.json',dict(models=audits,labels_used_as_input=False,face_coverage_complete=True,threshold=.5,fine_labels_present=any((r['semantic']>0).any() for r in rows),fine_output_validated=False,production_deployed=False))
 write(out/'status.json',dict(stage='complete',cases=len(rows),groups=len(groups),elapsed=time.time()-started))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',type=Path);a=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('high');train(a.data,a.cache,a.output,a.baseline)
