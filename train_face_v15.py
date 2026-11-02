"""Corrected-label controls and fold-matched bounded refinement; production unchanged."""
from pathlib import Path
import argparse,json,shutil,time,traceback
import torch,numpy as np
from meat_features import ROOT,sha,write
from data_face_v13 import load_rows
from face_decoder_v15 import build,pack
from weighted_face_v13 import loss as control_loss,fold_training
from loss_face_v15 import loss as refinement_loss
from predict_face_v15 import predict
from train_face_v9 import store_case,aggregate,evaluate
from model_service import lock
CONFIG=dict(variants=['original_labels_control_v15','corrected_labels_v15','bounded_refine_v15'],control_epochs=200,refinement_epochs=100,lr=.001,refinement_lr=.0005,eta_min=.0001,seed=42,fine_threshold=.5,dino_frozen=True,float32_matmul_precision='highest',refinement_base_frozen=True,refinement_logit_bound=1.,auxiliary_fine_source_weight=.5,rare_hard_loss_weight=.025,distillation_weight=.05,external_validation_tuning=False)

def fit(rows,path,status,variant,fold,init=None):
 torch.manual_seed(42);p=pack(rows,'cuda');net=build(p['x'].shape[1],variant).cuda();refine=variant=='bounded_refine_v15'
 if refine:
  ck=torch.load(init,map_location='cpu',weights_only=True);assert ck['train_ids']==[r['row']['id'] for r in rows];net.base.load_state_dict(ck['state_dict']);before={k:v.cpu().clone() for k,v in net.base.state_dict().items()}
 epochs=100 if refine else 200;opt=torch.optim.AdamW([v for v in net.parameters() if v.requires_grad],lr=.0005 if refine else .001,weight_decay=.01);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=.0001);start=time.time()
 with path.open('w') as stream:
  for epoch in range(1,epochs+1):
   net.train();opt.zero_grad(set_to_none=True)
   if refine:total,terms=refinement_loss(net,p,epoch)
   else:total,b,f=control_loss(net,p,epoch,balanced=True);terms=dict(binary=b,fine=f)
   assert torch.isfinite(total);total.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();schedule.step()
   if epoch==1 or epoch%20==0:
    z=dict(stage='training',variant=variant,fold=fold,epoch=epoch,epochs=epochs,loss=float(total.detach()),elapsed=time.time()-start,**{k:float(v.detach()) for k,v in terms.items()});write(status,z);stream.write(json.dumps(z)+'\n');stream.flush()
 net.eval()
 if refine:
  assert all(torch.equal(before[k],v.cpu()) for k,v in net.base.state_dict().items())
  with torch.no_grad():diff=(net(p)-net.base(p)).abs();assert float(diff.max())<=1.00001;assert not diff[~p['head']].any()
 return net

def save(path,net,rows,fp,data,variant):
 torch.save(dict(schema='mc-meat-face-v15',model_module='face_decoder_v15',variant=variant,input_dim=rows[0]['x'].shape[1],state_dict={k:v.cpu() for k,v in net.state_dict().items()},config=CONFIG,train_ids=[r['row']['id'] for r in rows],manifest_sha256=sha(data/'manifest.json'),feature_provenance=fp),path)

def train(a):
 assert not a.output.exists();out=a.output;out.mkdir();(out/'source').mkdir();start=time.time()
 for p in ROOT.glob('*.py'):shutil.copy2(p,out/'source'/p.name)
 shutil.copy2(ROOT/'surface_topology_v3.json',out/'source/surface_topology_v3.json');write(out/'source_hashes.json',{p.name:sha(p) for p in (out/'source').iterdir()})
 rows,fp=load_rows(a.data,a.cache);oldrows,_=load_rows(a.old_data,a.cache);om={r['row']['id']:r for r in oldrows};original_eval=json.loads((a.parent/'evaluation_manifest.json').read_text());humanids={r['id'] for r in original_eval['rows']};human=[r for r in rows if r['row']['id'] in humanids];aux=[r for r in rows if r['row']['id'] not in humanids and r['row'].get('training_eligible')];assert len(human)==169 and len(aux)==60;byid={r['row']['id']:r for r in human}
 changed=[r['row']['id'] for r in human if r['row']['semantic_metadata_sha256']!=om[r['row']['id']]['row']['semantic_metadata_sha256']];assert set(changed)=={'5D74V6D8LDF3ME42','4ZVTCXVRWTB9ZW4A'}
 for r in rows:
  old=om[r['row']['id']];assert torch.equal(r['y'],old['y']) and torch.equal(r['valid'],old['valid']) and torch.equal(r['x'],old['x'])
 folds=json.loads((a.parent/'folds.json').read_text());links=[(x['a'],x['b']) for x in json.loads((a.data/'coverage_audit.json').read_text())['identical_source_pairs']]
 write(out/'evaluation_manifest.json',dict(original_eval,rows=[r['row'] for r in human],annotation_revision='Two user-confirmed beard corrections; original OOF predictions re-scored against the same corrected targets'))
 for f,src in [('manifest.json',a.data/'manifest.json'),('folds.json',a.parent/'folds.json'),('production_before.json',ROOT/'service_runtime/production/active.json')]:shutil.copy2(src,out/f)
 write(out/'configuration.json',CONFIG);write(out/'feature_provenance.json',fp);write(out/'data_source.json',dict(data=str(a.data),old_data=str(a.old_data),cache=str(a.cache),parent=str(a.parent),corrections=changed))
 previous=json.loads((a.parent/'comparison.json').read_text());summary={};refmeta=previous.get('references',{}).copy()
 for r in human:
  rid=r['row']['id'];d=out/rid;d.mkdir();shutil.copy2(a.data/rid/'skin.png',d/'skin.png');np.savez_compressed(d/'truth.npz',valid=r['valid'].numpy(),labels=r['semantic'].numpy())
 for v,oldscore in previous['results'].items():
  per=[];meta={x['id']:x for x in oldscore['per_case']}
  for r in human:
   rid=r['row']['id'];src=a.parent/rid/(v+'.npz');saved=np.load(src);shutil.copy2(src,out/rid/(v+'.npz'));details=dict(labels=saved['labels'].tolist(),uncertain_count=len(saved.get('uncertain',[])));m=evaluate(r,saved['probability'],details);m.update(fold=meta[rid]['fold'],train_ids=meta[rid]['train_ids']);per.append(m)
  summary[v]=aggregate(per)
  if v not in refmeta:refmeta[v]=dict(supported_class_ids=[1,2,3,5,6,9,10])
 def update():write(out/'comparison.json',dict(cases=169,auxiliary_training_cases=60,results=summary,references=refmeta,folds=folds,evaluation='Corrected historical labels; all versions rescored on identical corrected targets. Original label metrics retained in parent releases. Not a new blind test.'))
 update();fa=[];reloads=[]
 for variant in CONFIG['variants']:
  vd=out/variant;vd.mkdir();per=[]
  for f in [*folds,dict(fold='full',train_ids=sorted(humanids),held_ids=[])]:
   tr,purged=fold_training(human,aux,f,links,True)
   if variant=='original_labels_control_v15':tr=[om[r['row']['id']] for r in tr]
   name='model.pt' if f['fold']=='full' else f"fold-{f['fold']}.pt";init=out/'corrected_labels_v15'/name if variant=='bounded_refine_v15' else None
   net=fit(tr,vd/(name+'.jsonl'),out/'status.json',variant,f['fold'],init);save(vd/name,net,tr,fp,a.old_data if variant=='original_labels_control_v15' else a.data,variant)
   fa.append(dict(variant=variant,fold=f['fold'],train_ids=[r['row']['id'] for r in tr],held_ids=f['held_ids'],purged=purged,initialized_from=str(init) if init else None));write(out/'fold_audit.json',fa)
   if f['fold']!='full':
    for rid in f['held_ids']:
     r=byid[rid];prob,q,details=predict(net,r);per.append(store_case(out,r,variant,prob,q,details,f['fold'],[s['row']['id'] for s in tr]))
    summary[variant]=aggregate(per);update()
   else:
    ck=torch.load(vd/name,map_location='cpu',weights_only=True);again=build(ck['input_dim'],variant).cuda();again.load_state_dict(ck['state_dict']);again.eval()
    for r in [human[0],human[-1]]:assert np.array_equal(predict(net,r)[1],predict(again,r)[1])
    reloads.append(dict(variant=variant,checkpoint_sha256=sha(vd/name),label_disagreements=0));del again
   del net;torch.cuda.empty_cache();print('FIT_DONE',variant,f['fold'],round(time.time()-start),flush=True)
 assert sha(ROOT/'service_runtime/production/active.json')==sha(out/'production_before.json');write(out/'inference_audit.json',dict(save_reload=reloads,base_frozen_verified=True,body_logits_unchanged_for_refinement=True,head_logit_delta_bound=1.,production_unchanged=True));write(out/'training_complete.json',dict(variants=CONFIG['variants'],human_cases=169,auxiliary_cases=60,beard_support=77,epochs=CONFIG,elapsed=time.time()-start));write(out/'status.json',dict(stage='training_complete'))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['data','old-data','cache','output','parent']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
 try:
  with lock(ROOT/'service_runtime/gpu.lock'):train(a)
 except Exception:
  if a.output.exists():write(a.output/'failure.json',dict(error=traceback.format_exc()))
  raise
