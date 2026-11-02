"""Matched human-only and weakly augmented training on unchanged held-out skins."""
import argparse,json,shutil,time,traceback,subprocess
from pathlib import Path
import numpy as np
import torch
from meat_features import ROOT,sha,write
from data_face_v9 import load_rows
from face_decoder_v9 import FaceDecoder,MEAT_IDS
from weighted_face_v12 import pack,loss,fold_training
from train_face_v9 import predict,store_case,aggregate

CONFIG=dict(model_module='face_decoder_v9',training_version='weak_expansion_v12',epochs=200,lr=.001,eta_min=.0001,
 weight_decay=.01,hidden=48,seed=42,folds=5,fine_threshold=.5,fine_loss_weight=.6,fine_warmup_epochs=20,
 dino_frozen=True,initializer='random per fold; no full-data checkpoint warm start',
 auxiliary_evaluation=False,auxiliary_binary_weight=.5,auxiliary_fine_weights='saved per-cell confidence; unknown ignored',
 variants=['ordinary_control','balanced_control','ordinary_augmented','balanced_augmented'],
 protocol='Historical held IDs unchanged; known same-source training counterparts purged in matched control and augmented arms; fixed epochs and confidence; no outer-fold tuning')


def fit(rows,path,status,variant,fold):
 torch.manual_seed(CONFIG['seed']);p=pack(rows,'cuda');net=FaceDecoder(p['x'].shape[1]).cuda()
 assert not p['x'].requires_grad and not any('dino' in n for n in net.state_dict())
 opt=torch.optim.AdamW(net.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay']);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,CONFIG['epochs'],eta_min=CONFIG['eta_min']);start=time.time()
 with path.open('w') as stream:
  for epoch in range(1,CONFIG['epochs']+1):
   net.train();opt.zero_grad(set_to_none=True);total,b,f=loss(net,p,epoch,balanced=variant.startswith('balanced'));assert torch.isfinite(total)
   total.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();schedule.step()
   if epoch==1 or epoch%20==0:
    val=dict(epoch=epoch,loss=float(total.detach()),binary=float(b.detach()),fine=float(f.detach()),elapsed=time.time()-start)
    stream.write(json.dumps(val)+'\n');stream.flush();write(status,dict(stage='training',variant=variant,fold=fold,training_cases=len(rows),**val))
 return net.eval()


def save(path,net,rows,fp,data,variant):
 torch.save(dict(schema='mc-meat-face-v9',model_module='face_decoder_v9',variant='face_v9',training_variant=variant,input_dim=rows[0]['x'].shape[1],state_dict={k:v.cpu() for k,v in net.state_dict().items()},config={**CONFIG,'balanced':variant.startswith('balanced')},train_ids=[r['row']['id'] for r in rows],manifest_sha256=sha(data/'manifest.json'),feature_provenance=fp),path)


def train(a):
 assert not a.output.exists(),'Use a fresh experiment directory';out=a.output;out.mkdir(parents=True);start=time.time()
 (out/'source').mkdir()
 for p in ROOT.glob('*.py'):shutil.copy2(p,out/'source'/p.name)
 shutil.copy2(ROOT/'surface_topology_v3.json',out/'source/surface_topology_v3.json')
 write(out/'source_hashes.json',{p.name:sha(p) for p in (out/'source').iterdir()})
 rows,fp=load_rows(a.data,a.cache);old=json.loads((a.parent/'manifest.json').read_text());humanids={r['id'] for r in old['rows']};human=[r for r in rows if r['row']['id'] in humanids];aux=[r for r in rows if r['row']['id'] not in humanids and r['row'].get('training_eligible') is True]
 assert len(human)==169 and len(aux)==42
 oldrows={r['id']:r for r in old['rows']}
 for r in human:
  rid=r['row']['id']
  for k in ['skin_sha256','mask_sha256','metadata_sha256','semantic_sha256','semantic_metadata_sha256','group']:assert oldrows[rid][k]==r['row'][k],(rid,k)
  assert r['reviewed'][r['valid']].all() and r['row']['evaluation_eligible'] is True
 for r in aux:
  assert r['row']['evaluation_eligible'] is False and not r['reviewed'].any() and r['row']['binary_training_weight']==.5
  assert float(r['fine_weights'].max())<=.8
 coverage=json.loads((a.data/'coverage_audit.json').read_text());links=[(r['a'],r['b']) for r in coverage['identical_source_pairs']]
 folds=json.loads((a.parent/'folds.json').read_text());byid={r['row']['id']:r for r in human};assert sorted(i for f in folds for i in f['held_ids'])==sorted(humanids)
 active=ROOT/'service_runtime/production/active.json';assert sha(active)==sha(a.parent/'production_before.json');shutil.copy2(active,out/'production_before.json')
 for name,src in [('manifest.json',a.data/'manifest.json'),('evaluation_manifest.json',a.parent/'manifest.json'),('folds.json',a.parent/'folds.json')]:shutil.copy2(src,out/name)
 write(out/'configuration.json',CONFIG);write(out/'feature_provenance.json',fp)
 write(out/'data_source.json',dict(data=str(a.data),cache=str(a.cache),parent=str(a.parent),balanced=str(a.balanced),guarded=str(a.guarded),commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()))
 references=[('current_reference',a.parent,'current_reference'),('ordinary_historical',a.parent,'face_v9'),('balanced_historical',a.balanced,'face_v9'),('guarded_v11',a.guarded,'rare_calibrated')];summary={};refmeta={}
 for name,parent,variant in references:
  value=json.loads((parent/'comparison.json').read_text())['results'][variant];summary[name]=value
  refmeta[name]=dict(run=str(parent),variant=variant,comparison_sha256=sha(parent/'comparison.json'),supported_class_ids=[1,2,3,5,6] if name=='current_reference' else [1,2,3,5,6,9,10],matched_training_protocol=False)
 for r in human:
  d=out/r['row']['id'];d.mkdir();shutil.copy2(a.data/r['row']['id']/'skin.png',d/'skin.png');shutil.copy2(a.parent/r['row']['id']/'truth.npz',d/'truth.npz')
  for name,parent,variant in references:shutil.copy2(parent/r['row']['id']/(variant+'.npz'),d/(name+'.npz'))
 fold_audits=[];reloads=[]
 def update():write(out/'comparison.json',dict(cases=len(human),auxiliary_training_cases=len(aux),results=summary,references=refmeta,folds=folds,evaluation='169 unchanged historical human cases; not new blind data. Assistant labels are training-only. New matched controls purge same-source overlaps identically to augmentation. Historical references use their original training protocols. Unsupported online classes must be reported as N/A.'))
 update()
 for variant in CONFIG['variants']:
  augmented=variant.endswith('augmented');vd=out/variant;vd.mkdir();per=[]
  for f in folds:
   tr,purged=fold_training(human,aux,f,links,augmented);held=[byid[i] for i in f['held_ids']]
   audit=dict(variant=variant,fold=f['fold'],held_ids=f['held_ids'],train_ids=[r['row']['id'] for r in tr],purged_same_source_ids=purged,auxiliary_train_ids=[r['row']['id'] for r in tr if r['row']['id'] not in humanids]);fold_audits.append(audit);write(out/'fold_audit.json',fold_audits)
   net=fit(tr,vd/f"fold-{f['fold']}.jsonl",out/'status.json',variant,f['fold']+1);save(vd/f"fold-{f['fold']}.pt",net,tr,fp,a.data,variant)
   for r in held:
    probability,q,detail=predict(net,r);per.append(store_case(out,r,variant,probability,q,detail,f['fold'],audit['train_ids']))
   summary[variant]=aggregate(per);update();del net;torch.cuda.empty_cache();print('FOLD_DONE',variant,f['fold']+1,round(time.time()-start),flush=True)
  final_rows=human+aux if augmented else human
  net=fit(final_rows,vd/'final_train.jsonl',out/'status.json',variant,None);save(vd/'model.pt',net,final_rows,fp,a.data,variant)
  ck=torch.load(vd/'model.pt',map_location='cpu',weights_only=True);reload=FaceDecoder(ck['input_dim']).cuda();reload.load_state_dict(ck['state_dict']);reload.eval()
  for r in [human[0],human[-1],aux[0]]:
   left=predict(net,r);right=predict(reload,r);assert np.array_equal(left[1],right[1]) and left[2]['labels']==right[2]['labels']
  reloads.append(dict(variant=variant,checkpoint_sha256=sha(vd/'model.pt'),max_score_difference=0,label_disagreements=0))
  del net,reload;torch.cuda.empty_cache()
 assert sha(active)==sha(out/'production_before.json')
 write(out/'inference_audit.json',dict(dino_frozen=True,features_use_labels=False,assistant_labels_used_for_evaluation=False,weights_verified=True,production_deployed=False,production_unchanged=True,save_reload=reloads))
 write(out/'training_complete.json',dict(human_cases=len(human),auxiliary_training_cases=len(aux),folds=5,epochs=200,elapsed=time.time()-start,variants=CONFIG['variants']))
 write(out/'status.json',dict(stage='training_complete',elapsed=time.time()-start))

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['data','cache','output','parent','balanced','guarded']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 from model_service import lock
 try:
  with lock(ROOT/'service_runtime/gpu.lock'):train(a)
 except Exception:
  if a.output.exists():write(a.output/'failure.json',dict(error=traceback.format_exc()))
  raise
