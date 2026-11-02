"""Nested grouped rare-class training; thresholds never see outer-fold labels."""
import argparse,json,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
from meat_features import ROOT,sha,write
from data_face_v9 import load_rows
from train_face_v9 import fit as fit_anchor,save as save_anchor,predict as predict_anchor,evaluate,aggregate,store_case
from face_decoder_v9 import FaceDecoder,MEAT_IDS
from rare_face_v11 import initialize,pack,loss,scores,decode,calibrate,RARE_IDS

CONFIG=dict(seed=42,rare_epochs=160,rare_lr=.0008,weight_decay=.05,anchor_epochs=200,
 hard_negatives_per_case=16,negative_weight=1.5,random_negative_weight=.25,presence_weight=.5,
 positive_balance='equal positive skin means per class',dino_frozen=True,
 frozen_anchor_backbone=True,rare_initializer='balanced_v9; ordinary fallback retained',presence_score='cell_probability*(0.5+0.5*image_presence)',threshold_search='all positive inner OOF score breakpoints',outer_folds=5,inner_folds=3,
 calibration='inner OOF F1 with precision>=0.35 when feasible; otherwise F0.5; no outer labels',
 scope='valid head cells only; original ordinary decoder fallback elsewhere')


def fit_balanced(rows,path,status,fold):
 from face_decoder_v9 import pack as bp
 from balanced_loss_v11 import loss as bl
 torch.manual_seed(42);p=bp(rows,'cuda');net=FaceDecoder(p['x'].shape[1]).cuda()
 opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,200,eta_min=.0001)
 with path.open('w') as stream:
  for epoch in range(1,201):
   net.train();opt.zero_grad(set_to_none=True);total,b,f=bl(net,p,epoch);assert torch.isfinite(total);total.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();sched.step()
   if epoch==1 or epoch%20==0:
    v=dict(epoch=epoch,loss=float(total.detach()),binary=float(b.detach()),fine=float(f.detach()));stream.write(json.dumps(v)+'\n');stream.flush();write(status,dict(stage='balanced_inner_anchor',fold=fold,**v))
 return net.eval()


def fit_rare(rows,anchor,path,status,phase):
 torch.manual_seed(42);net=initialize(anchor).cuda();p=pack(rows,'cuda')
 opt=torch.optim.AdamW([v for v in net.parameters() if v.requires_grad],lr=CONFIG['rare_lr'],weight_decay=.05)
 schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,CONFIG['rare_epochs'],eta_min=.00008);start=time.time()
 with path.open('w') as stream:
  for e in range(1,CONFIG['rare_epochs']+1):
   net.train();net.input.eval();net.messages.eval();net.norms.eval();opt.zero_grad(set_to_none=True);total,detail=loss(net,p);assert torch.isfinite(total)
   total.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();schedule.step()
   if e==1 or e%20==0:
    value=dict(epoch=e,loss=float(total.detach()),elapsed=time.time()-start,**{k:float(v) for k,v in detail.items()});stream.write(json.dumps(value)+'\n');stream.flush();write(status,dict(stage='rare_training',phase=phase,**value))
 return net.eval()


def save_rare(path,net,anchor,train_ids,fp,thresholds=None):
 torch.save(dict(schema='mc-meat-rare-v11',model_module='rare_face_v11',input_dim=anchor['input_dim'],state_dict={k:v.cpu() for k,v in net.state_dict().items()},anchor=anchor,train_ids=train_ids,config=CONFIG,feature_provenance=fp,thresholds=thresholds or [.5]*3),path)


def inner_splits(rows):
 # Labels distribute scarce positive groups across folds; all examples of a
 # duplicate group stay together. No outer labels enter the split or fit.
 groups=sorted({r['row']['group'] for r in rows});rng=np.random.default_rng(410);rng.shuffle(groups)
 pres={g:np.array([any(((r['semantic']==c)&r['valid']).any() for r in rows if r['row']['group']==g) for c in RARE_IDS],int) for g in groups}
 total=np.stack(list(pres.values())).sum(0).clip(1);groups.sort(key=lambda g:-float((pres[g]/total).sum()))
 counts=np.zeros((3,3));sizes=np.zeros(3);assign={}
 for g in groups:
  f=int((((counts+pres[g])**2/total).sum(1)-(counts**2/total).sum(1)+.2*sizes).argmin());assign[g]=f;counts[f]+=pres[g];sizes[f]+=1
 return [([r for r in rows if assign[r['row']['group']]!=f],[r for r in rows if assign[r['row']['group']]==f]) for f in range(3)]


def predict_pair(net,anchor_net,row):
 base,q,detail=predict_anchor(anchor_net,row);score,presence=scores(net,row)
 return dict(id=row['row']['id'],score=score,presence=presence,valid=row['valid'].numpy(),truth=row['semantic'].numpy(),base=np.array(detail['labels'],np.uint8),probability=base,q=q)


def train(a):
 assert not a.output.exists();a.output.mkdir(parents=True);out=a.output;start=time.time();(out/'source').mkdir()
 for p in ROOT.glob('*.py'):shutil.copy2(p,out/'source'/p.name)
 shutil.copy2(ROOT/'surface_topology_v3.json',out/'source/surface_topology_v3.json')
 write(out/'source_hashes.json',{p.name:sha(p) for p in (out/'source').iterdir()})
 rows,fp=load_rows(a.data,a.cache);byid={r['row']['id']:r for r in rows};folds=json.loads((a.parent/'folds.json').read_text())
 assert sha(a.data/'manifest.json')==sha(a.parent/'manifest.json')
 for r in rows:assert not ((r['semantic']>0)&torch.isin(r['semantic'],torch.tensor(RARE_IDS))&torch.from_numpy(r['cache']['region']%6!=0)).any()
 shutil.copy2(a.data/'manifest.json',out/'manifest.json');write(out/'configuration.json',CONFIG);write(out/'folds.json',folds);write(out/'data_source.json',dict(data=str(a.data),cache=str(a.cache),parent=str(a.parent),balanced=str(a.balanced),control=str(a.control)))
 active=ROOT/'service_runtime/production/active.json';shutil.copy2(active,out/'production_before.json')
 summary={};per_raw=[];per_cal=[];calibration=[];all_oof=[];audits=[]
 for name,parent in [('ordinary_v9',a.parent),('balanced_v9',a.balanced)]:
  c=json.loads((parent/'comparison.json').read_text());summary[name]=c['results']['face_v9']
 for r in rows:
  d=out/r['row']['id'];d.mkdir();shutil.copy2(a.parent/r['row']['id']/'truth.npz',d/'truth.npz');shutil.copy2(a.data/r['row']['id']/'skin.png',d/'skin.png')
  for name,parent in [('ordinary_v9',a.parent),('balanced_v9',a.balanced)]:shutil.copy2(parent/r['row']['id']/'face_v9.npz',d/(name+'.npz'))
 def update():write(out/'comparison.json',dict(cases=len(rows),results=summary,folds=folds,evaluation='Historical same outer grouped folds. New thresholds use nested inner OOF. Ordinary/balanced models are retained historical references; not new blind evaluation.'))
 update()
 for f in folds:
  tr=[byid[i] for i in f['train_ids']];held=[byid[i] for i in f['held_ids']];fd=out/f"fold-{f['fold']}";fd.mkdir();inner=[];inner_audit=[]
  for j,(itr,ival) in enumerate(inner_splits(tr)):
   assert not {r['row']['group'] for r in itr}&{r['row']['group'] for r in ival+held}
   phase=f"outer {f['fold']+1}/5 inner {j+1}/3"
   write(out/'status.json',dict(stage='inner_anchor',phase=phase))
   ap=a.control/f"fold-{f['fold']}"/f'inner-{j}-anchor.pt'
   ck=torch.load(ap,map_location='cpu',weights_only=True);assert set(ck['train_ids'])=={r['row']['id'] for r in itr}
   net=FaceDecoder(ck['input_dim']).cuda();net.load_state_dict(ck['state_dict']);net.eval()
   balanced_net=fit_balanced(itr,fd/f'inner-{j}-balanced.jsonl',out/'anchor_status.json',j+1)
   bp=fd/f'inner-{j}-balanced.pt';save_anchor(bp,balanced_net,'face_v9',itr,a.data,fp);rare_ck=torch.load(bp,map_location='cpu',weights_only=True);rare_ck['config']={**rare_ck['config'],'fine_class_balance':'equal mean per present training class','loss_module':'balanced_loss_v11'};torch.save(rare_ck,bp)
   del balanced_net;torch.cuda.empty_cache()
   rare=fit_rare(itr,rare_ck,fd/f'inner-{j}-rare.jsonl',out/'status.json',phase);save_rare(fd/f'inner-{j}-rare.pt',rare,ck,[r['row']['id'] for r in itr],fp)

   records=[]
   for r in ival:
    item=predict_pair(rare,net,r);inner.append(item);records.append(item['id'])
   inner_audit.append(dict(inner_fold=j,train_ids=ck['train_ids'],held_ids=records,outer_held_ids=f['held_ids']))
   del net,rare;torch.cuda.empty_cache();print('INNER_DONE',phase,round(time.time()-start),flush=True)
  thresholds,ca=calibrate(inner);calibration.append(dict(fold=f['fold'],thresholds=thresholds,calibration=ca,inner_folds=inner_audit))
  np.savez_compressed(fd/'inner_oof.npz',ids=np.array([x['id'] for x in inner]),score=np.stack([x['score'] for x in inner]),valid=np.stack([x['valid'] for x in inner]),truth=np.stack([x['truth'] for x in inner]))
  ck=torch.load(a.parent/'face_v9'/f"fold-{f['fold']}.pt",map_location='cpu',weights_only=True);assert set(ck['train_ids'])==set(f['train_ids'])
  net=FaceDecoder(ck['input_dim']).cuda();net.load_state_dict(ck['state_dict']);net.eval()
  rare_ck=torch.load(a.balanced/'face_v9'/f"fold-{f['fold']}.pt",map_location='cpu',weights_only=True);assert set(rare_ck['train_ids'])==set(f['train_ids'])
  rare=fit_rare(tr,rare_ck,fd/'rare.jsonl',out/'status.json',f"outer {f['fold']+1}/5 fit");save_rare(fd/'model.pt',rare,ck,f['train_ids'],fp,thresholds)
  for r in held:
   item=predict_pair(rare,net,r);all_oof.append(item);d=out/r['row']['id'];np.savez_compressed(d/'rare_scores.npz',score=item['score'],presence=item['presence'],base=item['base'])
   for name,th,per in [('rare_raw',[.5]*3,per_raw),('rare_calibrated',thresholds,per_cal)]:
    detail=decode(item['base'],item['score'],item['valid'],th);per.append(store_case(out,r,name,item['probability'],item['score'],detail,f['fold'],f['train_ids']))
  summary['rare_raw']=aggregate(per_raw);summary['rare_calibrated']=aggregate(per_cal);update();write(out/'calibration.json',dict(outer_folds=calibration))
  del net,rare;torch.cuda.empty_cache();print('OUTER_DONE',f['fold']+1,round(time.time()-start),flush=True)
 thresholds,ca=calibrate(all_oof)
 ck=torch.load(a.parent/'face_v9/model.pt',map_location='cpu',weights_only=True);assert set(ck['train_ids'])==set(byid)
 rare_ck=torch.load(a.balanced/'face_v9/model.pt',map_location='cpu',weights_only=True);assert set(rare_ck['train_ids'])==set(byid)
 rare=fit_rare(rows,rare_ck,out/'final_train.jsonl',out/'status.json','final all-data');save_rare(out/'model.pt',rare,ck,list(byid),fp,thresholds)
 saved=torch.load(out/'model.pt',map_location='cpu',weights_only=True);reload=initialize(saved['anchor']).cuda();reload.load_state_dict(saved['state_dict']);reload.eval()
 for r in rows[:3]:
  x=scores(rare,r);y=scores(reload,r);assert np.array_equal(x[0],y[0])
 del rare,reload;torch.cuda.empty_cache()
 write(out/'calibration.json',dict(outer_folds=calibration,final_thresholds=thresholds,final_calibration=ca,final_threshold_source='whole-data OOF; these thresholds do not replace outer-fold thresholds in evaluation'))
 assert sha(active)==sha(out/'production_before.json')
 write(out/'inference_audit.json',dict(dino_frozen=True,features_use_labels=False,outer_group_disjoint=True,inner_anchor_excludes_inner_and_outer_validation=True,save_reload_max_abs=0,production_deployed=False,production_unchanged=True))
 write(out/'training_complete.json',dict(cases=len(rows),folds=5,inner_folds=3,rare_epochs=CONFIG['rare_epochs'],elapsed=time.time()-start,checkpoint_sha256=sha(out/'model.pt')))
 write(out/'status.json',dict(stage='training_complete',elapsed=time.time()-start))

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['data','cache','output','parent','balanced','control']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 from model_service import lock
 try:
  with lock(ROOT/'service_runtime/gpu.lock'):train(a)
 except Exception:
  if a.output.exists():write(a.output/'failure.json',dict(error=traceback.format_exc()))
  raise
