"""Same-split comparison and candidate release for the five-category decoder."""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from meat_features import ROOT,sha,write
from train_face_v6 import load_rows,summarize
from train_supervised import metric
from surface_model_v4 import SurfaceHead,pack as old_pack
from train_surface_v4 import fit as old_fit
from semantic_preselection import decode_categories
from semantic_schema import CLASS_IDS
from face_decoder_v8 import FaceDecoder,pack,loss,decode,binary_logits,INTERNAL_IDS

CONFIG=dict(model_module='face_decoder_v8',semantic_class_ids=CLASS_IDS,internal_class_ids=INTERNAL_IDS,
 epochs=200,lr=.001,eta_min=.0001,weight_decay=.01,hidden=48,seed=42,folds=5,
 fine_loss_weight=.6,fine_warmup_epochs=20,fine_threshold=.5,binary_threshold=.5,
 dino_frozen=True,unknown_fine='binary marginal only; not other targets',
 other_fine='explicit whole-skin reviewed zeros only',validation='fixed five-fold duplicate-group stratification; all variants share exact held-out groups',
 variants=['surface_control','face_v8'],initializer='random per fold; never initialized from full-data weights')

def load(data,cache):
 rows,fp=load_rows(data,cache)
 for r in rows:
  sm=json.loads((data/r['row']['id']/'semantic.json').read_text()) if r['row'].get('semantic_sha256') else {}
  complete=sm.get('fine_review_complete') is True
  r['reviewed']=r['valid'].clone() if complete else torch.zeros(4096,dtype=torch.bool)
 return rows,fp

def splits(rows):
 groups=sorted({r['row']['group'] for r in rows});rng=np.random.default_rng(42);rng.shuffle(groups)
 presence={g:np.array([any(((r['semantic']==c)&r['valid']).any() for r in rows if r['row']['group']==g) for c in CLASS_IDS],int) for g in groups}
 total=np.stack(list(presence.values())).sum(0).clip(1)
 groups.sort(key=lambda g:-float((presence[g]/total).sum()))
 n=min(5,len(groups));counts=np.zeros((n,5));sizes=np.zeros(n);assigned={}
 for g in groups:
  score=((counts+presence[g])**2/total).sum(1)-((counts**2)/total).sum(1)+.25*sizes
  f=int(score.argmin());assigned[g]=f;counts[f]+=presence[g];sizes[f]+=1
 return [dict(fold=f,held_ids=[r['row']['id'] for r in rows if assigned[r['row']['group']]==f],train_ids=[r['row']['id'] for r in rows if assigned[r['row']['group']]!=f]) for f in range(n)]

def support(training,c):
 s=np.zeros((5,6,8,8),bool)
 for r in training:
  values=r['cache'];local=values['local'];lab=r['semantic'].numpy();w=r['fine_weights'].numpy()
  for i in np.flatnonzero(r['valid'].numpy()&(values['region']%6==0)&(lab>0)&(w>0)):
   k=CLASS_IDS.index(lab[i]);f=int(local[i,42:48].argmax());x,y=np.rint((local[i,49:51]+1)*4-.5).astype(int)
   s[k,f,max(0,y-1):min(8,y+2),max(0,x-1):min(8,x+2)]=True
 result=np.zeros((4096,5),bool);result[c['valid'].astype(bool),0]=True
 for i in np.flatnonzero(c['valid'].astype(bool)&(c['region']%6==0)):
  f=int(c['local'][i,42:48].argmax());x,y=np.rint((c['local'][i,49:51]+1)*4-.5).astype(int);result[i,1:]=s[1:,f,y,x]
 return result

@torch.inference_mode()
def predict(net,r,variant,training=None,device='cuda'):
 v=r['valid'].numpy();head=v&(r['cache']['region']%6==0);prob=np.zeros(4096,np.float32)
 if variant=='face_v8':
  logits=net(pack([r],device));q=np.zeros((4096,6),np.float32);q[v]=logits.softmax(-1).float().cpu().numpy()
  prob[v]=binary_logits(logits).sigmoid().float().cpu().numpy();details=decode(q,v,head,CONFIG['fine_threshold'])
 else:
  p=old_pack([r],device);b,s=net(p['x'],p['neighbors'],p['neighbor_mask'],p['region'])
  q=np.zeros((4096,5),np.float32);q[v]=s.softmax(-1).float().cpu().numpy();prob[v]=b.sigmoid().float().cpu().numpy()
  details=decode_categories(prob,q,v,head,class_support=support(training,r['cache']))
 return prob,q,details

def fit(rows,path,status,fold):
 torch.manual_seed(42);p=pack(rows,'cuda');net=FaceDecoder(p['x'].shape[1]).cuda()
 assert not p['x'].requires_grad
 opt=torch.optim.AdamW(net.parameters(),lr=CONFIG['lr'],weight_decay=.01);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,200,eta_min=CONFIG['eta_min'])
 start=time.time()
 with path.open('w') as f:
  for epoch in range(1,201):
   net.train();opt.zero_grad(set_to_none=True);total,binary,fine=loss(net,p,epoch);assert torch.isfinite(total)
   total.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();schedule.step()
   if epoch==1 or epoch%20==0:
    value=dict(epoch=epoch,loss=float(total.detach()),binary=float(binary.detach()),fine=float(fine.detach()),elapsed=time.time()-start)
    f.write(json.dumps(value)+'\n');f.flush();write(status,dict(stage='training',variant='face_v8',fold=fold,total=5,**value))
 return net.eval()

def save(path,net,variant,rows,data,fp):
 torch.save(dict(schema='mc-meat-face-v8' if variant=='face_v8' else 'mc-meat-surface-v3',model_module='face_decoder_v8' if variant=='face_v8' else 'surface_model_v4',variant=variant,input_dim=rows[0]['x'].shape[1],state_dict={k:v.cpu() for k,v in net.state_dict().items()},config=CONFIG,train_ids=[r['row']['id'] for r in rows],manifest_sha256=sha(data/'manifest.json'),feature_provenance=fp),path)

def evaluate(r,prob,details):
 v=r['valid'].numpy();lab=np.array(details['labels']);selected=np.isin(lab,[1,2,3]);gt=r['semantic'].numpy();known=v&(gt>0);human=known&(r['fine_weights'].numpy()==1)
 result=metric(r['y'].numpy(),selected.astype(float),v);result['raw_binary']=metric(r['y'].numpy(),prob,v)
 result.update(id=r['row']['id'],fine={},human_fine={})
 for target,key in [(known,'fine'),(human,'human_fine')]:
  for c in CLASS_IDS:
   truth=gt==c;pred=lab==c
   result[key][str(c)]={n:int((target&m).sum()) for n,m in [('tp',truth&pred),('fp',~truth&pred),('fn',truth&~pred)]}
 result['unknown_fine_predictions']=int((v&(gt==0)&np.isin(lab,[2,3,5,6])).sum())
 result['uncertain_count']=details.get('uncertain_count',0)
 return result

def aggregate(per):
 value=summarize(per)
 for kind in ['fine','human_fine']:
  result={}
  for c in CLASS_IDS:
   counts={k:sum(r[kind][str(c)][k] for r in per) for k in ['tp','fp','fn']};t,f,n=counts.values()
   result[str(c)]={**counts,'precision':t/max(t+f,1),'recall':t/max(t+n,1),'iou':t/max(t+f+n,1),'f1':2*t/max(2*t+f+n,1),'support':t+n}
  value[kind]=result
 value['fine_macro_f1']=float(np.mean([r['f1'] for r in value['fine'].values() if r['support']]))
 value['raw_binary_macro']={k:float(np.mean([r['raw_binary'][k] for r in per])) for k in ['iou','precision','recall']}
 return value

def store_case(out,r,variant,prob,q,details,fold,train_ids):
 case=out/r['row']['id'];case.mkdir(exist_ok=True)
 np.savez_compressed(case/(variant+'.npz'),probability=prob,category_probability=q,labels=np.array(details['labels'],np.uint8),uncertain=np.array(details.get('uncertain',[]),np.int64))
 result=evaluate(r,prob,details);result.update(fold=fold,train_ids=train_ids);return result

def train(data,cache,out,reference=None):
 assert not out.exists(),'Use a new experiment directory';out.mkdir(parents=True);(out/'source').mkdir()
 rows,fp=load(data,cache);folds=splits(rows);started=time.time()
 for f in ROOT.glob('*.py'):shutil.copy2(f,out/'source'/f.name)
 shutil.copy2(ROOT/'surface_topology_v3.json',out/'source/surface_topology_v3.json')
 shutil.copy2(data/'manifest.json',out/'manifest.json')
 write(out/'configuration.json',CONFIG);write(out/'folds.json',folds);write(out/'feature_provenance.json',fp);write(out/'data_source.json',dict(data=str(data),cache=str(cache)))
 write(out/'source_hashes.json',{p.name:sha(p) for p in (out/'source').glob('*')})
 for r in rows:
  case=out/r['row']['id'];case.mkdir();shutil.copy2(data/r['row']['id']/'skin.png',case/'skin.png')
  np.savez_compressed(case/'truth.npz',valid=r['valid'].numpy(),y=r['y'].numpy(),labels=r['semantic'].numpy(),weights=r['fine_weights'].numpy())
 summary={};audits=[]
 def update():write(out/'comparison.json',dict(cases=len(rows),folds=folds,results=summary,evaluation='Historical grouped five-fold comparison; mixed fine provenance. Fine precision only on known labels, not all unknown cells. Current reference has different training size. No threshold/epoch tuning on held-out results.'))
 if reference:
  oldrun=reference.parent.parent
  source=json.loads((oldrun/'data_source.json').read_text()) if (oldrun/'data_source.json').exists() else {}
  olddata=Path(source['data']) if source else oldrun.parent/'data';oldcache=Path(source['cache']) if source else oldrun.parent/'face_cache';oldrows,_=load(olddata,oldcache)
  om={r['row']['id']:r for r in oldrows};groups=sorted({r['row']['group'] for r in oldrows});per=[]
  is_face=torch.load(reference,map_location='cpu',weights_only=True)['model_module']=='face_decoder_v8'
  oldfolds=json.loads((oldrun/'folds.json').read_text()) if is_face else None
  ancestry=json.loads((oldrun/'parent.json').read_text()) if is_face and (oldrun/'parent.json').exists() else {}
  foldroot=Path(ancestry.get('base_training_path',ancestry.get('path',oldrun)))
  calibration=json.loads((oldrun/'calibration.json').read_text()) if (oldrun/'calibration.json').exists() else None
  support=json.loads((oldrun/'support_audit.json').read_text()) if (oldrun/'support_audit.json').exists() else None
  for r in rows:
   rid=r['row']['id'];oldfold=next((f['fold'] for f in oldfolds if rid in f['held_ids']),None) if is_face else None
   if rid in om:path=foldroot/'face_v8'/f'fold-{oldfold}.pt' if is_face else oldrun/'surface'/f"fold-{groups.index(om[rid]['row']['group'])}.pt"
   else:path=reference
   ck=torch.load(path,map_location='cpu',weights_only=True);held={s['row']['id'] for s in rows if s['row']['group']==r['row']['group']}
   assert not held.intersection(ck['train_ids']),'Reference leakage'
   net=(FaceDecoder(ck['input_dim']) if is_face else SurfaceHead(ck['input_dim'])).cuda();net.load_state_dict(ck['state_dict']);net.eval();tr=[s for s in oldrows if s['row']['id'] in ck['train_ids']]
   prob,q,details=predict(net,r,'face_v8' if is_face else 'surface_control',tr)
   if is_face and calibration:
    from calibrate_face_v8 import apply
    thresholds=next(f['thresholds'] for f in calibration['outer_folds'] if f['outer_fold']==oldfold) if oldfold is not None else calibration['final_thresholds']
    details=apply(q,r['valid'].numpy(),r['cache']['region'],thresholds)
   if is_face and support:
    from rejection_face_v8 import apply as reject_unsupported
    domain=next(f['domain'] for f in support['folds'] if f['fold']==oldfold) if oldfold is not None else support['final_support']
    assert not held.intersection(domain['train_ids']),'Reference support leakage'
    details=reject_unsupported(details,r['cache'],domain)
   per.append(store_case(out,r,'current_reference',prob,q,details,'old_oof' if rid in om else 'unseen_by_old',ck['train_ids']));del net
  summary['current_reference']=aggregate(per);update()
 for variant in CONFIG['variants']:
  vd=out/variant;vd.mkdir();per=[]
  for f in folds:
   tr=[r for r in rows if r['row']['id'] in f['train_ids']];held=[r for r in rows if r['row']['id'] in f['held_ids']]
   assert not {r['row']['group'] for r in tr}&{r['row']['group'] for r in held}
   write(out/'status.json',dict(stage='training',variant=variant,fold=f['fold']+1,total=len(folds)))
   net=fit(tr,vd/f"fold-{f['fold']}.jsonl",out/'status.json',f['fold']+1) if variant=='face_v8' else old_fit(tr,'surface',vd/f"fold-{f['fold']}.jsonl")
   save(vd/f"fold-{f['fold']}.pt",net,variant,tr,data,fp)
   for r in held:
    prob,q,details=predict(net,r,variant,tr);per.append(store_case(out,r,variant,prob,q,details,f['fold'],f['train_ids']))
   summary[variant]=aggregate(per);update();del net;torch.cuda.empty_cache()
   print('FOLD',variant,f['fold']+1,'elapsed',round(time.time()-started),flush=True)
  write(out/'status.json',dict(stage='final_fit',variant=variant))
  net=fit(rows,vd/'final_train.jsonl',out/'status.json',None) if variant=='face_v8' else old_fit(rows,'surface',vd/'final_train.jsonl')
  save(vd/'model.pt',net,variant,rows,data,fp)
  ck=torch.load(vd/'model.pt',map_location='cpu',weights_only=True);reload=(FaceDecoder(ck['input_dim']) if variant=='face_v8' else SurfaceHead(ck['input_dim'])).cuda();reload.load_state_dict(ck['state_dict']);reload.eval()
  maxdiff=0.;label_diff=0
  for r in rows[:3]:
   a=predict(net,r,variant,rows);b=predict(reload,r,variant,rows)
   maxdiff=max(maxdiff,float(np.abs(a[1]-b[1]).max()));label_diff+=sum(x!=y for x,y in zip(a[2]['labels'],b[2]['labels']))
  assert maxdiff==0 and label_diff==0
  audits.append(dict(variant=variant,save_reload_max_abs=maxdiff,label_disagreements=label_diff,checkpoint_sha256=sha(vd/'model.pt')))
  del net,reload;torch.cuda.empty_cache()
 new=summary['face_v8'];old=summary['surface_control']
 criteria=dict(binary_iou=new['macro']['iou']>=old['macro']['iou']-.01,binary_precision=new['macro']['precision']>=old['macro']['precision']-.01,fine_macro_f1=new['fine_macro_f1']>=old['fine_macro_f1']+.02,rare_classes=all(new['fine'][str(c)]['recall']>old['fine'][str(c)]['recall'] for c in [3,6] if new['fine'][str(c)]['support']))
 write(out/'quality_gate.json',dict(passed=all(criteria.values()),criteria=criteria,policy='Predeclared same-fold comparison, not a new blind test. Does not automatically deploy this experiment.'))
 write(out/'inference_audit.json',dict(models=audits,dino_frozen=True,features_use_labels=False,unknown_fine_used_as_other=False,fine_output_validated=True,production_deployed=False))
 write(out/'training_complete.json',dict(cases=len(rows),folds=len(folds),epochs=200,elapsed=time.time()-started,checkpoint_sha256=sha(out/'face_v8/model.pt')))
 write(out/'status.json',dict(stage='training_complete',elapsed=time.time()-started))
 return summary

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['data','cache','output','reference']:p.add_argument('--'+n,type=Path,required=n!='reference')
 a=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 from model_service import lock
 with lock(ROOT/'service_runtime/gpu.lock'):train(a.data,a.cache,a.output,a.reference)
