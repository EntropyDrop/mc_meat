"""Seven-category grouped OOF comparison; never promotes the production model."""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from meat_features import ROOT,sha,write
from data_face_v9 import load_rows
from train_face_v6 import summarize
from train_supervised import metric
from semantic_schema_v3 import CLASS_IDS
from face_decoder_v9 import FaceDecoder,pack,loss,decode,binary_logits,INTERNAL_IDS,MEAT_IDS

CONFIG=dict(model_module='face_decoder_v9',semantic_class_ids=CLASS_IDS,internal_class_ids=INTERNAL_IDS,
 epochs=200,lr=.001,eta_min=.0001,weight_decay=.01,hidden=48,seed=42,folds=5,
 fine_loss_weight=.6,fine_warmup_epochs=20,fine_threshold=.5,dino_frozen=True,
 unknown_fine='binary marginal only',reviewed_zero='internal other fine supervision',
 validation='fixed grouped stratified five-fold; same folds and joint loss for both architectures',
 variants=['surface_v9','face_v9'],initializer='random per fold, no full-data warm start',
 fine_scope='all reviewed visible UV cells; six-face inner/outer head refinement for face_v9')


def splits(rows):
 groups=sorted({r['row']['group'] for r in rows});rng=np.random.default_rng(42);rng.shuffle(groups)
 presence={g:np.array([any(((r['semantic']==c)&r['valid']).any() for r in rows if r['row']['group']==g) for c in CLASS_IDS],int) for g in groups}
 total=np.stack(list(presence.values())).sum(0).clip(1);groups.sort(key=lambda g:-float((presence[g]/total).sum()))
 n=min(5,len(groups));counts=np.zeros((n,len(CLASS_IDS)));sizes=np.zeros(n);assigned={}
 for g in groups:
  score=((counts+presence[g])**2/total).sum(1)-((counts**2)/total).sum(1)+.25*sizes
  f=int(score.argmin());assigned[g]=f;counts[f]+=presence[g];sizes[f]+=1
 return [dict(fold=f,held_ids=[r['row']['id'] for r in rows if assigned[r['row']['group']]==f],train_ids=[r['row']['id'] for r in rows if assigned[r['row']['group']]!=f]) for f in range(n)]


@torch.inference_mode()
def predict(net,r,device='cuda'):
 logits=net(pack([r],device));v=r['valid'].numpy();q=np.zeros((4096,len(INTERNAL_IDS)),np.float32);q[v]=logits.softmax(-1).float().cpu().numpy()
 prob=np.zeros(4096,np.float32);prob[v]=binary_logits(logits).sigmoid().float().cpu().numpy()
 details=decode(q,v,v&(r['cache']['region']%6==0),CONFIG['fine_threshold'])
 return prob,q,details


def fit(rows,path,status,fold,variant):
 torch.manual_seed(CONFIG['seed']);p=pack(rows,'cuda');net=FaceDecoder(p['x'].shape[1],facial=variant=='face_v9').cuda()
 assert not p['x'].requires_grad and not any('dino' in k for k in net.state_dict())
 opt=torch.optim.AdamW(net.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay']);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,CONFIG['epochs'],eta_min=CONFIG['eta_min'])
 started=time.time()
 with path.open('w') as stream:
  for epoch in range(1,CONFIG['epochs']+1):
   net.train();opt.zero_grad(set_to_none=True);total,binary,fine=loss(net,p,epoch);assert torch.isfinite(total)
   total.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();schedule.step()
   if epoch==1 or epoch%20==0:
    value=dict(epoch=epoch,loss=float(total.detach()),binary=float(binary.detach()),fine=float(fine.detach()),elapsed=time.time()-started)
    stream.write(json.dumps(value)+'\n');stream.flush();write(status,dict(stage='training',variant=variant,fold=fold,**value))
 return net.eval()


def save(path,net,variant,rows,data,fp):
 torch.save(dict(schema='mc-meat-face-v9',model_module='face_decoder_v9',variant=variant,input_dim=rows[0]['x'].shape[1],state_dict={k:v.cpu() for k,v in net.state_dict().items()},config=CONFIG,train_ids=[r['row']['id'] for r in rows],manifest_sha256=sha(data/'manifest.json'),feature_provenance=fp),path)


def evaluate(r,prob,details):
 v=r['valid'].numpy();lab=np.array(details['labels']);selected=np.isin(lab,MEAT_IDS);gt=r['semantic'].numpy();known=v&((gt>0)|r['reviewed'].numpy())
 result=metric(r['y'].numpy(),selected.astype(float),v);result['raw_binary']=metric(r['y'].numpy(),prob,v)
 result.update(id=r['row']['id'],fine={},uncertain_count=details.get('uncertain_count',0))
 for c in CLASS_IDS:
  truth=gt==c;pred=lab==c
  result['fine'][str(c)]={n:int((known&m).sum()) for n,m in [('tp',truth&pred),('fp',~truth&pred),('fn',truth&~pred)]}
 return result


def aggregate(per):
 value=summarize(per);value['fine']={}
 for c in CLASS_IDS:
  counts={k:sum(r['fine'][str(c)][k] for r in per) for k in ['tp','fp','fn']};t,f,n=counts.values()
  value['fine'][str(c)]={**counts,'precision':t/max(t+f,1),'recall':t/max(t+n,1),'iou':t/max(t+f+n,1),'f1':2*t/max(2*t+f+n,1),'support':t+n}
 value['fine_macro_f1']=float(np.mean([r['f1'] for r in value['fine'].values() if r['support']]))
 value['raw_binary_macro']={k:float(np.mean([r['raw_binary'][k] for r in per])) for k in ['iou','precision','recall']}
 return value


def store_case(out,r,variant,prob,q,details,fold,train_ids):
 case=out/r['row']['id'];case.mkdir(exist_ok=True)
 np.savez_compressed(case/(variant+'.npz'),probability=prob,category_probability=q,labels=np.array(details['labels'],np.uint8),uncertain=np.array(details.get('uncertain',[]),np.int64))
 result=evaluate(r,prob,details);result.update(fold=fold,train_ids=train_ids);return result


def current_reference(rows,reference,out):
 # Keep the five-class reference implementation and training inputs unchanged.
 from train_face_v8 import load as old_load,predict as old_predict
 from surface_model_v4 import SurfaceHead
 ck=torch.load(reference,map_location='cpu',weights_only=True);assert ck['model_module']=='surface_model_v4'
 oldrun=reference.parent.parent;source=json.loads((oldrun/'data_source.json').read_text()) if (oldrun/'data_source.json').exists() else {}
 olddata=Path(source['data']) if source else oldrun.parent/'data';oldcache=Path(source['cache']) if source else oldrun.parent/'face_cache'
 oldrows,_=old_load(olddata,oldcache);om={r['row']['id']:r for r in oldrows};groups=sorted({r['row']['group'] for r in oldrows})
 per=[];audit=[]
 for r in rows:
  rid=r['row']['id'];path=oldrun/'surface'/f"fold-{groups.index(om[rid]['row']['group'])}.pt" if rid in om else reference
  saved=torch.load(path,map_location='cpu',weights_only=True);held={s['row']['id'] for s in rows if s['row']['group']==r['row']['group']}
  assert not held.intersection(saved['train_ids']),'Reference group leakage'
  net=SurfaceHead(saved['input_dim']).cuda();net.load_state_dict(saved['state_dict']);net.eval();tr=[s for s in oldrows if s['row']['id'] in saved['train_ids']]
  prob,q,details=old_predict(net,r,'surface_control',tr)
  per.append(store_case(out,r,'current_reference',prob,q,details,'old_oof' if rid in om else 'unseen_by_old',saved['train_ids']))
  audit.append(dict(id=rid,checkpoint_sha256=sha(path),checkpoint=str(path),reference_scope=per[-1]['fold']))
  del net
 write(out/'current_reference_audit.json',audit)
 return aggregate(per)


def train(data,cache,out,reference):
 assert not out.exists(),'New experiment directory required';out.mkdir(parents=True);(out/'source').mkdir();started=time.time()
 rows,fp=load_rows(data,cache);folds=splits(rows)
 for p in ROOT.glob('*.py'):shutil.copy2(p,out/'source'/p.name)
 shutil.copy2(ROOT/'surface_topology_v3.json',out/'source/surface_topology_v3.json');shutil.copy2(data/'manifest.json',out/'manifest.json')
 write(out/'configuration.json',CONFIG);write(out/'folds.json',folds);write(out/'feature_provenance.json',fp)
 write(out/'data_source.json',dict(data=str(data),cache=str(cache)));write(out/'source_hashes.json',{p.name:sha(p) for p in (out/'source').iterdir()})
 active=ROOT/'service_runtime/production/active.json';shutil.copy2(active,out/'production_before.json')
 for r in rows:
  d=out/r['row']['id'];d.mkdir();shutil.copy2(data/r['row']['id']/'skin.png',d/'skin.png')
  np.savez_compressed(d/'truth.npz',valid=r['valid'].numpy(),y=r['y'].numpy(),labels=r['semantic'].numpy(),weights=r['fine_weights'].numpy())
 summary={};audits=[]
 def update():write(out/'comparison.json',dict(cases=len(rows),folds=folds,results=summary,evaluation='Historical grouped five-fold comparison on fully human-reviewed seven-class labels; not a new blind benchmark. Reference uses original held-out models or unseen examples; training size differs. Fine false positives include reviewed other cells. Fixed epochs and thresholds, no validation tuning.'))
 write(out/'status.json',dict(stage='current_reference'));summary['current_reference']=current_reference(rows,reference,out);update()
 for variant in CONFIG['variants']:
  vd=out/variant;vd.mkdir();per=[]
  for f in folds:
   tr=[r for r in rows if r['row']['id'] in f['train_ids']];held=[r for r in rows if r['row']['id'] in f['held_ids']]
   assert not {r['row']['group'] for r in tr}&{r['row']['group'] for r in held}
   net=fit(tr,vd/f"fold-{f['fold']}.jsonl",out/'status.json',f['fold']+1,variant);save(vd/f"fold-{f['fold']}.pt",net,variant,tr,data,fp)
   for r in held:
    prob,q,details=predict(net,r);per.append(store_case(out,r,variant,prob,q,details,f['fold'],f['train_ids']))
   summary[variant]=aggregate(per);update();del net;torch.cuda.empty_cache()
   print('FOLD',variant,f['fold']+1,'elapsed',round(time.time()-started),flush=True)
  net=fit(rows,vd/'final_train.jsonl',out/'status.json',None,variant);save(vd/'model.pt',net,variant,rows,data,fp)
  ck=torch.load(vd/'model.pt',map_location='cpu',weights_only=True);reloaded=FaceDecoder(ck['input_dim'],facial=variant=='face_v9').cuda();reloaded.load_state_dict(ck['state_dict']);reloaded.eval()
  maxdiff=0.;labeldiff=0
  for r in rows[:3]:
   a=predict(net,r);b=predict(reloaded,r);maxdiff=max(maxdiff,float(np.abs(a[1]-b[1]).max()));labeldiff+=sum(x!=y for x,y in zip(a[2]['labels'],b[2]['labels']))
  assert maxdiff==0 and labeldiff==0
  audits.append(dict(variant=variant,save_reload_max_abs=maxdiff,label_disagreements=labeldiff,checkpoint_sha256=sha(vd/'model.pt')))
  del net,reloaded;torch.cuda.empty_cache()
 # Paired uncertainty across entire held-out skins, without tuning any thresholds.
 a={r['id']:r for r in summary['face_v9']['per_case']};b={r['id']:r for r in summary['surface_v9']['per_case']}
 delta=np.array([a[k]['iou']-b[k]['iou'] for k in sorted(a)]);rng=np.random.default_rng(20261102)
 boot=delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(1)
 write(out/'paired_comparison.json',dict(metric='macro meat IoU',face_minus_surface=float(delta.mean()),bootstrap_95_percentile=np.quantile(boot,[.025,.975]).tolist(),units='whole historical skins',note='Exploratory interval; cross-validation training folds overlap.'))
 assert sha(active)==sha(out/'production_before.json'),'Production changed concurrently'
 write(out/'inference_audit.json',dict(models=audits,dino_frozen=True,features_use_labels=False,unknown_fine_used_as_other=False,reviewed_zeros_supervise_other=True,fine_output_validated=True,production_deployed=False,production_unchanged=True))
 write(out/'training_complete.json',dict(cases=len(rows),folds=len(folds),epochs=CONFIG['epochs'],elapsed=time.time()-started,checkpoint_sha256=sha(out/'face_v9/model.pt')))
 write(out/'status.json',dict(stage='training_complete',elapsed=time.time()-started))

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['data','cache','output','reference']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 from model_service import lock
 with lock(ROOT/'service_runtime/gpu.lock'):train(a.data,a.cache,a.output,a.reference)
