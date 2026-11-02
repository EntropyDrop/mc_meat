"""Nested two-fold calibration. Every outer-held prediction stays untouched until scoring."""
from pathlib import Path
import argparse,json,time,shutil,traceback
import numpy as np
import torch
from meat_features import ROOT,sha,write
from data_face_v13 import load_rows
from train_face_v14 import fit,save
from predict_face_v14 import predict
from train_face_v9 import aggregate,store_case
from calibration_v14 import fit_thresholds,decode_calibrated
from weighted_face_v12 import fold_training
from model_service import lock

def inner_splits(rows,links):
 ids={r['row']['id'] for r in rows};groups={i:i for i in ids}
 for a,b in links:
  if a in ids and b in ids:
   ga,gb=groups[a],groups[b]
   for k in groups:
    if groups[k]==ga:groups[k]=gb
 # Existing group relation is also honored.
 for a in rows:
  for b in rows:
   if a['row']['group']==b['row']['group']:
    ga,gb=groups[a['row']['id']],groups[b['row']['id']]
    for k in groups:
     if groups[k]==ga:groups[k]=gb
 names=sorted(set(groups.values()));rng=np.random.default_rng(42);rng.shuffle(names)
 presence={g:np.array([any(((r['semantic']==c)&r['valid']).any() for r in rows if groups[r['row']['id']]==g) for c in [6,9,10]],int) for g in names};tot=np.stack(list(presence.values())).sum(0).clip(1);names.sort(key=lambda g:-float((presence[g]/tot).sum()))
 counts=np.zeros((2,3));size=np.zeros(2);assigned={}
 for g in names:
  scores=(((counts+presence[g])**2-counts**2)/tot).sum(1)+.25*size;f=int(scores.argmin());assigned[g]=f;counts[f]+=presence[g];size[f]+=1
 return [dict(fold=f,held_ids=sorted(i for i in ids if assigned[groups[i]]==f),train_ids=sorted(i for i in ids if assigned[groups[i]]!=f)) for f in range(2)]

def train(a):
 assert not a.output.exists();out=a.output;out.mkdir();(out/'source').mkdir();start=time.time();parent=a.parent
 # Reuse the exact architecture and optimization code of the already frozen v14 run.
 for name,h in json.loads((parent/'source_hashes.json').read_text()).items():
  assert sha(parent/'source'/name)==h
  if (ROOT/name).exists():assert sha(ROOT/name)==h,'Changed shared source: '+name
  shutil.copy2(parent/'source'/name,out/'source'/name)
 for name in ['calibration_v14.py','train_calibration_v14.py','predict_calibrated_v14.py','analyze_calibrated_v14.py','report_calibrated_v14.py','finalize_calibration_v14.py','test_calibration_v14.py','predict_calibrated_v14_r2.py','train_calibration_v14_r2.py','finalize_calibration_v14_r2.py']:shutil.copy2(ROOT/name,out/'source'/name)
 write(out/'source_hashes.json',{p.name:sha(p) for p in (out/'source').iterdir()})
 rows,fp=load_rows(a.data,a.cache);ids={r['id'] for r in json.loads((parent/'evaluation_manifest.json').read_text())['rows']};human=[r for r in rows if r['row']['id'] in ids];aux=[r for r in rows if r['row']['id'] not in ids and r['row'].get('training_eligible')];byid={r['row']['id']:r for r in human};assert len(human)==169 and len(aux)==60
 links=[(x['a'],x['b']) for x in json.loads((a.data/'coverage_audit.json').read_text())['identical_source_pairs']];folds=json.loads((parent/'folds.json').read_text());prior=json.loads((parent/'comparison.json').read_text())
 for f in ['comparison.json','folds.json','evaluation_manifest.json','manifest.json','production_before.json','feature_provenance.json']:shutil.copy2(parent/f,out/f)
 for rid in ids:shutil.copytree(parent/rid,out/rid)
 write(out/'configuration.json',dict(variant='context_calibrated_v14',network='context_v14',outer_folds=5,inner_folds=2,fit_epochs=200,seed=42,threshold_grid=[round(x,2) for x in np.arange(.05,1,.05)],class_beta={'6':.5,'9':1.5,'10':1.5},threshold_data='Inner OOF predictions of outer training human skins only',synthetic_for_calibration=False,head_scope=True,parent_probability_cutoff=.5,selection='predeclared context architecture; no outer-score-based choice'))
 write(out/'data_source.json',dict(data=str(a.data),cache=str(a.cache),parent=str(parent)));all_audits=[];thresholds_by_fold={};per=[]
 for outer in [*folds,dict(fold='full',train_ids=sorted(ids),held_ids=[])]:
  tr,purged=fold_training(human,aux,outer,links,True);h=[r for r in tr if r['row']['id'] in ids];auxtr=[r for r in tr if r['row']['id'] not in ids];held_outer=set(outer['held_ids']);inner=inner_splits(h,links);records=[];od=out/('calibration-'+str(outer['fold']));od.mkdir()
  for f in inner:
   trainrows,removed=fold_training(h,auxtr,f,links,True);held=[byid[i] for i in f['held_ids']];trainids=[r['row']['id'] for r in trainrows];assert not set(trainids)&held_outer and not set(f['held_ids'])&held_outer
   audit=dict(outer_fold=outer['fold'],inner_fold=f['fold'],train_ids=trainids,calibration_ids=f['held_ids'],outer_held_ids=sorted(held_outer),purged=removed);all_audits.append(audit);write(out/'nested_audit.json',all_audits)
   net=fit(trainrows,od/f"inner-{f['fold']}.jsonl",out/'status.json','context_v14',str(outer['fold'])+'-inner-'+str(f['fold']));save(od/f"inner-{f['fold']}.pt",net,trainrows,fp,a.data,'context_v14')
   for r in held:
    prob,q,detail=predict(net,r);v=r['valid'].numpy();rec=dict(id=r['row']['id'],q=q,valid=v,head=v&(r['cache']['region']%6==0),labels=r['semantic'].numpy());records.append(rec);np.savez_compressed(od/(r['row']['id']+'.npz'),q=q,valid=v,labels=rec['labels'],head=rec['head'])
   del net;torch.cuda.empty_cache()
  assert {r['id'] for r in records}=={r['row']['id'] for r in h}
  thresholds,curves=fit_thresholds(records);thresholds_by_fold[str(outer['fold'])]=thresholds;write(od/'calibration.json',dict(thresholds=thresholds,curves=curves,calibration_ids=[r['id'] for r in records],outer_held_ids=outer['held_ids']))
  if outer['fold']!='full':
   trainids=[r['row']['id'] for r in tr]
   for rid in outer['held_ids']:
    r=byid[rid];npz=np.load(parent/rid/'context_v14.npz');q=npz['category_probability'];v=r['valid'].numpy();detail=decode_calibrated(q,v,v&(r['cache']['region']%6==0),thresholds);per.append(store_case(out,r,'context_calibrated_v14',npz['probability'],q,detail,outer['fold'],trainids))
   prior['results']['context_calibrated_v14']=aggregate(per);write(out/'comparison.json',prior)
  else:
   bundle=out/'context_calibrated_v14';bundle.mkdir();shutil.copy2(parent/'context_v14/model.pt',bundle/'model.pt');write(bundle/'calibration.json',dict(schema='mc-meat-nested-calibration-v14',thresholds=thresholds,parent_model_sha256=sha(bundle/'model.pt'),scope='head rare categories with binary-parent eligibility',source='two-fold full-training human OOF',human_calibration_cases=len(records)))
  write(out/'thresholds_by_fold.json',thresholds_by_fold);print('CALIBRATION_DONE',outer['fold'],thresholds,round(time.time()-start),flush=True)
 assert len(per)==169;assert sha(ROOT/'service_runtime/production/active.json')==sha(out/'production_before.json')
 write(out/'training_complete.json',dict(nested_fits=12,outer_cases=169,inner_folds=2,model_reused_from=str(parent/'context_v14/model.pt'),elapsed=time.time()-start));write(out/'status.json',dict(stage='calibration_complete'))

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['data','cache','output','parent']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
 try:
  with lock(ROOT/'service_runtime/gpu.lock'):train(a)
 except Exception:
  if a.output.exists():write(a.output/'failure.json',dict(error=traceback.format_exc()))
  raise
