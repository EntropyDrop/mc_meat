"""Body confidence calibration with nested grouped cross-fitting, never outer labels.

The head's five-category decision is unchanged. Body thresholds optimize a
predeclared pooled F0.5 objective on three inner held-out folds. Final thresholds
use full-data OOF predictions, not in-sample probabilities of the final model.
"""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from meat_features import ROOT,sha,write
from face_decoder_v8 import decode
from train_face_v8 import load,splits,fit,predict,aggregate,store_case

POLICY=dict(version='body-nested-f05-v1',inner_folds=3,objective='pooled F0.5',grid=[.5,.6,.7,.8,.9,.95,.975,.99,.995,.999],min_positive=32,min_negative=32,head_threshold_changed=False,outer_labels_used_for_calibration=False)

def calibrate(records):
 result={}
 for part in range(1,6):
  y=np.concatenate([r['y'][r['valid']&(r['region']%6==part)] for r in records])>.5
  p=np.concatenate([r['prob'][r['valid']&(r['region']%6==part)] for r in records])
  candidates=[]
  for threshold in POLICY['grid']:
   pred=p>=threshold;tp=int((pred&y).sum());fp=int((pred&~y).sum());fn=int((~pred&y).sum())
   score=1.25*tp/max(1.25*tp+.25*fn+fp,1);candidates.append(dict(threshold=threshold,score=score,tp=tp,fp=fp,fn=fn))
  best=max(candidates,key=lambda c:(c['score'],-c['threshold'])) if y.sum()>=32 and (~y).sum()>=32 else candidates[0]
  result[str(part)]=dict(**best,positive=int(y.sum()),negative=int((~y).sum()),candidates=candidates)
 return result

def apply(q,valid,region,thresholds):
 head=valid&(region%6==0);result=decode(q,valid,head,.5);labels=np.array(result['labels'],np.uint8);uncertain=set(result['uncertain'])
 for part in range(1,6):
  threshold=thresholds[str(part)]['threshold'];body=valid&(region%6==part)
  rejected=body&(labels==1)&(q[:,1]<threshold);labels[rejected]=0;uncertain.update(np.flatnonzero(rejected).tolist())
 result.update(labels=labels.tolist(),selected=np.flatnonzero(np.isin(labels,[1,2,3])).tolist(),uncertain=sorted(uncertain),uncertain_count=len(uncertain),category_counts={str(c):int((labels==c).sum()) for c in [1,2,3,5,6]},calibration=POLICY['version'])
 return result

def run(parent,out,data,cache):
 assert not out.exists();out.mkdir(parents=True);(out/'inner').mkdir();start=time.time();rows,fp=load(data,cache)
 for n in ['manifest.json','folds.json','feature_provenance.json','data_source.json']:shutil.copy2(parent/n,out/n)
 write(out/'calibration_policy.json',POLICY);write(out/'parent.json',dict(path=str(parent),comparison_sha256=sha(parent/'comparison.json'),checkpoint_sha256=sha(parent/'face_v8/model.pt')))
 comparison=json.loads((parent/'comparison.json').read_text());folds=comparison['folds'];audits=[];per=[]
 for r in rows:
  d=out/r['row']['id'];d.mkdir()
  for p in (parent/r['row']['id']).glob('*.npz'):shutil.copy2(p,d/p.name)
  shutil.copy2(parent/r['row']['id']/'skin.png',d/'skin.png')
 for outer in folds:
  train=[r for r in rows if r['row']['id'] in outer['train_ids']];held=[r for r in rows if r['row']['id'] in outer['held_ids']]
  groups=sorted({r['row']['group'] for r in train});rng=np.random.default_rng(142+outer['fold']);rng.shuffle(groups)
  assignment={g:i%3 for i,g in enumerate(groups)};records=[];audit=[]
  for inner in range(3):
   tr=[r for r in train if assignment[r['row']['group']]!=inner];val=[r for r in train if assignment[r['row']['group']]==inner]
   trainids=[r['row']['id'] for r in tr];valids=[r['row']['id'] for r in val]
   assert not set(trainids+valids)&set(outer['held_ids'])
   net=fit(tr,out/'inner'/f"outer-{outer['fold']}-inner-{inner}.jsonl",out/'status.json',outer['fold']+1)
   for r in val:
    probability,q,_=predict(net,r,'face_v8');records.append(dict(y=r['y'].numpy(),valid=r['valid'].numpy(),region=r['cache']['region'],prob=probability))
   audit.append(dict(train_ids=trainids,held_ids=valids));del net;torch.cuda.empty_cache()
  thresholds=calibrate(records);audits.append(dict(outer_fold=outer['fold'],outer_held_ids=outer['held_ids'],inner_folds=audit,thresholds=thresholds))
  for r in held:
   pred=np.load(parent/r['row']['id']/'face_v8.npz');q=pred['category_probability'];prob=pred['probability'];details=apply(q,r['valid'].numpy(),r['cache']['region'],thresholds)
   per.append(store_case(out,r,'face_v8_calibrated',prob,q,details,outer['fold'],outer['train_ids']))
  print('CALIBRATION',outer['fold']+1,'elapsed',round(time.time()-start),flush=True)
 records=[]
 for r in rows:
  pred=np.load(parent/r['row']['id']/'face_v8.npz');records.append(dict(y=r['y'].numpy(),valid=r['valid'].numpy(),region=r['cache']['region'],prob=pred['probability']))
 thresholds=calibrate(records);write(out/'calibration.json',dict(policy=POLICY,outer_folds=audits,final_thresholds=thresholds,final_source='five-fold OOF of latest data; no final-model training-set predictions'))
 comparison['results']['face_v8_calibrated']=aggregate(per);comparison['evaluation']+=' Body calibration uses nested three-fold predictions of each outer training partition only.';write(out/'comparison.json',comparison)
 (out/'face_v8').mkdir();ck=torch.load(parent/'face_v8/model.pt',map_location='cpu',weights_only=True)
 ck['body_calibration']=thresholds;ck['calibration_policy']=POLICY;torch.save(ck,out/'face_v8/model.pt')
 old=comparison['results']['surface_control'];new=comparison['results']['face_v8_calibrated'];current=comparison['results']['current_reference']
 criteria=dict(binary_iou=new['macro']['iou']>=old['macro']['iou']-.01,binary_precision=new['macro']['precision']>=old['macro']['precision'],fine_macro_f1=new['fine_macro_f1']>old['fine_macro_f1']+.02,rare_classes=all(new['fine'][str(c)]['recall']>old['fine'][str(c)]['recall'] for c in [3,6] if new['fine'][str(c)]['support']),false_positive_count=new['pooled']['fp']<=current['pooled']['fp'])
 write(out/'quality_gate.json',dict(passed=all(criteria.values()),criteria=criteria,policy='Nested body-only F0.5 calibration; compare fixed criteria on outer OOF.'))
 audit=json.loads((parent/'inference_audit.json').read_text());audit['body_calibration_nested']=True;audit['models']=[dict(variant='face_v8',checkpoint_sha256=sha(out/'face_v8/model.pt'),parent_weights_unchanged=True)]
 write(out/'inference_audit.json',audit);write(out/'training_complete.json',dict(parent_training_sha256=sha(parent/'training_complete.json'),inner_fits=15,cases=len(rows),elapsed=time.time()-start,checkpoint_sha256=sha(out/'face_v8/model.pt')))
 write(out/'status.json',dict(stage='training_complete',elapsed=time.time()-start))

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['parent','output','data','cache']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 from model_service import lock
 with lock(ROOT/'service_runtime/gpu.lock'):run(a.parent,a.output,a.data,a.cache)
