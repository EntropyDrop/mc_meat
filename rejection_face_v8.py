"""Positive-only learned spatial support; unsupported fine classes are uncertain.

This is an abstention boundary, not a claim that unknown labels are other.
Each OOF support map is built only from its training partition. No case-specific
mask, target label or binary annotation is read during inference.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import torch
from semantic_schema import CLASS_IDS
from meat_features import sha,write

def build(rows):
 support=np.zeros((5,6,8,8),bool)
 for r in rows:
  c=r['cache'];sem=r['semantic'].numpy();w=r['fine_weights'].numpy()
  for i in np.flatnonzero(r['valid'].numpy()&(c['region']%6==0)&(sem>0)&(w>0)):
   k=CLASS_IDS.index(sem[i]);face=int(c['local'][i,42:48].argmax());x,y=np.rint((c['local'][i,49:51]+1)*4-.5).astype(int)
   support[k,face,max(0,y-1):min(8,y+2),max(0,x-1):min(8,x+2)]=True
 return dict(schema='face-v8-positive-support-v1',support=support.astype(int).tolist(),class_ids=CLASS_IDS,radius=1,unknown_used_as_negative=False,train_ids=[r['row']['id'] for r in rows])

def apply(details,values,domain):
 assert domain['class_ids']==CLASS_IDS
 support=np.array(domain['support'],bool);labels=np.array(details['labels'],np.uint8);uncertain=set(details.get('uncertain',[]));rejected=[]
 for i in np.flatnonzero(np.isin(labels,[2,3,5,6])):
  c=int(labels[i]);face=int(values['local'][i,42:48].argmax());x,y=np.rint((values['local'][i,49:51]+1)*4-.5).astype(int)
  if values['region'][i]%6!=0 or not support[CLASS_IDS.index(c),face,y,x]:labels[i]=0;uncertain.add(int(i));rejected.append(int(i))
 return {**details,'labels':labels.tolist(),'selected':np.flatnonzero(np.isin(labels,[1,2,3])).tolist(),'category_counts':{str(c):int((labels==c).sum()) for c in CLASS_IDS},'uncertain':sorted(uncertain),'uncertain_count':len(uncertain),'unsupported_fine_count':len(rejected),'fine_support_schema':domain['schema']}

def run(parent,out):
 from train_face_v8 import load,aggregate,store_case
 from face_decoder_v8 import decode
 from calibrate_face_v8 import apply as calibrated
 assert not out.exists();out.mkdir(parents=True)
 source=json.loads((parent/'data_source.json').read_text());rows,_=load(Path(source['data']),Path(source['cache']))
 for name in ['manifest.json','folds.json','data_source.json','feature_provenance.json','calibration.json','calibration_policy.json']:
  shutil.copy2(parent/name,out/name)
 for r in rows:
  d=out/r['row']['id'];d.mkdir()
  for p in (parent/r['row']['id']).glob('*.npz'):shutil.copy2(p,d/p.name)
  shutil.copy2(parent/r['row']['id']/'skin.png',d/'skin.png')
 comparison=json.loads((parent/'comparison.json').read_text());cal=json.loads((parent/'calibration.json').read_text());per={k:[] for k in ['face_v8_guarded_balanced','face_v8_guarded']};audits=[]
 for fold in comparison['folds']:
  tr=[r for r in rows if r['row']['id'] in fold['train_ids']];domain=build(tr);assert not set(domain['train_ids'])&set(fold['held_ids'])
  thresholds=next(f['thresholds'] for f in cal['outer_folds'] if f['outer_fold']==fold['fold']);audits.append(dict(fold=fold['fold'],domain=domain))
  for r in rows:
   if r['row']['id'] not in fold['held_ids']:continue
   pred=np.load(parent/r['row']['id']/'face_v8.npz');q=pred['category_probability'];valid=r['valid'].numpy();head=valid&(r['cache']['region']%6==0)
   for variant,base in [('face_v8_guarded_balanced',decode(q,valid,head)),('face_v8_guarded',calibrated(q,valid,r['cache']['region'],thresholds))]:
    details=apply(base,r['cache'],domain);per[variant].append(store_case(out,r,variant,pred['probability'],q,details,fold['fold'],fold['train_ids']))
 for k,v in per.items():comparison['results'][k]=aggregate(v)
 write(out/'comparison.json',comparison);write(out/'support_audit.json',dict(folds=audits,final_support=build(rows),note='Known positive fine labels only; unsupported predictions abstain. Not a learned semantic negative.'))
 (out/'face_v8').mkdir();ck=torch.load(parent/'face_v8/model.pt',map_location='cpu',weights_only=True);ck['fine_support']=build(rows);torch.save(ck,out/'face_v8/model.pt')
 new=comparison['results']['face_v8_guarded'];old=comparison['results']['surface_control'];current=comparison['results']['current_reference']
 criteria=dict(binary_iou=new['macro']['iou']>=old['macro']['iou']-.01,binary_precision=new['macro']['precision']>=old['macro']['precision'],fine_macro_f1=new['fine_macro_f1']>old['fine_macro_f1']+.02,rare_classes=all(new['fine'][str(c)]['recall']>old['fine'][str(c)]['recall'] for c in [3,6] if new['fine'][str(c)]['support']),false_positive_count=new['pooled']['fp']<=current['pooled']['fp'])
 write(out/'quality_gate.json',dict(passed=all(criteria.values()),criteria=criteria,policy='Same fixed grouped OOF, nested body calibration, training-only positive spatial support.'))
 write(out/'parent.json',dict(path=str(parent),base_training_path=json.loads((parent/'parent.json').read_text())['path'],checkpoint_sha256=sha(parent/'face_v8/model.pt')))
 audit=json.loads((parent/'inference_audit.json').read_text());audit['spatial_support_training_only']=True;audit['models']=[dict(variant='face_v8',checkpoint_sha256=sha(out/'face_v8/model.pt'),parent_weights_unchanged=True)];write(out/'inference_audit.json',audit)
 write(out/'training_complete.json',dict(parent_training_sha256=sha(parent/'training_complete.json'),cases=len(rows),checkpoint_sha256=sha(out/'face_v8/model.pt')))
 write(out/'source_hashes.json',{Path(__file__).name:sha(Path(__file__))});write(out/'status.json',dict(stage='training_complete',cases=len(rows)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);run(a.parent,a.output)
