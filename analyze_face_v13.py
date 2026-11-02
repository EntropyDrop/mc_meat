"""Paired rare-class analysis using human held-out labels only."""
from pathlib import Path
import argparse,json
import numpy as np
from meat_features import write
RARE=[6,9,10]

def analyze(run):
 c=json.loads((run/'comparison.json').read_text());manifest=json.loads((run/'evaluation_manifest.json').read_text());ids=[r['id'] for r in manifest['rows']]
 assert c['cases']==len(ids)==169
 for name,value in c['results'].items():assert sorted(p['id'] for p in value['per_case'])==sorted(ids),(name,'incomplete evaluation')
 source=json.loads((run/'data_source.json').read_text());coverage=json.loads((Path(source['data'])/'coverage_audit.json').read_text())
 group={i:i for i in ids}
 for pair in coverage['identical_source_pairs']:
  a,b=pair['a'],pair['b']
  if a in group and b in group:
   old=group[a];target=group[b]
   for k in group:
    if group[k]==old:group[k]=target
 gids=sorted(set(group.values()));casegroup=np.array([gids.index(group[i]) for i in ids]);out={};casefacts={i:{} for i in ids};labels={}
 for name,value in c['results'].items():
  supported=c.get('references',{}).get(name,{}).get('supported_class_ids',[1,2,3,5,6,9,10]);out[name]={}
  predictions=[np.load(run/i/(name+'.npz'))['labels'] for i in ids];labels[name]=predictions
  for cls in RARE:
   if cls not in supported:out[name][str(cls)]={'supported':False};continue
   matrix=[];dest={};orig={};positive_skins=0;detected_skins=0;negative_skins=0;negative_hit=0;binary_errors=0
   for rid,pred in zip(ids,predictions):
    gt=np.load(run/rid/'truth.npz');v=gt['valid'].astype(bool);truth=gt['labels'];y=v&(truth==cls);take=v&(pred==cls)
    tp=int((y&take).sum());fp=int((~y&take&v).sum());fn=int((y&~take).sum());neg=int((v&~y).sum());matrix.append([tp,fp,fn,neg])
    if y.any():positive_skins+=1;detected_skins+=int(tp>0)
    else:negative_skins+=1;negative_hit+=int(fp>0)
    for k in [0,1,2,3,5,6,9,10]:
     dest[str(k)]=dest.get(str(k),0)+int((y&(pred==k)).sum());orig[str(k)]=orig.get(str(k),0)+int((take&(truth==k)&v).sum())
    selected=np.isin(pred,[1,2,3,9,10]);binary_errors+=int((y&(selected if cls==6 else ~selected)).sum())
    casefacts[rid].setdefault(str(cls),{})[name]=dict(tp=tp,fp=fp,fn=fn,positive_cells=int(y.sum()))
   arr=np.array(matrix);tp,fp,fn,neg=arr.sum(0)
   out[name][str(cls)]=dict(supported=True,tp=int(tp),fp=int(fp),fn=int(fn),support=int(tp+fn),precision=float(tp/max(tp+fp,1)),recall=float(tp/max(tp+fn,1)),f1=float(2*tp/max(2*tp+fp+fn,1)),false_positives_per_1000_negative_cells=float(fp/max(neg,1)*1000),positive_skins=positive_skins,detected_positive_skins=detected_skins,negative_skins=negative_skins,negative_skins_with_false_positive=negative_hit,binary_parent_errors_on_positive_cells=binary_errors,ground_truth_cells_predicted_as=dest,predicted_cells_ground_truth=orig,per_case_counts=arr.tolist())
 comparisons={};rng=np.random.default_rng(20261102);draw=rng.integers(len(gids),size=(3000,len(gids)))
 def metrics(a):
  t,f,n=a[...,0],a[...,1],a[...,2]
  return np.stack([t/np.maximum(t+f,1),t/np.maximum(t+n,1),2*t/np.maximum(2*t+f+n,1)],-1)
 for baseline,candidate in [('ordinary_augmented','ordinary_boundary'),('balanced_augmented','balanced_boundary'),('balanced_positive_only','balanced_boundary')]:
  pair={}
  for cls in RARE:
   aa=np.array(out[baseline][str(cls)]['per_case_counts']);bb=np.array(out[candidate][str(cls)]['per_case_counts']);ga=np.stack([aa[casegroup==k].sum(0) for k in range(len(gids))]);gb=np.stack([bb[casegroup==k].sum(0) for k in range(len(gids))]);delta=metrics(gb[draw].sum(1))-metrics(ga[draw].sum(1))
   pair[str(cls)]=dict(delta_tp=int(bb[:,0].sum()-aa[:,0].sum()),delta_fp=int(bb[:,1].sum()-aa[:,1].sum()),delta_fn=int(bb[:,2].sum()-aa[:,2].sum()),precision_recall_f1_delta=(metrics(bb.sum(0))-metrics(aa.sum(0))).tolist(),exploratory_95_percentile_interval=np.quantile(delta,[.025,.975],axis=0).tolist())
  comparisons[baseline+'__'+candidate]=pair
 result=dict(evaluation_human_cases=len(ids),source_groups=len(gids),metrics=out,matched_comparisons=comparisons,cases=casefacts,note='Rare FN means a missing semantic category, not necessarily an unselected meat cell. Binary parent errors are reported separately. Intervals resample whole source groups and are exploratory: historical development data, few positive skins, overlapping CV training folds.')
 write(run/'rare_analysis.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();analyze(a.run)
