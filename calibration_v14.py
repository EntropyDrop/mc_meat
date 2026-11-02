"""Training-only rare operating-point calibration; no external validation tuning."""
import numpy as np
from face_decoder_v9 import decode,INTERNAL_IDS
RARE=[6,9,10];INDEX=[5,6,7];BETA=[.5,1.5,1.5]
GRID=np.round(np.arange(.05,1.,.05),2)

def decode_calibrated(q,valid,head,thresholds):
 q=np.asarray(q);valid=np.asarray(valid,bool);head=np.asarray(head,bool)
 detail=decode(q,valid,head,.5);labels=np.array(detail['labels'],np.uint8)
 meat=q[:,[1,2,3,6,7]].sum(-1);eligible=np.stack([meat<.5,meat>=.5,meat>=.5],1)&(valid&head)[:,None]
 tau=np.array([thresholds[str(c)] for c in RARE]);score=q[:,INDEX]/tau[None,:];ok=(score>=1)&eligible
 # A rejected rare prediction returns to its binary parent, never a new UI type.
 for c,parent in [(6,0),(9,1),(10,1)]:labels[labels==c]=parent
 chosen=np.where(ok,score,-1).argmax(1);take=ok.any(1);labels[take]=np.array(RARE,np.uint8)[chosen[take]];labels[~valid]=0
 detail['uncertain']=np.flatnonzero(valid&(labels==0)&(q.max(1)<.5)).tolist();detail['uncertain_count']=len(detail['uncertain']);detail['labels']=labels.tolist();detail['selected']=np.flatnonzero(np.isin(labels,[1,2,3,9,10])).tolist();detail['category_counts']={str(c):int((labels==c).sum()) for c in INTERNAL_IDS[1:]};detail['calibration_thresholds']=thresholds;detail['preselection_policy']='nested_training_calibrated';return detail

def fit_thresholds(records):
 # Each record is an INNER held-out human prediction from the outer TRAIN portion.
 q=np.concatenate([r['q'][r['valid']] for r in records]);y=np.concatenate([r['labels'][r['valid']] for r in records]);head=np.concatenate([r['head'][r['valid']] for r in records]);meat=q[:,[1,2,3,6,7]].sum(-1);thresholds={};curves={}
 for c,k,beta in zip(RARE,INDEX,BETA):
  truth=y==c;eligible=head&((meat<.5) if c==6 else (meat>=.5));items=[]
  for tau in GRID:
   p=(q[:,k]>=tau)&eligible;t=int((truth&p).sum());f=int((~truth&p).sum());n=int((truth&~p).sum());fscore=(1+beta*beta)*t/max((1+beta*beta)*t+beta*beta*n+f,1)
   items.append(dict(threshold=float(tau),tp=t,fp=f,fn=n,f_beta=fscore))
  best=max(items,key=lambda a:(a['f_beta'],a['threshold']));thresholds[str(c)]=best['threshold'];curves[str(c)]=dict(beta=beta,support=int(truth.sum()),selected=best,curve=items)
 return thresholds,curves
