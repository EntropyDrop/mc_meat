"""Training-only positional support for the closed-set auxiliary classifier.

Unknown fine cells are not used as negatives. Support uses only known positive
fine labels, pools inner/outer layers, and permits a one-cell neighborhood on
each head face. It rejects extrapolation to unseen regions (e.g. rear hair as a
beard); it does not label an input from a training mask or change binary meat.
"""
import argparse,json
from pathlib import Path
import numpy as np
from meat_features import ROOT,sha,write
from semantic_schema import CLASS_IDS

def support_path(checkpoint):return ROOT/'service_runtime/semantic_support'/(sha(checkpoint)+'.json')

def build(checkpoint,data,cache):
 import torch
 from fine_supervision import weights
 ck=torch.load(checkpoint,weights_only=True,map_location='cpu');manifest=data/'manifest.json'
 assert sha(manifest)==ck['manifest_sha256']
 assert json.loads((cache/'provenance.json').read_text())==ck['feature_provenance']
 support=np.zeros((5,6,8,8),bool);counts={str(c):0 for c in CLASS_IDS};sources=[]
 for row in json.loads(manifest.read_text())['rows']:
  if not row.get('semantic_sha256'):continue
  d=data/row['id'];meta=json.loads((d/'semantic.json').read_text())
  assert sha(d/'semantic.json')==row['semantic_metadata_sha256'] and sha(d/'semantic.png')==row['semantic_sha256']
  cp=cache/(row['id']+'.npz');audit=json.loads(cp.with_suffix('.json').read_text())
  assert audit['sha256']==sha(cp) and audit['skin_sha256']==row['skin_sha256']
  values=np.load(cp);valid=values['valid'].astype(bool);local=values['local'];labels=np.array(meta['labels']);confidence=weights(meta)
  known=valid&(values['region']%6==0)&(labels>0)&(confidence>0)
  for i in np.flatnonzero(known):
   c=int(labels[i]);k=CLASS_IDS.index(c);face=int(local[i,42:48].argmax())
   x,y=np.rint((local[i,49:51]+1)*4-.5).astype(int);assert 0<=x<8 and 0<=y<8
   support[k,face,max(0,y-1):min(8,y+2),max(0,x-1):min(8,x+2)]=True;counts[str(c)]+=1
  sources.append(dict(id=row['id'],semantic_metadata_sha256=row['semantic_metadata_sha256']))
 p=support_path(checkpoint);p.parent.mkdir(parents=True,exist_ok=True)
 value=dict(schema='mc-meat-fine-support-v1',checkpoint_sha256=sha(checkpoint),manifest_sha256=sha(manifest),class_ids=CLASS_IDS,support=support.astype(np.uint8).tolist(),known_head_cells=counts,sources=sources,unknown_cells_used_as_negatives=False,layer_pooling=True,face_neighborhood_radius=1)
 if p.exists():assert json.loads(p.read_text())==value,'Use a new checkpoint when training support changes'
 else:write(p,value)
 return p

def allowed(checkpoint,values):
 p=support_path(checkpoint)
 if not p.exists():return None,None
 meta=json.loads(p.read_text());assert meta['checkpoint_sha256']==sha(checkpoint) and meta['class_ids']==CLASS_IDS
 support=np.asarray(meta['support'],bool);assert support.shape==(5,6,8,8)
 valid=values['valid'].astype(bool);out=np.zeros((4096,5),bool);out[valid,0]=True
 local=values['local'];head=np.flatnonzero(valid&(values['region']%6==0))
 for i in head:
  f=int(local[i,42:48].argmax());x,y=np.rint((local[i,49:51]+1)*4-.5).astype(int)
  out[i,1:]=support[1:,f,y,x]
 return out,sha(p)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['checkpoint','data','cache']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();print(build(a.checkpoint,a.data,a.cache))
