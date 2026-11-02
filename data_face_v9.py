"""Strict seven-class dataset and frozen feature loader."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from meat_features import sha,matrix as old_matrix
from face_features_v3 import matrix,provenance
from fine_supervision import weights as fine_weights
from semantic_schema_v3 import validate_labels,encode_labels,CLASS_IDS

def load_rows(data,cache):
 manifest=json.loads((data/'manifest.json').read_text());fp=json.loads((cache/'provenance.json').read_text());assert fp==provenance()
 rows=[]
 for r in manifest['rows']:
  d=data/r['id'];npz=cache/(r['id']+'.npz');cm=json.loads(npz.with_suffix('.json').read_text())
  assert cm['sha256']==sha(npz) and cm['skin_sha256']==r['skin_sha256'] and cm['labels_read'] is False
  for f,k in [('skin.png','skin_sha256'),('meat.png','mask_sha256'),('annotation.json','metadata_sha256'),('valid.png','valid_sha256'),('visible_skin.png','visible_skin_sha256')]:assert sha(d/f)==r[k]
  c=dict(np.load(npz));valid=c['valid'].astype(bool);assert np.array_equal(valid,np.array(Image.open(d/'valid.png')).flatten()>0)
  assert np.all(c['face_coverage'][valid]==1)
  y=(np.array(Image.open(d/'meat.png'))[...,0].flatten()>0).astype(np.float32);sem=np.zeros(4096,np.int64);fw=np.zeros(4096,np.float32)
  if r.get('semantic_sha256'):
   assert sha(d/'semantic.png')==r['semantic_sha256'] and sha(d/'semantic.json')==r['semantic_metadata_sha256']
   sm=json.loads((d/'semantic.json').read_text());sem=np.array(validate_labels(sm['labels'],valid,np.flatnonzero(y).tolist()),np.int64);fw=fine_weights(sm)
   assert (d/'semantic.png').read_bytes()==encode_labels(sem.tolist(),valid)
  reviewed=bool(r.get('fine_review_complete'))
  if reviewed:assert sm.get('fine_review_complete') is True and sm.get('reviewed_class_ids')==CLASS_IDS
  rows.append(dict(reviewed=torch.from_numpy(valid.copy()) if reviewed else torch.zeros(4096,dtype=torch.bool),row=r,cache=c,valid=torch.from_numpy(valid),y=torch.from_numpy(y),semantic=torch.from_numpy(sem),fine_weights=torch.from_numpy(fw),x=matrix(c),old_x=old_matrix(c,'dino_local')))
 return rows,fp
