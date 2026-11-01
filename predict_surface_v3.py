"""UV-only inference: never reads annotation files; no finetuning on inference input."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from meat_features import sha,make_dino,make_renderer,extract,read_skin
from face_features_v3 import extract_faces,matrix,provenance
from surface_model_v3b import SurfaceHead as SurfaceHeadV3,pack,forward
from surface_model_v4 import SurfaceHead as SurfaceHeadV4
from train_supervised import MeatHead
from uv_visibility import annotation_mask
from semantic_preselection import decode_categories

def prediction_details(checkpoint,skin,cache=None):
 ck=torch.load(checkpoint,weights_only=True,map_location='cpu');assert ck.get('model_module') in {'surface_model_v3b','surface_model_v4','face_decoder_v5b'};assert ck['feature_provenance']==provenance(),'Feature implementation differs from checkpoint; use frozen implementation'
 values=None;skinsha=sha(skin)
 if cache and (cache/'provenance.json').exists() and json.loads((cache/'provenance.json').read_text())==ck['feature_provenance']:
  for m in cache.glob('*.json'):
   meta=json.loads(m.read_text());p=m.with_suffix('.npz')
   if meta.get('skin_sha256')==skinsha and p.exists() and meta.get('sha256')==sha(p):values=dict(np.load(p,allow_pickle=False));break
 if values is None:
  model=make_dino();renderer=make_renderer('cuda');values,_=extract(skin,model,renderer)
  a,parts,valid=read_skin(skin);f,c,audit=extract_faces(a,parts,valid,model);values.update(face_dino=f,face_coverage=c)
  del model,renderer;torch.cuda.empty_cache()
 facial=ck['model_module']=='face_decoder_v5b'
 if facial:
  from face_decoder_v5b import FaceDecoder,pack as face_pack
  net=FaceDecoder(ck['input_dim'])
 else:
  SurfaceHead=SurfaceHeadV4 if ck['model_module']=='surface_model_v4' else SurfaceHeadV3
  net=(MeatHead(ck['input_dim']) if ck['variant']=='all_faces' else SurfaceHead(ck['input_dim']))
 net.load_state_dict(ck['state_dict']);net.eval()
 # Dummy labels are never consumed by forward; pack only needs x, geometry, and valid.
 row=dict(x=matrix(values),valid=torch.from_numpy(values['valid'].astype(bool)),cache=values,y=torch.zeros(4096),semantic=torch.zeros(4096,dtype=torch.long))
 fine=None
 with torch.no_grad():
  packed=face_pack([row]) if facial else pack([row])
  if facial:logits,sem=net(packed)
  elif ck['variant']=='all_faces':logits=net(packed['x']);sem=None
  else:logits,sem=net(packed['x'],packed['neighbors'],packed['neighbor_mask'],packed['region'])
  p=logits.sigmoid().numpy()
  # Old eight-class auxiliary checkpoints retain binary-only compatibility.
  if sem is not None and ck['model_module'] in {'surface_model_v4','face_decoder_v5b'}:
   from semantic_schema import CLASS_IDS
   assert ck['config'].get('semantic_class_ids',CLASS_IDS if facial else None)==CLASS_IDS and sem.shape[1]==len(CLASS_IDS)
   fine=np.zeros((4096,len(CLASS_IDS)),np.float32);fine[values['valid'].astype(bool)]=sem.softmax(-1).numpy()
 valid=values['valid'].astype(bool);prob=np.zeros(4096,np.float32);prob[valid]=p
 assert np.isfinite(prob).all() and not prob[~valid].any()
 scope=valid&(values['region']%6==0)
 if facial:scope&=values['local'][:,46]>0  # This decoder's auxiliary loss is front-only.
 from semantic_support import allowed
 support,support_sha=allowed(checkpoint,values)
 # Unvalidated legacy auxiliary heads must not force every hair cell into a
 # facial class. Keep binary-only compatibility until support is prepared.
 if support is None:fine=None
 details=decode_categories(prob,fine,valid,scope,class_support=support)
 details['semantic_support_sha256']=support_sha
 details.update(fine_scope='head_front' if facial else 'head',human_label=False)
 return prob,valid,details

def probabilities(checkpoint,skin,cache=None):
 prob,valid,_=prediction_details(checkpoint,skin,cache)
 return prob,valid

def predict(checkpoint,skin,output,cache=None):
 prob,valid=probabilities(checkpoint,skin,cache);selected=np.flatnonzero((prob>=.5)&valid).tolist();output.write_bytes(annotation_mask(selected,valid.astype(np.uint8).tobytes()));return selected
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path);a=p.parse_args();torch.set_num_threads(4);predict(a.checkpoint,a.input,a.output,a.cache)
