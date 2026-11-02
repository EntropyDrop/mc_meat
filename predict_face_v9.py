"""UV-only inference for the eight-state internal / seven-category UI decoder."""
import json
from pathlib import Path
import numpy as np
import torch
from meat_features import sha,read_skin,extract,make_dino,make_renderer
from face_features_v3 import matrix,provenance,extract_faces
from face_decoder_v9 import FaceDecoder,pack,decode

def prediction_details(checkpoint,skin,cache=None,policy='fixed'):
 assert policy=='fixed','This experiment uses the predeclared fixed threshold'
 ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
 assert ck['schema']=='mc-meat-face-v9' and ck['model_module']=='face_decoder_v9'
 assert ck['feature_provenance']==provenance(),'Frozen features no longer match checkpoint'
 values=None;skinsha=sha(skin)
 if cache and (cache/'provenance.json').exists() and json.loads((cache/'provenance.json').read_text())==ck['feature_provenance']:
  for meta in cache.glob('*.json'):
   m=json.loads(meta.read_text());p=meta.with_suffix('.npz')
   if m.get('skin_sha256')==skinsha and p.exists() and m.get('sha256')==sha(p):values=dict(np.load(p));break
 if values is None:
  model=make_dino();renderer=make_renderer('cuda');values,_=extract(skin,model,renderer)
  a,parts,valid=read_skin(skin);f,count,_=extract_faces(a,parts,valid,model);values.update(face_dino=f,face_coverage=count)
  del model,renderer;torch.cuda.empty_cache()
 valid=values['valid'].astype(bool);head=valid&(values['region']%6==0)
 if not valid.any():raise ValueError('Input has no visible cells')
 net=FaceDecoder(ck['input_dim'],facial=ck['variant']=='face_v9');net.load_state_dict(ck['state_dict']);net.eval()
 # Labels are dummy placeholders used only by the training pack interface.
 row=dict(cache=values,x=matrix(values),valid=torch.from_numpy(valid),y=torch.zeros(4096),semantic=torch.zeros(4096,dtype=torch.long))
 with torch.inference_mode():logits=net(pack([row]));q=np.zeros((4096,8),np.float32);q[valid]=logits.softmax(-1).numpy()
 result=decode(q,valid,head,ck['config']['fine_threshold'])
 result.update(human_label=False,fine_scope='all valid cells; six-face head refinement',internal_other=True,preselection_policy=policy)
 return q[:,[1,2,3,6,7]].sum(-1),valid,result

if __name__=='__main__':
 import argparse
 from uv_visibility import annotation_mask
 from semantic_schema_v3 import encode_labels
 p=argparse.ArgumentParser()
 for n in ['checkpoint','input','output','cache']:p.add_argument('--'+n,type=Path,required=n!='cache')
 p.add_argument('--policy',choices=['fixed'],default='fixed');a=p.parse_args();torch.set_num_threads(4)
 a.output.mkdir(parents=True,exist_ok=True)
 if any((a.output/n).exists() for n in ['prediction.json','meat.png','semantic.png']):raise FileExistsError('Use a new prediction output directory')
 probability,valid,result=prediction_details(a.checkpoint,a.input,a.cache,a.policy)
 (a.output/'meat.png').write_bytes(annotation_mask(result['selected'],valid.astype('uint8').tobytes()))
 (a.output/'semantic.png').write_bytes(encode_labels(result['labels'],valid))
 (a.output/'prediction.json').write_text(json.dumps(result,indent=2))
