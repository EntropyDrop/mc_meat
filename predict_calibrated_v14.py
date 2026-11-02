"""UV-only inference with archived image features, never labels as inputs."""
from pathlib import Path
import json
import numpy as np
import torch
from face_decoder_v14 import build,pack
from calibration_v14 import decode_calibrated
from face_decoder_v9 import decode,binary_logits,INTERNAL_IDS
from face_features_v3 import matrix,provenance,extract_faces
from meat_features import sha,read_skin,extract,make_dino,make_renderer

@torch.inference_mode()
def predict(net,r,device='cuda'):
 logits=net(pack([r],device));v=r['valid'].numpy();q=np.zeros((4096,8),np.float32);q[v]=logits.softmax(-1).float().cpu().numpy()
 prob=np.zeros(4096,np.float32);prob[v]=binary_logits(logits).sigmoid().float().cpu().numpy()
 return prob,q,decode(q,v,v&(r['cache']['region']%6==0),.5)

def prediction_details(checkpoint,skin,cache=None):
 ck=torch.load(checkpoint,map_location='cpu',weights_only=True);assert ck['schema']=='mc-meat-face-v14';calibration=json.loads((checkpoint.parent/'calibration.json').read_text());assert calibration['parent_model_sha256']==sha(checkpoint);assert ck['feature_provenance']==provenance();values=None;s=sha(skin)
 if cache and (cache/'provenance.json').exists() and json.loads((cache/'provenance.json').read_text())==ck['feature_provenance']:
  for meta in cache.glob('*.json'):
   m=json.loads(meta.read_text());p=meta.with_suffix('.npz')
   if m.get('skin_sha256')==s and p.exists() and m.get('sha256')==sha(p):values=dict(np.load(p));break
 if values is None:
  model=make_dino();renderer=make_renderer('cuda');values,_=extract(skin,model,renderer);a,parts,valid=read_skin(skin);f,count,_=extract_faces(a,parts,valid,model);values.update(face_dino=f,face_coverage=count);del model,renderer;torch.cuda.empty_cache()
 valid=values['valid'].astype(bool);assert valid.any()
 net=build(ck['input_dim'],ck['variant']);net.load_state_dict(ck['state_dict']);net.eval()
 row=dict(cache=values,x=matrix(values),valid=torch.from_numpy(valid),y=torch.zeros(4096),semantic=torch.zeros(4096,dtype=torch.long))
 prob,q,raw=predict(net,row,'cpu');details=decode_calibrated(q,valid,valid&(values['region']%6==0),calibration['thresholds']);details.update(human_label=False,model_variant='context_calibrated_v14')
 return prob,valid,details

if __name__=='__main__':
 import argparse
 from uv_visibility import annotation_mask
 from semantic_schema_v3 import encode_labels
 p=argparse.ArgumentParser()
 for n in ['checkpoint','input','output','cache']:p.add_argument('--'+n,type=Path,required=n!='cache')
 a=p.parse_args();torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=True)
 assert not any((a.output/n).exists() for n in ['prediction.json','meat.png','semantic.png'])
 probability,valid,result=prediction_details(a.checkpoint,a.input,a.cache)
 (a.output/'meat.png').write_bytes(annotation_mask(result['selected'],valid.astype('uint8').tobytes()));(a.output/'semantic.png').write_bytes(encode_labels(result['labels'],valid));(a.output/'prediction.json').write_text(json.dumps(result,indent=2))
