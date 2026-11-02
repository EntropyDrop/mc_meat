"""UV-only seven-category inference with a conservative ordinary fallback."""
import json,importlib
from pathlib import Path
import numpy as np
import torch
from meat_features import sha,read_skin,extract,make_dino,make_renderer
from face_features_v3 import matrix,provenance,extract_faces
from face_decoder_v9 import FaceDecoder,decode as base_decode,pack as base_pack
from rare_guard import decode


def prediction_details(checkpoint,skin,cache=None,policy='calibrated'):
 assert policy == 'calibrated', 'Guarded release supports calibrated policy only'
 ck=torch.load(checkpoint,map_location='cpu',weights_only=True);assert ck['schema']=='mc-meat-rare-guarded-v1'
 module=importlib.import_module(ck['parent_model_module']);initialize=module.initialize;scores=module.scores
 assert ck['feature_provenance']==provenance(),'Feature provenance mismatch'
 values=None;digest=sha(skin)
 if cache and (cache/'provenance.json').exists() and json.loads((cache/'provenance.json').read_text())==ck['feature_provenance']:
  for meta in cache.glob('*.json'):
   m=json.loads(meta.read_text());p=meta.with_suffix('.npz')
   if m.get('skin_sha256')==digest and p.exists() and m.get('sha256')==sha(p):values=dict(np.load(p));break
 if values is None:
  dino=make_dino();renderer=make_renderer('cuda');values,_=extract(skin,dino,renderer)
  a,parts,valid=read_skin(skin);f,count,_=extract_faces(a,parts,valid,dino);values.update(face_dino=f,face_coverage=count)
  del dino,renderer;torch.cuda.empty_cache()
 valid=values['valid'].astype(bool);assert valid.any(),'No visible skin cells'
 row=dict(cache=values,x=matrix(values),valid=torch.from_numpy(valid),y=torch.zeros(4096),semantic=torch.zeros(4096,dtype=torch.long),reviewed=torch.zeros(4096,dtype=torch.bool))
 anchor=ck['anchor'];base=FaceDecoder(anchor['input_dim']);base.load_state_dict(anchor['state_dict']);base.eval()
 with torch.inference_mode():
  logits=base(base_pack([row]));q=np.zeros((4096,8),np.float32);q[valid]=logits.softmax(-1).numpy()
 result=base_decode(q,valid,valid&(values['region']%6==0),anchor['config']['fine_threshold'])
 net=initialize(anchor);net.load_state_dict(ck['state_dict']);net.eval();score,presence=scores(net,row,'cpu')
 thresholds=ck['thresholds'] if policy=='calibrated' else [.5]*3
 output=decode(result['labels'],score,valid,thresholds,ck['calibration_enabled']);output.update(policy=policy,rare_thresholds=thresholds,rare_presence=presence.tolist(),skin_sha256=digest,checkpoint_sha256=sha(checkpoint))
 return output,score,valid

if __name__=='__main__':
 import argparse
 from uv_visibility import annotation_mask
 from semantic_schema_v3 import encode_labels
 p=argparse.ArgumentParser()
 for n in ['checkpoint','input','output','cache']:p.add_argument('--'+n,type=Path,required=n!='cache')
 p.add_argument('--policy',choices=['calibrated'],default='calibrated');a=p.parse_args();torch.set_num_threads(4)
 assert not a.output.exists(),'Use a new output directory';a.output.mkdir(parents=True)
 result,score,valid=prediction_details(a.checkpoint,a.input,a.cache,a.policy)
 (a.output/'prediction.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 (a.output/'meat.png').write_bytes(annotation_mask(result['selected'],valid.astype(np.uint8).tobytes()))
 (a.output/'semantic.png').write_bytes(encode_labels(result['labels'],valid))
 np.save(a.output/'rare_scores.npy',score)
