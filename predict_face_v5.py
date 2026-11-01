"""UV-only inference: never reads annotation files; no finetuning on inference input."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from meat_features import sha,make_dino,make_renderer,extract,read_skin
from face_features_v3 import extract_faces,matrix,provenance
from face_decoder_v5 import FaceDecoder,pack,forward
from train_supervised import MeatHead
from uv_visibility import annotation_mask

def probabilities(checkpoint,skin,cache=None):
 ck=torch.load(checkpoint,weights_only=True,map_location='cpu');assert ck.get('model_module') in {'face_decoder_v5','face_decoder_v5b'};assert ck['feature_provenance']==provenance(),'Feature implementation differs from checkpoint; use frozen implementation'
 values=None;skinsha=sha(skin)
 if cache and (cache/'provenance.json').exists() and json.loads((cache/'provenance.json').read_text())==ck['feature_provenance']:
  for m in cache.glob('*.json'):
   meta=json.loads(m.read_text());p=m.with_suffix('.npz')
   if meta.get('skin_sha256')==skinsha and p.exists() and meta.get('sha256')==sha(p):values=dict(np.load(p,allow_pickle=False));break
 if values is None:
  model=make_dino();renderer=make_renderer('cuda');values,_=extract(skin,model,renderer)
  a,parts,valid=read_skin(skin);f,c,audit=extract_faces(a,parts,valid,model);values.update(face_dino=f,face_coverage=c)
  del model,renderer;torch.cuda.empty_cache()
 for code in ['face_decoder_v5.py','surface_model_v4.py']+(['face_decoder_v5b.py'] if ck['model_module']=='face_decoder_v5b' else []):
  frozen=Path(checkpoint).parent.parent/'source'/code
  assert frozen.exists() and sha(frozen)==sha(Path(__file__).parent/code),'Use the frozen decoder implementation shipped with this checkpoint'
 net=FaceDecoder(ck['input_dim']);net.load_state_dict(ck['state_dict']);net.eval()
 # Dummy labels are never consumed by forward; pack only needs x, geometry, and valid.
 row=dict(x=matrix(values),valid=torch.from_numpy(values['valid'].astype(bool)),cache=values,y=torch.zeros(4096),semantic=torch.zeros(4096,dtype=torch.long))
 with torch.no_grad():p=forward(net,pack([row]),ck['variant']).sigmoid().numpy()
 valid=values['valid'].astype(bool);prob=np.zeros(4096,np.float32);prob[valid]=p
 assert np.isfinite(prob).all() and not prob[~valid].any()
 return prob,valid

def predict(checkpoint,skin,output,cache=None):
 prob,valid=probabilities(checkpoint,skin,cache);selected=np.flatnonzero((prob>=.5)&valid).tolist();output.write_bytes(annotation_mask(selected,valid.astype(np.uint8).tobytes()));return selected
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path);a=p.parse_args();torch.set_num_threads(4);predict(a.checkpoint,a.input,a.output,a.cache)
