import json,time,copy
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from meat_features import ROOT,write
from face_features_v3 import matrix
from face_decoder_v5 import FaceDecoder,pack,loss
from fine_supervision import weights

def rows():
 data=ROOT/'data/face_v5_20261101';cache=ROOT/'cache/semantic_v3_20261101';result=[]
 for r in json.loads((data/'manifest.json').read_text())['rows']:
  c=dict(np.load(cache/(r['id']+'.npz')));sm=json.loads((data/r['id']/'semantic.json').read_text())
  result.append(dict(row=r,cache=c,x=matrix(c),valid=torch.from_numpy(c['valid'].astype(bool)),y=torch.from_numpy((np.array(Image.open(data/r['id']/'meat.png'))[...,0].flatten()>0).astype(np.float32)),semantic=torch.tensor(sm['labels']),fine_weights=torch.from_numpy(weights(sm))))
 return result

if __name__=='__main__':
 from model_service import lock
 torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.manual_seed(42)
 with lock(ROOT/'service_runtime/gpu.lock'):
  r=rows();p=pack(r,'cuda');n=FaceDecoder(p['x'].shape[-1]).cuda().eval()
  assert not any('dino' in k.lower() for k,_ in n.named_parameters())
  with torch.no_grad():
   a=n(p)[0];other=dict(p);other['y']=1-p['y'];other['semantic']=torch.zeros_like(p['semantic']);b=n(other)[0];assert torch.equal(a,b)
   one=pack([r[0]],'cuda');single=n(one)[0];batch_kernel_diff=float((single-a[:len(single)]).abs().max());perturbed=dict(p);perturbed['x']=p['x'].clone();perturbed['x'][len(single):]=torch.randn_like(p['x'][len(single):]);changed=n(perturbed)[0];assert torch.equal(a[:len(single)],changed[:len(single)])
  blank=dict(p);blank['semantic']=torch.zeros_like(p['semantic']);blank['fine_weights']=torch.zeros_like(p['fine_weights']);total,binary,fine=loss(n,blank,'face_fine',True);assert fine.item()==0;assert total.item()==binary.item()
  n.zero_grad();total,binary,fine=loss(n,p,'face_fine',True);fine.backward();assert n.semantic.weight.grad.abs().sum()>0 and n.face_input.weight.grad.abs().sum()>0
  assert torch.isfinite(total)
  opt=torch.optim.AdamW(n.parameters(),lr=.002);torch.cuda.synchronize();start=time.time()
  for _ in range(5):n.train();opt.zero_grad();l=loss(n,p,'face_fine');l.backward();opt.step()
  torch.cuda.synchronize();elapsed=(time.time()-start)/5
  report=dict(dino_frozen=True,trainable_parameters=sum(v.numel() for v in n.parameters()),unknown_fine_loss_zero=True,fine_gradient_to_face_decoder=True,no_label_input=True,no_cross_case_attention=True,batch_kernel_logit_difference=batch_kernel_diff,seconds_per_epoch=elapsed,peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,finite=True)
  write(ROOT/'experiments/face_v5_preflight.json',report);print(json.dumps(report))
