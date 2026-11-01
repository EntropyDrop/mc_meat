"""V5 ablation: auxiliary loss restricted to the head front; binary remains full skin."""
import torch
import torch.nn.functional as F
from semantic_schema import CLASS_IDS
from face_decoder_v5 import FaceDecoder,pack,forward

def loss(net,p,variant,details=False):
 logits,sem=net(p);y=p['y'];pos=y.sum();pw=((len(y)-pos)/pos.clamp_min(1)).sqrt().clamp(1,5)
 per=F.binary_cross_entropy_with_logits(logits,y,pos_weight=pw,reduction='none');count=torch.bincount(p['owner']).to(y)
 binary=(per/count[p['owner']]).sum()/len(count)
 face=torch.zeros_like(p['semantic'],dtype=torch.bool);face[p['front_indices'][p['front_mask']]]=True
 known=(p['semantic']>0)&(p['fine_weights']>0)&face
 fine=sem.sum()*0
 if variant=='face_front_fine' and known.any():
  target=(p['semantic'][known,None]==p['semantic'].new_tensor(CLASS_IDS)[None,:]).long().argmax(1)
  n=torch.bincount(target,minlength=5).to(y);balance=(n.sum()/n.clamp_min(1)).sqrt().clamp(max=8);balance=balance/balance[n>0].mean()
  # Absolute confidence scales loss; it is not cancelled by normalization.
  weighted=F.cross_entropy(sem[known],target,reduction='none')*p['fine_weights'][known]*balance[target]
  fine=weighted.sum()/balance[target].sum().clamp_min(1)
 total=binary+.3*fine
 return (total,binary,fine) if details else total
