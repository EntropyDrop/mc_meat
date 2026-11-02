"""Training-only class-balanced fine supervision; binary loss stays unchanged."""
import torch
import torch.nn.functional as F
from face_decoder_v9 import binary_logits,INTERNAL_IDS


def loss(net,p,epoch=200):
 logits=net(p);y=p['y'];b=binary_logits(logits)
 pos=y.sum();pw=((len(y)-pos)/pos.clamp_min(1)).sqrt().clamp(1,3)
 per=F.binary_cross_entropy_with_logits(b,y,pos_weight=pw,reduction='none')
 counts=p['region_counts'].to(y);occupied=(counts.reshape(p['cases'],12)>0).sum(1)
 binary=(per/counts[p['region']].clamp_min(1)/occupied[p['owner']]/p['cases']).sum()
 known=((p['semantic']>0)&(p['fine_weights']>0))|p['reviewed'];fine=logits.sum()*0
 if known.any():
  matched=p['semantic'][known,None]==p['semantic'].new_tensor(INTERNAL_IDS)[None,:]
  assert matched.any(1).all(),'Unsupported class id'
  target=matched.long().argmax(1)
  confidence=torch.where(p['reviewed'][known],torch.ones_like(y[known]),p['fine_weights'][known])
  ce=F.cross_entropy(logits[known],target,reduction='none')
  # Each present training class contributes one mean, irrespective of its cell count.
  # Absent classes are never invented; partial unknowns do not become negatives.
  means=[]
  for c in range(len(INTERNAL_IDS)):
   choose=target==c
   if choose.any():means.append((ce[choose]*confidence[choose]).mean())
  fine=torch.stack(means).mean()
 return binary+.6*min(1.,epoch/20)*fine,binary,fine
