"""Pixel-balanced trusted labels with capped auxiliary supervision and a small hard-negative term."""
import torch
import torch.nn.functional as F
from face_decoder_v9 import binary_logits
from loss_face_v14 import known_targets,rare_loss

def pixel_fine(logits,p):
 known,conf,target=known_targets(p);ce=F.cross_entropy(logits,target,reduction='none');human=p['human_case'][p['owner']];gw=p['group_weight'][p['owner']];terms=[]
 for cls in range(8):
  a=known&(target==cls)&human;b=known&(target==cls)&~human
  h=(ce*a).sum()/a.sum().clamp_min(1);w=(ce*conf*b*gw).sum()/(b*gw).sum().clamp_min(1)
  terms.append(h+.5*w)
 return torch.stack(terms).mean()/1.5

def loss(net,p,epoch):
 logits=net(p);y=p['y'];mask=p['binary_cell_weights'];w=p['case_weights'][p['owner']]*mask;pos=(w*y).sum();pw=((w.sum()-pos)/pos.clamp_min(1)).sqrt().clamp(1,3)
 b=F.binary_cross_entropy_with_logits(binary_logits(logits),y,pos_weight=pw,reduction='none');counts=y.new_zeros(p['cases']*12).scatter_add_(0,p['region'],mask);occupied=(counts.reshape(p['cases'],12)>0).sum(1).clamp_min(1);binary=(b*w/counts[p['region']].clamp_min(1)/occupied[p['owner']]/p['cases']).sum()
 fine=pixel_fine(logits,p);hard=rare_loss(logits,p)
 with torch.no_grad():prior=net.base(p).softmax(-1)
 distill=F.kl_div(logits[p['head']].log_softmax(-1),prior[p['head']],reduction='batchmean')
 total=binary+.6*fine+.025*hard+.05*distill
 return total,dict(binary=binary,fine=fine,hard=hard,distill=distill)
