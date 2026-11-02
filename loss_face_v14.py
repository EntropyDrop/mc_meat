"""Human-anchored source-balanced loss and training-only rare hard negatives."""
import torch
import torch.nn.functional as F
from face_decoder_v9 import binary_logits,INTERNAL_IDS
from face_decoder_v14 import RARE_IDS,RARE_INDEX

def weighted_mean(v,mask,weight):
 w=mask.to(v)*weight;return (v*w).sum()/w.sum().clamp_min(1e-8)

def source_mean(v,mask,p):
 human=p['human_case'];w=p['group_weight']
 return weighted_mean(v,mask&human,w)+.25*weighted_mean(v,mask&~human,w)

def known_targets(p):
 sem=p['semantic'];neg=p['fine_negative_weights'];known=((sem>0)&(p['fine_weights']>0))|p['reviewed']|(neg>0)
 confidence=torch.where(p['reviewed'],torch.ones_like(neg),torch.where(sem>0,p['fine_weights'],neg))
 target=(sem[:,None]==sem.new_tensor(INTERNAL_IDS)[None,:]).long().argmax(1)
 return known,confidence,target

def fine_loss(logits,p):
 known,confidence,target=known_targets(p);ce=F.cross_entropy(logits,target,reduction='none');flat=p['owner']*8+target
 sums=ce.new_zeros(p['cases']*8).scatter_add_(0,flat,ce*known*confidence).reshape(-1,8)
 counts=ce.new_zeros(p['cases']*8).scatter_add_(0,flat,known.to(ce)).reshape(-1,8)
 value=sums/counts.clamp_min(1);present=counts>0
 terms=[source_mean(value[:,k],present[:,k],p) for k in range(8)]
 return torch.stack(terms).mean()/1.25

def rare_loss(logits,p):
 known,confidence,_=known_targets(p);ids=p['head_indices'];valid=p['head_valid'];safe=ids.clamp_min(0)
 kp=known[safe]&valid;cf=confidence[safe];sem=p['semantic'][safe];q=logits.softmax(-1)[safe].clamp(1e-6,1-1e-6);parts=[]
 for c,k,neg_scale in zip(RARE_IDS,RARE_INDEX,[1.,.35,.35]):
  pos=kp&(sem==c);neg=kp&(sem!=c);posloss=-q[:,:,k].log();negloss=-(1-q[:,:,k]).log()
  pp=(posloss*cf*pos).sum(1)/pos.sum(1).clamp_min(1)
  # A fixed 16 hardest KNOWN negatives per head: no held labels or pseudo-negatives.
  hard=(negloss*cf).masked_fill(~neg,-1).topk(16,dim=1).values
  nn=hard.clamp_min(0).sum(1)/neg.sum(1).clamp(1,16)
  parts.append(source_mean(pp,pos.any(1),p)+neg_scale*source_mean(nn,neg.any(1),p))
 return torch.stack(parts).mean()/1.25

def presence_loss(logits,p):
 if logits is None:return p['x'].sum()*0
 ids=p['head_indices'].clamp_min(0);valid=p['head_valid'];sem=p['semantic'][ids];parts=[]
 for k,c in enumerate(RARE_IDS):
  positive=((sem==c)&valid&(p['fine_weights'][ids]>0)).any(1)
  # Only full human review proves whole-head absence. Partial auxiliary absence is unknown.
  negative=p['human_case']&~positive;ce=F.binary_cross_entropy_with_logits(logits[:,k],positive.float(),reduction='none')
  parts.append(source_mean(ce,positive,p)+weighted_mean(ce,negative,p['group_weight']))
 return torch.stack(parts).mean()/2

def loss(net,p,epoch=200):
 if hasattr(net,'forward_details'):logits,presence=net.forward_details(p)
 else:logits=net(p);presence=None
 y=p['y'];b=binary_logits(logits);mask=p['binary_cell_weights'];cw=p['case_weights']*p['group_weight'];w=cw[p['owner']]*mask
 pos=(y*w).sum();pw=((w.sum()-pos)/pos.clamp_min(1)).sqrt().clamp(1,3)
 per=F.binary_cross_entropy_with_logits(b,y,pos_weight=pw,reduction='none');counts=y.new_zeros(p['cases']*12).scatter_add_(0,p['region'],mask);occupied=(counts.reshape(p['cases'],12)>0).sum(1).clamp_min(1)
 binary=(per*w/counts[p['region']].clamp_min(1)/occupied[p['owner']]/p['group_weight'].sum()).sum()
 fine=fine_loss(logits,p);rare=rare_loss(logits,p);presence_term=presence_loss(presence,p);warm=min(1.,epoch/20)
 total=binary+warm*(.6*fine+.25*rare+.1*presence_term)
 return total,dict(binary=binary,fine=fine,rare=rare,presence=presence_term)
