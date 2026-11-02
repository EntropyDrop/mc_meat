"""Confidence-aware partial labels; explicit negatives never imply full review."""
import torch
import torch.nn.functional as F
from face_decoder_v9 import binary_logits,INTERNAL_IDS
from weighted_face_v12 import pack as old_pack,fold_training

def pack(rows,device='cpu',use_negative=True):
    p=old_pack(rows,device)
    for key,default in [('binary_cell_weights',1.),('fine_negative_weights',0.)]:
        p[key]=torch.cat([r.get(key,torch.full((4096,),default))[r['valid']] for r in rows]).to(device)
    if not use_negative:p['fine_negative_weights'].zero_()
    return p

def loss(net,p,epoch=200,balanced=False):
    logits=net(p);y=p['y'];b=binary_logits(logits)
    cw=p.get('case_weights',y.new_ones(p['cases']));mask=p.get('binary_cell_weights',torch.ones_like(y));weight=cw[p['owner']]*mask
    pos=(y*weight).sum();pw=((weight.sum()-pos)/pos.clamp_min(1)).sqrt().clamp(1,3)
    per=F.binary_cross_entropy_with_logits(b,y,pos_weight=pw,reduction='none')
    counts=y.new_zeros(p['cases']*12).scatter_add_(0,p['region'],mask)
    occupied=(counts.reshape(p['cases'],12)>0).sum(1).clamp_min(1)
    binary=(per*weight/counts[p['region']].clamp_min(1)/occupied[p['owner']]/p['cases']).sum()
    neg=p.get('fine_negative_weights',torch.zeros_like(y))
    assert not neg[p['semantic']>0].any()
    known=((p['semantic']>0)&(p['fine_weights']>0))|p['reviewed']|(neg>0);fine=logits.sum()*0
    if known.any():
        matched=p['semantic'][known,None]==p['semantic'].new_tensor(INTERNAL_IDS)[None,:]
        assert matched.any(1).all()
        target=matched.long().argmax(1)
        weak=torch.where(p['semantic'][known]>0,p['fine_weights'][known],neg[known])
        confidence=torch.where(p['reviewed'][known],torch.ones_like(weak),weak)
        n=torch.bincount(target,minlength=len(INTERNAL_IDS)).to(y);ce=F.cross_entropy(logits[known],target,reduction='none')
        if balanced:fine=(ce*confidence/n[target].clamp_min(1)).sum()/(n>0).sum().clamp_min(1)
        else:
            balance=(n.sum()/n.clamp_min(1)).sqrt().clamp(max=12);balance=balance/balance[n>0].mean()
            fine=(ce*confidence*balance[target]).sum()/balance[target].sum().clamp_min(1)
    return binary+.6*min(1.,epoch/20)*fine,binary,fine
