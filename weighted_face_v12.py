"""Confidence-aware joint loss for unchanged and weakly augmented face decoders."""
import torch
import torch.nn.functional as F
from face_decoder_v9 import pack as base_pack, binary_logits, INTERNAL_IDS


def pack(rows,device='cpu'):
    p=base_pack(rows,device)
    weights=[float(r['row'].get('binary_training_weight',1.)) for r in rows]
    assert all(0<=x<=1 for x in weights)
    p['case_weights']=p['y'].new_tensor(weights)
    return p


def loss(net,p,epoch=200,balanced=False):
    logits=net(p);y=p['y'];b=binary_logits(logits)
    cw=p.get('case_weights',y.new_ones(p['cases']));weight=cw[p['owner']]
    pos=(y*weight).sum();total_weight=weight.sum()
    pw=((total_weight-pos)/pos.clamp_min(1)).sqrt().clamp(1,3)
    per=F.binary_cross_entropy_with_logits(b,y,pos_weight=pw,reduction='none')
    counts=p['region_counts'].to(y);occupied=(counts.reshape(p['cases'],12)>0).sum(1)
    binary=(per*weight/counts[p['region']].clamp_min(1)/occupied[p['owner']]/p['cases']).sum()
    known=((p['semantic']>0)&(p['fine_weights']>0))|p['reviewed'];fine=logits.sum()*0
    if known.any():
        matched=p['semantic'][known,None]==p['semantic'].new_tensor(INTERNAL_IDS)[None,:]
        assert matched.any(1).all(),'Unsupported semantic class'
        target=matched.long().argmax(1)
        confidence=torch.where(p['reviewed'][known],torch.ones_like(y[known]),p['fine_weights'][known])
        n=torch.bincount(target,minlength=len(INTERNAL_IDS)).to(y)
        ce=F.cross_entropy(logits[known],target,reduction='none')
        if balanced:
            fine=(ce*confidence/n[target].clamp_min(1)).sum()/(n>0).sum().clamp_min(1)
        else:
            balance=(n.sum()/n.clamp_min(1)).sqrt().clamp(max=12);balance=balance/balance[n>0].mean()
            fine=(ce*confidence*balance[target]).sum()/balance[target].sum().clamp_min(1)
    return binary+.6*min(1.,epoch/20)*fine,binary,fine


def fold_training(human,auxiliary,fold,duplicate_links,augmented):
    """Keep historical held IDs; purge known same-source training counterparts."""
    held=set(fold['held_ids']);blocked=set(held)
    for a,b in duplicate_links:
        if a in held:blocked.add(b)
        if b in held:blocked.add(a)
    train=[r for r in human if r['row']['id'] in fold['train_ids'] and r['row']['id'] not in blocked]
    if augmented:
        train += [r for r in auxiliary if r['row'].get('training_eligible') is True and r['row']['id'] not in blocked]
    assert not {r['row']['id'] for r in train}&held
    assert not {r['row']['group'] for r in train}&{r['row']['group'] for r in human if r['row']['id'] in held}
    return train,sorted(set(fold['train_ids'])-set(r['row']['id'] for r in train))
