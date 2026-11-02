"""Rare facial specialists with per-skin positives and mined hard negatives."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from face_decoder_v9 import pack as base_pack
RARE_IDS=[6,9,10]

def means(h,group,size):
 order=torch.argsort(group,stable=True)
 return torch.segment_reduce(h[order],reduce='mean',lengths=torch.bincount(group,minlength=size)).nan_to_num(0)

class RareDecoder(nn.Module):
 def __init__(self,dim,hidden=48,facial=True):
  super().__init__()
  self.facial=facial
  self.presence=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Dropout(.2),nn.Linear(hidden,3))
  self.input=nn.Sequential(nn.Linear(dim,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(.1))
  self.messages=nn.ModuleList([nn.Sequential(nn.Linear(hidden*4,hidden),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden,hidden)) for _ in range(2)])
  self.norms=nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
  self.classes=nn.Linear(hidden,3)
  self.local=nn.Sequential(nn.Conv2d(hidden+1,hidden,3,padding=1),nn.GELU(),nn.Conv2d(hidden,hidden,3,padding=1))
  self.position=nn.Linear(9,hidden,bias=False)
  self.attention=nn.TransformerEncoderLayer(hidden,4,hidden*2,dropout=.1,batch_first=True,norm_first=True)
  self.head_norm=nn.LayerNorm(hidden)
  self.head_classes=nn.Linear(hidden,3)
  nn.init.zeros_(self.head_classes.weight);nn.init.zeros_(self.head_classes.bias)

 def forward(self,p):
  h=self.input(p['x']);n=p['cases']
  for message,norm in zip(self.messages,self.norms):
   adjacent=(h[p['neighbors']]*p['neighbor_mask'][...,None]).sum(1)/p['neighbor_mask'].sum(1).clamp_min(1)[:,None]
   region=means(h,p['region'],n*12)[p['region']]
   global_context=means(h,p['owner'],n)[p['owner']]
   h=norm(h+message(torch.cat([h,adjacent,region,global_context],-1)))
  logits=self.classes(h)
  if not len(p['face_indices']):return logits,self.presence(h.new_zeros((n,h.shape[-1])))
  indices=p['face_indices'];mask=p['face_mask'];pairs=len(indices)
  face=h[indices.clamp_min(0)]*mask[...,None]
  grid=face.reshape(pairs*2,8,8,-1).permute(0,3,1,2);valid=mask.reshape(pairs*2,1,8,8)
  conv=self.local(torch.cat([grid,valid.to(grid)],1))*valid
  face=face+conv.permute(0,2,3,1).reshape(pairs,128,-1)
  context=means(h,p['region'],n*12).reshape(n,12,-1)[p['pair_owner']]
  tok=torch.cat([face,context],1)+self.position(p['token_position'])
  context_mask=p['region_counts'].reshape(n,12)[p['pair_owner']]==0
  tok=self.attention(tok,src_key_padding_mask=torch.cat([~mask,context_mask],1))
  ids=indices[mask];logits=logits.clone()
  logits[ids]=logits[ids]+self.head_classes(self.head_norm(tok[:,:128][mask]))
  context=h.clone();context[ids]=context[ids]+self.head_norm(tok[:,:128][mask])
  pooled=means(context[p['head']],p['owner'][p['head']],n)
  return logits,self.presence(pooled)


def pack(rows,device='cpu'):
 p=base_pack(rows,device);p['uv']=torch.cat([r['valid'].nonzero().flatten() for r in rows]).to(device)
 p['target']=(p['semantic'][:,None]==p['semantic'].new_tensor(RARE_IDS)[None,:]).float()
 p['known']=p['head']&(p['reviewed']|(p['semantic']>0))
 return p


def initialize(anchor):
 net=RareDecoder(anchor['input_dim']);state=net.state_dict()
 for k,v in anchor['state_dict'].items():
  if k in state and state[k].shape==v.shape:state[k]=v.clone()
 net.load_state_dict(state)
 # The ordinary decoder supplies frozen global image context. Adapt only the
 # facial refinement, rare logits and image-level presence heads.
 for module in [net.input,net.messages,net.norms]:module.requires_grad_(False)
 nn.init.constant_(net.classes.bias,-2.);nn.init.zeros_(net.classes.weight)
 nn.init.constant_(net.head_classes.bias,0.);nn.init.zeros_(net.head_classes.weight)
 return net


def loss(net,p):
 logits,presence=net(p);target=p['target'];known=p['known'];owner=p['owner'];n=p['cases']
 ce=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
 pos=target*known[:,None];neg=(1-target)*known[:,None]
 lengths=torch.bincount(owner,minlength=n)
 pc=torch.segment_reduce(pos,reduce='sum',lengths=lengths)
 nc=torch.segment_reduce(neg,reduce='sum',lengths=lengths)
 sums=torch.segment_reduce(ce*pos,reduce='sum',lengths=lengths)
 present=pc>0
 positive=((sums/pc.clamp_min(1))*present).sum(0)/present.sum(0).clamp_min(1)
 all_negative=(ce*neg).sum(0)/neg.sum(0).clamp_min(1)
 # Top supervised negative cells per skin and class; bodies/ignored/unknown
 # cells cannot enter this mining operation.
 grid=logits.new_zeros((n*4096,3));grid[owner*4096+p['uv']]=ce*neg
 hard=grid.reshape(n,4096,3).topk(16,dim=1).values.sum(1)/nc.clamp(1,16)
 hard=(hard*(nc>0)).sum(0)/(nc>0).sum(0).clamp_min(1)
 image_target=present.float();image_ce=F.binary_cross_entropy_with_logits(presence,image_target,reduction='none')
 presence_loss=.5*((image_ce*image_target).sum(0)/image_target.sum(0).clamp_min(1)+(image_ce*(1-image_target)).sum(0)/(1-image_target).sum(0).clamp_min(1))
 active=(pc.sum(0)>0).float()
 per=positive+1.5*hard+.25*all_negative+.5*presence_loss
 total=(per*active).sum()/active.sum().clamp_min(1)
 return total,dict(positive=positive.mean().detach(),hard_negative=hard.mean().detach(),presence=presence_loss.mean().detach())


@torch.inference_mode()
def scores(net,row,device='cuda'):
 p=pack([row],device);cell,image=net(p);valid=row['valid'].numpy();head=valid&(row['cache']['region']%6==0)
 a=np.zeros((4096,3),np.float32);a[valid]=cell.sigmoid().float().cpu().numpy()
 image=image.sigmoid().float().cpu().numpy()[0]
 a[~head]=0
 return a*(.5+.5*image)[None,:],image


def decode(base,score,valid,thresholds):
 from face_decoder_v9 import MEAT_IDS
 labels=np.asarray(base,dtype=np.uint8).copy();valid=np.asarray(valid,bool)
 labels[labels==6]=0;labels[np.isin(labels,[9,10])]=1
 thresholds=np.asarray(thresholds,float);ratio=score/thresholds[None,:]
 winner=ratio.argmax(1);take=valid&(ratio.max(1)>=1)
 labels[take]=np.asarray(RARE_IDS,np.uint8)[winner[take]];labels[~valid]=0
 return dict(labels=labels.tolist(),selected=np.flatnonzero(np.isin(labels,MEAT_IDS)).tolist(),uncertain=[],uncertain_count=0,semantic_available=True,class_ids=[1,2,3,5,6,9,10],label_schema='mc-meat-semantics-v3',human_label=False)


def calibrate(items):
 thresholds=[];audit=[]
 for k,c in enumerate(RARE_IDS):
  value=np.concatenate([x['score'][x['valid'],k] for x in items]);truth=np.concatenate([x['truth'][x['valid']]==c for x in items]);candidates=[]
  # Search every positive score breakpoint; an image-level score can be
  # much lower than .05 without being uninformative. This uses inner OOF only.
  order=np.argsort(-value,kind='stable');v=value[order];y=truth[order];tp=np.cumsum(y);fp=np.cumsum(~y)
  ends=np.r_[np.flatnonzero(v[:-1]!=v[1:]),len(v)-1];ends=ends[v[ends]>0]
  for i in ends:
   t=float(v[i]);a=int(tp[i]);b=int(fp[i]);n=int(truth.sum()-a)
   candidates.append(dict(threshold=t,tp=a,fp=b,fn=n,precision=a/max(a+b,1),recall=a/max(a+n,1),f1=2*a/max(2*a+b+n,1),f05=1.25*a/max(1.25*a+b+.25*n,1)))
  candidates.append(dict(threshold=1.001,tp=0,fp=0,fn=int(truth.sum()),precision=0.,recall=0.,f1=0.,f05=0.))
  feasible=[x for x in candidates if x['precision']>=.35 and x['tp']>0]
  best=max(feasible,key=lambda x:(x['f1'],x['precision'],x['threshold'])) if feasible else max(candidates,key=lambda x:(x['f05'],x['precision'],x['threshold']))
  if not truth.any():best=candidates[-1]
  thresholds.append(best['threshold']);audit.append(dict(class_id=c,**best,positive_support=int(truth.sum())))
 return thresholds,audit
