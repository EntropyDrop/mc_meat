"""Seven editable categories plus internal other; frozen image-only DINO features.

Reviewed zeros supervise other; partial unknown cells retain only binary loss.
Both model variants use identical joint supervision. Facial refinement can be
disabled to isolate its effect on the same grouped folds.
"""
import torch
from torch import nn
import torch.nn.functional as F
from surface_model_v4 import pack as surface_pack

INTERNAL_IDS=[0,1,2,3,5,6,9,10]
MEAT_IDS=[1,2,3,9,10]

def means(h,group,size):
 order=torch.argsort(group,stable=True)
 return torch.segment_reduce(h[order],reduce='mean',lengths=torch.bincount(group,minlength=size)).nan_to_num(0)

class FaceDecoder(nn.Module):
 def __init__(self,dim,hidden=48,facial=True):
  super().__init__()
  self.facial=facial
  self.input=nn.Sequential(nn.Linear(dim,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(.1))
  self.messages=nn.ModuleList([nn.Sequential(nn.Linear(hidden*4,hidden),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden,hidden)) for _ in range(2)])
  self.norms=nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
  self.classes=nn.Linear(hidden,len(INTERNAL_IDS))
  self.local=nn.Sequential(nn.Conv2d(hidden+1,hidden,3,padding=1),nn.GELU(),nn.Conv2d(hidden,hidden,3,padding=1))
  self.position=nn.Linear(9,hidden,bias=False)
  self.attention=nn.TransformerEncoderLayer(hidden,4,hidden*2,dropout=.1,batch_first=True,norm_first=True)
  self.head_norm=nn.LayerNorm(hidden)
  self.head_classes=nn.Linear(hidden,len(INTERNAL_IDS))
  nn.init.zeros_(self.head_classes.weight);nn.init.zeros_(self.head_classes.bias)

 def forward(self,p):
  h=self.input(p['x']);n=p['cases']
  for message,norm in zip(self.messages,self.norms):
   adjacent=(h[p['neighbors']]*p['neighbor_mask'][...,None]).sum(1)/p['neighbor_mask'].sum(1).clamp_min(1)[:,None]
   region=means(h,p['region'],n*12)[p['region']]
   global_context=means(h,p['owner'],n)[p['owner']]
   h=norm(h+message(torch.cat([h,adjacent,region,global_context],-1)))
  logits=self.classes(h)
  if not self.facial or not len(p['face_indices']):return logits
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
  return logits

def pack(rows,device='cpu'):
 p=surface_pack(rows);indices=[];masks=[];positions=[];pair_owner=[];heads=[];reviewed=[];offset=0
 for j,r in enumerate(rows):
  ids=r['valid'].nonzero().flatten();c=r['cache'];local=torch.from_numpy(c['local'][ids.numpy()]);region=torch.from_numpy(c['region'][ids.numpy()]).long();head=region%6==0
  heads.append(head);reviewed.append(r.get('reviewed',torch.zeros(4096,dtype=torch.bool))[ids])
  face_id=local[:,42:48].argmax(1);xy=torch.round((local[:,49:51]+1)*4-.5).long()
  for f in range(6):
   inds=torch.full((128,),-1,dtype=torch.long);keep=head&(face_id==f)
   pos=(region[keep]//6)*64+xy[keep,1]*8+xy[keep,0]
   assert (pos>=0).all() and (pos<128).all() and len(pos.unique())==len(pos)
   inds[pos]=torch.arange(len(ids))[keep]+offset
   if not (inds>=0).any():continue
   geom=torch.zeros(140,9);geom[:128,f]=1
   for k in range(128):geom[k,6:]=torch.tensor([k//64,(k%8)/7,(k%64//8)/7])
   # Context positions encode part and layer, distinct from within-face pixels.
   for k in range(12):geom[128+k,k%6]=1;geom[128+k,6:]=torch.tensor([k//6,-1.,-1.])
   indices.append(inds);masks.append(inds>=0);positions.append(geom);pair_owner.append(j)
  offset+=len(ids)
 p.update(cases=len(rows),face_indices=torch.stack(indices) if indices else torch.empty((0,128),dtype=torch.long),face_mask=torch.stack(masks) if masks else torch.empty((0,128),dtype=torch.bool),token_position=torch.stack(positions) if positions else torch.empty((0,140,9)),pair_owner=torch.tensor(pair_owner,dtype=torch.long),head=torch.cat(heads),reviewed=torch.cat(reviewed),region_counts=torch.bincount(p['region'],minlength=len(rows)*12))
 return {k:v.to(device) if torch.is_tensor(v) else v for k,v in p.items()}

def binary_logits(logits):
 return torch.logsumexp(logits[:,[1,2,3,6,7]],-1)-torch.logsumexp(logits[:,[0,4,5]],-1)

def loss(net,p,epoch=200):
 logits=net(p);y=p['y'];b=binary_logits(logits)
 pos=y.sum();pw=((len(y)-pos)/pos.clamp_min(1)).sqrt().clamp(1,3)
 per=F.binary_cross_entropy_with_logits(b,y,pos_weight=pw,reduction='none')
 # Equal weight per occupied part/layer, then per skin, preserves small faces.
 counts=p['region_counts'].to(y);occupied=(counts.reshape(p['cases'],12)>0).sum(1)
 binary=(per/counts[p['region']].clamp_min(1)/occupied[p['owner']]/p['cases']).sum()
 known=((p['semantic']>0)&(p['fine_weights']>0))|p['reviewed']
 fine=logits.sum()*0
 if known.any():
  target=(p['semantic'][known,None]==p['semantic'].new_tensor(INTERNAL_IDS)[None,:]).long().argmax(1)
  confidence=torch.where(p['reviewed'][known],torch.ones_like(y[known]),p['fine_weights'][known])
  n=torch.bincount(target,minlength=len(INTERNAL_IDS)).to(y)
  balance=(n.sum()/n.clamp_min(1)).sqrt().clamp(max=12);balance=balance/balance[n>0].mean()
  fine=(F.cross_entropy(logits[known],target,reduction='none')*confidence*balance[target]).sum()/balance[target].sum().clamp_min(1)
 total=binary+.6*min(1.,epoch/20)*fine
 return total,binary,fine

def decode(probability,valid,head,confidence=.5):
 import numpy as np
 v=np.asarray(valid,bool);q=np.asarray(probability);head=np.asarray(head,bool)
 assert q.shape==(4096,len(INTERNAL_IDS)) and np.isfinite(q).all() and np.allclose(q[v].sum(1),1,atol=1e-5)
 winner=np.asarray(INTERNAL_IDS,dtype=np.uint8)[q.argmax(1)];uncertain=v&(q.max(1)<confidence)
 labels=winner.copy();labels[~v|uncertain]=0
 selected=np.flatnonzero(np.isin(labels,MEAT_IDS)).tolist()
 return dict(labels=labels.tolist(),selected=selected,uncertain=np.flatnonzero(uncertain).tolist(),uncertain_count=int(uncertain.sum()),semantic_available=True,category_counts={str(c):int((labels==c).sum()) for c in INTERNAL_IDS[1:]},class_ids=INTERNAL_IDS[1:],label_schema='mc-meat-semantics-v3',fine_threshold=confidence,threshold=.5,internal_other=True)
