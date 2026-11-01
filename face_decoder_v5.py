"""Frozen-DINO surface backbone + dedicated two-layer facial spatial decoder.

Only valid facial cells participate in convolution/attention. Each skin has its
own 8x8 inner/outer face grids and twelve head-face context tokens. No UV-atlas
adjacency, labels, other skins, or body pixels enter facial attention.
"""
import torch
from torch import nn
import torch.nn.functional as F
from surface_model_v4 import SurfaceHead,pack as surface_pack
from semantic_schema import CLASS_IDS

class FaceDecoder(nn.Module):
 def __init__(self,dim,hidden=48):
  super().__init__();self.surface=SurfaceHead(dim,hidden)
  self.face_input=nn.Linear(dim,hidden)
  self.local=nn.Sequential(nn.Conv2d(hidden+1,hidden,3,padding=1),nn.GELU(),nn.Conv2d(hidden,hidden,3,padding=1))
  self.position=nn.Linear(9,hidden,bias=False)
  self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(hidden,4,hidden*2,dropout=.1,batch_first=True,norm_first=True) for _ in range(2)])
  self.norm=nn.LayerNorm(hidden)
  self.binary=nn.Linear(hidden,1);self.semantic=nn.Linear(hidden,len(CLASS_IDS))
  # Start from a surface prediction; learn facial corrections rather than discard it.
  nn.init.zeros_(self.binary.weight);nn.init.zeros_(self.binary.bias)
 def forward(self,p):
  binary,semantic=self.surface(p['x'],p['neighbors'],p['neighbor_mask'],p['region'])
  h=self.face_input(p['x']);indices=p['front_indices'];mask=p['front_mask'];batch=indices.shape[0]
  face=h[indices.clamp_min(0)]*mask[...,None]
  grid=face.reshape(batch*2,8,8,-1).permute(0,3,1,2)
  valid=mask.reshape(batch*2,1,8,8)
  local=self.local(torch.cat([grid,valid.to(grid)],1))*valid
  face=face+local.permute(0,2,3,1).reshape(batch,128,-1)
  # Sorted segment mean is deterministic and excludes hidden/transparent cells.
  head=h[p['head_indices']]
  context=torch.segment_reduce(head[p['head_order']],reduce='mean',lengths=p['head_counts']).nan_to_num(0).reshape(batch,12,-1)
  tok=torch.cat([face,context],1)+self.position(p['token_position'])
  padding=torch.cat([~mask,p['head_counts'].reshape(batch,12)==0],1)
  for block in self.blocks:tok=block(tok,src_key_padding_mask=padding)
  face=self.norm(tok[:,:128])[mask];ids=indices[mask]
  binary=binary.clone();semantic=semantic.clone()
  binary[ids]=binary[ids]+self.binary(face).squeeze(-1)
  semantic[ids]=self.semantic(face)
  return binary,semantic

def pack(rows,device='cpu'):
 p=surface_pack(rows);indices=[];masks=[];pos=[];head_indices=[];head_group=[];weights=[];offset=0
 for j,r in enumerate(rows):
  ids=r['valid'].nonzero().flatten();lookup=torch.full((4096,),-1,dtype=torch.long);lookup[ids]=torch.arange(len(ids))+offset
  front=torch.tensor([(8+y)*64+u+x for u in (8,40) for y in range(8) for x in range(8)])
  inds=lookup[front];indices.append(inds);masks.append(inds>=0)
  geom=torch.zeros((140,9));geom[:128,4]=1
  for k in range(128):geom[k,6:]=torch.tensor([k//64,(k%8)/7,(k%64//8)/7])
  for k in range(12):geom[128+k,k%6]=1;geom[128+k,6:]=torch.tensor([k//6,.5,.5])
  pos.append(geom)
  local=torch.from_numpy(r['cache']['local'][ids.numpy()]);regions=torch.from_numpy(r['cache']['region'][ids.numpy()]);head=regions%6==0
  head_indices.append(torch.arange(len(ids))[head]+offset)
  head_group.append(local[head,42:48].argmax(1)+(regions[head]//6)*6+j*12)
  weights.append(r.get('fine_weights',torch.zeros(4096))[ids]);offset+=len(ids)
 group=torch.cat(head_group).long();order=torch.argsort(group,stable=True)
 p.update(front_indices=torch.stack(indices),front_mask=torch.stack(masks),token_position=torch.stack(pos),head_indices=torch.cat(head_indices),head_order=order,head_counts=torch.bincount(group,minlength=len(rows)*12),fine_weights=torch.cat(weights))
 return {k:v.to(device) for k,v in p.items()}

def forward(net,p,variant):return net(p)[0]

def loss(net,p,variant,details=False):
 logits,sem=net(p);y=p['y'];pos=y.sum();pw=((len(y)-pos)/pos.clamp_min(1)).sqrt().clamp(1,5)
 per=F.binary_cross_entropy_with_logits(logits,y,pos_weight=pw,reduction='none');count=torch.bincount(p['owner']).to(y)
 binary=(per/count[p['owner']]).sum()/len(count)
 known=(p['semantic']>0)&(p['fine_weights']>0)
 fine=sem.sum()*0
 if variant=='face_fine' and known.any():
  target=(p['semantic'][known,None]==p['semantic'].new_tensor(CLASS_IDS)[None,:]).long().argmax(1)
  n=torch.bincount(target,minlength=5).to(y);balance=(n.sum()/n.clamp_min(1)).sqrt().clamp(max=8);balance=balance/balance[n>0].mean()
  # Absolute confidence scales loss; it is not cancelled by normalization.
  weighted=F.cross_entropy(sem[known],target,reduction='none')*p['fine_weights'][known]*balance[target]
  fine=weighted.sum()/balance[target].sum().clamp_min(1)
 total=binary+.3*fine
 return (total,binary,fine) if details else total
