"""Cuboid-surface decoder. Connections never cross cases, parts, or layers."""
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

class SurfaceHead(nn.Module):
 def __init__(self,dim,hidden=48):
  super().__init__();self.input=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Dropout(.1))
  self.messages=nn.ModuleList([nn.Sequential(nn.Linear(hidden*3,hidden),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden,hidden)) for _ in range(2)])
  self.norms=nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
  self.binary=nn.Linear(hidden,1);self.semantic=nn.Linear(hidden,8)
 def forward(self,x,neighbors,neighbor_mask,region):
  h=self.input(x)
  count=torch.bincount(region,minlength=int(region.max())+1).to(h).clamp_min(1)
  for layer,norm in zip(self.messages,self.norms):
   adjacent=(h[neighbors]*neighbor_mask[...,None]).sum(1)/neighbor_mask.sum(1).clamp_min(1)[:,None]
   pooled=h.new_zeros(len(count),h.shape[1]).index_add_(0,region,h)/count[:,None]
   h=norm(h+layer(torch.cat([h,adjacent,pooled[region]],-1)))
  return self.binary(h).squeeze(-1),self.semantic(h)

def pack(rows,device='cpu'):
 topology=json.loads((Path(__file__).parent/'surface_topology_v3.json').read_text())['neighbors']
 xs=[];ys=[];ss=[];owners=[];edges=[];masks=[];regions=[];offset=0
 for j,r in enumerate(rows):
  valid=r['valid'].numpy();ids=valid.nonzero()[0];lookup=torch.full((4096,),-1,dtype=torch.long);lookup[ids]=torch.arange(len(ids))
  neighbors=torch.tensor([topology[i] if topology[i] else [i]*4 for i in ids]);mapped=lookup[neighbors];mask=mapped>=0
  edges.append(mapped.clamp_min(0)+offset);masks.append(mask);regions.append(torch.from_numpy(r['cache']['region'][ids]).long()+12*j)
  xs.append(r['x'][ids]);ys.append(r['y'][ids]);ss.append(r['semantic'][ids]);owners.append(torch.full((len(ids),),j));offset+=len(ids)
 return {k:torch.cat(v).to(device) for k,v in dict(x=xs,y=ys,semantic=ss,owner=owners,neighbors=edges,neighbor_mask=masks,region=regions).items()}

def forward(net,p,variant):
 return net(p['x']) if variant=='all_faces' else net(p['x'],p['neighbors'],p['neighbor_mask'],p['region'])[0]

def loss(net,p,variant):
 if variant=='all_faces':logits=net(p['x']);sem=None
 else:logits,sem=net(p['x'],p['neighbors'],p['neighbor_mask'],p['region'])
 y=p['y'];pos=y.sum();pw=((len(y)-pos)/pos.clamp_min(1)).sqrt().clamp(1,5)
 cells=F.binary_cross_entropy_with_logits(logits,y,pos_weight=pw,reduction='none')
 count=torch.bincount(p['owner']).to(y);weight=(1/count[p['owner']])/len(count)
 binary=(cells*weight).sum();known=p['semantic']>0
 # Unlabelled fine cells are ignored, never background/negative examples.
 fine=F.cross_entropy(sem[known],p['semantic'][known]-1) if sem is not None and known.any() else binary*0
 return binary+.3*fine
