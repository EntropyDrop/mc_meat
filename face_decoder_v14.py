"""Cross-face head context and learned rare-category presence.

Seven public categories are unchanged. Gates redistribute rare probability to
its binary parent, so gating alone cannot change the meat probability.
"""
from collections import Counter
import torch
from torch import nn
import torch.nn.functional as F
from face_decoder_v9 import FaceDecoder,INTERNAL_IDS,MEAT_IDS
from weighted_face_v13 import pack as previous_pack

RARE_IDS=[6,9,10]
RARE_INDEX=[INTERNAL_IDS.index(c) for c in RARE_IDS]
PARENT_INDEX=[0,1,1]

def pack(rows,device='cpu'):
 p=previous_pack(rows,'cpu');n=len(rows);indices=torch.full((n,768),-1,dtype=torch.long);offset=0
 for j,r in enumerate(rows):
  ids=r['valid'].nonzero().flatten();loc=torch.from_numpy(r['cache']['local'][ids.numpy()]);reg=torch.from_numpy(r['cache']['region'][ids.numpy()]);head=reg%6==0;face=loc[:,42:48].argmax(1);xy=torch.round((loc[:,49:51]+1)*4-.5).long();pos=face*128+(reg//6)*64+xy[:,1]*8+xy[:,0]
  assert len(pos[head].unique())==int(head.sum())
  indices[j,pos[head]]=torch.arange(len(ids))[head]+offset;offset+=len(ids)
 p['head_indices']=indices;p['head_valid']=indices>=0
 groups=[r.get('row',{}).get('group',str(j)) for j,r in enumerate(rows)];counts=Counter(groups)
 p['group_weight']=torch.tensor([1/counts[g] for g in groups],dtype=torch.float32)
 p['human_case']=torch.tensor([bool(r.get('row',{}).get('evaluation_eligible',False)) for r in rows])
 return {k:v.to(device) if torch.is_tensor(v) else v for k,v in p.items()}

def gate_distribution(logits,presence,owners,head):
 q=logits.softmax(-1);out=q.clone()
 for k,(cls,parent) in enumerate(zip(RARE_INDEX,PARENT_INDEX)):
  keep=torch.where(head,presence[owners,k].sigmoid(),torch.ones_like(owners,dtype=q.dtype))
  removed=q[:,cls]*(1-keep);out[:,cls]=q[:,cls]-removed;out[:,parent]=out[:,parent]+removed
 return out.clamp_min(1e-12).log()

class ContextDecoder(nn.Module):
 def __init__(self,dim,hidden=48,gated=True):
  super().__init__();self.gated=gated;self.base=FaceDecoder(dim,hidden)
  self.pixel=nn.Sequential(nn.Linear(dim,hidden),nn.LayerNorm(hidden),nn.GELU())
  self.texture=nn.Sequential(nn.Linear(57,hidden),nn.GELU())
  self.local=nn.Sequential(nn.Conv2d(hidden+1,hidden,3,padding=1),nn.GELU(),nn.Conv2d(hidden,hidden,3,padding=1))
  self.positions=nn.Parameter(torch.randn(1,48,hidden)*.02)
  layer=nn.TransformerEncoderLayer(hidden,4,hidden*2,dropout=.1,batch_first=True,norm_first=True)
  self.context=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
  self.expert=nn.Sequential(nn.Linear(hidden*3,hidden*2),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden*2,8))
  self.presence=nn.Sequential(nn.Linear(hidden*2,hidden),nn.GELU(),nn.Linear(hidden,3))
  nn.init.zeros_(self.expert[-1].weight);nn.init.zeros_(self.expert[-1].bias)
  nn.init.zeros_(self.presence[-1].weight);nn.init.constant_(self.presence[-1].bias,1.)

 def forward_details(self,p):
  logits=self.base(p);ids=p['head_indices'];valid=p['head_valid'];n=p['cases'];x=p['x'][ids.clamp_min(0)]
  h=(self.pixel(x)+self.texture(x[...,:57]))*valid[...,None]
  grids=h.reshape(n*12,8,8,-1).permute(0,3,1,2);m=valid.reshape(n*12,1,8,8)
  grids=(grids+self.local(torch.cat([grids,m.to(grids)],1)))*m
  pooled=F.avg_pool2d(grids,4)/F.avg_pool2d(m.to(grids),4).clamp_min(.0625)
  tokens=pooled.permute(0,2,3,1).reshape(n,48,-1);tm=F.max_pool2d(m.float(),4).reshape(n,48)>0
  assert tm.any(1).all(),'Every skin must have at least one visible head cell'
  tokens=self.context(tokens+self.positions,src_key_padding_mask=~tm)*tm[...,None]
  # Eight quadrant tokens describe both layers of the front face (face index 4).
  front=(tokens[:,32:40].sum(1)/tm[:,32:40].sum(1).clamp_min(1)[:,None]);whole=tokens.sum(1)/tm.sum(1).clamp_min(1)[:,None]
  presence=self.presence(torch.cat([front,whole],-1))
  expanded=tokens.reshape(n*12,2,2,-1).repeat_interleave(4,1).repeat_interleave(4,2).reshape(n,768,-1)
  refined=grids.permute(0,2,3,1).reshape(n,768,-1)
  residual=self.expert(torch.cat([refined,expanded,front[:,None,:].expand(-1,768,-1)],-1))
  logits=logits.clone();logits[ids[valid]]=logits[ids[valid]]+residual[valid]
  if self.gated:logits=gate_distribution(logits,presence,p['owner'],p['head'])
  return logits,presence

 def forward(self,p):return self.forward_details(p)[0]

def build(dim,variant):
 if variant=='loss_v14':return FaceDecoder(dim)
 assert variant in ['context_v14','gated_v14']
 return ContextDecoder(dim,gated=variant=='gated_v14')
