"""All 72 isolated part/layer/face orthographic texture views; no labels read.

A cuboid's face-normal orthographic view is an affine UV rectangle. We rasterize
that plane directly with its physical enlarged width/height, avoiding limb and
opposite-face occlusion. All valid UV cells receive a separate face feature.
Original multi-view context remains a separate channel, never averaged away.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from meat_features import ROOT,sha,write,read_skin,make_dino,matrix as old_matrix,provenance as old_provenance

def provenance():
 return dict(version='mc-meat-all-faces-v3',parent=old_provenance(),size=224,padding=14,geometry='exact face-normal affine UV projection; isolated part/layer; physical inflation and bottom UV flip',code={n:sha(ROOT/n) for n in ['face_features_v3.py','surface_topology_v3.json']},projection_seed=7920)

def face_views(a,parts,valid):
 for part in parts:
  for face,(u,v,w,h,*flags) in part['faces'].items():
   if not valid[v:v+h,u:u+w].any():continue
   pw,ph=w+part['inflate'],h+part['inflate'];scale=196/max(pw,ph);rw,rh=max(1,round(pw*scale)),max(1,round(ph*scale));x0=(224-rw)//2;y0=(224-rh)//2
   xx=np.minimum((np.arange(rw)+.5)/rw*w,w-1e-5).astype(int);yy=np.minimum((np.arange(rh)+.5)/rh*h,h-1e-5).astype(int)
   if flags and flags[0]:xx=w-1-xx
   if len(flags)>1 and flags[1]:yy=h-1-yy
   ids=np.full((224,224),-1,np.int64);ids[y0:y0+rh,x0:x0+rw]=(v+yy[:,None])*64+u+xx[None,:]
   good=(ids>=0)&valid.flatten()[ids.clip(0)];rgb=np.full((224,224,3),.5,np.float32);rgb[good]=a.reshape(-1,4)[ids[good],:3]/255
   yield f'{part["name"]}_{part["layer"]}_{face}',rgb,ids,good

@torch.inference_mode()
def extract_faces(a,parts,valid,model,device='cuda'):
 values=np.zeros((4096,1024),np.float32);count=np.zeros(4096,np.int16);audits=[]
 views=list(face_views(a,parts,valid))
 for start in range(0,len(views),4):
  batch=views[start:start+4];im=torch.from_numpy(np.stack([r[1] for r in batch])).permute(0,3,1,2).to(device)
  im=(im-im.new_tensor([.485,.456,.406])[None,:,None,None])/im.new_tensor([.229,.224,.225])[None,:,None,None]
  with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):tok=model.forward_features(im)['x_norm_patchtokens']
  maps=tok.float().transpose(1,2).reshape(len(batch),1024,16,16)
  for b,(name,rgb,ids,good) in enumerate(batch):
   selected=np.unique(ids[good]);centers=np.array([[np.where((ids==i)&good)[1].mean(),np.where((ids==i)&good)[0].mean()] for i in selected],np.float32)
   grid=torch.from_numpy((centers+.5)/224*2-1).to(device)[None,:,None,:]
   f=F.grid_sample(maps[b:b+1],grid,align_corners=False)[0,:,:,0].T
   f=F.normalize(f,dim=-1).cpu().numpy();values[selected]+=f;count[selected]+=1
   audits.append(dict(view=name,uv_cells=len(selected),raster_pixels=int(good.sum())))
 assert np.all(count[valid.flatten()]==1),'Each valid UV must have exactly one face-normal feature'
 assert not count[~valid.flatten()].any()
 return values.astype(np.float16),count,audits

def matrix(c):
 old=old_matrix(c,'dino_local');g=torch.Generator().manual_seed(7920);proj=torch.randn(1024,128,generator=g)/128**.5
 f=F.normalize(torch.from_numpy(c['face_dino']).float()@proj,dim=-1)*128**.5
 return torch.cat([old,f],1)

def prepare(data,old_cache,out):
 data,old_cache,out=map(Path,[data,old_cache,out]);out.mkdir(parents=True,exist_ok=True);fp=provenance()
 assert json.loads((old_cache/'provenance.json').read_text())==fp['parent']
 if (out/'provenance.json').exists():assert json.loads((out/'provenance.json').read_text())==fp
 else:write(out/'provenance.json',fp)
 rows=json.loads((data/'manifest.json').read_text())['rows'];model=None
 for j,r in enumerate(rows):
  name=r['id'];target=out/(name+'.npz');meta=out/(name+'.json')
  if target.exists() and meta.exists():
   m=json.loads(meta.read_text());assert m['skin_sha256']==r['skin_sha256'] and m['sha256']==sha(target);continue
  write(out/'status.json',dict(stage='face_features',case=name,done=j,total=len(rows)))
  source=old_cache/(name+'.npz');sm=json.loads(source.with_suffix('.json').read_text());assert sm['input_sha256']==r['skin_sha256'] and sm['cache_sha256']==sha(source)
  assert sha(data/name/'skin.png')==r['skin_sha256']
  c=dict(np.load(source));a,parts,valid=read_skin(data/name/'skin.png');assert np.array_equal(a,c['rgba'])
  if model is None:model=make_dino()
  f,count,audit=extract_faces(a,parts,valid,model);c.update(face_dino=f,face_coverage=count)
  np.savez_compressed(target,**c);write(meta,dict(skin_sha256=r['skin_sha256'],sha256=sha(target),face_views=audit,valid_cells=int(valid.sum()),covered_cells=int((count>0).sum()),labels_read=False))
  if j==0:
   # Inspectable exact texture views; original layer and UV alpha retained.
   preview=out/'first_case_views';preview.mkdir(exist_ok=True)
   for view,rgb,ids,good in face_views(a,parts,valid):Image.fromarray((rgb*255).round().astype(np.uint8)).save(preview/(view+'.png'))
  print('FACE_FEATURE',j+1,len(rows),name,flush=True)
 if model is not None:del model;torch.cuda.empty_cache()
 write(out/'status.json',dict(stage='complete',count=len(rows),all_valid_covered=True))

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--old-cache',required=True);p.add_argument('--output',required=True);a=p.parse_args();prepare(a.data,a.old_cache,a.output)
