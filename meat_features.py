"""Label-free input features: local face texture and frozen multi-view DINOv2."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from uv_annotation_format import skin_parts
from uv_visibility import POLICY, visible_skin
from local_paths import dependency_path

ROOT = Path(__file__).resolve().parent
DINO = dependency_path('dino_source', Path.home() / '.cache/torch/hub/facebookresearch_dinov2_main')
BACKBONE = dependency_path('backbone', Path.home() / '.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth')
RENDERER = dependency_path('renderer', ROOT.parent / 'differentiable_minecraft_renderer/differentiable_renderer.py')
ASSETS = dependency_path('assets', ROOT / 'data/renderer_assets')
VIEWS = ['front_left','back_left','front_right','back_right','head_above_front','head_above_back']
PARTS = ['head','body','rightArm','leftArm','rightLeg','leftLeg']
FACES = ['right','left','top','bottom','front','back']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path, value):
    path=Path(path); tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n'); tmp.replace(path)


def provenance():
    return dict(version='frozen_dino_local_orthographic_v2',visibility_policy=POLICY,backbone=str(BACKBONE),backbone_sha256=sha(BACKBONE),
                dino_source=str(DINO), dino_code={str(p.relative_to(DINO)):sha(p) for p in sorted((DINO/'dinov2').rglob('*.py'))},
                renderer=str(RENDERER), renderer_sha256=sha(RENDERER),assets=str(ASSETS),
                mapping_sha256={v:sha(ASSETS/(v+'_mapping.pt')) for v in VIEWS},views=VIEWS,
                modes=['combined','inner','outer'],full_size=[448,224],head_size=[224,224],
                projection='fixed seed 7919 Gaussian 1024 -> 128; no fitting to evaluation data',
                local='face-isolated 3x3 RGBA, part/face/layer/xy, face RGB mean/std; no targets',
                code={f:sha(ROOT/f) for f in ['meat_features.py','uv_annotation_format.py','uv_visibility.py','local_paths.py']})


def read_skin(path):
    im=Image.open(path)
    if im.size != (64,64):raise ValueError('Input must be a 64x64 UV skin')
    a=np.array(im.convert('RGBA')).copy()
    parts,valid,model=skin_parts(a.tobytes())
    if model != 'classic':raise ValueError('This baseline has only validated classic mappings; slim is unsupported')
    if not np.isin(a[...,3],[0,255]).all():raise ValueError('Only binary alpha is supported')
    shown,valid,_=visible_skin(a.tobytes(),parts,valid)
    a=np.frombuffer(shown,np.uint8).reshape(64,64,4).copy()
    valid=np.frombuffer(valid,np.uint8).astype(bool).copy().reshape(64,64)
    a[~valid]=0
    return a,parts,valid


def local_features(a,parts,valid):
    # Clamp neighborhoods to the same cuboid face, never across unrelated atlas seams.
    features=np.zeros((64,64,57),np.float32); region=np.full((64,64),-1,np.int64)
    arr=a.astype(np.float32)/255
    for part in parts:
        pi=PARTS.index(part['name'])
        for face,box in part['faces'].items():
            u,v,w,h,*_=box;fi=FACES.index(face)
            tile=arr[v:v+h,u:u+w].copy();visible=valid[v:v+h,u:u+w]
            tile[~visible]=0
            pad=np.pad(tile,((1,1),(1,1),(0,0)),mode='edge')
            patch=np.concatenate([pad[dy:dy+h,dx:dx+w] for dy in range(3) for dx in range(3)],-1)
            patch=(patch-.5)*2
            ctx=np.zeros((h,w,21),np.float32);ctx[...,pi]=1;ctx[...,6+fi]=1;ctx[...,12]=part['layer']
            yy,xx=np.mgrid[:h,:w];ctx[...,13]=(xx+.5)/w*2-1;ctx[...,14]=(yy+.5)/h*2-1
            colors=tile[visible,:3]
            if len(colors):ctx[...,15:18]=colors.mean(0);ctx[...,18:21]=colors.std(0)
            features[v:v+h,u:u+w]=np.concatenate([patch,ctx],-1)
            region[v:v+h,u:u+w]=pi+part['layer']*6
    features[~valid]=0
    return features.reshape(4096,57),region


def make_renderer(device='cpu'):
    spec=importlib.util.spec_from_file_location('meat_renderer',RENDERER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.DifferentiableRenderer(str(ASSETS),bg_color=(.5,.5,.5)).to(device).eval()


def make_dino(device='cuda'):
    sys.path.insert(0,str(DINO))
    from dinov2.hub.backbones import dinov2_vitl14_reg
    model=dinov2_vitl14_reg(pretrained=False)
    model.load_state_dict(torch.load(BACKBONE,map_location='cpu',weights_only=True),strict=True)
    return model.eval().requires_grad_(False).to(device)


@torch.inference_mode()
def observe(model,rgb,ids,valid,size):
    h,w=ids.shape;device=rgb.device
    image=F.interpolate(rgb[None],size,mode='bilinear',align_corners=False,antialias=True)
    image=(image-image.new_tensor([.485,.456,.406])[None,:,None,None])/image.new_tensor([.229,.224,.225])[None,:,None,None]
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        tokens=model.forward_features(image)['x_norm_patchtokens']
    fmap=tokens.float().transpose(1,2).reshape(1,1024,size[0]//14,size[1]//14)
    selected=ids[valid];counts=torch.bincount(selected,minlength=4096).float()
    y,x=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
    centers=torch.zeros(4096,2,device=device)
    for j,coord in enumerate([x,y]):centers[:,j].scatter_add_(0,selected,coord[valid].float())
    centers/=counts[:,None].clamp_min(1)
    centers[:,0]=(centers[:,0]+.5)/w*2-1;centers[:,1]=(centers[:,1]+.5)/h*2-1
    values=F.grid_sample(fmap,centers[None,:,None],align_corners=False)[0,:,:,0].T
    support=counts>0
    return F.normalize(values,dim=-1)*support[:,None],support


@torch.inference_mode()
def extract(path,model,renderer,device='cuda'):
    a,parts,valid=read_skin(path);local,region=local_features(a,parts,valid)
    uv=torch.from_numpy(a).permute(2,0,1).float().to(device)/255
    active=torch.from_numpy(valid.flatten()).to(device)
    head=torch.from_numpy(((region%6)==0).flatten()).to(device)&active
    layers=torch.from_numpy(region//6).to(device)
    ids_texture=torch.arange(4096,device=device).reshape(64,64)/4095
    sums=torch.zeros(4096,1024,device=device);count=torch.zeros(4096,device=device);audit=[]
    for mode in ['combined','inner','outer']:
        skin=uv.clone()
        if mode=='inner':skin[3,layers==1]=0
        if mode=='outer':skin[3,layers==0]=0
        encoded_skin=skin.clone();encoded_skin[:3]=ids_texture[None]
        for view in VIEWS:
            rendered=renderer.forward_view(skin[None],view)[0]
            encoded=renderer.forward_view(encoded_skin[None],view)[0]
            ids=(encoded[0]*4095).round().long().clamp(0,4095)
            opaque=rendered[3]>.999
            matched=(rendered[:3]-skin[:3].flatten(1)[:,ids]).abs().amax(0)<.002
            good=opaque & matched & active[ids]
            ratio=float(good.sum()/opaque.sum().clamp_min(1))
            audit.append(dict(mode=mode,view=view,opaque_pixels=int(opaque.sum()),mapped_fraction=ratio))
            if opaque.any() and ratio<.999:raise ValueError('UV correspondence mismatch')
            if not good.any():continue
            values,seen=observe(model,rendered[:3],ids,good,(448,224));sums+=values;count+=seen
            hp=good & head[ids]
            if hp.any():
                yy,xx=hp.nonzero().T;y0=max(0,int(yy.min())-8);y1=min(ids.shape[0],int(yy.max())+9)
                x0=max(0,int(xx.min())-8);x1=min(ids.shape[1],int(xx.max())+9)
                values,seen=observe(model,rendered[:3,y0:y1,x0:x1],ids[y0:y1,x0:x1],hp[y0:y1,x0:x1],(224,224))
                sums+=values;count+=seen
    values=F.normalize(sums/count[:,None].clamp_min(1),dim=-1)
    return dict(local=local,dino=values.cpu().numpy().astype(np.float16),coverage=count.cpu().numpy(),
                valid=valid.flatten(),region=region.flatten(),rgba=a),audit


def matrix(cache,variant):
    local=torch.from_numpy(cache['local']).float()
    if variant=='local':return local
    if variant!='dino_local':raise ValueError(variant)
    g=torch.Generator().manual_seed(7919)
    proj=torch.randn(1024,128,generator=g)/128**.5
    f=F.normalize(torch.from_numpy(cache['dino']).float()@proj,dim=-1)*128**.5
    return torch.cat([local,f,torch.from_numpy((cache['coverage']>0).astype(np.float32))[:,None]],1)


def save_tensor(path,t):
    a=t.detach().float().clamp(0,1).cpu().numpy().transpose(1,2,0)
    Image.fromarray((a*255).round().astype(np.uint8)).save(path)


@torch.inference_mode()
def render_views(rgba,renderer,path,mask=None):
    t=torch.from_numpy(rgba.copy()).permute(2,0,1).float()/255
    if mask is not None:
        tint=torch.tensor([[.2,.35,.9],[.05,.95,.35]])[torch.as_tensor(mask).long().reshape(64,64)]
        t[:3]=.35*t[:3]+.65*tint.permute(2,0,1)
    save_tensor(path,torch.cat([renderer.forward_view(t[None],v)[0,:3] for v in VIEWS[:2]],2))
