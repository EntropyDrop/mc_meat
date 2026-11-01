"""Compare semantic-free DINO clustering with dino.txt UV pseudo-labels.

This is a feasibility experiment, not a trained or validated meat classifier.
Only train-pool textures are used. No original skin or human annotation is changed.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
from local_paths import dependency_path
DINO = dependency_path('dino_source', Path.home() / '.cache/torch/hub/facebookresearch_dinov2_main')
BACKBONE = dependency_path('backbone', Path.home() / '.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth')
V1 = dependency_path('probe_layout', ROOT / 'data/probe_layout')
RENDERER = dependency_path('renderer', ROOT.parent / 'differentiable_minecraft_renderer/differentiable_renderer.py')
ASSETS = dependency_path('assets', ROOT / 'data/renderer_assets')
QUEUE = dependency_path('probe_queue', ROOT / 'data/annotation_queue/train.jsonl')
VIEWS = ['front_left', 'back_left', 'front_right', 'back_right', 'head_above_front', 'head_above_back']
MEAT = ['bare facial skin', 'eyes', 'lips and mouth', 'bare hands', 'bare arms', 'bare legs', 'bare feet', 'bare torso skin']
NON_MEAT = ['hair', 'beard', 'eyebrows', 'eyeglasses', 'hat', 'crown', 'shirt', 'jacket', 'pants', 'shoes', 'gloves', 'bracelet', 'necklace', 'hood']
CONCEPTS = MEAT + NON_MEAT + ['background']
PROMPTS = ['{}', "a Minecraft character's {}", '{} on a pixel art character']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, data):
    p = Path(path); t = p.with_suffix('.tmp')
    t.write_text(json.dumps(data, ensure_ascii=False, indent=2)); t.replace(p)


def matching_agreement(labels, votes, total):
    """Agreement must support the emitted class, including when mean scores disagree with votes."""
    chosen = votes.gather(0, labels.long().clamp(0, 1)[None])[0]
    return torch.where((labels == 0) | (labels == 1), chosen / total.clamp_min(1), 0.)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def make_model():
    sys.path.insert(0, str(DINO))
    from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l
    from dinov2.hub.text.tokenizer import Tokenizer
    originals = torch.hub.load_state_dict_from_url
    allowed = {BACKBONE.name: BACKBONE}
    provenance = json.loads((ROOT / '.models/provenance.json').read_text())
    for r in provenance:
        p = ROOT / '.models' / r['file']
        assert sha(p) == r['sha256'], 'Downloaded file changed: ' + r['file']
        if p.suffix == '.pth':
            allowed[p.name] = p
    def offline_load(url, *args, **kwargs):
        name = url.rsplit('/', 1)[-1]
        if name not in allowed:
            raise ValueError('Unexpected network checkpoint request: ' + name)
        return torch.load(allowed[name], map_location='cpu', weights_only=True)
    torch.hub.load_state_dict_from_url = offline_load
    try:
        model = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().eval().requires_grad_(False).cuda()
    finally:
        torch.hub.load_state_dict_from_url = originals
    tokenizer = Tokenizer(vocab_path=str(ROOT / '.models/bpe_simple_vocab_16e6.txt.gz'))
    texts = [template.format(c) for c in CONCEPTS for template in PROMPTS]
    embeddings = []
    for i in range(0, len(texts), 12):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            e = model.encode_text(tokenizer.tokenize(texts[i:i+12]).cuda())[:, 1024:]
        embeddings.append(F.normalize(e.float(), dim=-1))
    prototypes = torch.cat(embeddings).reshape(len(CONCEPTS), len(PROMPTS), -1).mean(1)
    return model, F.normalize(prototypes, dim=-1), provenance


def save_image(path, tensor):
    a = tensor.detach().float().clamp(0, 1).cpu().numpy()
    if a.ndim == 3:
        a = a.transpose(1, 2, 0)
    Image.fromarray((a * 255).round().astype(np.uint8)).save(path)


@torch.no_grad()
def observe(model, prototypes, rgb, ids, valid, target_hw, raw_features=False):
    """Per-view averages, so big projected cells do not get extra view votes."""
    h, w = ids.shape
    image = F.interpolate(rgb[None], target_hw, mode='bilinear', align_corners=False, antialias=True)
    mean = image.new_tensor([.485, .456, .406])[None, :, None, None]
    std = image.new_tensor([.229, .224, .225])[None, :, None, None]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        cls, patches, registers = model.visual_model.get_backbone_features((image - mean) / std)
        tokens = model.visual_model.head(torch.cat([cls[:, None], registers, patches], 1))
        local = tokens[:, model.visual_model.num_register_tokens + 1:]
    hh, ww = target_hw[0] // 14, target_hw[1] // 14
    local = F.normalize(local.float(), dim=-1)
    scores = (local @ prototypes.T).transpose(1, 2).reshape(1, len(CONCEPTS), hh, ww)
    scores = F.interpolate(scores, (h, w), mode='bilinear', align_corners=False)[0]
    selected_ids = ids[valid]
    count = torch.bincount(selected_ids, minlength=4096).float()
    total = torch.zeros(len(CONCEPTS), 4096, device='cuda')
    total.scatter_add_(1, selected_ids[None].expand(len(CONCEPTS), -1), scores[:, valid])
    supported = count > 0
    total /= count.clamp_min(1)
    features = None
    if raw_features:
        yy, xx = torch.meshgrid(torch.arange(h, device='cuda'), torch.arange(w, device='cuda'), indexing='ij')
        centers = torch.zeros(4096, 2, device='cuda')
        for j, coords in enumerate([xx, yy]):
            centers[:, j].scatter_add_(0, selected_ids, coords[valid].float())
        centers /= count[:, None].clamp_min(1)
        centers[:, 0] = (centers[:, 0] + .5) / w * 2 - 1
        centers[:, 1] = (centers[:, 1] + .5) / h * 2 - 1
        fmap = patches.float().transpose(1, 2).reshape(1, 1024, hh, ww)
        features = F.grid_sample(fmap, centers[None, :, None], align_corners=False)[0, :, :, 0].T
        features = F.normalize(features, dim=-1) * supported[:, None]
    return total, supported, features


@torch.no_grad()
def case_probe(row, directory, model, prototypes, renderer, layout):
    directory.mkdir()
    path = Path(row['input_path']); assert sha(path) == row['input_sha256']
    a = np.array(Image.open(path).convert('RGBA')).copy()
    assert a.shape == (64, 64, 4) and np.isin(a[..., 3], [0, 255]).all()
    a[~layout.VALID] = 0; a[a[..., 3] == 0, :3] = 0
    uv = torch.from_numpy(a).permute(2, 0, 1).float().cuda() / 255
    active = (uv[3] > .5).flatten()
    ids_uv = torch.arange(4096, device='cuda').float().reshape(64, 64) / 4095
    outer = torch.from_numpy(layout.OUTER).cuda(); inner = torch.from_numpy(layout.INNER).cuda()
    head = torch.from_numpy((layout.REGION % 6 == 0) & layout.VALID).cuda().flatten()
    total = torch.zeros(len(CONCEPTS), 4096, device='cuda'); count = torch.zeros(4096, device='cuda')
    meat_votes = torch.zeros_like(count); seen_views = torch.zeros(6, 4096, device='cuda', dtype=torch.bool)
    raw_sum = torch.zeros(4096, 1024, device='cuda'); raw_count = torch.zeros_like(count)
    mapping_audit = []
    for mode in ['combined', 'inner', 'outer']:
        skin = uv.clone()
        if mode == 'inner': skin[3, outer] = 0
        if mode == 'outer': skin[3, inner] = 0
        id_texture = skin.clone(); id_texture[:3] = ids_uv[None]
        for vi, view in enumerate(VIEWS):
            rendered = renderer.forward_view(skin[None], view)[0]
            encoded = renderer.forward_view(id_texture[None], view)[0]
            ids = (encoded[0] * 4095).round().long().clamp(0, 4095)
            opaque = rendered[3] > .999
            matched = (rendered[:3] - skin[:3].flatten(1)[:, ids]).abs().amax(0) < .002
            valid = opaque & matched & active[ids]
            ratio = float(valid.sum() / opaque.sum().clamp_min(1))
            mapping_audit.append(dict(mode=mode, view=view, opaque_pixels=int(opaque.sum()), matching_fraction=ratio))
            if int(opaque.sum()): assert ratio > .999, 'Renderer UV correspondence mismatch'
            if mode == 'combined' and view in VIEWS[:2]: save_image(directory / (view + '.png'), rendered[:3])
            if not valid.any(): continue
            observations = [(rendered[:3], ids, valid, (448, 224), mode == 'combined')]
            hp = valid & head[ids]
            if hp.any():
                y, x = hp.nonzero().T; y0, y1 = max(0, int(y.min())-8), min(ids.shape[0], int(y.max())+9)
                x0, x1 = max(0, int(x.min())-8), min(ids.shape[1], int(x.max())+9)
                observations.append((rendered[:3, y0:y1, x0:x1], ids[y0:y1, x0:x1], hp[y0:y1, x0:x1], (224, 224), False))
            for rgb, oid, ov, size, get_raw in observations:
                score, support, raw = observe(model, prototypes, rgb, oid, ov, size, get_raw)
                total += score * support; count += support
                meat_score = score[:len(MEAT)].amax(0); other_score = score[len(MEAT):-1].amax(0)
                meat_votes += ((meat_score > other_score) & support)
                seen_views[vi] |= support
                if raw is not None: raw_sum += raw; raw_count += support
    support = (count > 0) & active
    mean = total / count.clamp_min(1)
    meat_score = mean[:len(MEAT)].amax(0); other_score = mean[len(MEAT):-1].amax(0)
    margin = meat_score - other_score; binary = margin > 0
    agreement = matching_agreement(binary.long(), torch.stack([count-meat_votes, meat_votes]), count)
    background_wins = mean[-1] >= torch.maximum(meat_score, other_score)
    labels = torch.full((4096,), 255, device='cuda', dtype=torch.uint8)
    labels[support] = binary[support].to(torch.uint8)
    # These are experimental gates, not calibrated probabilities or accuracy claims.
    reliable = support & (seen_views.sum(0) >= 2) & (agreement >= .8) & (margin.abs() >= .015) & ~background_wins
    pseudo = torch.full_like(labels, 255); pseudo[reliable] = labels[reliable]
    raw_features = F.normalize(raw_sum / raw_count[:, None].clamp_min(1), dim=-1).cpu()
    active_cpu = active.cpu().numpy(); supported_cpu = (raw_count > 0).cpu().numpy()
    Image.fromarray(a).save(directory / 'uv.png')
    Image.fromarray(labels.reshape(64,64).cpu().numpy()).save(directory / 'candidate_labels.png')
    Image.fromarray(pseudo.reshape(64,64).cpu().numpy()).save(directory / 'pseudo_labels.png')
    np.savez_compressed(directory / 'evidence.npz', margin=margin.cpu().numpy().reshape(64,64),
                        meat_votes=meat_votes.cpu().numpy().reshape(64,64),
                        agreement=agreement.cpu().numpy().reshape(64,64), observations=count.cpu().numpy().reshape(64,64),
                        distinct_views=seen_views.sum(0).cpu().numpy().reshape(64,64), active=active_cpu.reshape(64,64))
    colors = torch.tensor([[.25,.45,1.],[.15,.9,.4],[1.,.75,.1]],device='cuda')
    for title, label in [('candidate', labels), ('filtered', pseudo)]:
        index = label.long().clone(); index[index == 255] = 2
        tint = colors[index].T.reshape(3,64,64)
        overlay = uv.clone(); overlay[:3] = .35*uv[:3] + .65*tint
        views = [renderer.forward_view(overlay[None], v)[0,:3] for v in VIEWS[:2]]
        save_image(directory / (title+'_views.png'), torch.cat(views,dim=2))
    summary = dict(input_path=str(path), input_sha256=sha(path), split=row['split'], family=row['family'],
                   active_cells=int(active.sum()), observed_cells=int(support.sum()),
                   pseudo_labeled_cells=int(reliable.sum()), meat_candidate_cells=int((labels==1).sum()),
                   meat_pseudo_cells=int((pseudo==1).sum()), non_meat_pseudo_cells=int((pseudo==0).sum()),
                   pseudo_label_coverage=float(reliable.sum()/active.sum().clamp_min(1)),
                   observed_mean_agreement=float(agreement[support].mean()) if support.any() else None,
                   mapping_checks=mapping_audit, human_ground_truth=False, accuracy=None)
    write(directory/'audit.json',summary)
    return summary, raw_features, supported_cpu & active_cpu, uv.cpu()


def cluster(features, support):
    """Per-image unsupervised clusters: IDs have no meat/non-meat meaning."""
    x = features[support].float(); x = F.normalize(x,dim=1)
    centers = x[torch.linspace(0,len(x)-1,8).long()].clone()
    for _ in range(20):
        labels = (x @ centers.T).argmax(1)
        for k in range(8):
            if (labels==k).any():centers[k]=F.normalize(x[labels==k].mean(0),dim=0)
    result = torch.full((4096,),8,dtype=torch.long); result[support]=labels
    return result


def make_report(out, summaries):
    def image(path):
        return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()
    rows=[]
    for name,s in summaries.items():
        d=out/name
        rows.append('<section><h2>'+name+' · '+Path(s['input_path']).name+'</h2><p>自动候选覆盖 '+f"{s['pseudo_label_coverage']:.1%}"+'；不是准确率。未得到语义证据的格子保留未知。</p><div class="row">'+''.join('<figure><figcaption>'+label+'</figcaption><img src="'+image(d/file)+'"></figure>' for label,file in [('原始皮肤前后视图','original_views.png'),('普通 DINOv2 聚类 · 颜色无类别含义','cluster_views.png'),('dino.txt 二类候选 · 绿=meat，蓝=非 meat','candidate_views.png'),('多视角过滤 · 黄=未知/不参与监督','filtered_views.png')])+'</div></section>')
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MC Meat · 无人工标注可行性</title><style>body{font:16px system-ui;background:#f5f6f8;color:#202936;margin:30px auto;max-width:1500px;padding:0 20px}section{background:white;padding:20px;border-radius:14px;margin:24px 0}.row{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;image-rendering:pixelated}figcaption{min-height:50px;font-size:14px}p{line-height:1.7}@media(max-width:850px){.row{grid-template-columns:repeat(2,minmax(0,1fr))}}</style><h1>MC Meat：无人工标注的可行性实验</h1><p>8 个训练池样本；没有人工逐格标签，也没有训练新的 meat 权重。普通 DINOv2 聚类只展示相似纹理；dino.txt 依赖预训练的图文语义。绿色是自动肉身候选，蓝色是自动非肉身候选，黄色表示未被过滤规则接受。覆盖率和跨视图一致率均不是准确率；一致的预测也可能一起出错。</p>'''+''.join(rows)+'</html>'
    (out/'probe_report_portable.html').write_text(html)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=8);a=p.parse_args()
    assert not a.output.exists(),'Preserve prior experiment outputs'
    a.output.mkdir(parents=True)
    torch.set_num_threads(4);torch.manual_seed(42);torch.set_float32_matmul_precision('high')
    rows=[json.loads(x) for x in QUEUE.read_text().splitlines()][:a.limit]
    assert all(r['split']=='train' for r in rows)
    layout=load_module('mc_probe_layout',V1/'layout.py')
    renderer_module=load_module('mc_probe_renderer',RENDERER)
    renderer=renderer_module.DifferentiableRenderer(str(ASSETS),bg_color=(.5,.5,.5)).cuda()
    model,prototypes,provenance=make_model()
    write(a.output/'configuration.json',dict(concepts=CONCEPTS,meat_count=len(MEAT),prompts=PROMPTS,views=VIEWS,
        input_queue=str(QUEUE),queue_sha256=sha(QUEUE),renderer_sha256=sha(RENDERER),backbone_sha256=sha(BACKBONE),
        dinotxt_code_sha256=sha(DINO/'dinov2/hub/dinotxt.py'),weights=provenance,script_sha256=sha(__file__),
        human_labels_used=0,thresholds=dict(agreement=.8,cosine_margin=.015,distinct_views=2),thresholds_calibrated=False))
    summaries={};start=time.time()
    palette=torch.tensor([[.8,.1,.1],[.1,.7,.2],[.2,.3,1],[.9,.7,.1],[.8,.2,.9],[.1,.8,.9],[.9,.4,.1],[.4,.3,.7],[.65,.65,.65]])
    for i,row in enumerate(rows):
        name=f'case-{i:02d}';write(a.output/'status.json',dict(stage='probe',done=i,total=len(rows),case=name))
        s,features,support,uv=case_probe(row,a.output/name,model,prototypes,renderer,layout)
        clusters=cluster(features,support);color=uv.clone();color[:3]=palette[clusters].T.reshape(3,64,64)
        rendered=[renderer.forward_view(color[None].cuda(),v)[0,:3] for v in VIEWS[:2]]
        save_image(a.output/name/'cluster_views.png',torch.cat(rendered,2))
        images=[Image.open(a.output/name/(v+'.png')).convert('RGB') for v in VIEWS[:2]]
        joined=Image.new('RGB',(sum(x.width for x in images),images[0].height));offset=0
        for im in images:joined.paste(im,(offset,0));offset+=im.width
        joined.save(a.output/name/'original_views.png')
        summaries[name]=s;write(a.output/'summary.json',summaries);make_report(a.output,summaries)
        print(json.dumps({k:v for k,v in s.items() if k!='mapping_checks'}),flush=True)
    write(a.output/'status.json',dict(stage='complete',done=len(rows),total=len(rows),seconds=time.time()-start,
        human_labels_used=0,trained_meat_weights=False,accuracy=None))


if __name__=='__main__':
    main()
