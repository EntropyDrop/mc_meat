"""Second independent teacher; disagreement remains unlabeled, never ground truth."""
import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from probe_no_labels import ROOT,V1,VIEWS,RENDERER,load_module,sha,write,save_image,matching_agreement

# Feet is deliberately unknown: this teacher has no separate shoe category.
# Face is only a coarse proposal; the dino.txt agreement gate still applies.
POSITIVE = [1,12,13,14,16]
NEGATIVE = [2,3,4,5,6,7,8,9,10,11,17]


def report(source,out,results):
    def embed(p):return 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MC Meat · 双教师无人工标注探测</title><style>body{font:16px system-ui;background:#f5f6f8;color:#202936;margin:30px auto;max-width:1500px;padding:0 20px}section{background:white;padding:20px;border-radius:14px;margin:24px 0}.row{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;image-rendering:pixelated}figcaption{min-height:50px;font-size:14px}p{line-height:1.7}@media(max-width:850px){.row{grid-template-columns:repeat(2,minmax(0,1fr))}}</style><h1>MC Meat：能否不用人工标注？</h1><p>这 8 例用于可行性观察，均来自训练素材池。没有人工逐格真值，也没有训练新分类器。绿=自动 meat 候选，蓝=自动非 meat 候选，黄=未知。第四列只保留两个教师均通过筛选且相互一致的格子。覆盖率是自动候选的数量，不是准确率；两个教师也可能同时犯错。</p><p>人体分割模型没有单独的“鞋”类别，因此它的 feet 类保留未知。face 类也不能自动解决胡须、眉毛和眼镜边缘，必须结合局部语义证据。</p>'''
    for name,s in results.items():
        html+='<section><h2>'+name+' · '+s['file']+'</h2><p>双教师一致的候选覆盖 '+f"{s['accepted_coverage']:.1%}"+'；候选 meat '+str(s['meat_cells'])+' 格，非 meat '+str(s['non_meat_cells'])+' 格。</p><div class="row">'
        for title,path in [('原始前后视图',source/name/'original_views.png'),('dino.txt 候选',source/name/'candidate_views.png'),('人体/服饰解析候选',out/name/'parser_views.png'),('双教师过滤后的候选',out/name/'agreement_views.png')]:
            html+='<figure><figcaption>'+title+'</figcaption><img src="'+embed(path)+'"></figure>'
        html+='</div></section>'
    (out/'comparison_report_portable.html').write_text(html+'</html>')


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();a.output.mkdir(parents=True)
    torch.set_num_threads(4);torch.manual_seed(42)
    provenance=json.loads((ROOT/'.models/fashn-human-parser/provenance.json').read_text());weights=Path(provenance['path'])
    for r in provenance['files']:assert sha(weights/r['file'])==r['sha256']
    model=SegformerForSemanticSegmentation.from_pretrained(weights,local_files_only=True).cuda().eval().requires_grad_(False)
    layout=load_module('parser_probe_layout',V1/'layout.py')
    renderer=load_module('parser_probe_renderer',RENDERER).DifferentiableRenderer(str(V1/'assets'),bg_color=(.5,.5,.5)).cuda()
    source=json.loads((a.source/'summary.json').read_text());results={};start=time.time()
    colors=torch.tensor([[.25,.45,1.],[.15,.9,.4],[1.,.75,.1]],device='cuda')
    write(a.output/'configuration.json',dict(teacher=provenance,preprocessing='RGB uint8; cv2.INTER_AREA resize to 384x576; ImageNet normalization; bf16 inference',
        preprocessing_reference='https://github.com/fashn-AI/fashn-human-parser/blob/main/src/fashn_human_parser/parser.py',
        shared_helpers_sha256=sha(ROOT/'probe_no_labels.py'),
        source_probe=str(a.source),source_summary_sha256=sha(a.source/'summary.json'),positive_classes=POSITIVE,
        negative_classes=NEGATIVE,ignored_classes=[0,15],class_probability_gate=.7,agreement_gate=.8,
        distinct_views_gate=2,thresholds_calibrated=False,human_labels=0,script_sha256=sha(__file__)))
    for i,(name,entry) in enumerate(source.items()):
        d=a.output/name;d.mkdir();write(a.output/'status.json',dict(stage='probe',done=i,total=len(source)))
        uv=torch.from_numpy(np.array(Image.open(a.source/name/'uv.png'))).permute(2,0,1).float().cuda()/255
        active=(uv[3]>.5).flatten();outer=torch.from_numpy(layout.OUTER).cuda();inner=torch.from_numpy(layout.INNER).cuda()
        all_scores=torch.zeros(18,4096,device='cuda');count=torch.zeros(4096,device='cuda')
        votes=torch.zeros(2,4096,device='cuda');views_seen=torch.zeros(6,4096,dtype=torch.bool,device='cuda')
        for mode in ['combined','inner','outer']:
            skin=uv.clone()
            if mode=='inner':skin[3,outer]=0
            if mode=='outer':skin[3,inner]=0
            id_uv=skin.clone();id_uv[:3]=torch.arange(4096,device='cuda').reshape(1,64,64)/4095
            for vi,view in enumerate(VIEWS):
                rgb=renderer.forward_view(skin[None],view)[0];encoded=renderer.forward_view(id_uv[None],view)[0]
                ids=(encoded[0]*4095).round().long().clamp(0,4095)
                opaque=rgb[3]>.999
                valid=opaque & active[ids] & ((rgb[:3]-skin[:3].flatten(1)[:,ids]).abs().amax(0)<.002)
                if not valid.any():continue
                assert float(valid.sum()/opaque.sum().clamp_min(1))>.999
                pixels=(rgb[:3].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
                pixels=cv2.resize(pixels,(384,576),interpolation=cv2.INTER_AREA).astype(np.float32)/255
                pixels=(pixels-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
                batch=torch.from_numpy(pixels.transpose(2,0,1).copy())[None].cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(pixel_values=batch).logits
                probs=F.interpolate(logits.float(),ids.shape,mode='bilinear',align_corners=False)[0].softmax(0)
                which=ids[valid];n=torch.bincount(which,minlength=4096).float();support=n>0
                s=torch.zeros(18,4096,device='cuda');s.scatter_add_(1,which[None].expand(18,-1),probs[:,valid]);s/=n.clamp_min(1)
                all_scores+=s*support;count+=support;views_seen[vi]|=support
                winner=s.argmax(0)
                votes[1]+=torch.isin(winner,torch.tensor(POSITIVE,device='cuda'))&support
                votes[0]+=torch.isin(winner,torch.tensor(NEGATIVE,device='cuda'))&support
        mean=all_scores/count.clamp_min(1);maximum,winner=mean.max(0)
        labels=torch.full((4096,),255,dtype=torch.uint8,device='cuda')
        labels[torch.isin(winner,torch.tensor(POSITIVE,device='cuda'))&active]=1
        labels[torch.isin(winner,torch.tensor(NEGATIVE,device='cuda'))&active]=0
        agreement=matching_agreement(labels,votes,count)
        good=(maximum>=.7)&(agreement>=.8)&(views_seen.sum(0)>=2)&active&(labels!=255)
        previous=torch.from_numpy(np.array(Image.open(a.source/name/'pseudo_labels.png')).flatten()).cuda()
        accepted=good&(previous!=255)&(previous==labels)
        combined=torch.full_like(labels,255);combined[accepted]=labels[accepted]
        Image.fromarray(labels.reshape(64,64).cpu().numpy()).save(d/'parser_candidate_labels.png')
        Image.fromarray(combined.reshape(64,64).cpu().numpy()).save(d/'agreement_pseudo_labels.png')
        np.savez_compressed(d/'evidence.npz',class_confidence=maximum.cpu().numpy().reshape(64,64),
            votes=votes.cpu().numpy().reshape(2,64,64),observations=count.cpu().numpy().reshape(64,64),
            agreement=agreement.cpu().numpy().reshape(64,64),concept=winner.cpu().numpy().reshape(64,64),
            distinct_views=views_seen.sum(0).cpu().numpy().reshape(64,64))
        for filename,label in [('parser_views.png',labels),('agreement_views.png',combined)]:
            idx=label.long().clone();idx[idx==255]=2;overlay=uv.clone();overlay[:3]=.35*uv[:3]+.65*colors[idx].T.reshape(3,64,64)
            imgs=[renderer.forward_view(overlay[None],v)[0,:3] for v in VIEWS[:2]];save_image(d/filename,torch.cat(imgs,2))
        results[name]=dict(file=Path(entry['input_path']).name,active_cells=int(active.sum()),
            accepted_cells=int(accepted.sum()),accepted_coverage=float(accepted.sum()/active.sum().clamp_min(1)),
            meat_cells=int((combined==1).sum()),non_meat_cells=int((combined==0).sum()),
            human_ground_truth=False,accuracy=None)
        write(d/'audit.json',results[name]);write(a.output/'summary.json',results);report(a.source,a.output,results)
        print(json.dumps(results[name]),flush=True)
    write(a.output/'status.json',dict(stage='complete',done=len(source),total=len(source),seconds=time.time()-start,
         trained_meat_weights=False,human_labels=0,accuracy=None))


if __name__=='__main__':main()
