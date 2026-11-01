"""Portable same-split OOF report. Fine labels are never inferred from binary masks."""
import argparse,base64,html,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from meat_features import make_renderer,render_views,sha,write,PARTS
from train_supervised import metric

def report(run,cache):
 comparison=json.loads((run/'comparison.json').read_text());rows=json.loads((run/'manifest.json').read_text())['rows'];variants=list(comparison['results']);renderer=make_renderer('cpu')
 labels={'baseline':'当前预选模型','all_faces':'补齐六面 · 同一分类头','surface':'六面＋表面邻接模型'}
 def image(p):return 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()
 def figures(fs):return '<div class="grid">'+''.join(f'<figure><figcaption>{html.escape(l)}</figcaption><img loading="lazy" src="{image(p)}"></figure>' for l,p in fs)+'</div>'
 cards=[];regions={v:{p:dict(y=[],p=[],valid=[]) for p in PARTS+['bottom','head_front','outer']} for v in variants}
 for row in rows:
  d=run/row['id'];c=dict(np.load(cache/(row['id']+'.npz')));valid=c['valid'].astype(bool);y=np.array(Image.open(d/'human_meat.png'))[...,0].flatten()>0
  render_views(c['rgba'],renderer,d/'original_views.png');render_views(c['rgba'],renderer,d/'human_views.png',y)
  fs=[('原皮肤',d/'original_views.png'),('人工 · 绿 meat / 蓝非 meat',d/'human_views.png')];uv=[('原 UV',d/'skin.png'),('人工 meat',d/'human_meat.png')]
  for v in variants:
   prob=np.load(d/(v+'_probability.npy')).flatten();pred=prob>=.5;render_views(c['rgba'],renderer,d/(v+'_views.png'),pred);m=next(t for t in comparison['results'][v]['per_case'] if t['id']==row['id'])
   fs.append((labels.get(v,v)+f" · IoU {m['iou']:.1%}",d/(v+'_views.png')))
   error=np.zeros((4096,4),np.uint8);error[valid]=[68,75,86,255];error[valid&y&pred]=[25,205,100,255];error[valid&~y&pred]=[255,70,85,255];error[valid&y&~pred]=[255,190,45,255]
   Image.fromarray(error.reshape(64,64,4)).save(d/(v+'_error.png'));uv.append((labels.get(v,v)+' · 逐格误差',d/(v+'_error.png')))
   for part in regions[v]:
    select=valid & ((c['region']%6==PARTS.index(part)) if part in PARTS else (c['local'][:,45]>0 if part=='bottom' else (c['region']%6==0)&(c['local'][:,46]>0) if part=='head_front' else c['region']>=6))
    r=regions[v][part];r['y'].extend(y[select]);r['p'].extend(prob[select]);r['valid'].extend([True]*int(select.sum()))
  cards.append('<section data-id="'+row['id']+'"><h2>'+row['id']+'</h2>'+figures(fs)+'<details><summary>UV 逐格误差：绿正确 meat，红误选，黄漏选，灰正确非 meat</summary>'+figures(uv)+'</details></section>')
 breakdown={v:{k:metric(r['y'],r['p'],np.array(r['valid'],bool)) for k,r in parts.items()} for v,parts in regions.items()};write(run/'region_comparison.json',breakdown)
 rowshtml=''.join('<tr><th>'+labels.get(v,v)+'</th>'+''.join(f'<td>{comparison["results"][v]["macro"][k]:.1%}</td>' for k in ['iou','precision','recall'])+f'<td>{comparison["results"][v]["pooled"]["fp"]}</td><td>{comparison["results"][v]["pooled"]["fn"]}</td></tr>' for v in variants)
 intro=f'''<h1>MC Meat · 六面特征与表面结构对比</h1><p>{len(rows)} 张既有人工标注，按整张皮肤分组留一验证。每组固定 200 轮、相同 0.5 阈值。下图来自未训练过该皮肤的折模型；最终全量训练模型不用于这张表。本报告不是全新盲测，也未使用留出标签选择阈值或轮次。</p><p>有效格包含同面正对正交投影仍可见的内层，以及有效外层。已补齐每部件每层的六面视图，保留原多视角上下文。表面模型沿真实立方体面接缝聚合邻格与同部件上下文，不将 UV 图上无关相邻块混合。</p><p>本轮细分类格数（可能含助手补标，来源见数据快照）：<b>{comparison['fine_labeled_cells']}</b>。未细标格仍用于二分类；细分类损失仅监督明确标注，未知不当负例。当前无法据此声称模型已学会区分眼睛、眉毛与胡子。标注仍需复核，异常肤色、衣饰和少量外层 meat 仍可能误判。</p><table><tr><th>模型</th><th>平均 IoU</th><th>平均精确率</th><th>平均召回</th><th>误选格</th><th>漏选格</th></tr>{rowshtml}</table><p>平均按每张皮肤计算；误选、漏选是全体有效格计数。颜色覆盖仅用于显示，原皮肤和人工标注未改写。</p><input id="search" placeholder="按皮肤 ID 搜索" oninput="document.querySelectorAll('section[data-id]').forEach(e=>e.hidden=!e.dataset.id.toLowerCase().includes(this.value.toLowerCase()))">'''
 style='body{font:16px system-ui;background:#f3f5f8;color:#223047;max-width:1700px;margin:30px auto;padding:0 24px}p{line-height:1.8}section{background:white;padding:22px;margin:24px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}figure{margin:0}figcaption{font-size:13px;min-height:45px}img{width:100%;image-rendering:pixelated}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border:1px solid #ddd;text-align:left}summary{cursor:pointer;margin:20px 0}input{padding:12px;width:350px}section[hidden]{display:none}@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}}'
 (run/'review_report_portable.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MC Meat 六面特征与结构对比</title><style>'+style+'</style>'+intro+''.join(cards)+'</html>')
 write(run/'report_images.json',{str(p.relative_to(run)):sha(p) for p in run.glob('*/*.png')})
 # Preserve the checkpoint/feature training provenance; report generation has its own provenance.
 write(run/'report_provenance.json',dict(code_sha256=sha(Path(__file__)),comparison_sha256=sha(run/'comparison.json'),report_sha256=sha(run/'review_report_portable.html')))
 write(run/'release.json',dict(schema='mc-meat-surface-v3',status='experimental_not_deployed',manifest_sha256=sha(run/'manifest.json'),models={v:dict(path=str((run/v/'model.pt').resolve()),sha256=sha(run/v/'model.pt')) for v in variants if v!='baseline'},files={f:sha(run/f) for f in ['comparison.json','inference_audit.json','review_report_portable.html','region_comparison.json','report_images.json','report_provenance.json']},fine_class_predictions_validated=False))
if __name__=='__main__':
 torch.set_num_threads(4)
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args();report(a.run,a.cache)
