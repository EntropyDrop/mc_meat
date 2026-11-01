"""Portable same-split OOF report. Fine labels are never inferred from binary masks."""
import argparse,base64,html,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from meat_features import make_renderer,render_views,sha,write,PARTS
from train_supervised import metric

def report(run,cache):
 data=run.parent.parent/'data/face_v5_20261101';annotation_totals={str(c):0 for c in [1,2,3,5,6]};unknown_meat=0;conflicts=[]
 comparison=json.loads((run/'comparison.json').read_text());rows=json.loads((run/'manifest.json').read_text())['rows'];variants=list(comparison['results']);renderer=make_renderer('cpu')
 labels={'baseline':'当前在线模型','surface':'上一轮六面＋表面模型','face_binary':'面部解码器 · 仅二分类','face_fine':'面部＋全身细分类辅助','face_front_fine':'面部＋仅面部细分类辅助'}
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
  sm=json.loads((data/row['id']/'semantic.json').read_text());sem=np.array(sm['labels']).reshape(64,64);color=np.zeros((64,64,4),np.uint8)
  palette={1:[38,187,128,255],2:[0,169,232,255],3:[241,123,157,255],5:[234,154,39,255],6:[199,124,71,255]}
  for cls,rgba in palette.items():color[sem==cls]=rgba;annotation_totals[str(cls)]+=int((sem==cls).sum())
  Image.fromarray(color).save(d/'assistant_fine.png');uv.insert(2,('助手细标 · 绿皮肤/蓝眼睛/粉嘴/橙眉/棕胡',d/'assistant_fine.png'))
  unknown_meat+=int((y&(sem.flatten()==0)).sum());conflicts.extend([dict(id=row['id'],**note) for note in sm.get('review_notes',[])])
  original=Image.open(d/'skin.png').convert('RGBA').crop((8,8,16,16)).resize((256,256),Image.Resampling.NEAREST);colored=Image.fromarray(color).crop((8,8,16,16)).resize((256,256),Image.Resampling.NEAREST)
  original.save(d/'face_original.png');colored.save(d/'face_semantic.png')
  detail=figures([('头部正面原色',d/'face_original.png'),('头部正面细标签（空白=未知）',d/'face_semantic.png')])
  notes='<p>待复核格（UV x,y）：'+html.escape(', '.join(str(n['uv']) for n in sm.get('review_notes',[])))+'</p>' if sm.get('review_notes') else ''
  cards.append('<section data-id="'+row['id']+'"><h2>'+row['id']+'</h2>'+figures(fs)+'<details><summary>UV 逐格误差：绿正确 meat，红误选，黄漏选，灰正确非 meat</summary>'+figures(uv)+detail+notes+'</details></section>')
 breakdown={v:{k:metric(r['y'],r['p'],np.array(r['valid'],bool)) for k,r in parts.items()} for v,parts in regions.items()};write(run/'region_comparison.json',breakdown)
 rowshtml=''.join('<tr><th>'+labels.get(v,v)+'</th>'+''.join(f'<td>{comparison["results"][v]["macro"][k]:.1%}</td>' for k in ['iou','precision','recall'])+f'<td>{comparison["results"][v]["pooled"]["fp"]}</td><td>{comparison["results"][v]["pooled"]["fn"]}</td></tr>' for v in variants)
 write(run/'annotation_audit.json',dict(cases=len(rows),class_counts=annotation_totals,unknown_meat_cells=unknown_meat,conflict_review=conflicts,binary_unchanged=True,human_fine_labels=False,annotation_source='assistant_visual_review_20261101',fine_labels_are_weak_supervision=True))
 intro=f'''<h1>MC Meat · 面部解码器与五类辅助监督</h1><p>34 张既有人工 meat 标注，按整张皮肤分组留一验证；固定 200 轮、阈值 0.5、种子 42。所有对比图来自未训练过该皮肤的折模型。DINO 全程冻结。历史验证集不是全新盲测。</p><p>新模型增加脸部 8×8 空间卷积、内外层面部注意力和头部六面上下文；身体沿用表面邻接网络。三组新实验分别为：不加辅助、全身细分类辅助、仅面部细分类辅助。第三组在发现第二组身体退步后增加，因此是针对同一历史验证集的追加实验，不能视作独立盲测。全量最终模型另存，未替换当前在线预选模型。</p><p>助手补标：裸露皮肤 {annotation_totals['1']} 格，眼睛 {annotation_totals['2']} 格，嘴巴 {annotation_totals['3']} 格，眉毛 {annotation_totals['5']} 格，胡子 {annotation_totals['6']} 格。这些是助手判断，不是人工细分类真值；既有 meat 的剩余格暂按皮肤低权重监督。{unknown_meat} 个旧 meat 格存在细分类冲突，继续只参加二分类。胡子没有有效训练样本，眉毛仅 2 格，不能声称已经学会这两类。</p><p>辅助权重：剩余皮肤 0.35、明确眼睛 0.65、嘴巴与眉毛 0.55；未知格为 0。二分类人工标签保持不变；头发、眼镜和衣饰没有新增分类。图中细标签可在 DDJ 标注工具中人工修正。</p><table><tr><th>模型</th><th>平均 IoU</th><th>平均精确率</th><th>平均召回</th><th>误选格</th><th>漏选格</th></tr>{rowshtml}</table><p>绿为正确 meat，红为误选，黄为漏选，灰为正确非 meat。平均按皮肤计算，误选/漏选按全体格子计数。细分类预测尚缺人工独立验证。</p><input id="search" placeholder="按皮肤 ID 搜索" oninput="document.querySelectorAll('section[data-id]').forEach(e=>e.hidden=!e.dataset.id.toLowerCase().includes(this.value.toLowerCase()))">'''

 style='body{font:16px system-ui;background:#f3f5f8;color:#223047;max-width:1700px;margin:30px auto;padding:0 24px}p{line-height:1.8}section{background:white;padding:22px;margin:24px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:12px}figure{margin:0}figcaption{font-size:13px;min-height:45px}img{width:100%;image-rendering:pixelated}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border:1px solid #ddd;text-align:left}summary{cursor:pointer;margin:20px 0}input{padding:12px;width:350px}section[hidden]{display:none}@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}}'
 (run/'review_report_portable.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MC Meat 面部解码器对比</title><style>'+style+'</style>'+intro+''.join(cards)+'</html>')
 write(run/'report_images.json',{str(p.relative_to(run)):sha(p) for p in run.glob('*/*.png')})
 # Preserve the checkpoint/feature training provenance; report generation has its own provenance.
 write(run/'report_provenance.json',dict(code_sha256=sha(Path(__file__)),comparison_sha256=sha(run/'comparison.json'),report_sha256=sha(run/'review_report_portable.html')))
 write(run/'release.json',dict(schema='mc-meat-surface-v3',status='experimental_not_deployed',manifest_sha256=sha(run/'manifest.json'),models={v:dict(path=str((run/v/'model.pt').resolve()),sha256=sha(run/v/'model.pt')) for v in ['face_front_fine']},files={f:sha(run/f) for f in ['comparison.json','inference_audit.json','review_report_portable.html','region_comparison.json','report_images.json','report_provenance.json','annotation_audit.json']},fine_class_predictions_validated=False))
if __name__=='__main__':
 torch.set_num_threads(4)
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args();report(a.run,a.cache)
