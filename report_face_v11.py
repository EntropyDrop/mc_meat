"""Portable held-out seven-category comparisons and release integrity manifest."""
import base64,html,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from meat_features import make_renderer,render_views,sha,write
from semantic_schema_v3 import CLASSES
from face_decoder_v9 import MEAT_IDS

NAMES={'ordinary_v9':'普通七类 v9','balanced_v9':'全局均衡 v9b','rare_raw':'专用头 v11 · 固定阈值','rare_calibrated':'专用头 v11 · 嵌套校准'}
COLORS=np.zeros((11,4),np.uint8);COLORS[0]=[65,70,80,255]
for c in CLASSES:COLORS[c['id']]=[*bytes.fromhex(c['color'][1:]),255]


def report(run,cache):
 c=json.loads((run/'comparison.json').read_text());manifest=json.loads((run/'manifest.json').read_text());rows=manifest['rows'];variants=list(c['results']);renderer=make_renderer('cpu');source=json.loads((run/'data_source.json').read_text());parent=Path(source['parent']);balanced=Path(source['balanced'])
 def image(p):return 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()
 def figs(items):return '<div class="grid">'+''.join(f'<figure><figcaption>{html.escape(t)}</figcaption><img loading="lazy" src="{image(p)}"></figure>' for t,p in items)+'</div>'
 def table(headers,values):return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in values)+'</table>'
 def category(labels,valid,path):
  color=COLORS[labels].copy();color[~valid]=0;Image.fromarray(color.reshape(64,64,4)).save(path)
 metrics={v:{p['id']:p for p in c['results'][v]['per_case']} for v in variants}
 cards=[]
 for row in sorted(rows,key=lambda r:metrics['rare_calibrated'][r['id']]['iou']):
  d=run/row['id'];values=dict(np.load(cache/(row['id']+'.npz')));t=np.load(d/'truth.npz');valid=t['valid'];y=t['y']>.5
  shutil.copy2(parent/row['id']/'original_views.png',d/'original_views.png');shutil.copy2(parent/row['id']/'truth_views.png',d/'truth_views.png')
  fs=[('原皮肤',d/'original_views.png'),('人工复核 · 绿=meat / 蓝=非 meat',d/'truth_views.png')];uv=[('原 UV',d/'skin.png')];errors=[]
  category(t['labels'],valid,d/'fine_truth.png');uv.append(('人工七类标注',d/'fine_truth.png'))
  for v in variants:
   labels=np.load(d/(v+'.npz'))['labels'];selected=np.isin(labels,MEAT_IDS)

   if v in ['ordinary_v9','balanced_v9']:
    prior=parent if v=='ordinary_v9' else balanced;shutil.copy2(prior/row['id']/'face_v9_views.png',d/(v+'_views.png'))
   else:render_views(values['rgba'],renderer,d/(v+'_views.png'),selected)
   met=metrics[v][row['id']]
   fs.append((NAMES[v]+f" · IoU {met['iou']:.1%}",d/(v+'_views.png')))
   category(labels,valid,d/(v+'_fine.png'));uv.append((NAMES[v],d/(v+'_fine.png')))
   error=np.zeros((4096,4),np.uint8);error[valid]=[65,70,80,255];error[valid&y&selected]=[38,187,128,255];error[valid&~y&selected]=[255,70,85,255];error[valid&y&~selected]=[255,190,45,255]
   Image.fromarray(error.reshape(64,64,4)).save(d/(v+'_error.png'));errors.append((NAMES[v]+' · 红=误选，黄=漏选',d/(v+'_error.png')))
  cards.append('<section data-id="'+row['id']+'"><h2>'+row['id']+'</h2>'+figs(fs)+'<details open><summary>七类 UV 对照</summary>'+figs(uv)+'</details><details><summary>二分类逐格误差</summary>'+figs(errors)+'</details></section>')
 overall=table(['版本','meat IoU','精确率','召回率','误选格','漏选格','七类宏平均 F1'],[[NAMES[v]]+[f"{r['macro'][k]:.1%}" for k in ['iou','precision','recall']]+[r['pooled']['fp'],r['pooled']['fn'],f"{r['fine_macro_f1']:.1%}"] for v,r in c['results'].items()])
 fine=table(['版本','类别','正例皮肤数','正例格数','精确率','召回率','F1'],[[NAMES[v],cl['label'],manifest['class_counts'][str(cl['id'])]['skins'],r['fine'][str(cl['id'])]['support']]+[f"{r['fine'][str(cl['id'])][k]:.1%}" for k in ['precision','recall','f1']] for v,r in c['results'].items() for cl in CLASSES])
 legend=' '.join(f'<span style="border-left:12px solid {cl["color"]};padding:0 12px 0 6px">{cl["label"]}</span>' for cl in CLASSES)
 intro=f'''<h1>胡子、眼影、鼻子专项 · v11 对比</h1><p>169 张完整人工复核，数据固定于提交 {manifest['dataset_commit'][:8]}。所有新旧结果使用相同五折整张皮肤分组。普通版与全局均衡版为历史对照。</p><p>v11 保留普通模型作为默认结果，另外训练三类专用头：按正例皮肤均衡，挖掘每张皮肤最难的 16 个已标注负例，并联合学习整张皮肤的类别存在性。专用头从均衡版继承特征，普通版仍作为默认结果；类别存在性采用有下限的柔性权重，校准搜索全部正分数断点。DINO 和继承的特征主干冻结，面部卷积/注意力及专用头可训练。只在专用头通过阈值时替换默认类别，允许纠正原来的 meat 漏检。</p><p>固定阈值版用于区分训练改动与校准收益。校准版在每个外层训练折内再做三折留出；包括通用主干在内的内层模型均未训练对应校准样本。每类在可达到 35% 精确率时最大化 F1，否则选择 F0.5；外层验证标签不参与选阈值。最终部署用阈值来自全数据 OOF，未用于修改这里的外层预测。</p><p><b>历史开发集，不是全新盲测。</b>胡子仅 5 张、眼影 8 张、鼻子 9 张正例。请同时查看精确率、召回、误选格数及具体 UV，不能只看召回上升。所有展示的新模型预测均来自排除该皮肤及其重复组的外层模型；全量训练权重单独保存。线上预选未替换。</p>{overall}<details open><summary>逐类精确率与召回</summary>{fine}</details><p>{legend} 灰色=未选/其他；透明=忽略。</p><p>案例按校准版 meat IoU 从低到高排列。</p><input placeholder="搜索皮肤 ID" oninput="document.querySelectorAll('section[data-id]').forEach(s=>s.hidden=!s.dataset.id.toLowerCase().includes(this.value.toLowerCase()))">'''

 style='body{font:16px system-ui;background:#f3f5f8;color:#223047;max-width:1800px;margin:28px auto;padding:0 24px}p{line-height:1.8}section{background:#fff;border-radius:12px;padding:20px;margin:22px 0}.grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px;margin:20px 0}figure{margin:0}figcaption{font-size:12px;min-height:45px}img{width:100%;image-rendering:pixelated;background:repeating-conic-gradient(#e7e7e7 0% 25%,#fafafa 0% 50%) 50%/12px 12px}table{border-collapse:collapse;width:100%;background:#fff}th,td{padding:9px;border:1px solid #ddd;text-align:left}summary{cursor:pointer;margin:16px 0}input{padding:12px;width:360px}section[hidden]{display:none}@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}}'
 (run/'review_report_portable.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MC Meat 稀有类别专项 v11</title><style>'+style+'</style>'+intro+''.join(cards)+'</html>')
 write(run/'report_images.json',{str(p.relative_to(run)):sha(p) for p in run.glob('*/*.png')})
 write(run/'report_provenance.json',dict(code_sha256=sha(Path(__file__)),comparison_sha256=sha(run/'comparison.json'),report_sha256=sha(run/'review_report_portable.html')))
 write(run/'release.json',dict(schema='mc-meat-rare-v11',status='experimental_not_deployed',manifest_sha256=sha(run/'manifest.json'),models={'rare_v11':dict(path=str(run/'model.pt'),sha256=sha(run/'model.pt'))},files={f:sha(run/f) for f in ['comparison.json','calibration.json','inference_audit.json','training_complete.json','review_report_portable.html','report_images.json','report_provenance.json','standalone_inference_audit.json','inference_source_hashes.json']},fine_class_predictions_validated=True))


if __name__=='__main__':
 import argparse,torch
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);report(a.run,a.cache)
