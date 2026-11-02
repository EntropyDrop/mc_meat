"""Portable held-out seven-category comparisons and release integrity manifest."""
import base64,html,json
from pathlib import Path
import numpy as np
from PIL import Image
from meat_features import make_renderer,render_views,sha,write
from semantic_schema_v3 import CLASSES
from face_decoder_v9 import MEAT_IDS

NAMES={'current_reference':'当前线上模型参考（五类）','surface_v9':'七类表面解码器 · 同数据对照','face_v9':'七类面部解码器 · 新模型'}
COLORS=np.zeros((11,4),np.uint8);COLORS[0]=[65,70,80,255]
for c in CLASSES:COLORS[c['id']]=[*bytes.fromhex(c['color'][1:]),255]


def report(run,cache):
 c=json.loads((run/'comparison.json').read_text());manifest=json.loads((run/'manifest.json').read_text());rows=manifest['rows'];variants=list(c['results']);renderer=make_renderer('cpu')
 def image(p):return 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()
 def figs(items):return '<div class="grid">'+''.join(f'<figure><figcaption>{html.escape(t)}</figcaption><img loading="lazy" src="{image(p)}"></figure>' for t,p in items)+'</div>'
 def table(headers,values):return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in values)+'</table>'
 def category(labels,valid,path):
  color=COLORS[labels].copy();color[~valid]=0;Image.fromarray(color.reshape(64,64,4)).save(path)
 metrics={v:{p['id']:p for p in c['results'][v]['per_case']} for v in variants}
 cards=[]
 for row in sorted(rows,key=lambda r:metrics['face_v9'][r['id']]['iou']):
  d=run/row['id'];values=dict(np.load(cache/(row['id']+'.npz')));t=np.load(d/'truth.npz');valid=t['valid'];y=t['y']>.5
  render_views(values['rgba'],renderer,d/'original_views.png');render_views(values['rgba'],renderer,d/'truth_views.png',y)
  fs=[('原皮肤',d/'original_views.png'),('人工复核 · 绿=meat / 蓝=非 meat',d/'truth_views.png')];uv=[('原 UV',d/'skin.png')];errors=[]
  category(t['labels'],valid,d/'fine_truth.png');uv.append(('人工七类标注',d/'fine_truth.png'))
  for v in variants:
   labels=np.load(d/(v+'.npz'))['labels'];selected=np.isin(labels,MEAT_IDS)
   render_views(values['rgba'],renderer,d/(v+'_views.png'),selected);met=metrics[v][row['id']]
   fs.append((NAMES[v]+f" · IoU {met['iou']:.1%}",d/(v+'_views.png')))
   category(labels,valid,d/(v+'_fine.png'));uv.append((NAMES[v],d/(v+'_fine.png')))
   error=np.zeros((4096,4),np.uint8);error[valid]=[65,70,80,255];error[valid&y&selected]=[38,187,128,255];error[valid&~y&selected]=[255,70,85,255];error[valid&y&~selected]=[255,190,45,255]
   Image.fromarray(error.reshape(64,64,4)).save(d/(v+'_error.png'));errors.append((NAMES[v]+' · 红=误选，黄=漏选',d/(v+'_error.png')))
  cards.append('<section data-id="'+row['id']+'"><h2>'+row['id']+'</h2>'+figs(fs)+'<details open><summary>七类 UV 对照</summary>'+figs(uv)+'</details><details><summary>二分类逐格误差</summary>'+figs(errors)+'</details></section>')
 overall=table(['版本','meat IoU','精确率','召回率','误选格','漏选格','七类宏平均 F1'],[[NAMES[v]]+[f"{r['macro'][k]:.1%}" for k in ['iou','precision','recall']]+[r['pooled']['fp'],r['pooled']['fn'],f"{r['fine_macro_f1']:.1%}"] for v,r in c['results'].items()])
 fine=table(['版本','类别','正例皮肤数','正例格数','精确率','召回率','F1'],[[NAMES[v],cl['label'],manifest['class_counts'][str(cl['id'])]['skins'],r['fine'][str(cl['id'])]['support']]+[f"{r['fine'][str(cl['id'])][k]:.1%}" for k in ['precision','recall','f1']] for v,r in c['results'].items() for cl in CLASSES])
 legend=' '.join(f'<span style="border-left:12px solid {cl["color"]};padding:0 12px 0 6px">{cl["label"]}</span>' for cl in CLASSES)
 intro=f'''<h1>169 张完整复核 · 七类均衡损失训练</h1><p>训练数据固定于 DDJ 提交 {manifest['dataset_commit'][:8]}。169 张均已人工完整复核；保持正交遮挡排除、类别互斥以及眼影/鼻子归 meat、眉毛/胡子归非 meat 的定义。</p><p>冻结 DINO，多视角与独立 UV 面特征相同。七类表面解码器和七类面部解码器使用完全相同的五折整张皮肤分组、随机初始化、200 轮及固定 0.5 置信度。新模型追加内外层头部六面卷积与注意力。两者均包含内部“其他”状态，界面仍只有七类。本组细分类损失按每个训练折中存在的类别分别求平均，再对类别求平均；二分类损失不变。该实验是在先前训练的训练集拟合诊断中发现稀有类别没有学会后增加的对照，属于探索性改进。</p><p><b>以下均为留出预测，不是全量模型在训练集上的拟合图。</b>当前线上五类模型使用原留出检查点；从未用于其训练的新皮肤才使用原全量模型。它的数据量不同、也不能预测眼影/鼻子，应与同数据七类对照区分。该数据集已多次用于开发，属于历史分组验证，不是新盲测。</p><p>本轮所有格已完整复核，细类误选统计覆盖所有有效格，包括人工未选的“其他”格。少量正例（胡子 5 张、眼影 8 张、鼻子 9 张）会导致逐类指标波动，不能只看总 IoU。新权重尚未替换线上预选。</p>{overall}<details><summary>展开逐类指标</summary>{fine}</details><p>{legend} <span>灰色=未选/其他；透明=忽略</span></p><p>案例按新模型 IoU 从低到高排列，便于优先复核失败例。</p><input placeholder="搜索皮肤 ID" oninput="document.querySelectorAll('section[data-id]').forEach(s=>s.hidden=!s.dataset.id.toLowerCase().includes(this.value.toLowerCase()))">'''
 style='body{font:16px system-ui;background:#f3f5f8;color:#223047;max-width:1800px;margin:28px auto;padding:0 24px}p{line-height:1.8}section{background:#fff;border-radius:12px;padding:20px;margin:22px 0}.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:20px 0}figure{margin:0}figcaption{font-size:12px;min-height:45px}img{width:100%;image-rendering:pixelated;background:repeating-conic-gradient(#e7e7e7 0% 25%,#fafafa 0% 50%) 50%/12px 12px}table{border-collapse:collapse;width:100%;background:#fff}th,td{padding:9px;border:1px solid #ddd;text-align:left}summary{cursor:pointer;margin:16px 0}input{padding:12px;width:360px}section[hidden]{display:none}@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}}'
 (run/'review_report_portable.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MC Meat 七类重新训练</title><style>'+style+'</style>'+intro+''.join(cards)+'</html>')
 write(run/'report_images.json',{str(p.relative_to(run)):sha(p) for p in run.glob('*/*.png')})
 write(run/'report_provenance.json',dict(code_sha256=sha(Path(__file__)),comparison_sha256=sha(run/'comparison.json'),report_sha256=sha(run/'review_report_portable.html')))
 write(run/'release.json',dict(schema='mc-meat-face-v9',status='experimental_not_deployed',manifest_sha256=sha(run/'manifest.json'),models={v:dict(path=str(run/v/'model.pt'),sha256=sha(run/v/'model.pt')) for v in ['surface_v9','face_v9']},files={f:sha(run/f) for f in ['comparison.json','paired_comparison.json','inference_audit.json','training_complete.json','review_report_portable.html','report_images.json','report_provenance.json','standalone_inference_audit.json']},fine_class_predictions_validated=True))

if __name__=='__main__':
 import argparse,torch
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);report(a.run,a.cache)
