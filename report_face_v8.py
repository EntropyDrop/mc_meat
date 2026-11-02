"""Portable UV/category and rendered binary comparisons, using held-out outputs."""
import base64,html,json
from pathlib import Path
import numpy as np
from PIL import Image
from meat_features import make_renderer,render_views,sha,write

NAMES={'current_reference':'当前模型参考 · 原留出/新增未见样本','surface_control':'旧结构 · 最新数据同划分重训','face_v8':'新面部解码器 · 同划分','face_v8_calibrated':'新解码器＋身体校准 · 嵌套验证'}
NAMES.update(face_v8_guarded_balanced='候选 · 均衡召回＋细类拒选',face_v8_guarded='候选 · 减少误选＋细类拒选')
CLASSES={1:'裸露皮肤',2:'眼睛',3:'嘴巴',5:'眉毛',6:'胡子'}
COLORS=np.array([[65,70,80,255],[38,187,128,255],[0,169,232,255],[241,123,157,255],[0,0,0,0],[234,154,39,255],[199,124,71,255]],np.uint8)

def report(run,cache):
 comparison=json.loads((run/'comparison.json').read_text());rows=json.loads((run/'manifest.json').read_text())['rows'];variants=list(comparison['results']);renderer=make_renderer('cpu')
 def image(p):return 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()
 def figs(items):return '<div class="grid">'+''.join(f'<figure><figcaption>{html.escape(t)}</figcaption><img loading="lazy" src="{image(p)}"></figure>' for t,p in items)+'</div>'
 def category(labels,valid,path):
  color=COLORS[labels].copy();color[~valid]=0;Image.fromarray(color.reshape(64,64,4)).save(path)
 def table(headers,values):return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in values)+'</table>'
 cards=[];case_audit=[]
 for row in rows:
  d=run/row['id'];c=dict(np.load(cache/(row['id']+'.npz')));t=np.load(d/'truth.npz');valid=t['valid'];y=t['y']>.5
  render_views(c['rgba'],renderer,d/'original_views.png');render_views(c['rgba'],renderer,d/'truth_views.png',y)
  fs=[('原皮肤',d/'original_views.png'),('保存的 meat 标注 · 绿为 meat',d/'truth_views.png')];uv=[('原 UV',d/'skin.png')]
  category(t['labels'],valid,d/'fine_truth.png');uv.append(('已有细标 · 灰色为未知/未选',d/'fine_truth.png'));heads=[]
  for v in variants:
   result=np.load(d/(v+'.npz'));labels=result['labels'];selected=np.isin(labels,[1,2,3]);render_views(c['rgba'],renderer,d/(v+'_views.png'),selected)
   met=next(p for p in comparison['results'][v]['per_case'] if p['id']==row['id'])
   fs.append((NAMES[v]+f" · IoU {met['iou']:.1%}",d/(v+'_views.png')))
   category(labels,valid,d/(v+'_fine.png'));uv.append((NAMES[v],d/(v+'_fine.png')))
   error=np.zeros((4096,4),np.uint8);error[valid]=[65,70,80,255];error[valid&y&selected]=[38,187,128,255];error[valid&~y&selected]=[255,70,85,255];error[valid&y&~selected]=[255,190,45,255]
   Image.fromarray(error.reshape(64,64,4)).save(d/(v+'_error.png'))
   source=Image.open(d/(v+'_fine.png'));panel=Image.new('RGBA',(32,8))
   for i,(x,z) in enumerate([(8,8),(40,8),(0,8),(16,8)]):panel.paste(source.crop((x,z,x+8,z+8)),(i*8,0))
   panel.save(d/(v+'_head.png'));heads.append((NAMES[v]+' · 内正/外正/内右/内左',d/(v+'_head.png')))
  errors=[(NAMES[v]+' · 红误选/黄漏选',d/(v+'_error.png')) for v in variants]
  cards.append('<section data-id="'+row['id']+'"><h2>'+row['id']+'</h2>'+figs(fs)+'<details open><summary>五类 UV：绿皮肤、蓝眼睛、粉嘴巴、橙眉毛、棕胡子；灰为未选</summary>'+figs(uv)+figs(heads)+'</details><details><summary>二分类逐格误差</summary>'+figs(errors)+'</details></section>')
  case_audit.append(dict(id=row['id'],skin_sha256=sha(d/'skin.png')))
 overall=table(['版本','实际预选 IoU','精确率','召回','误选格','漏选格','五类 F1（已知细标）'],[[NAMES[v]]+[f"{r['macro'][k]:.1%}" for k in ['iou','precision','recall']]+[r['pooled']['fp'],r['pooled']['fn'],f"{r['fine_macro_f1']:.1%}"] for v,r in comparison['results'].items()])
 fine=table(['版本','类别','已有格数','已知细标内精确率','召回','F1'],[[NAMES[v],CLASSES[int(c)],r['support']]+[f"{r[k]:.1%}" for k in ['precision','recall','f1']] for v,s in comparison['results'].items() for c,r in s['fine'].items()])
 gate=json.loads((run/'quality_gate.json').read_text())
 intro=f'''<h1>五类预选：面部解码器与当前模型对比</h1><p>{len(rows)} 张最新有效标注，固定五折整张/重复组划分。旧结构与新结构都从随机解码器初始化，使用相同训练/验证案例；冻结 DINO。当前模型参考使用原留出权重，新案例才使用它的全量模型，训练量不同，属于辅助参考。</p><p>界面仍是五类。模型内部有“其他”，不确定格保持未选并提示。未细标格只提供二分类边际监督；只有明确勾选整张复核的未选格才用于“其他”细分类监督。本轮不会将旧的 partial 标签自动转为完整真值。</p><p><b>历史验证，不是新盲测。</b>已有细标含人工与助手来源，五类精确率只在已知细标范围计算，未知格不当负例；报告包括实际应用置信度后的预选结果，不能只看原始二分类概率。训练轮数固定；身体阈值用内层分组验证选择。拒选机制是在开发过程中发现头发误判后补充，属于历史开发集上的探索性改进，仍需新盲测。</p><p>细类拒选仅使用各训练折已知正标注的位置范围：超出支持范围的眼睛、嘴巴、眉毛、胡子标为不确定。它不能证明区域是头发或衣饰，也可能漏掉少见造型；未标注区域没有被当作负例。最终模型的训练集拟合图不作为泛化证据。</p>{overall}<details><summary>展开逐类结果</summary>{fine}</details><p>预先设定的候选质量门槛：<b>{'通过' if gate['passed'] else '未全部通过'}</b>。完成训练不等于已经替换线上模型，部署状态见交付说明。</p><input placeholder="输入皮肤 ID" oninput="document.querySelectorAll('section[data-id]').forEach(s=>s.hidden=!s.dataset.id.toLowerCase().includes(this.value.toLowerCase()))">'''
 style='body{font:16px system-ui;background:#f3f5f8;color:#223047;max-width:1800px;margin:28px auto;padding:0 24px}p{line-height:1.8}section{background:#fff;border-radius:12px;padding:20px;margin:22px 0}.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:20px 0}figure{margin:0}figcaption{font-size:12px;min-height:45px}img{width:100%;image-rendering:pixelated;background:repeating-conic-gradient(#e7e7e7 0% 25%,#fafafa 0% 50%) 50%/12px 12px}table{border-collapse:collapse;width:100%;background:#fff}th,td{padding:10px;border:1px solid #ddd;text-align:left}summary{cursor:pointer;margin:16px 0}input{padding:12px;width:360px}section[hidden]{display:none}@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}}'
 (run/'review_report_portable.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MC Meat 面部解码器对比</title><style>'+style+'</style>'+intro+''.join(cards)+'</html>')
 write(run/'report_images.json',{str(p.relative_to(run)):sha(p) for p in run.glob('*/*.png')})
 write(run/'report_provenance.json',dict(code_sha256=sha(Path(__file__)),comparison_sha256=sha(run/'comparison.json'),report_sha256=sha(run/'review_report_portable.html')))
 write(run/'release.json',dict(schema='mc-meat-face-v8',status='candidate',manifest_sha256=sha(run/'manifest.json'),models={'face_v8':dict(sha256=sha(run/'face_v8/model.pt'))},files={f:sha(run/f) for f in ['comparison.json','quality_gate.json','inference_audit.json','training_complete.json','review_report_portable.html','report_images.json','report_provenance.json']},fine_class_predictions_validated=True))

if __name__=='__main__':
 import argparse,torch
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);report(a.run,a.cache)
