"""Historical human OOF comparison; never score new assistant labels as truth."""
import argparse,base64,html,json
from pathlib import Path
import numpy as np
from PIL import Image
from meat_features import sha,write,make_renderer,render_views
from train_supervised import metric

NAMES={'online_scheme_oof':'线上方案历史 OOF','old_head_latest_labels':'原小分类器 · 34 例重训','previous_face_v6':'上轮面部解码器 · 34 例','assistant20_face':'本轮面部解码器 · 加 20 例助手标注'}

def report(run):
 src=json.loads((run/'data_source.json').read_text());data=Path(src['data']);cache=Path(src['cache']);control=Path(src['control'])
 comp=json.loads((run/'comparison.json').read_text());assert comp['complete']
 rows=[r for r in json.loads((run/'manifest.json').read_text())['rows'] if r['evaluation_eligible']]
 variants=list(comp['results']);renderer=make_renderer('cpu');cards=[];front={v:dict(y=[],p=[]) for v in variants}
 def img(p):return 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()
 def figs(items):return '<div class="grid">'+''.join('<figure><figcaption>'+html.escape(label)+'</figcaption><img loading="lazy" src="'+img(path)+'"></figure>' for label,path in items)+'</div>'
 for row in rows:
  rid=row['id'];d=run/rid;c=dict(np.load(cache/(rid+'.npz')));valid=c['valid'].astype(bool);y=np.array(Image.open(d/'reference_meat.png'))[...,0].flatten()>0
  face=valid&(c['region']%6==0)&(c['local'][:,46]>0)
  figures=[('原皮肤',control/rid/'original_views.png'),('人工 meat 参考 · 绿为 meat',control/rid/'human_views.png')];uv=[('原始 UV',d/'skin.png'),('人工 meat · 白色',d/'reference_meat.png')]
  for v in variants:
   p=np.load(d/(v+'_probability.npy')).flatten();pred=p>=.5;front[v]['y'].extend(y[face]);front[v]['p'].extend(p[face])
   error=np.zeros((4096,4),np.uint8);error[valid]=[68,75,86,255];error[valid&y&pred]=[25,205,100,255];error[valid&~y&pred]=[255,70,85,255];error[valid&y&~pred]=[255,190,45,255]
   Image.fromarray(error.reshape(64,64,4)).save(d/(v+'_error.png'));uv.append((NAMES[v],d/(v+'_error.png')))
   if v in ['previous_face_v6','assistant20_face']:
    render_views(c['rgba'],renderer,d/(v+'_views.png'),pred)
    m=next(x for x in comp['results'][v]['per_case'] if x['id']==rid)
    figures.append((NAMES[v]+f" · IoU {m['iou']:.1%}",d/(v+'_views.png')))
  cards.append('<section data-id="'+rid+'"><h2>'+rid+'</h2>'+figs(figures)+'<details><summary>查看 UV 逐格误差</summary>'+figs(uv)+'</details></section>')
 regions={v:metric(r['y'],r['p'],np.ones(len(r['y']),bool)) for v,r in front.items()};write(run/'head_front_comparison.json',regions)
 old={r['id']:r['iou'] for r in comp['results']['previous_face_v6']['per_case']};new={r['id']:r['iou'] for r in comp['results']['assistant20_face']['per_case']}
 delta=np.array([new[r['id']]-old[r['id']] for r in rows]);rng=np.random.default_rng(42);boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
 paired=dict(mean_iou_delta=float(delta.mean()),bootstrap_95_interval=np.quantile(boot,[.025,.975]).tolist(),improved=int((delta>1e-10).sum()),regressed=int((delta< -1e-10).sum()),tied=int((abs(delta)<=1e-10).sum()),per_case=[dict(id=r['id'],delta=float(d)) for r,d in zip(rows,delta)])
 write(run/'paired_comparison.json',paired)
 table='<table><tr><th>方案</th><th>平均 IoU</th><th>面部汇总 IoU</th><th>平均精确率</th><th>平均召回率</th><th>误选格</th><th>漏选格</th></tr>'
 for v,m in comp['results'].items():table+='<tr><td>'+NAMES[v]+'</td>'+f'<td>{m["macro"]["iou"]:.2%}</td><td>{regions[v]["iou"]:.2%}</td><td>{m["macro"]["precision"]:.2%}</td><td>{m["macro"]["recall"]:.2%}</td><td>{m["pooled"]["fp"]}</td><td>{m["pooled"]["fn"]}</td></tr>'
 table+='</table>'
 intro=f'''<h1>新增 20 例助手标注 · 冻结 DINO 面部解码器</h1><p>共 54 例：原有 34 例人工二分类标签保持不变；新增 20 例为助手逐格视觉标注，只参加训练，二分类权重 0.5，细分类保留逐格置信权重并乘以样本权重。没有把助手标注称为人工真值。五类为裸露皮肤、眼睛、嘴巴、眉毛、胡须；后三者中的眉毛和胡须属于非 meat。</p><p>原有 34 例逐组留一验证，每折从固定随机初始化训练面部解码器 200 轮。留出皮肤及其近重复分组不参与该折训练。DINO 冻结；未知细分类仅参与二分类。与上轮比较使用完全相同的人工标签和逐格特征，已检查所有哈希及特征数值。这些历史案例反复用于改进，不是全新盲测。</p>{table}<p>相对上轮面部解码器平均 IoU 变化 {paired['mean_iou_delta']*100:+.2f} 个百分点，改善 {paired['improved']} 例，退步 {paired['regressed']} 例。配对重采样 95% 区间 [{paired['bootstrap_95_interval'][0]*100:+.2f}, {paired['bootstrap_95_interval'][1]*100:+.2f}]，仅为探索性估计，没有覆盖多随机种子与历史方案选择的不确定性。</p><p>原有验证集没有人工胡须细分类样本，因此不能据此宣称胡须识别已得到验证。新增 20 例上的拟合只用于检查训练流程，不计作准确率。最终全量模型已另训并核验保存重载；没有自动替换线上模型。</p><p>UV 误差：绿=正确 meat，红=误选，黄=漏选，灰=正确非 meat；透明为无效或正交投影完全遮挡区。</p><input placeholder="按 ID 查找" oninput="document.querySelectorAll('section[data-id]').forEach(e=>e.hidden=!e.dataset.id.includes(this.value.toUpperCase()))">'''
 css='body{font:16px system-ui;background:#f3f5f8;color:#223047;margin:24px auto;padding:20px;max-width:1600px}p{line-height:1.8}section{background:white;padding:22px;margin:24px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}figure{margin:0}figcaption{font-size:13px;min-height:48px}img{width:100%;image-rendering:pixelated;background:#ddd}td,th{padding:10px;border:1px solid #ddd}table{border-collapse:collapse;width:100%;background:white}summary{cursor:pointer;margin:16px 0}input{padding:12px}section[hidden]{display:none}@media(max-width:800px){.grid{grid-template-columns:repeat(2,1fr)}}'
 page=run/'review_report_portable.html';page.write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MC Meat 新增20例训练对比</title><style>'+css+'</style>'+intro+''.join(cards)+'</html>')
 write(run/'report_images.json',{str(p.relative_to(run)):sha(p) for p in run.glob('*/*.png')})
 write(run/'release.json',dict(checkpoint='face_front_fine/model.pt',checkpoint_sha256=sha(run/'face_front_fine/model.pt'),report_sha256=sha(page),comparison_sha256=sha(run/'comparison.json'),inference_audit_sha256=sha(run/'inference_audit.json'),report_images_sha256=sha(run/'report_images.json'),production_deployed=False,usage='Reuse existing FaceDecoder feature/provenance-checked inference; threshold 0.5; do not infer fine labels outside the supervised front scope.'))
 write(run/'status.json',dict(stage='complete',report=str(page),checkpoint=str(run/'face_front_fine/model.pt')))
 print(json.dumps(dict(stage='complete',paired=paired,macro={v:m['macro'] for v,m in comp['results'].items()})),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);report(p.parse_args().run)
