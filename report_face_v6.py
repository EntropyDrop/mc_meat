"""Latest-snapshot report with OOF region/class audits and paired skin intervals."""
import argparse,base64,html,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from meat_features import make_renderer,render_views,sha,write,PARTS
from train_supervised import metric

LABELS={'current_oof':'当前线上方案 · 旧标注 OOF','previous_face':'上一轮面部解码器 · 旧标注 OOF','baseline_retrained':'原分类器 · 最新标注重训','face_binary':'新面部解码器 · 仅二分类','face_front_fine':'新面部解码器 · 五类辅助监督'}
CLASSES={1:'裸露皮肤',2:'眼睛',3:'嘴巴',5:'眉毛',6:'胡子'}
PALETTE={1:[38,187,128,255],2:[0,169,232,255],3:[241,123,157,255],5:[234,154,39,255],6:[199,124,71,255]}

def report(run):
 source=json.loads((run/'data_source.json').read_text());data=Path(source['data']);cache=Path(source['cache'])
 comparison=json.loads((run/'comparison.json').read_text());rows=json.loads((run/'manifest.json').read_text())['rows'];variants=list(comparison['results']);renderer=make_renderer('cpu')
 def image(p):return 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()
 def figures(fs):return '<div class="grid">'+''.join(f'<figure><figcaption>{html.escape(l)}</figcaption><img loading="lazy" src="{image(p)}"></figure>' for l,p in fs)+'</div>'
 regions={v:{part:dict(y=[],p=[],valid=[]) for part in PARTS+['head_front','outer']} for v in variants}
 class_binary={v:{str(k):dict(total=0,correct=0) for k in CLASSES} for v in variants}
 fine={scope:np.zeros((5,5),np.int64) for scope in ['all_known','explicit_human_changed']};cards=[]
 for row in rows:
  rid=row['id'];d=run/rid;c=dict(np.load(cache/(rid+'.npz')));valid=c['valid'].astype(bool);y=np.array(Image.open(d/'human_meat.png'))[...,0].flatten()>0
  sm=json.loads((data/rid/'semantic.json').read_text());sem=np.array(sm['labels']);front=valid&(c['region']%6==0)&(c['local'][:,46]>0)
  render_views(c['rgba'],renderer,d/'original_views.png');render_views(c['rgba'],renderer,d/'human_views.png',y)
  fs=[('原皮肤',d/'original_views.png'),('最新 meat · 绿=meat，蓝=非 meat',d/'human_views.png')];uv=[('原 UV',d/'skin.png'),('最新 meat · 白色',d/'human_meat.png')];heads=[]
  for v in variants:
   prob=np.load(d/(v+'_probability.npy')).flatten();pred=prob>=.5;render_views(c['rgba'],renderer,d/(v+'_views.png'),pred)
   m=next(t for t in comparison['results'][v]['per_case'] if t['id']==rid);fs.append((LABELS[v]+f" · IoU {m['iou']:.1%}",d/(v+'_views.png')))
   error=np.zeros((4096,4),np.uint8);error[valid]=[68,75,86,255];error[valid&y&pred]=[25,205,100,255];error[valid&~y&pred]=[255,70,85,255];error[valid&y&~pred]=[255,190,45,255]
   err=Image.fromarray(error.reshape(64,64,4));err.save(d/(v+'_error.png'));uv.append((LABELS[v]+' · UV 误差',d/(v+'_error.png')))
   face=Image.new('RGBA',(16,8));face.paste(err.crop((8,8,16,16)),(0,0));face.paste(err.crop((40,8,48,16)),(8,0));face.save(d/(v+'_face_error.png'));heads.append((LABELS[v],d/(v+'_face_error.png')))
   for part in regions[v]:
    select=valid&((c['region']%6==PARTS.index(part)) if part in PARTS else ((c['region']%6==0)&(c['local'][:,46]>0)) if part=='head_front' else c['region']>=6)
    r=regions[v][part];r['y'].extend(y[select]);r['p'].extend(prob[select]);r['valid'].extend([True]*int(select.sum()))
   for cls in CLASSES:
    selected=valid&(sem==cls);s=class_binary[v][str(cls)];s['total']+=int(selected.sum());s['correct']+=int((selected&(pred==y)).sum())
  color=np.zeros((4096,4),np.uint8)
  for cls,rgba in PALETTE.items():color[sem==cls]=rgba
  Image.fromarray(color.reshape(64,64,4)).save(d/'latest_fine.png');uv.insert(2,('最新细标：绿皮肤/蓝眼睛/粉嘴/橙眉/棕胡',d/'latest_fine.png'))
  for name,img in [('face_original',Image.open(d/'skin.png').convert('RGBA')),('face_semantic',Image.fromarray(color.reshape(64,64,4)))]:
   panel=Image.new('RGBA',(16,8));panel.paste(img.crop((8,8,16,16)),(0,0));panel.paste(img.crop((40,8,48,16)),(8,0));panel.save(d/(name+'.png'))
  fpred=np.load(d/'face_front_fine_fine_prediction.npy').flatten();human=np.zeros(4096,bool);human[sm.get('human_reviewed_cells',[])]=True
  if sm.get('annotation_source') in ['human','human_reviewed']:human[:]=True
  known=front&(sem>0)
  for scope,mask in [('all_known',known),('explicit_human_changed',known&human)]:
   for i,cls in enumerate(CLASSES):
    for j,pcls in enumerate(CLASSES):fine[scope][i,j]+=int((mask&(sem==cls)&(fpred==pcls)).sum())
  face_figs=[('头部正面原色 · 左内层，右外层',d/'face_original.png'),('最新细标签 · 透明=未细标',d/'face_semantic.png')]+heads
  cards.append(f'<section data-id="{rid}"><h2>{rid}</h2>'+figures(fs)+'<h3>面部逐格误差</h3>'+figures(face_figs)+'<details><summary>展开全部 UV：绿=正确 meat，红=误选，黄=漏选，灰=正确非 meat</summary>'+figures(uv)+'</details></section>')
 breakdown={v:{k:metric(r['y'],r['p'],np.array(r['valid'],bool)) for k,r in parts.items()} for v,parts in regions.items()};write(run/'region_comparison.json',breakdown)
 for v in variants:
  for s in class_binary[v].values():s['binary_correct_fraction']=s['correct']/s['total'] if s['total'] else None
 fine_audit={}
 for scope,mat in fine.items():
  fine_audit[scope]=dict(class_ids=list(CLASSES),confusion=mat.tolist(),known_cells=int(mat.sum()),by_class={str(c):dict(support=int(mat[i].sum()),correct=int(mat[i,i]),recall=float(mat[i,i]/mat[i].sum()) if mat[i].sum() else None) for i,c in enumerate(CLASSES)})
 write(run/'class_comparison.json',dict(binary=class_binary,fine_front_only=fine_audit,notes='Binary metrics evaluate latest meat truth. Auxiliary fine agreement uses known front cells only; mixed provenance, not fully independent human ground truth. Zero-support class has no accuracy estimate.'))
 rng=np.random.default_rng(42);ordered=[r['id'] for r in rows];paired={}
 for other in ['current_oof','previous_face','baseline_retrained','face_binary']:
  a={r['id']:r['iou'] for r in comparison['results']['face_front_fine']['per_case']};b={r['id']:r['iou'] for r in comparison['results'][other]['per_case']};diff=np.array([a[k]-b[k] for k in ordered]);boots=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(1)
  paired[other]=dict(mean_iou_delta=float(diff.mean()),bootstrap_95_interval=np.quantile(boots,[.025,.975]).tolist(),improved=int((diff>1e-10).sum()),regressed=int((diff< -1e-10).sum()),tied=int((abs(diff)<=1e-10).sum()),per_case=[dict(id=k,delta=float(d)) for k,d in zip(ordered,diff)])
 write(run/'paired_comparison.json',paired)
 a=json.loads((run/'annotation_audit.json').read_text());counts=a['counts'];counttext='，'.join(f'{label} {counts.get(str(k),0)} 格（{a["class_cases"].get(str(k),0)} 张）' for k,label in CLASSES.items())
 table='<table><tr><th>方案</th><th>平均 IoU</th><th>面部 IoU（汇总）</th><th>平均精确率</th><th>平均召回</th><th>误选格</th><th>漏选格</th></tr>'
 for v in variants:
  m=comparison['results'][v];table+='<tr><th>'+LABELS[v]+'</th>'+f'<td>{m["macro"]["iou"]:.2%}</td><td>{breakdown[v]["head_front"]["iou"]:.2%}</td>'+''.join(f'<td>{m["macro"][k]:.2%}</td>' for k in ['precision','recall'])+f'<td>{m["pooled"]["fp"]}</td><td>{m["pooled"]["fn"]}</td></tr>'
 table+='</table>'
 ct='<table><tr><th>实际已标类别</th><th>格数</th>'+''.join('<th>'+LABELS[v]+'</th>' for v in variants)+'</tr>'
 for k,label in CLASSES.items():
  ct+=f'<tr><td>{label}</td><td>{counts.get(str(k),0)}</td>'+''.join('<td>'+('未标注' if class_binary[v][str(k)]['total']==0 else f"{class_binary[v][str(k)]['correct']}/{class_binary[v][str(k)]['total']}")+'</td>' for v in variants)+'</tr>'
 ct+='</table>'
 ft='<details><summary>细分类辅助头：已细标面部格的一致率</summary><p>仅统计有细标的正面内外层格；未标格不计为任何类别。全部细标包含保留的助手预标；显式人工改动子集很小，不代表完整人工测试集。</p><table><tr><th>类别</th><th>全部细标：判对/总数</th><th>显式人工改动：判对/总数</th></tr>'
 for k,label in CLASSES.items():
  ft+='<tr><td>'+label+'</td>'
  for scope in ['all_known','explicit_human_changed']:
   x=fine_audit[scope]['by_class'][str(k)];ft+='<td>'+('无样本' if not x['support'] else str(x['correct'])+'/'+str(x['support']))+'</td>'
  ft+='</tr>'
 ft+='</table></details>'
 p=paired['face_binary'];delta=f'{p["mean_iou_delta"]*100:+.2f}';ci=p['bootstrap_95_interval'];aux=f'五类辅助相对同架构二分类的平均 IoU 变化 {delta} 个百分点；按皮肤配对重采样的 95% 区间 [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]。'
 intro=f'''<h1>最新五类标注 · 冻结 DINO 的面部解码器重训</h1><p>冻结快照：{html.escape(str(data))}。{len(rows)} 张皮肤、34 个独立分组，逐组留一验证，固定 200 轮、阈值 0.5、种子 42。DINO 不参与训练，只复用经输入/代码哈希核验的无标签视觉特征。五类辅助仅施加于头部正面内外层，未知细标格继续参加二分类。训练初始化没有接触留出皮肤；全量模型在验证结束后另训。</p><p>本次标签：{counttext}。眉毛和胡子按现有定义属于非 meat；眼睛、嘴巴、裸露皮肤属于 meat。没有新增头发、眼镜、衣饰类别。胡子无样本，嘴巴样本很少，不能宣称已经学会这两类。</p><p>29 张记录了人工保存，5 张仍保留助手预标来源。文件中记录的人工改动格使用权重 1；未改动的预标格保留原置信权重。所有二分类指标都对照最新保存的 meat 标签；未把旧指标复制进来。细分类一致性另存 class_comparison.json，保留来源区分。</p><p>“当前线上方案”与“上一轮面部解码器”使用旧标注训练时的留出预测，重新对照最新标签计分；并非当前全量模型在训练样本上的拟合分数。“原分类器 · 最新标注重训”用于区分标签更新与结构改进的影响。此处都是反复使用的历史样本，不是全新盲测；置信区间是探索性的，未反映多随机种子及历史方案选择影响。当前线上模型未替换。</p>{table}<p>{aux}</p><h2>已标细类的 meat / 非 meat 判对格数</h2><p>这是二分类结果在各细类上的切片：眼睛/嘴巴应保留为 meat，眉毛/胡子应排除。不是五分类准确率。</p>{ct}{ft}<p>下方全部为留出预测。绿色 meat、蓝色非 meat；面部误差与 UV 误差中绿色正确 meat、红色误选、黄色漏选、灰色正确非 meat。透明表示无效/遮挡排除区域。点击 UV 可检查身体和隐藏面的情况。</p><input id="search" placeholder="按皮肤 ID 搜索" oninput="document.querySelectorAll('section[data-id]').forEach(e=>e.hidden=!e.dataset.id.toLowerCase().includes(this.value.toLowerCase()))">'''
 style='body{font:16px system-ui;background:#f3f5f8;color:#223047;max-width:1800px;margin:30px auto;padding:0 24px}p{line-height:1.8}section{background:white;padding:22px;margin:24px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:12px}figure{margin:0}figcaption{font-size:13px;min-height:52px}img{width:100%;image-rendering:pixelated;background:#e4e7eb}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border:1px solid #ddd;text-align:left}summary{cursor:pointer;margin:20px 0}input{padding:12px;width:350px}section[hidden]{display:none}@media(max-width:1000px){.grid{grid-template-columns:repeat(3,1fr)}}'
 (run/'review_report_portable.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MC Meat 最新五类重训对比</title><style>'+style+'</style>'+intro+''.join(cards)+'</html>')
 write(run/'report_images.json',{str(p.relative_to(run)):sha(p) for p in run.glob('*/*.png')})
 write(run/'report_provenance.json',dict(code_sha256=sha(Path(__file__)),comparison_sha256=sha(run/'comparison.json'),report_sha256=sha(run/'review_report_portable.html')))
 print(json.dumps(dict(results={v:s['macro'] for v,s in comparison['results'].items()},head_front={v:breakdown[v]['head_front'] for v in variants},class_binary=class_binary,paired=paired,fine=fine_audit),ensure_ascii=False))

if __name__=='__main__':
 torch.set_num_threads(4);p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();report(a.run)
