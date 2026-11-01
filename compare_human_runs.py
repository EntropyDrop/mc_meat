"""Compare unchanged held-out cases across two completed human-label runs."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path


def compare(previous,current,output):
    assert not output.exists(), 'Preserve existing report'
    old=json.loads((previous/'comparison.json').read_text());new=json.loads((current/'comparison.json').read_text())
    om={r['id']:r for r in json.loads((previous/'manifest.json').read_text())['rows']}
    nm={r['id']:r for r in json.loads((current/'manifest.json').read_text())['rows']}
    common=sorted(set(om)&set(nm));assert common
    for key in common:
        for field in ['skin_sha256','mask_sha256']:assert om[key][field]==nm[key][field]
    matched={}
    for variant in ['local','dino_local']:
        a={r['id']:r for r in old['results'][variant]['per_case']};b={r['id']:r for r in new['results'][variant]['per_case']}
        for key in common:
            assert key not in a[key]['train_ids'] and key not in b[key]['train_ids']
        matched[variant]={name:{metric:sum(rows[k][metric] for k in common)/len(common) for metric in ['iou','precision','recall','dice']} for name,rows in [('previous',a),('current',b)]}
    result=dict(common_cases=common,matched=matched,current_all={v:s['macro'] for v,s in new['results'].items()},
                note='Existing DDJ cases; same unchanged skins/masks; each prediction excludes the whole case from its training fold. Exploratory historical validation, not a new blind test.')
    output.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    def img(p):return 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()
    cards=[]
    for key in common:
        a=next(r for r in old['results']['dino_local']['per_case'] if r['id']==key)
        b=next(r for r in new['results']['dino_local']['per_case'] if r['id']==key)
        figs=[('原皮肤',current/key/'original_views.png'),('人工标注',current/key/'human_views.png'),('原版 · 用其他 5 张训练',previous/key/'dino_local_views.png'),('新版 · 用其他 12 张训练',current/key/'dino_local_views.png')]
        cards.append('<section><h2>'+html.escape(key)+'</h2><p>meat IoU '+f"{a['iou']:.1%} → {b['iou']:.1%}"+'</p><div class="grid">'+''.join('<figure><figcaption>'+label+'</figcaption><img src="'+img(p)+'"></figure>' for label,p in figs)+'</div></section>')
    summary=matched['dino_local']
    report='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MC Meat · 增加人工标注前后对比</title><style>body{font:16px system-ui;color:#223047;background:#f2f4f7;max-width:1500px;margin:30px auto;padding:0 20px}section{background:white;padding:22px;margin:24px 0;border-radius:14px}p{line-height:1.75}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}figure{margin:0}figcaption{min-height:48px}img{width:100%;image-rendering:pixelated}@media(max-width:850px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}</style><h1>增加人工标注后的变化</h1><p>人工标注由 6 张增加到 13 张。下方仅比较原来 6 张同样的皮肤和标签；每次预测都未使用该张皮肤的标签训练。绿色是 meat，蓝色是非 meat。</p><p>相同 6 张案例的 DINO＋纹理宏平均 IoU：'''+f"<strong>{summary['previous']['iou']:.1%} → {summary['current']['iou']:.1%}</strong>。"+'''新版全部 13 张的宏平均 IoU 为 '''+f"{new['results']['dino_local']['macro']['iou']:.1%}"+'''。</p><p>模型结构、训练轮数和阈值保持相同；新增的 7 张参与训练。旧案例已被用于开发分析，这不是全新盲测，样本仍然很少。所有图都是留一预测，最终使用全部 13 张重训的权重没有混入这些验证图。</p><p><a href="review_report_portable.html">查看新版全部 13 张及逐格 UV 误差</a></p>'''+''.join(cards)+'</html>'
    output.write_text(report)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--current',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();compare(a.previous,a.current,a.output)
