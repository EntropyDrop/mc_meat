"""Human-supervised baseline with held-out-skin predictions and texture ablation.

Frozen DINO extraction never accesses labels. All head hyperparameters and the
0.5 decision threshold are fixed before leave-one-group-out evaluation. No fold's
validation labels select epochs, normalization, weights, or thresholds.
"""
import argparse
import base64
import html
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from uv_visibility import POLICY, annotation_mask

from meat_features import (ROOT, extract, make_dino, make_renderer, matrix,
                           provenance, render_views, sha, write)

CONFIG=dict(visibility_policy=POLICY,epochs=200,lr=.002,weight_decay=.01,hidden=[64,32],dropout=.1,
            seed=42,threshold=.5,loss='per-skin normalized BCE; sqrt imbalance weight clipped to [1,5]',
            variants=['local','dino_local'],validation='leave-one-duplicate-group-out',
            epoch_selection='fixed count, no validation selection',threshold_calibrated=False,
            augmentation='none; no invented labels or pseudo-labels')


class MeatHead(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x):return self.net(x).squeeze(-1)


def weighted_bce(logits,y,skin_indices):
    pos=y.sum();negative=len(y)-pos
    posweight=(negative/pos.clamp_min(1)).sqrt().clamp(1,5)
    losses=F.binary_cross_entropy_with_logits(logits,y,pos_weight=posweight,reduction='none')
    # Equal skin weighting, including complete all-negative examples.
    return torch.stack([losses[skin_indices==i].mean() for i in skin_indices.unique()]).mean()


def fit(train,log_path):
    torch.manual_seed(CONFIG['seed'])
    x=torch.cat([r['x'][r['valid']] for r in train]);y=torch.cat([r['y'][r['valid']] for r in train])
    owners=torch.cat([torch.full((int(r['valid'].sum()),),i) for i,r in enumerate(train)])
    assert y.sum()>0 and y.sum()<len(y),'Training fold must contain both classes'
    net=MeatHead(x.shape[1]);opt=torch.optim.AdamW(net.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,CONFIG['epochs'],eta_min=.0002)
    with log_path.open('w') as log:
        for epoch in range(1,CONFIG['epochs']+1):
            net.train();opt.zero_grad(set_to_none=True)
            loss=weighted_bce(net(x),y,owners)
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step();scheduler.step()
            if epoch==1 or epoch%20==0:
                log.write(json.dumps(dict(epoch=epoch,train_loss=float(loss.detach())))+'\n');log.flush()
    return net.eval()


def metric(y,p,valid):
    y=np.asarray(y,dtype=bool)[valid];pred=np.asarray(p)[valid]>=CONFIG['threshold']
    tp=int((y&pred).sum());fp=int((~y&pred).sum());fn=int((y&~pred).sum());tn=int((~y&~pred).sum())
    return counts_metric(tp,fp,fn,tn)


def counts_metric(tp,fp,fn,tn):
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
                iou=tp/max(tp+fp+fn,1),dice=2*tp/max(2*tp+fp+fn,1),
                non_meat_iou=tn/max(tn+fp+fn,1))


def checkpoint(path,net,variant,manifest,feature_provenance,train_ids):
    torch.save(dict(state_dict=net.state_dict(),input_dim=net.net[0].in_features,variant=variant,
                    config=CONFIG,feature_provenance=feature_provenance,manifest_sha256=sha(manifest),
                    train_ids=train_ids,schema='mc-meat-supervised-v1',
                    semantics=dict(meat='bare skin, eyes, lips/mouth, bare hands/feet/body',
                                   non_meat='hair, beard, eyebrows, glasses, headwear, clothing, accessories'),
                    mask_encoding='RGBA white meat; black non-meat; alpha=0 ignore incl fully occluded inner cells'),path)


def prepare_cache(data,out):
    manifest=json.loads((data/'manifest.json').read_text())
    out.mkdir(parents=True,exist_ok=True)
    fp=provenance()
    if (out/'provenance.json').exists():
        assert json.loads((out/'provenance.json').read_text())==fp,'Feature implementation changed; use a new cache'
    else:write(out/'provenance.json',fp)
    missing=[]
    for row in manifest['rows']:
        d=data/row['id'];assert sha(d/'skin.png')==row['skin_sha256']
        target=out/(row['id']+'.npz');meta=out/(row['id']+'.json')
        if target.exists() and meta.exists():
            a=json.loads(meta.read_text())
            assert a['input_sha256']==row['skin_sha256'] and a['cache_sha256']==sha(target)
        else:missing.append(row)
    if missing:
        model=make_dino();renderer=make_renderer('cuda')
        for i,row in enumerate(missing):
            write(out/'status.json',dict(stage='extract',case=row['id'],remaining=len(missing)-i))
            values,audit=extract(data/row['id']/'skin.png',model,renderer)
            assert np.array_equal(values['rgba'],np.asarray(Image.open(data/row['id']/'visible_skin.png'))),'Processed input mismatch'
            assert not values['rgba'][~values['valid'].reshape(64,64)].any(),'Ignored cells entered features'
            path=out/(row['id']+'.npz');np.savez_compressed(path,**values)
            write(out/(row['id']+'.json'),dict(input_sha256=row['skin_sha256'],cache_sha256=sha(path),
                 active_cells=int(values['valid'].sum()),observed_cells=int(((values['coverage']>0)&values['valid']).sum()),
                 mapping_audit=audit,labels_read=False))
            print('FEATURES',row['id'],flush=True)
        del model,renderer;torch.cuda.empty_cache()
    write(out/'status.json',dict(stage='complete',count=len(manifest['rows'])))


def train(data,cache,out):
    assert not out.exists(),'Use a fresh run directory; do not overwrite evaluation'
    manifest_path=data/'manifest.json';manifest=json.loads(manifest_path.read_text());rows=manifest['rows']
    fp=json.loads((cache/'provenance.json').read_text());assert fp==provenance()
    out.mkdir(parents=True);(out/'source').mkdir()
    for name in ['prepare_human.py','meat_features.py','train_supervised.py','predict_meat.py','uv_annotation_format.py','uv_visibility.py','local_paths.py','test_supervised.py']:
        shutil.copy2(ROOT/name,out/'source'/name)
    shutil.copy2(manifest_path,out/'manifest.json');write(out/'configuration.json',CONFIG);write(out/'feature_provenance.json',fp)
    all_data=[]
    for row in rows:
        d=data/row['id'];c=cache/(row['id']+'.npz');cm=json.loads((cache/(row['id']+'.json')).read_text())
        assert cm['input_sha256']==row['skin_sha256'] and cm['cache_sha256']==sha(c)
        for f,key in [('skin.png','skin_sha256'),('meat.png','mask_sha256'),('annotation.json','metadata_sha256'),('valid.png','valid_sha256'),('visible_skin.png','visible_skin_sha256')]:
            assert sha(d/f)==row[key], 'Frozen input changed'
        values=dict(np.load(c));valid=values['valid'].astype(bool)
        assert np.array_equal(valid,np.array(Image.open(d/'valid.png')).flatten()>0)
        y=torch.from_numpy((np.array(Image.open(d/'meat.png'))[...,0].flatten()>0).astype(np.float32))
        all_data.append(dict(row=row,cache=values,valid=torch.from_numpy(valid),y=y))
        case=out/row['id'];case.mkdir();shutil.copy2(d/'skin.png',case/'skin.png');shutil.copy2(d/'meat.png',case/'human_meat.png')
    groups=sorted(set(r['group'] for r in rows));summary={};audits=[]
    for variant in CONFIG['variants']:
        subset=[dict(r,x=matrix(r['cache'],variant)) for r in all_data]
        variant_dir=out/variant;variant_dir.mkdir();per_case=[]
        for fold,group in enumerate(groups):
            train_rows=[r for r in subset if r['row']['group']!=group];held=[r for r in subset if r['row']['group']==group]
            train_ids=[r['row']['id'] for r in train_rows];held_ids=[r['row']['id'] for r in held]
            assert not set(train_ids)&set(held_ids)
            write(out/'status.json',dict(stage='cross_validation',variant=variant,fold=fold+1,total=len(groups),held_out=held_ids))
            net=fit(train_rows,variant_dir/f'fold-{fold}.jsonl')
            checkpoint(variant_dir/f'fold-{fold}.pt',net,variant,manifest_path,fp,train_ids)
            for r in held:
                with torch.no_grad():prob=net(r['x']).sigmoid().numpy()
                valid=r['valid'].numpy();prob[~valid]=0;case=out/r['row']['id']
                np.save(case/(variant+'_probability.npy'),prob.reshape(64,64))
                (case/(variant+'_meat.png')).write_bytes(annotation_mask(np.flatnonzero((prob>=.5)&valid).tolist(),valid.astype(np.uint8).tobytes()))
                metrics=metric(r['y'].numpy(),prob,valid)
                metrics.update(id=r['row']['id'],fold=fold,train_ids=train_ids,held_out_group=group)
                per_case.append(metrics)
            print('FOLD',variant,fold+1,per_case[-1],flush=True)
        pooled=counts_metric(*(sum(r[k] for r in per_case) for k in ['tp','fp','fn','tn']))
        macro={k:float(np.mean([r[k] for r in per_case])) for k in ['precision','recall','iou','dice','non_meat_iou']}
        summary[variant]=dict(per_case=per_case,macro=macro,pooled=pooled)
        write(out/'comparison.json',dict(evaluation='held-out whole-skin groups; existing historical DDJ cases',cases=len(rows),results=summary))
        write(out/'status.json',dict(stage='final_fit',variant=variant,training_cases=len(rows)))
        net=fit(subset,variant_dir/'final_train.jsonl')
        path=variant_dir/'model.pt';checkpoint(path,net,variant,manifest_path,fp,[r['id'] for r in rows])
        saved=torch.load(path,map_location='cpu',weights_only=True);reloaded=MeatHead(saved['input_dim']);reloaded.load_state_dict(saved['state_dict']);reloaded.eval()
        with torch.no_grad():
            delta=max(float((net(r['x'])-reloaded(r['x'])).abs().max()) for r in subset)
        assert delta==0
        audits.append(dict(variant=variant,checkpoint_sha256=sha(path),save_reload_max_abs=delta,trained_cases=len(rows),
                           held_out_predictions='generated only by fold models that excluded the entire case/group'))
    write(out/'inference_audit.json',dict(models=audits,mask_threshold=.5,invalid_excluded=True,
                                      features_use_labels=False,visibility_policy=POLICY,threshold_calibrated=False,production_validated=False))
    make_report(out,all_data,summary)
    write(out/'release.json',dict(status='experimental_baseline',manifest_sha256=sha(manifest_path),
          models={v:dict(path=str((out/v/'model.pt').resolve()),sha256=sha(out/v/'model.pt')) for v in CONFIG['variants']},
          files={f:sha(out/f) for f in ['comparison.json','inference_audit.json','review_report_portable.html']},
          note=f'Not deployed. {len(rows)} skins cannot establish broad generalization. OOF estimates are exploratory.'))
    write(out/'status.json',dict(stage='complete',cases=len(rows),groups=len(groups),epochs=CONFIG['epochs']))


def make_report(out,all_data,summary):
    renderer=make_renderer('cpu')
    def image(p):return 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()
    cards=[]
    for r in all_data:
        case=out/r['row']['id'];rgba=r['cache']['rgba'];valid=r['valid'].numpy().reshape(64,64)
        truth=r['y'].numpy().reshape(64,64)>0
        render_views(rgba,renderer,case/'original_views.png');render_views(rgba,renderer,case/'human_views.png',truth)
        figs=[('原皮肤',case/'original_views.png'),('人工标注 · 绿=meat / 蓝=非 meat',case/'human_views.png')]
        for variant in CONFIG['variants']:
            p=np.load(case/(variant+'_probability.npy'))>=.5
            render_views(rgba,renderer,case/(variant+'_views.png'),p)
            figs.append(('仅局部纹理 OOF' if variant=='local' else 'DINO + 局部纹理 OOF',case/(variant+'_views.png')))
        uv_figs=[('UV 原图',case/'skin.png'),('UV 人工 · 白=meat',case/'human_meat.png')]
        for variant in CONFIG['variants']:
            pred=np.load(case/(variant+'_probability.npy'))>=.5
            error=np.zeros((64,64,4),np.uint8);error[valid]=[60,68,80,255]
            error[valid & truth & pred]=[30,210,100,255]
            error[valid & ~truth & pred]=[255,75,80,255]
            error[valid & truth & ~pred]=[255,190,40,255]
            Image.fromarray(error).save(case/(variant+'_error.png'))
            uv_figs.append((variant+' UV 误差',case/(variant+'_error.png')))
        def figures(fs):return '<div class="grid">'+''.join('<figure><figcaption>'+html.escape(label)+'</figcaption><img src="'+image(path)+'"></figure>' for label,path in fs)+'</div>'
        metrics=[]
        for v in CONFIG['variants']:
            m=next(m for m in summary[v]['per_case'] if m['id']==r['row']['id'])
            metrics.append(f"{v}: IoU {m['iou']:.1%} · precision {m['precision']:.1%} · recall {m['recall']:.1%}")
        cards.append('<section><h2>'+html.escape(r['row']['id'])+'</h2><p>'+'<br>'.join(metrics)+'</p>'+figures(figs)+'<details><summary>展开逐格 UV 对比（绿=正确 meat、红=误选、黄=漏选）</summary>'+figures(uv_figs)+'</details></section>')
    table='<table><tr><th>整张留一验证</th><th>宏平均 IoU</th><th>宏平均 Precision</th><th>宏平均 Recall</th><th>宏平均 Dice</th></tr>'
    for v,s in summary.items():table+='<tr><td>'+v+'</td>'+''.join(f'<td>{s["macro"][k]:.1%}</td>' for k in ['iou','precision','recall','dice'])+'</tr>'
    table+='</table>'
    report='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MC Meat · 人工标注监督基线</title><style>body{font:16px system-ui;background:#f2f4f7;color:#223047;max-width:1500px;margin:30px auto;padding:0 20px}section{background:white;padding:22px;margin:24px 0;border-radius:14px}p{line-height:1.75}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}figure{margin:0}figcaption{min-height:50px;font-size:14px}img{width:100%;image-rendering:pixelated}table{border-collapse:collapse;width:100%;background:white}td,th{text-align:left;padding:12px;border:1px solid #ddd}summary{cursor:pointer;margin:20px 0}@media(max-width:850px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}</style><h1>MC Meat：人工标注监督基线</h1><p>逐格分类，内外层均可为 meat 或非 meat。仅采用当前有效人工标注；透明格、UV 空白区，以及被同部件外层完全遮住的内层格均忽略；部分露出仍参与训练。下方为整张皮肤留一验证预测（OOF），不是用这张皮肤训练后的拟合图。DINOv2 冻结；训练的是小型分类头。两组使用相同划分、200 轮和固定 0.5 阈值，未根据留出标签挑选轮次或阈值。</p><p>当前仅 6 张既有 DDJ 皮肤，样本量很小，尚不足以证明泛化能力；这不是全新盲测。人工标签仍是初步标注。未采样到的语义和肤色需要后续数据覆盖。绿色表示 meat，蓝色表示非 meat；底面及局部可见区域请展开 UV 查看。最终部署候选权重使用全部标注重训，其训练集拟合指标未混入下表。</p>'''+table+''.join(cards)+'</html>'
    report=report.replace('当前仅 6 张',f'当前仅 {len(all_data)} 张')
    (out/'review_report_portable.html').write_text(report)
    write(out/'report_images.json',{str(p.relative_to(out)):sha(p) for p in out.glob('*/*.png')})


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--features-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    prepare_cache(a.data,a.cache)
    if not a.features_only:train(a.data,a.cache,a.output)


if __name__=='__main__':main()
