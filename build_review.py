"""Build an offline review of completed probes; never modifies their labels or metrics."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--notes', type=Path)
    a = parser.parse_args()
    assert not a.output.exists(), 'Use a new review directory'
    s = json.loads((a.source/'summary.json').read_text())
    t = json.loads((a.teacher/'summary.json').read_text())
    assert list(s) == list(t)
    for root, script in [(a.source, 'probe_no_labels.py'), (a.teacher, 'probe_parser.py')]:
        assert json.loads((root/'status.json').read_text())['stage'] == 'complete'
        cfg = json.loads((root/'configuration.json').read_text())
        assert sha(root/'source'/script) == cfg['script_sha256']
        if 'shared_helpers_sha256' in cfg:
            assert sha(root/'source/probe_no_labels.py') == cfg['shared_helpers_sha256']
    assert json.loads((a.teacher/'configuration.json').read_text())['source_summary_sha256'] == sha(a.source/'summary.json')
    notes = json.loads(a.notes.read_text()) if a.notes else {}
    if notes:
        assert notes['source_summary_sha256'] == sha(a.source/'summary.json')
        assert notes['teacher_summary_sha256'] == sha(a.teacher/'summary.json')
    assets = {}
    def image(p):
        with Image.open(p) as im:
            im.load()
        assets[str(p)] = sha(p)
        return 'data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode()
    sections = []
    for key, row in s.items():
        d, h = a.source/key, a.teacher/key
        uv = np.array(Image.open(d/'uv.png')); active = uv[...,3] > 0
        candidate = np.array(Image.open(d/'candidate_labels.png'))
        pseudo = np.array(Image.open(d/'pseudo_labels.png'))
        both = np.array(Image.open(h/'agreement_pseudo_labels.png'))
        for label in [candidate, pseudo, both]:
            assert label.shape == (64,64) and np.isin(label, [0,1,255]).all()
            assert (label[~active] == 255).all()
        assert (both[both != 255] == pseudo[both != 255]).all()
        assert int(active.sum()) == row['active_cells'] == t[key]['active_cells']
        assert int((pseudo != 255).sum()) == row['pseudo_labeled_cells']
        assert int((both != 255).sum()) == t[key]['accepted_cells']
        assert Path(row['input_path']).name == t[key]['file']
        assert all(v['matching_fraction'] > .999 for v in row['mapping_checks'] if v['opaque_pixels'])
        evidence = np.load(d/'evidence.npz')
        if 'meat_votes' in evidence:
            count = evidence['observations']; v = evidence['meat_votes']
            expected = np.where(candidate == 1, v, count-v)/np.maximum(count,1)
            assert np.allclose(evidence['agreement'][active], expected[active])
        frames = [('原始皮肤', d/'original_views.png'), ('DINOv2 聚类（颜色无类别含义）', d/'cluster_views.png'),
                  ('dino.txt · 已筛选候选', d/'filtered_views.png'), ('人体/服饰解析 · 未筛选候选', h/'parser_views.png'),
                  ('双教师严格交集', h/'agreement_views.png')]
        cards = ''.join('<figure><figcaption>'+html.escape(title)+'</figcaption><img loading="lazy" src="'+image(path)+'"></figure>' for title,path in frames)
        sections.append('<section><h2>'+html.escape(Path(row['input_path']).name)+'</h2><p>'+html.escape(notes.get('cases',{}).get(key,''))+'</p><div class="row">'+cards+'</div></section>')
    active = sum(x['active_cells'] for x in s.values())
    accepted = sum(x['pseudo_labeled_cells'] for x in s.values())
    both = sum(x['accepted_cells'] for x in t.values())
    meat = sum(x['meat_cells'] for x in t.values())
    observations = ''.join('<p>'+html.escape(x)+'</p>' for x in notes.get('observations',[]))
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MC Meat 无人工标注可行性对比</title><style>body{font:16px system-ui;background:#f4f6f9;color:#202b38;max-width:1800px;margin:30px auto;padding:0 24px}p{line-height:1.7}section,.intro{background:white;border-radius:14px;padding:22px;margin:24px 0}.row{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;image-rendering:pixelated}figcaption{min-height:54px;font-size:14px}strong{color:#184d6e}@media(max-width:1000px){.row{grid-template-columns:repeat(2,minmax(0,1fr))}}</style><h1>MC Meat：无人工标注可行性对比</h1><div class="intro"><p><strong>这是教师模型的可行性探测，没有训练或发布 meat 分类器。</strong></p><p>绿 = meat 候选；蓝 = non_meat 候选；黄 = 未知／被过滤。DINO 聚类列的颜色没有类别含义。</p>'''
    page += f'<p>{len(s)} 个训练素材池样本，{active} 个有效不透明 UV 格。dino.txt 筛选后接受 {accepted} 格（{accepted/active:.2%}）；双教师交集仅 {both} 格（{both/active:.2%}），其中 meat 为 {meat} 格。</p>'
    page += '<p><strong>这些数值是候选覆盖量，不是准确率。</strong>没有人工逐格真值，阈值未校准；历史训练池样本也不是盲测。两个教师相互同意仍可能一起出错。</p>'+observations+'</div>'+''.join(sections)+'</html>'
    a.output.mkdir(parents=True)
    report = a.output/'review_report_portable.html'; report.write_text(page)
    verification = dict(status='passed', cases=len(s), human_ground_truth=False, trained_meat_weights=False, accuracy=None,
                        source_summary_sha256=sha(a.source/'summary.json'), teacher_summary_sha256=sha(a.teacher/'summary.json'),
                        active_cells=active, dino_accepted_cells=accepted, intersection_cells=both, intersection_meat_cells=meat,
                        report_sha256=sha(report), image_sha256=assets, report_builder_sha256=sha(__file__),
                        checks=['completed_runs', 'source_snapshot_hashes', 'source_linkage', 'input_identity', 'mask_values_and_transparency',
                                'counts', 'intersection_subset', 'uv_mapping_audit', 'vote_agreement', 'embedded_images_decoded'])
    (a.output/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in verification.items() if k!='image_sha256'},ensure_ascii=False))


if __name__ == '__main__':
    main()
