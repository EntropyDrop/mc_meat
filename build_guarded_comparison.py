"""Derive guarded inference artifacts without modifying trained experiments."""
import argparse, copy, importlib, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from meat_features import ROOT, sha, write
from model_service import lock
from data_face_v9 import load_rows
from train_face_v9 import store_case, aggregate, predict as base_predict
from face_decoder_v9 import FaceDecoder
from rare_guard import guarded_thresholds, decode
from predict_rare_guard import prediction_details
from report_rare_guard import report


def derive(parent, out):
    assert not out.exists(), 'Preserve previous experiments'
    out.mkdir(parents=True)
    source = json.loads((parent/'data_source.json').read_text())
    data, cache = Path(source['data']), Path(source['cache'])
    rows, _ = load_rows(data, cache); byid = {r['row']['id']: r for r in rows}
    original = json.loads((parent/'comparison.json').read_text())
    calibration = json.loads((parent/'calibration.json').read_text())
    for filename in ['manifest.json', 'folds.json', 'production_before.json']:
        shutil.copy2(parent/filename, out/filename)
    assert sha(ROOT/'service_runtime/production/active.json') == sha(out/'production_before.json')
    source['guard_parent'] = str(parent); write(out/'data_source.json', source)
    shutil.copytree(parent/'inference_source', out/'inference_source')
    for name in ['rare_guard.py', 'predict_rare_guard.py', 'report_rare_guard.py', 'build_guarded_comparison.py', 'test_rare_guard.py']:
        shutil.copy2(ROOT/name, out/'inference_source'/name)
    write(out/'inference_source_hashes.json', {p.name:sha(p) for p in (out/'inference_source').iterdir() if p.is_file()})
    summary = {k: original['results'][k] for k in ['ordinary_v9', 'balanced_v9', 'rare_raw']}
    summary['rare_unrestricted'] = original['results']['rare_calibrated']
    for r in rows:
        d=out/r['row']['id']; d.mkdir()
        for name in ['skin.png', 'truth.npz', 'rare_scores.npz', 'ordinary_v9.npz', 'balanced_v9.npz', 'rare_raw.npz']:
            shutil.copy2(parent/r['row']['id']/name, d/name)
        shutil.copy2(parent/r['row']['id']/'rare_calibrated.npz', d/'rare_unrestricted.npz')
    guarded = []; per = []
    for fold, cal in zip(original['folds'], calibration['outer_folds']):
        assert fold['fold'] == cal['fold']
        thresholds, enabled = guarded_thresholds(cal['calibration'])
        guarded.append(dict(fold=fold['fold'], thresholds=thresholds, enabled=enabled,
                            evidence=cal['calibration'], source='inner OOF only'))
        for rid in fold['held_ids']:
            r=byid[rid]; values=np.load(parent/rid/'rare_scores.npz')
            prob=np.load(parent/rid/'rare_raw.npz')['probability']
            details=decode(values['base'], values['score'], r['valid'].numpy(), thresholds, enabled)
            per.append(store_case(out,r,'rare_calibrated',prob,values['score'],details,fold['fold'],fold['train_ids']))
    summary['rare_calibrated'] = aggregate(per)
    write(out/'comparison.json', dict(cases=len(rows), results=summary, folds=original['folds'],
        evaluation='Exploratory historical grouped OOF; guard uses inner calibration evidence only. Not a new blind test. No new training in this derivative.'))
    thresholds, enabled=guarded_thresholds(calibration['final_calibration'])
    write(out/'calibration.json', dict(outer_folds=guarded, final_thresholds=thresholds, final_enabled=enabled,
         final_calibration=calibration['final_calibration'], policy='precision>=0.35 and tp>0; otherwise retain ordinary class',
         final_threshold_source='whole-data OOF, not used to replace outer validation thresholds'))
    ck=torch.load(parent/'model.pt',map_location='cpu',weights_only=True)
    original_ck=copy.deepcopy(ck); module=importlib.import_module(ck['model_module'])
    ck.update(schema='mc-meat-rare-guarded-v1', parent_model_module=ck['model_module'], model_module='rare_guard',
              calibration_enabled=enabled, thresholds=thresholds, parent_checkpoint_sha256=sha(parent/'model.pt'))
    torch.save(ck,out/'model.pt')
    saved=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(saved['state_dict'][k],v) for k,v in original_ck['state_dict'].items())
    assert all(torch.equal(saved['anchor']['state_dict'][k],v) for k,v in original_ck['anchor']['state_dict'].items())
    write(out/'training_complete.json', dict(operation='calibration_only_derivative', additional_training_steps=0,
        parent=str(parent), parent_training_complete=json.loads((parent/'training_complete.json').read_text()),
        checkpoint_sha256=sha(out/'model.pt'), model_tensors_unchanged=True))
    write(out/'inference_audit.json', dict(model_tensors_unchanged=True, outer_group_disjoint=True,
        thresholds_use_inner_OOF_only=True, labels_used_as_input=False, production_deployed=False))
    with lock(ROOT/'service_runtime/gpu.lock'):
        anchor=FaceDecoder(ck['input_dim']);anchor.load_state_dict(ck['anchor']['state_dict']);anchor.eval()
        net=module.initialize(ck['anchor']);net.load_state_dict(ck['state_dict']);net.eval()
        tests=[]
        for row in rows[:2]:
            base=base_predict(anchor,row,'cpu')[2]['labels']; score,_=module.scores(net,row,'cpu')
            expected=decode(base,score,row['valid'].numpy(),thresholds,enabled)
            actual,z,v=prediction_details(out/'model.pt',data/row['row']['id']/'skin.png',cache)
            assert actual['labels']==expected['labels'];assert np.max(np.abs(score-z))<1e-6
            tests.append(dict(id=row['row']['id'],label_disagreements=0,max_score_error=float(np.abs(score-z).max())))
        skin=data/rows[0]['row']['id']/'skin.png'
        cached=prediction_details(out/'model.pt',skin,cache);fresh=prediction_details(out/'model.pt',skin,None)
        delta=float(np.abs(cached[1]-fresh[1]).max());assert delta<1e-4 and cached[0]['labels']==fresh[0]['labels']
    result=subprocess.run([sys.executable,str(out/'inference_source/predict_rare_guard.py'),
        '--checkpoint',str(out/'model.pt'),'--input',str(skin),'--cache',str(cache),'--output',str(out/'cli_smoke')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    write(out/'standalone_inference_audit.json',dict(cases=tests,uncached_max_score_error=delta,
        uncached_label_disagreements=0,archived_cli_passed=True,labels_used_as_input=False))
    report(out,cache)
    release=json.loads((out/'release.json').read_text())
    release.update(schema='mc-meat-rare-guarded-v1', parent_checkpoint_sha256=sha(parent/'model.pt'),
        models={'guarded':dict(path=str(out/'model.pt'),sha256=sha(out/'model.pt'))})
    write(out/'release.json',release)
    for n,h in release['files'].items(): assert sha(out/n)==h,n
    assert sha(ROOT/'service_runtime/production/active.json')==sha(out/'production_before.json')
    write(out/'status.json',dict(stage='complete',time=time.time(),production_deployed=False))
    print(out.name,summary['rare_calibrated']['macro'],summary['rare_calibrated']['fine'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('high');derive(a.parent,a.output)
