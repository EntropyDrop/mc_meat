"""Audit nested calibration, verify UV-only inference, and publish comparison."""
from pathlib import Path
import argparse,json,shutil,subprocess,sys
import numpy as np
import torch
from meat_features import ROOT,sha,write
from data_face_v13 import load_rows
from face_decoder_v14 import build
from predict_face_v14 import predict
from calibration_v14 import decode_calibrated
from predict_calibrated_v14 import prediction_details
from report_calibrated_v14 import report
from model_service import lock

def finish(a):
 out=a.run;parent=Path(json.loads((out/'data_source.json').read_text())['parent']);config=json.loads((out/'configuration.json').read_text());audits=json.loads((out/'nested_audit.json').read_text());assert len(audits)==12
 for r in audits:
  assert not set(r['train_ids'])&set(r['calibration_ids']);assert not (set(r['train_ids'])|set(r['calibration_ids']))&set(r['outer_held_ids'])
  ck=torch.load(out/('calibration-'+str(r['outer_fold']))/f"inner-{r['inner_fold']}.pt",map_location='cpu',weights_only=True);assert ck['train_ids']==r['train_ids']
 for name,h in json.loads((out/'source_hashes.json').read_text()).items():assert sha(out/'source'/name)==h
 rows,fp=load_rows(a.data,a.cache);selected=[next(r for r in rows if r['row']['evaluation_eligible'] and (r['semantic']==c).any()) for c in [6,9,10]];path=out/'context_calibrated_v14/model.pt';cal=json.loads(path.with_name('calibration.json').read_text());assert sha(path)==sha(parent/'context_v14/model.pt')==cal['parent_model_sha256']
 checks=[]
 with lock(ROOT/'service_runtime/gpu.lock'):
  ck=torch.load(path,map_location='cpu',weights_only=True);net=build(ck['input_dim'],ck['variant']);net.load_state_dict(ck['state_dict']);net.eval()
  for r in selected:
   prob,q,_=predict(net,r,'cpu');v=r['valid'].numpy();expected=decode_calibrated(q,v,v&(r['cache']['region']%6==0),cal['thresholds']);actual=prediction_details(path,a.data/r['row']['id']/'skin.png',a.cache)
   delta=float(np.abs(actual[0]-prob).max());assert delta<1e-5 and actual[2]['labels']==expected['labels'];checks.append(dict(id=r['row']['id'],max_probability_error=delta,label_disagreements=0))
  skin=a.data/selected[0]['row']['id']/'skin.png';cached=prediction_details(path,skin,a.cache);fresh=prediction_details(path,skin,None);delta=float(np.abs(cached[0]-fresh[0]).max());assert delta<1e-4 and cached[2]['labels']==fresh[2]['labels']
 shutil.copytree(out/'source',out/'inference_source');smoke=out/'standalone_smoke'
 cmd=[sys.executable,str(out/'inference_source/predict_calibrated_v14.py'),'--checkpoint',str(path),'--input',str(skin),'--cache',str(a.cache),'--output',str(smoke)];p=subprocess.run(cmd,capture_output=True,text=True);assert p.returncode==0,p.stderr;actual=json.loads((smoke/'prediction.json').read_text());assert actual['labels']==cached[2]['labels']
 write(out/'inference_audit.json',dict(held_labels_used_in_training_or_calibration=False,synthetic_used_for_calibration=False,checkpoints_verified=12,checks=checks,uncached_error=delta,archived_cli_passed=True,production_deployed=False))
 report(out);assert sha(ROOT/'service_runtime/production/active.json')==sha(out/'production_before.json')
 files=['configuration.json','comparison.json','rare_analysis.json','nested_audit.json','thresholds_by_fold.json','folds.json','evaluation_manifest.json','manifest.json','training_complete.json','inference_audit.json','review_report_portable.html','report_provenance.json','report_images.json','source_hashes.json']
 release=dict(schema='mc-meat-nested-v14',status='experimental_not_deployed',models={'context_calibrated_v14':dict(path=str(path),sha256=sha(path),calibration_sha256=sha(path.with_name('calibration.json')))},files={f:sha(out/f) for f in files},human_evaluation_cases=169,production_unchanged=True)
 write(out/'release.json',release);write(out/'status.json',dict(stage='complete'))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['run','data','cache']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');finish(a)
