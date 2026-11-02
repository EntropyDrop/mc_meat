"""Verify frozen UV-only inference and publish an offline experiment report."""
from pathlib import Path
import argparse,json,shutil,time,traceback,subprocess,sys
import numpy as np
import torch
from meat_features import ROOT,sha,write
from data_face_v13 import load_rows
from face_decoder_v14 import build
from face_decoder_v9 import MEAT_IDS
from predict_face_v14 import predict
from predict_face_v14_r2 import prediction_details
from report_face_v14 import report
from model_service import lock


def finish(a):
 out=a.run
 def status(stage,**kw):write(a.job/'finalize_status.json',dict(stage=stage,time=time.time(),**kw))
 while not (out/'training_complete.json').exists():
  if (out/'failure.json').exists():raise RuntimeError((out/'failure.json').read_text())
  status('waiting_training');time.sleep(10)
 status('verify_artifacts')
 for name,h in json.loads((out/'source_hashes.json').read_text()).items():assert sha(out/'source'/name)==h
 for item in json.loads((out/'inference_audit.json').read_text())['save_reload']:assert sha(out/item['variant']/'model.pt')==item['checkpoint_sha256']
 assert sha(out/'folds.json')==sha(Path(json.loads((out/'data_source.json').read_text())['parent'])/'folds.json')
 rows,_=load_rows(a.data,a.cache);byid={r['row']['id']:r for r in rows};folds=json.loads((out/'fold_audit.json').read_text())
 for f in folds:
  assert not set(f['train_ids'])&set(f['held_ids']);assert not set(f['auxiliary_train_ids'])&set(f['held_ids'])
  ck=torch.load(out/f['variant']/f"fold-{f['fold']}.pt",map_location='cpu',weights_only=True);assert ck['train_ids']==f['train_ids']
 status('uv_only_inference_audit');checks=[]
 sample=[next(r for r in rows if r['row'].get('evaluation_eligible') and (r['semantic']==k).any()) for k in [6,9,10]]
 sample.append(next(r for r in rows if r['row'].get('training_eligible')))
 with lock(ROOT/'service_runtime/gpu.lock'):
  for variant in ['loss_v14','context_v14','gated_v14']:
   path=out/variant/'model.pt';ck=torch.load(path,map_location='cpu',weights_only=True);net=build(ck['input_dim'],variant);net.load_state_dict(ck['state_dict']);net.eval()
   for r in sample:
    expected=predict(net,r,'cpu');actual=prediction_details(path,a.data/r['row']['id']/'skin.png',a.cache)
    delta=float(np.abs(actual[0]-expected[0]).max());assert delta<1e-5 and actual[2]['labels']==expected[2]['labels']
    labels=np.array(actual[2]['labels']);assert not labels[~actual[1]].any();assert actual[2]['selected']==np.flatnonzero(np.isin(labels,MEAT_IDS)).tolist()
    checks.append(dict(variant=variant,id=r['row']['id'],max_probability_error=delta,label_disagreements=0))
  skin=a.data/sample[0]['row']['id']/'skin.png';cached=prediction_details(out/'gated_v14/model.pt',skin,a.cache);fresh=prediction_details(out/'gated_v14/model.pt',skin,None)
  delta=float(np.abs(cached[0]-fresh[0]).max());assert delta<1e-4 and cached[2]['labels']==fresh[2]['labels']
 assert not (out/'inference_source').exists();shutil.copytree(out/'source',out/'inference_source');shutil.copy2(ROOT/'predict_face_v14_r2.py',out/'inference_source/predict_face_v14_r2.py');write(out/'inference_source_hashes.json',{p.name:sha(p) for p in (out/'inference_source').iterdir() if p.is_file()})
 cmd=[sys.executable,str(out/'inference_source/predict_face_v14_r2.py'),'--checkpoint',str(out/'gated_v14/model.pt'),'--input',str(skin),'--cache',str(a.cache),'--output',str(a.job/'standalone_smoke')]
 result=subprocess.run(cmd,capture_output=True,text=True);assert result.returncode==0,result.stderr
 actual=json.loads((a.job/'standalone_smoke/prediction.json').read_text());assert actual['labels']==cached[2]['labels']
 write(out/'standalone_inference_audit.json',dict(cases=checks,uncached_max_probability_error=delta,uncached_label_disagreements=0,archived_cli_passed=True,labels_used_as_input=False))
 status('report');report(out)
 assert sha(ROOT/'service_runtime/production/active.json')==sha(out/'production_before.json')
 files=['configuration.json','comparison.json','rare_analysis.json','fold_audit.json','folds.json','evaluation_manifest.json','manifest.json','training_complete.json','inference_audit.json','standalone_inference_audit.json','review_report_portable.html','report_provenance.json','report_images.json','source_hashes.json','inference_source_hashes.json']
 release=dict(schema='mc-meat-structured-v14',status='experimental_not_deployed',models={v:dict(path=str(out/v/'model.pt'),sha256=sha(out/v/'model.pt')) for v in ['loss_v14','context_v14','gated_v14']},files={name:sha(out/name) for name in files},human_evaluation_cases=169,weak_training_cases=60,real_revised_cases=42,synthetic_makeup_cases=12,partial_hard_negative_cases=6,production_unchanged=True)
 write(out/'release.json',release)
 for name,h in release['files'].items():assert sha(out/name)==h
 status('complete',report=str(out/'review_report_portable.html'),production_deployed=False)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['run','job','data','cache']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
 try:finish(a)
 except Exception:write(a.job/'finalize_status.json',dict(stage='failed',error=traceback.format_exc()));raise
