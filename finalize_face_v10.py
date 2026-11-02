"""Freeze a standalone inference package, audit it, then render the comparison."""
import argparse,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from meat_features import ROOT,sha,write
from model_service import lock
from data_face_v9 import load_rows
from face_decoder_v9 import FaceDecoder,MEAT_IDS
from train_face_v9 import predict as base_predict
from rare_face_v10 import initialize,scores,decode
from predict_face_v10 import prediction_details
from report_face_v10 import report


def run(a):
 out=a.run
 def status(stage,**kw):write(a.job/'finalize_status.json',dict(stage=stage,time=time.time(),**kw))
 try:
  while not (out/'training_complete.json').exists():
   if (out/'failure.json').exists():raise RuntimeError((out/'failure.json').read_text())
   status('waiting_training');time.sleep(10)
  assert not (out/'inference_source').exists()
  shutil.copytree(out/'source',out/'inference_source')
  shutil.copy2(ROOT/'predict_face_v10.py',out/'inference_source/predict_face_v10.py')
  write(out/'inference_source_hashes.json',{p.name:sha(p) for p in (out/'inference_source').iterdir() if p.is_file()})
  rows,_=load_rows(a.data,a.cache);ck=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
  tests=[];status('inference_audit')
  with lock(ROOT/'service_runtime/gpu.lock'):
   anchor=FaceDecoder(ck['input_dim']);anchor.load_state_dict(ck['anchor']['state_dict']);anchor.eval()
   net=initialize(ck['anchor']);net.load_state_dict(ck['state_dict']);net.eval()
   for row in rows[:2]:
    b=base_predict(anchor,row,'cpu')[2]['labels'];s,_=scores(net,row,'cpu')
    for policy,thresholds in [('fixed',[.5]*3),('calibrated',ck['thresholds'])]:
     expected=decode(b,s,row['valid'].numpy(),thresholds)
     actual,z,v=prediction_details(out/'model.pt',a.data/row['row']['id']/'skin.png',a.cache,policy)
     assert actual['labels']==expected['labels'] and not np.array(actual['labels'])[~v].any()
     assert actual['selected']==np.flatnonzero(np.isin(actual['labels'],MEAT_IDS)).tolist()
     tests.append(dict(id=row['row']['id'],policy=policy,max_score_error=float(np.abs(z-s).max()),label_disagreements=0))
   row=rows[0];skin=a.data/row['row']['id']/'skin.png'
   ca=prediction_details(out/'model.pt',skin,a.cache);fresh=prediction_details(out/'model.pt',skin,None)
   delta=float(np.abs(ca[1]-fresh[1]).max());assert delta<1e-4 and ca[0]['labels']==fresh[0]['labels']
  # Invoke the frozen standalone CLI, with only the UV file and image cache.
  cmd=[sys.executable,str(out/'inference_source/predict_face_v10.py'),'--checkpoint',str(out/'model.pt'),'--input',str(a.data/rows[0]['row']['id']/'skin.png'),'--cache',str(a.cache),'--output',str(a.job/'standalone_smoke')]
  result=subprocess.run(cmd,capture_output=True,text=True);assert result.returncode==0,result.stderr
  write(out/'standalone_inference_audit.json',dict(cases=tests,uncached_max_score_error=delta,uncached_label_disagreements=0,archived_cli_passed=True,labels_used_as_input=False))
  status('report');report(out,a.cache)
  assert sha(ROOT/'service_runtime/production/active.json')==sha(out/'production_before.json')
  release=json.loads((out/'release.json').read_text())
  for n,h in release['files'].items():assert sha(out/n)==h,n
  status('complete',checkpoint=str(out/'model.pt'),report=str(out/'review_report_portable.html'),production_deployed=False)
 except Exception:
  status('failed',error=traceback.format_exc());raise

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['run','job','data','cache']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('high');run(a)
