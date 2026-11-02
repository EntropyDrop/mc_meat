"""Run a compare-only seven-class experiment after feature preparation completes."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from meat_features import ROOT,write,sha
from model_service import lock
from data_face_v9 import load_rows
from train_face_v9b import train,predict
from face_decoder_v9 import FaceDecoder,MEAT_IDS
from predict_face_v9 import prediction_details
from report_face_v9b import report


def run(a):
 def status(stage,**kw):write(a.job/'training_status.json',dict(stage=stage,time=time.time(),**kw))
 try:
  while True:
   s=json.loads((a.job/'status.json').read_text()) if (a.job/'status.json').exists() else {}
   if s.get('stage')=='failed':raise RuntimeError('Feature preparation failed: '+s.get('error',''))
   if s.get('stage')=='features_complete':break
   status('waiting_features',feature_stage=s.get('stage'));time.sleep(10)
  status('tests')
  p=subprocess.run([sys.executable,str(ROOT/'test_face_v9b.py')],capture_output=True,text=True)
  (a.job/'tests.log').write_text(p.stdout+p.stderr);assert p.returncode==0,'Seven-class unit tests failed'
  status('waiting_gpu')
  with lock(ROOT/'service_runtime/gpu.lock'):
   status('training');train(a.data,a.cache,a.output,a.reference)
   status('standalone_inference')
   rows,_=load_rows(a.data,a.cache);audits=[]
   for variant in ['surface_v9','face_v9']:
    checkpoint=a.output/variant/'model.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    net=FaceDecoder(ck['input_dim'],facial=variant=='face_v9');net.load_state_dict(ck['state_dict']);net.eval()
    for r in rows[:2]:
     expected=predict(net,r,device='cpu')
     prob,valid,result=prediction_details(checkpoint,a.data/r['row']['id']/'skin.png',a.cache)
     assert np.max(np.abs(prob-expected[0]))<1e-5 and result['labels']==expected[2]['labels']
     assert not np.array(result['labels'])[~valid].any()
     assert result['selected']==np.flatnonzero(np.isin(result['labels'],MEAT_IDS)).tolist()
     audits.append(dict(variant=variant,id=r['row']['id'],max_probability_error=float(np.max(np.abs(prob-expected[0]))),label_differences=0))
   # Exercise true UV-only extraction without a cache or any annotation files.
   r=rows[0];ck=a.output/'face_v9/model.pt';skin=a.data/r['row']['id']/'skin.png'
   cached=prediction_details(ck,skin,a.cache);fresh=prediction_details(ck,skin,None)
   diff=float(np.max(np.abs(cached[0]-fresh[0])));ld=sum(x!=y for x,y in zip(cached[2]['labels'],fresh[2]['labels']))
   assert diff<1e-4 and ld==0,'Uncached inference differs from cached image features'
   write(a.output/'standalone_inference_audit.json',dict(cases=audits,uncached=dict(id=r['row']['id'],max_probability_error=diff,label_differences=ld),labels_used_as_input=False,production_unchanged=sha(ROOT/'service_runtime/production/active.json')==sha(a.output/'production_before.json')))
  status('report');report(a.output,a.cache)
  release=json.loads((a.output/'release.json').read_text())
  for name,h in release['files'].items():assert sha(a.output/name)==h
  for model in release['models'].values():assert sha(Path(model['path']))==model['sha256']
  assert sha(ROOT/'service_runtime/production/active.json')==sha(a.output/'production_before.json')
  status('complete',report=str(a.output/'review_report_portable.html'),checkpoint=str(a.output/'face_v9/model.pt'),production_deployed=False)
 except Exception:
  status('failed',error=traceback.format_exc());raise

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['data','cache','output','reference','job']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('high');run(a)
