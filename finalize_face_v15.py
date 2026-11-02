"""Audit corrected-label experiments and export offline reports, never deploy."""
from pathlib import Path
import argparse,json,shutil,subprocess,sys
import torch,numpy as np
from meat_features import ROOT,sha,write
from data_face_v13 import load_rows
from face_decoder_v15 import build
from face_decoder_v9 import MEAT_IDS
from predict_face_v15 import predict,prediction_details
from report_face_v15 import report
from diagnose_faces_v15 import diagnose
from model_service import lock
VARIANTS=['original_labels_control_v15','corrected_labels_v15','bounded_refine_v15']

def finish(a):
 out=a.run;assert (out/'training_complete.json').exists()
 for name,h in json.loads((out/'source_hashes.json').read_text()).items():assert sha(out/'source'/name)==h
 parent=Path(json.loads((out/'data_source.json').read_text())['parent']);assert sha(out/'folds.json')==sha(parent/'folds.json')
 rows,_=load_rows(a.data,a.cache);byid={r['row']['id']:r for r in rows};folds=json.loads((out/'fold_audit.json').read_text());assert len(folds)==18
 checks=[]
 for f in folds:
  assert not set(f['train_ids'])&set(f['held_ids']);name='model.pt' if f['fold']=='full' else f"fold-{f['fold']}.pt";path=out/f['variant']/name;ck=torch.load(path,map_location='cpu',weights_only=True);assert ck['train_ids']==f['train_ids']
  if f['variant']=='bounded_refine_v15':
   base=torch.load(out/'corrected_labels_v15'/name,map_location='cpu',weights_only=True);assert ck['train_ids']==base['train_ids'];assert all(torch.equal(v,ck['state_dict']['base.'+k]) for k,v in base['state_dict'].items())
  checks.append(dict(variant=f['variant'],fold=f['fold'],sha256=sha(path),held_train_disjoint=True))
 write(out/'checkpoint_audit.json',checks)
 sample=[next(r for r in rows if r['row'].get('evaluation_eligible') and (r['semantic']==k).any()) for k in [6,9,10]];sample.append(next(r for r in rows if r['row'].get('hard_negative_only')));checks=[]
 with lock(ROOT/'service_runtime/gpu.lock'):
  for v in VARIANTS:
   path=out/v/'model.pt';ck=torch.load(path,map_location='cpu',weights_only=True);net=build(ck['input_dim'],v);net.load_state_dict(ck['state_dict']);net.eval()
   for r in sample:
    expected=predict(net,r,'cpu');actual=prediction_details(path,a.data/r['row']['id']/'skin.png',a.cache);delta=float(np.abs(actual[0]-expected[0]).max());assert delta<1e-5 and actual[2]['labels']==expected[2]['labels']
    labels=np.array(actual[2]['labels']);assert not labels[~actual[1]].any();assert actual[2]['selected']==np.flatnonzero(np.isin(labels,MEAT_IDS)).tolist();checks.append(dict(variant=v,id=r['row']['id'],max_probability_error=delta,label_disagreements=0))
  skin=a.data/sample[0]['row']['id']/'skin.png';cached=prediction_details(out/'bounded_refine_v15/model.pt',skin,a.cache);fresh=prediction_details(out/'bounded_refine_v15/model.pt',skin,None);delta=float(np.abs(cached[0]-fresh[0]).max());assert delta<1e-4 and cached[2]['labels']==fresh[2]['labels']
 cli=out/'source/predict_face_v15.py';result=subprocess.run([sys.executable,str(cli),'--checkpoint',str(out/'bounded_refine_v15/model.pt'),'--input',str(skin),'--cache',str(a.cache),'--output',str(a.job/'standalone_smoke')],capture_output=True,text=True);assert result.returncode==0,result.stderr
 assert json.loads((a.job/'standalone_smoke/prediction.json').read_text())['labels']==cached[2]['labels']
 write(out/'standalone_inference_audit.json',dict(cases=checks,uncached_max_probability_error=delta,uncached_label_disagreements=0,archived_cli_passed=True,labels_used_as_input=False))
 for name in ['correction_audit.json','preflight.json']:shutil.copy2(a.job/name,out/name)
 diagnose(out);report(out)
 assert sha(ROOT/'service_runtime/production/active.json')==sha(out/'production_before.json')
 files=['configuration.json','comparison.json','rare_analysis.json','face_distribution_audit.json','fold_audit.json','checkpoint_audit.json','folds.json','evaluation_manifest.json','manifest.json','training_complete.json','inference_audit.json','standalone_inference_audit.json','correction_audit.json','preflight.json','review_report_portable.html','report_provenance.json','report_images.json','source_hashes.json']
 release=dict(schema='mc-meat-corrected-v15',status='experimental_not_deployed',models={v:dict(path=str(out/v/'model.pt'),sha256=sha(out/v/'model.pt')) for v in VARIANTS},files={n:sha(out/n) for n in files},human_evaluation_cases=169,weak_training_cases=60,beard_support=77,correction_cases=2,correction_cells=88,production_unchanged=True)
 write(out/'release.json',release)
 for n,h in release['files'].items():assert sha(out/n)==h
 print(json.dumps(dict(status='complete',models=release['models'],production_unchanged=True)))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['run','job','data','cache']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');finish(a)
