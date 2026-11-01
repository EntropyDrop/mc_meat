"""Audit released models, fold exclusion, and UV-only inference without labels."""
import argparse,base64,json,re,shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from meat_features import ROOT,sha,write
from model_service import lock
from predict_face_v5 import probabilities,predict

def audit(run):
 source=json.loads((run/'data_source.json').read_text());data=Path(source['data']);cache=Path(source['cache'])
 rows=json.loads((run/'manifest.json').read_text())['rows'];folds=json.loads((run/'folds.json').read_text());variants=['baseline_retrained','face_binary','face_front_fine'];count=0
 assert sha(data/'manifest.json')==source['manifest_sha256']==sha(run/'manifest.json')
 for variant in variants:
  for fold in folds:
   ck=torch.load(run/variant/f"fold-{fold['fold']}.pt",weights_only=True,map_location='cpu')
   assert ck['train_ids']==fold['train_ids'] and set(ck['train_ids']).isdisjoint(fold['held_ids'])
   assert ck['manifest_sha256']==sha(run/'manifest.json');count+=1
  ck=torch.load(run/variant/'model.pt',weights_only=True,map_location='cpu');assert set(ck['train_ids'])=={r['id'] for r in rows}
  assert not any('dino' in k for k in ck['state_dict'])
 stage=run/'uv_only_audit';stage.mkdir(exist_ok=True);skin=stage/'input.png';shutil.copy2(data/rows[0]['id']/'skin.png',skin)
 assert not any(p.name in ['meat.png','semantic.png','annotation.json','semantic.json'] for p in stage.iterdir())
 with lock(ROOT/'service_runtime/gpu.lock'):
  checkpoint=run/'face_front_fine/model.pt';cached,valid=probabilities(checkpoint,skin,cache);fresh,freshvalid=probabilities(checkpoint,skin,None)
  assert np.array_equal(valid,freshvalid);delta=float(abs(cached-fresh).max());flips=int(((cached>=.5)!=(fresh>=.5)).sum());assert delta==0 and flips==0
  predict(checkpoint,skin,stage/'prediction.png',cache)
 mask=np.asarray(Image.open(stage/'prediction.png'));assert mask.shape==(64,64,4) and np.array_equal(mask[...,3].flatten()>0,valid)
 active=ROOT/'service_runtime/production/active.json';assert sha(active)==sha(run/'production_before.json')
 images=json.loads((run/'report_images.json').read_text());assert all(sha(run/k)==v for k,v in images.items())
 report=(run/'review_report_portable.html').read_text();embedded=re.findall('src="data:image/png;base64,([^\"]+)"',report)
 import hashlib
 hashes={hashlib.sha256(base64.b64decode(v)).hexdigest() for v in embedded};assert hashes.issubset(set(images.values()))
 a=dict(checked_folds=count,whole_group_exclusion=True,uv_sha256=sha(skin),fresh_vs_cache_max_abs=delta,mask_flips=flips,fresh_cases=1,labels_present_in_input_directory=False,dino_frozen=True,dino_backbone_sha256=json.loads((run/'feature_provenance.json').read_text())['parent']['backbone_sha256'],production_unchanged=True,production_id=json.loads(active.read_text())['id'],report_embedded_images=len(embedded),report_image_hashes_verified=True,mask_shape=list(mask.shape),invalid_output_transparent=True)
 a['audit_code_sha256']=sha(Path(__file__));a['inference_code_sha256']=sha(ROOT/'predict_face_v5.py')
 report_source=run/'report_source';report_source.mkdir(exist_ok=True)
 for f in ['report_face_v6.py','audit_face_v6.py']:shutil.copy2(ROOT/f,report_source/f)
 write(run/'final_audit.json',a)
 models={v:dict(path=str(run/v/'model.pt'),sha256=sha(run/v/'model.pt')) for v in variants}
 files=['comparison.json','region_comparison.json','class_comparison.json','paired_comparison.json','annotation_audit.json','inference_audit.json','final_audit.json','review_report_portable.html','report_images.json','report_provenance.json','manifest.json','configuration.json']
 write(run/'release.json',dict(status='experimental_not_deployed',data=str(data),manifest_sha256=sha(run/'manifest.json'),models=models,files={f:sha(run/f) for f in files},primary_candidate='face_front_fine',limitations=['historical OOF, not a fresh blind test','34 skins and one seed only','no beard examples; mouth scarce','fine labels include retained assistant proposals']))
 write(run/'status.json',dict(stage='complete',cases=len(rows),groups=len(folds),variants=variants,audit_passed=True,production_deployed=False));print(json.dumps(a))

if __name__=='__main__':
 torch.set_num_threads(4);p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();audit(args.run)
