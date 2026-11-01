"""Verify shipped decoder using only a UV image, with and without feature cache."""
import json,shutil
from pathlib import Path
import numpy as np
import torch
from meat_features import ROOT,sha,write
from model_service import lock
from predict_face_v5 import probabilities
torch.set_num_threads(4)
run=ROOT/'runs/face_v5b_20261101';base=ROOT/'runs/face_v5_20261101';data=ROOT/'data/face_v5_20261101';cache=ROOT/'cache/semantic_v3_20261101'
manifest=json.loads((data/'manifest.json').read_text());row=manifest['rows'][0];stage=run/'uv_only_audit';stage.mkdir(exist_ok=True)
skin=stage/'input.png';shutil.copy2(data/row['id']/'skin.png',skin)
# Directory contains no human/fine labels. Inference cannot depend on sidecars.
assert not any(p.name.endswith(('meat.png','semantic.png','annotation.json','semantic.json')) for p in stage.iterdir())
audits=[]
with lock(ROOT/'service_runtime/gpu.lock'):
 for directory,variant in [(base,'face_binary'),(base,'face_fine'),(run,'face_front_fine')]:
  ck=directory/variant/'model.pt';p,valid=probabilities(ck,skin,cache);assert np.isfinite(p).all() and not p[~valid].any()
  audits.append(dict(variant=variant,checkpoint_sha256=sha(ck),valid_count=int(valid.sum()),finite=True,invalid_output_zero=True))
 ck=run/'face_front_fine/model.pt';cached,valid=probabilities(ck,skin,cache);fresh,freshvalid=probabilities(ck,skin,None)
 assert np.array_equal(valid,freshvalid);delta=float(np.abs(cached-fresh).max());flips=int(((cached>=.5)!=(fresh>=.5)).sum());assert delta==0 and flips==0
active=json.loads((ROOT/'service_runtime/production/active.json').read_text());assert active['id']=='manual-2026-10-31-f3eb6076'
write(run/'uv_only_audit.json',dict(uv_sha256=sha(skin),case=row['id'],labels_present_in_input_directory=False,labels_used_as_input=False,models=audits,fresh_vs_cache_max_abs=delta,mask_flips=flips,production_model_unchanged=active['id'],checked_fresh_cases=1))
print('UV-only inference verified',delta,flips)
