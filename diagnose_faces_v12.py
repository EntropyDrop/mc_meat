"""Post-training surface coverage diagnostics; no label or threshold changes."""
import argparse,json
from pathlib import Path
import numpy as np
from meat_features import FACES,write

def diagnose(run):
 src=json.loads((run/'data_source.json').read_text());data=Path(src['data']);cache=Path(src['cache']);rows=json.loads((data/'manifest.json').read_text())['rows'];human={r['id'] for r in json.loads((run/'evaluation_manifest.json').read_text())['rows']};out={}
 variants=['ordinary_control','balanced_control','ordinary_augmented','balanced_augmented']
 for cls in [6,9,10]:
  hist={n:0 for n in FACES};added={n:0 for n in FACES};mets={v:{n:{'tp':0,'fp':0,'fn':0} for n in [*FACES,'outside_head']} for v in variants}
  for r in rows:
   if r['id'] not in human and not r.get('training_eligible'):continue
   a=np.load(cache/(r['id']+'.npz'));valid=a['valid'].astype(bool);head=a['region']%6==0;face=a['local'][:,42:48].argmax(1);truth=np.array(json.loads((data/r['id']/'semantic.json').read_text())['labels']);y=truth==cls
   target=hist if r['id'] in human else added
   for i,n in enumerate(FACES):target[n]+=int((valid&head&(face==i)&y).sum())
   if r['id'] not in human:continue
   for variant in variants:
    p=np.load(run/r['id']/(variant+'.npz'))['labels']==cls
    for i,n in enumerate([*FACES,'outside_head']):
     mask=valid&(~head if n=='outside_head' else head&(face==i))
     for k,z in [('tp',y&p),('fp',~y&p),('fn',y&~p)]:mets[variant][n][k]+=int((mask&z).sum())
  out[str(cls)]=dict(human_positive_cells_by_head_face=hist,added_positive_cells_by_head_face=added,metrics=mets)
 write(run/'face_distribution_audit.json',out);print(json.dumps(out))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();diagnose(a.run)
