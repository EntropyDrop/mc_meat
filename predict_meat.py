"""UV-only inference. Never write into DDJ label directories implicitly."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from meat_features import extract,make_dino,make_renderer,matrix,provenance,sha,write,read_skin,local_features
from train_supervised import MeatHead
from uv_visibility import POLICY, annotation_mask


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');a=p.parse_args()
    if a.output.exists():raise ValueError('Refusing to overwrite an existing output')
    torch.set_num_threads(4)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    assert ck['schema']=='mc-meat-supervised-v1'
    if ck['variant']=='dino_local':
        assert ck['feature_provenance']==provenance(),'Feature provenance changed'
        model=make_dino(a.device);renderer=make_renderer(a.device)
        values,audit=extract(a.input,model,renderer,a.device)
    else:
        rgba,parts,valid=read_skin(a.input);local,region=local_features(rgba,parts,valid)
        values=dict(local=local,valid=valid.flatten());audit=[]
    head=MeatHead(ck['input_dim']);head.load_state_dict(ck['state_dict']);head.eval()
    with torch.no_grad():prob=head(matrix(values,ck['variant'])).sigmoid().numpy().reshape(64,64)
    valid=values['valid'].reshape(64,64);prob[~valid]=0
    assert np.isfinite(prob).all()
    pred=((prob>=ck['config']['threshold'])&valid).astype(np.uint8)*255
    manifest_path=a.checkpoint.parent.parent/'manifest.json'
    seen=None
    if manifest_path.exists() and sha(manifest_path)==ck['manifest_sha256']:
        manifest=json.loads(manifest_path.read_text())
        seen=any(r['skin_sha256']==sha(a.input) and r['id'] in ck['train_ids'] for r in manifest['rows'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(annotation_mask(np.flatnonzero(pred).tolist(),valid.astype(np.uint8).tobytes()))
    np.save(a.output.with_suffix('.probability.npy'),prob)
    write(a.output.with_suffix('.json'),dict(schema='mc-meat-prediction-v1',human_label=False,
        visibility_policy=POLICY,input_sha256=sha(a.input),checkpoint_sha256=sha(a.checkpoint),mask_sha256=sha(a.output),
        threshold=ck['config']['threshold'],valid_cells=int(valid.sum()),meat_cells=int((pred>0).sum()),
        mapping_audit=audit,input_seen_during_training=seen,
        warning='Experimental model; predictions are not saved human annotations.'))
    print(json.dumps(dict(output=str(a.output),meat_cells=int((pred>0).sum()))))


if __name__=='__main__':main()
