"""Freeze committed seven-class reviews and reuse only hash-matched image features."""
import argparse,json,shutil,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
from prepare_human import prepare,git_bytes,digest
from semantic_schema_v3 import SCHEMA,CLASSES,CLASS_IDS,PARENT,validate_labels,encode_labels
from fine_supervision import weights
from meat_features import ROOT,sha,write,provenance
from face_features_v3 import provenance as face_provenance,prepare as prepare_faces
from train_supervised import prepare_cache
from model_service import lock


def snapshot(dataset,revision,out):
    prepare(dataset,revision,out)
    manifest=json.loads((out/'manifest.json').read_text())
    counts={str(c):dict(cells=0,skins=0) for c in CLASS_IDS}
    for r in manifest['rows']:
        d=out/r['id'];source=r['source_annotation'].removesuffix('_meat.json')+'_semantic'
        mb=git_bytes(dataset,manifest['dataset_commit'],source+'.json');png=git_bytes(dataset,manifest['dataset_commit'],source+'.png');m=json.loads(mb)
        assert m['schema']==SCHEMA and m['classes']==CLASSES and m['id']==r['id']
        assert m['skin_sha256']==r['skin_sha256'] and m['mask_sha256']==digest(png)
        valid=np.array(Image.open(d/'valid.png')).flatten()>0
        selected=np.flatnonzero(np.array(Image.open(d/'meat.png'))[...,0].flatten()).tolist()
        labels=validate_labels(m['labels'],valid,selected)
        assert m['valid_sha256']==digest(valid.astype(np.uint8).tobytes())
        assert m['fine_review_complete'] is True and m['reviewed_class_ids']==CLASS_IDS and m['reviewed_valid_sha256']==m['valid_sha256']
        assert m['human_saved'] is True and m['annotation_source']=='human_reviewed'
        assert m['known_count']==sum(c>0 for c in labels)
        assert png==encode_labels(labels,valid)
        assert np.flatnonzero(np.isin(labels,[c for c in CLASS_IDS if PARENT[c]])).tolist()==selected
        assert all(w==1 for c,w in zip(labels,weights(m)) if c)
        (d/'semantic.json').write_bytes(mb);(d/'semantic.png').write_bytes(png)
        r.update(semantic_sha256=sha(d/'semantic.png'),semantic_metadata_sha256=sha(d/'semantic.json'),semantic_count=m['known_count'],fine_review_complete=True)
        for c in CLASS_IDS:
            n=labels.count(c);counts[str(c)]['cells']+=n;counts[str(c)]['skins']+=int(n>0)
    manifest.update(semantic_schema=SCHEMA,semantic_class_ids=CLASS_IDS,reviewed_count=len(manifest['rows']),class_counts=counts)
    write(out/'manifest.json',manifest)
    write(out/'annotation_audit.json',dict(committed_revision=manifest['dataset_commit'],cases=len(manifest['rows']),all_reviewed=True,conflicts=0,classes=counts,hidden_cells_excluded=True))
    return manifest


def seed(data,out,fp,face=False):
    out.mkdir(parents=True);write(out/'provenance.json',fp)
    candidates=list((ROOT/'cache').glob('*'))+list((ROOT/'service_runtime/production/jobs').glob('*/face_cache' if face else '*/cache'))
    compatible=[p for p in candidates if p!=out and (p/'provenance.json').exists() and json.loads((p/'provenance.json').read_text())==fp]
    n=0
    for row in json.loads((data/'manifest.json').read_text())['rows']:
        for src in compatible:
            p=src/(row['id']+'.npz');m=p.with_suffix('.json')
            if not p.exists() or not m.exists():continue
            meta=json.loads(m.read_text())
            if meta.get('skin_sha256' if face else 'input_sha256')!=row['skin_sha256']:continue
            if meta.get('sha256' if face else 'cache_sha256')!=sha(p):continue
            shutil.copy2(p,out/p.name);shutil.copy2(m,out/m.name);n+=1;break
    return n


def run(a):
    a.job.mkdir(parents=True,exist_ok=True)
    def status(stage,**kw):write(a.job/'status.json',dict(stage=stage,time=time.time(),**kw))
    try:
        status('snapshot');manifest=snapshot(a.dataset,a.revision,a.data)
        write(a.job/'dataset.json',dict(path=str(a.data),manifest_sha256=sha(a.data/'manifest.json'),cases=len(manifest['rows']),classes=manifest['class_counts']))
        status('waiting_gpu')
        with lock(ROOT/'service_runtime/gpu.lock'):
            cached=seed(a.data,a.cache,provenance());faces=seed(a.data,a.faces,face_provenance(),True)
            write(a.job/'cache_reuse.json',dict(multiview=cached,faces=faces,total=len(manifest['rows'])))
            status('multiview_features',reused=cached,total=len(manifest['rows']));prepare_cache(a.data,a.cache)
            status('face_features',reused=faces,total=len(manifest['rows']));prepare_faces(a.data,a.cache,a.faces)
        status('features_complete',cases=len(manifest['rows']))
    except Exception:
        status('failed',error=traceback.format_exc());raise

if __name__=='__main__':
    import torch
    p=argparse.ArgumentParser()
    for n in ['dataset','data','cache','faces','job']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--revision',required=True);a=p.parse_args()
    torch.set_num_threads(4);run(a)
