"""Import explicitly reviewed assistant labels without overwriting saved labels.

This importer does not infer labels. It preserves assistant provenance and builds
a new immutable training snapshot; existing human annotations are hash-checked.
"""
import argparse, base64, hashlib, json, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def encoded(obj): return (json.dumps(obj, ensure_ascii=False, indent=2)+'\n').encode()

def main(a):
 sys.path.insert(0,str(a.dataset/'best_annotator'))
 from app import Dataset
 from meat_annotations import digest
 from uv_visibility import annotation_mask, POLICY, SCHEMA
 from semantic_schema import validate_labels, encode_labels, CLASSES, SCHEMA as FINE_SCHEMA
 ds=Dataset(a.dataset); batch=json.loads(a.annotations.read_text()); old_manifest=json.loads((a.old_data/'manifest.json').read_text())
 before=json.loads(a.review_manifest.read_text())['existing_annotation_hashes']
 def preserved():
  for r in before: assert sha(a.dataset/r['path'])==r['sha256'], 'Existing annotation changed: '+r['path']
 preserved(); assert not a.output.exists(), 'Use a new snapshot directory'
 cases=batch['cases']; assert len(cases)==20 and len({c['id'] for c in cases})==20
 plans=[]; now=datetime.now(timezone.utc).isoformat()
 for c in cases:
  r=ds.records[c['key']]; current=ds.meat.load(c['key']); prior=json.loads((a.review/c['id']/'state.json').read_text())
  for k in ['skin_revision','revision','meat_revision','valid']: assert current[k]==prior[k], (c['id'],k)
  assert current['meat_status']=='missing' and current['skin_revision']==c['skin_sha256']
  valid=base64.b64decode(current['valid']); labels=c['labels']; selected=[i for i,x in enumerate(labels) if x in (1,2,3)]
  labels=validate_labels(labels,valid,selected); w=c['fine_weights']
  assert len(w)==4096 and all(np.isfinite(x) and 0<=x<=.8 and (labels[i]>0 or x==0) for i,x in enumerate(w))
  png=annotation_mask(selected,valid)
  meta=dict(schema=SCHEMA,id=c['id'],skin_file=current['skin_file'],skin_sha256=current['skin_revision'],mask_sha256=digest(png),model=current['model'],selected=selected,selected_count=len(selected),encoding={'meat':255,'non_meat':0,'invalid':'RGBA alpha=0; excluded from labels and loss'},visibility_policy=POLICY,valid_sha256=digest(valid),occluded_count=current['occluded_count'],complete=True,updated_at=now,annotation_source=c['annotation_source'],human_saved=False,binary_training_weight=.5,review_notes=c['review_notes'],selection_audit=c['selection_audit'],review_batch_sha256=sha(a.annotations))
  mb=encoded(meta); meat_revision=digest(b'|'.join(digest(b).encode() for b in (png,mb)))
  fine=encode_labels(labels,valid)
  fm=dict(schema=FINE_SCHEMA,classes=CLASSES,id=c['id'],labels=labels,known_count=sum(x>0 for x in labels),mask_sha256=digest(fine),skin_file=current['skin_file'],skin_sha256=current['skin_revision'],meat_revision=meat_revision,visibility_policy=POLICY,valid_sha256=digest(valid),human_saved=False,annotation_source=c['annotation_source'],fine_weights=w,review_notes=c['review_notes'],partial=True,unknown_class=0,encoding='RGBA class IDs 1,2,3,5,6; valid unknown=0; alpha=0 ignore',updated_at=now,review_batch_sha256=sha(a.annotations))
  files=list(zip(list(ds.meat.paths(r))+list(ds.semantics.paths(r)),[png,mb,fine,encoded(fm)]))
  assert all(not p.exists() and not p.is_symlink() for p,_ in files)
  assert all(len(b)<=131072 for _,b in files)
  plans.append((c,files))
 created=[]
 try:
  for c,files in plans:
   assert ds.meat.load(c['key'])['meat_status']=='missing'
   for p,b in files:
    with p.open('xb') as f:f.write(b)
    created.append((p,b))
   s=ds.semantics.load(c['key']); m=ds.meat.load(c['key'])
   assert s['status']=='saved' and s['conflict_count']==0 and m['meat_status']=='saved'
   assert s['labels']==c['labels'] and m['selected']==[i for i,x in enumerate(c['labels']) if x in (1,2,3)]
  preserved()
 except BaseException:
  for p,b in reversed(created):
   if p.is_file() and p.read_bytes()==b:p.unlink()
  raise
 # Copy old labels byte-for-byte; assistant annotations have separate provenance.
 a.output.mkdir(); rows=[]
 for r in old_manifest['rows']:
  d=a.old_data/r['id']; annotation=a.dataset/r['source_annotation']
  assert sha(annotation)==r['metadata_sha256'] and sha(annotation.with_suffix('.png'))==r['mask_sha256']
  shutil.copytree(d,a.output/r['id']); row=dict(r,annotation_source='existing_human_binary',binary_training_weight=1.,evaluation_eligible=True);rows.append(row)
 for c,_ in plans:
  d=a.output/c['id']; d.mkdir(); r=ds.records[c['key']]; skin,_,_=ds.meat.effective_skin(r)
  for source,name in [(skin,'skin.png'),*zip(ds.meat.paths(r),['meat.png','annotation.json']),*zip(ds.semantics.paths(r),['semantic.png','semantic.json'])]:shutil.copy2(source,d/name)
  for name in ['visible_skin.png','valid.png']:shutil.copy2(a.features/c['id']/name,d/name)
  row=dict(id=c['id'],model='classic',source_skin=str(skin.relative_to(a.dataset)),source_annotation=str(ds.meat.paths(r)[1].relative_to(a.dataset)),valid_cells=c['valid_cells'],meat_cells=c['meat_cells'],non_meat_cells=c['valid_cells']-c['meat_cells'],semantic_count=sum(c['class_counts'].values()),annotation_source=c['annotation_source'],binary_training_weight=.5,evaluation_eligible=False)
  for name,key in [('skin.png','skin_sha256'),('meat.png','mask_sha256'),('annotation.json','metadata_sha256'),('visible_skin.png','visible_skin_sha256'),('valid.png','valid_sha256'),('semantic.png','semantic_sha256'),('semantic.json','semantic_metadata_sha256')]:row[key]=sha(d/name)
  rows.append(row)
 sys.path.insert(0,str(a.repo));from prepare_human import groups_for
 groups,audit=groups_for([np.array(Image.open(a.output/r['id']/'skin.png').convert('RGBA')) for r in rows],[r['id'] for r in rows])
 old_groups={r['id']:r['group'] for r in old_manifest['rows']}
 for r,g in zip(rows,groups):
  if r['id'] in old_groups:assert old_groups[r['id']]==g,'Duplicate-group change requires a new control'
  r['group']=g
 manifest=dict(schema='mc-meat-human-plus-assistant-v1',rows=rows,source_human_manifest_sha256=sha(a.old_data/'manifest.json'),assistant_batch_sha256=sha(a.annotations),visibility_policy=POLICY,duplicate_audit=audit,independent_groups=len(set(groups)),snapshot_at=now,note='34 unchanged human binary labels for historical grouped OOF; 20 assistant visual annotations are training-only weak labels with binary weight 0.5; original fine-label confidence preserved.')
 (a.output/'manifest.json').write_bytes(encoded(manifest))
 audit=dict(added_cases=len(cases),unchanged_existing_files=before,created_files=[dict(path=str(p.relative_to(a.dataset)),sha256=sha(p)) for p,_ in created],canonical_roundtrip_passed=True,hidden_cells_ignored=True,mutually_exclusive_classes=True,assistant_not_human=True,counts={str(k):sum(c['class_counts'][str(k)] for c in cases) for k in [1,2,3,5,6]})
 (a.output/'annotation_audit.json').write_bytes(encoded(audit));shutil.copy2(a.annotations,a.output/'reviewed_annotations.json')
 preserved(); print(json.dumps(dict(added=len(cases),total=len(rows),groups=len(set(groups)),counts=audit['counts'],original_files_unchanged=len(before))))

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ['dataset','repo','annotations','review_manifest','review','old_data','features','output']:p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
 main(p.parse_args())
