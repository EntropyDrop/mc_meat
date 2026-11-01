"""Freeze verified, committed human meat labels; never read recovery history as examples."""
import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from uv_annotation_format import decode_rgba, skin_parts
from uv_visibility import POLICY, SCHEMA, visible_skin, annotation_mask


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git_bytes(root, revision, name):
    blob = subprocess.check_output(['git', 'show', revision + ':' + name], cwd=root)
    if blob.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
        oid = next(x.split(':', 1)[1] for x in blob.decode().splitlines() if x.startswith('oid sha256:'))
        # Read committed LFS object, not the possibly changing annotation working tree.
        common = subprocess.check_output(['git', 'rev-parse', '--git-common-dir'], cwd=root, text=True).strip()
        obj = (root / common / 'lfs/objects' / oid[:2] / oid[2:4] / oid)
        blob = obj.read_bytes()
        assert digest(blob) == oid, 'LFS content does not match committed pointer'
    return blob


def validate(skin, png, meta):
    assert meta['schema'] == SCHEMA and meta['complete'] is True, 'Migrate legacy labels before training'
    assert digest(skin) == meta['skin_sha256'] and digest(png) == meta['mask_sha256']
    raw = decode_rgba(skin)
    rgba = np.frombuffer(raw, np.uint8).reshape(64, 64, 4)
    parts, occupied, model = skin_parts(raw)
    shown, valid, hidden = visible_skin(raw, parts, occupied)
    assert meta['visibility_policy'] == POLICY and meta['valid_sha256'] == digest(valid)
    assert annotation_mask(meta['selected'], valid) == png, 'RGBA mask/validity mismatch'
    rgba = np.frombuffer(shown, np.uint8).reshape(64,64,4).copy()
    valid = np.frombuffer(valid, np.uint8).astype(bool).reshape(64, 64)
    mask_im = Image.open(io.BytesIO(png))
    assert mask_im.mode == 'RGBA' and mask_im.size == (64, 64)
    pixels = np.array(mask_im)
    assert np.array_equal(pixels[...,3]>0,valid)
    mask = pixels[...,0]
    assert np.isin(mask, [0, 255]).all()
    assert not ((mask > 0) & ~valid).any()
    assert np.flatnonzero(mask).tolist() == sorted(meta['selected'])
    assert int((mask > 0).sum()) == meta['selected_count']
    assert meta['model'] == model
    # The cached renderer has classic mappings. Do not silently misrender slim or partial alpha.
    assert model == 'classic', 'Need separately validated slim renderer before training slim examples'
    assert np.isin(rgba[..., 3], [0, 255]).all(), 'Only binary alpha is currently supported'
    return rgba, valid, mask


def groups_for(images, names):
    """Conservative near-duplicate grouping independent of human labels."""
    parent = list(range(len(images)))
    def find(x):
        while x != parent[x]:
            x = parent[x]
        return x
    audit = []
    for i in range(len(images)):
        for j in range(i):
            a, b = images[i], images[j]
            union = (a[..., 3] > 0) | (b[..., 3] > 0)
            equal = float((np.abs(a.astype(float)-b.astype(float)).max(-1)[union] <= 3).mean())
            duplicate = equal >= .98
            audit.append(dict(a=names[i], b=names[j], fraction_cells_with_rgba_delta_le_3=equal, grouped=duplicate))
            if duplicate:
                parent[find(i)] = find(j)
    return [names[find(i)] for i in range(len(images))], audit


def prepare(root, revision, out):
    revision = subprocess.check_output(['git', 'rev-parse', revision], cwd=root, text=True).strip()
    files = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision], cwd=root, text=True).splitlines()
    labels = [p for p in files if p.endswith('_meat.json') and '.meat_history' not in Path(p).parts]
    assert labels, 'No committed human labels'
    assert not out.exists(), 'Use a new snapshot path'
    validated = []
    for name in labels:
        meta_bytes = git_bytes(root, revision, name)
        meta = json.loads(meta_bytes)
        assert Path(meta['skin_file']).name == meta['skin_file']
        skin_path = str(Path(name).parent / meta['skin_file'])
        png_path = name[:-5] + '.png'
        skin, png = git_bytes(root, revision, skin_path), git_bytes(root, revision, png_path)
        a, valid, mask = validate(skin, png, meta)
        a = a.copy(); a[~valid] = 0  # ignore padding and invisible RGB during duplicate comparison
        validated.append((name, meta, meta_bytes, skin_path, skin, png, a, valid, mask))
    assert len({x[1]['id'] for x in validated}) == len(validated)
    groups, audit = groups_for([x[6] for x in validated], [x[1]['id'] for x in validated])
    assert len(set(groups)) >= 3, 'Too few independent groups for cross-validation'
    out.mkdir(parents=True)
    rows = []
    for group, (name, meta, mb, sp, skin, png, a, valid, mask) in zip(groups, validated):
        d = out / meta['id']; d.mkdir()
        (d/'skin.png').write_bytes(skin); (d/'meat.png').write_bytes(png); (d/'annotation.json').write_bytes(mb)
        Image.fromarray(valid.astype(np.uint8)*255).save(d/'valid.png')
        Image.fromarray(a).save(d/'visible_skin.png')
        rows.append(dict(id=meta['id'], group=group, model=meta['model'], source_skin=sp, source_annotation=name,
                         skin_sha256=digest(skin), mask_sha256=digest(png), metadata_sha256=digest(mb),
                         visible_skin_sha256=digest((d/'visible_skin.png').read_bytes()),valid_sha256=digest((d/'valid.png').read_bytes()), valid_cells=int(valid.sum()),
                         meat_cells=int((mask > 0).sum()), non_meat_cells=int((valid & (mask == 0)).sum())))
    manifest = dict(schema='mc-meat-human-snapshot-v1', dataset_root=str(root), dataset_commit=revision,
                    labels='human RGBA masks; white=meat, black=non-meat, transparent=ignore',visibility_policy=POLICY,
                    rows=rows, duplicate_audit=audit, independent_groups=len(set(groups)),
                    note='All are existing DDJ cases, not a new blind benchmark. Missing labels and history excluded.')
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(snapshot=str(out), count=len(rows), groups=len(set(groups)), meat_cells=sum(r['meat_cells'] for r in rows)), ensure_ascii=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--revision',default='HEAD'); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); prepare(a.dataset,a.revision,a.output)
