"""Partial binary and explicit fine-negative supervision on frozen UV features."""
import json
import numpy as np
import torch
from data_face_v9 import load_rows as base_load

def load_rows(data,cache):
    rows,fp=base_load(data,cache)
    for r in rows:
        d=data/r['row']['id'];am=json.loads((d/'annotation.json').read_text());sm=json.loads((d/'semantic.json').read_text())
        bw=np.asarray(am.get('binary_cell_weights',[1.]*4096),np.float32)
        nw=np.asarray(sm.get('fine_negative_weights',[0.]*4096),np.float32)
        assert bw.shape==nw.shape==(4096,)
        assert np.isfinite(bw).all() and np.isin(bw,[0.,1.]).all()
        assert np.isfinite(nw).all() and (nw>=0).all() and (nw<=.8).all()
        assert not nw[r['semantic'].numpy()>0].any(),'Fine-negative weight on a positive category'
        assert not nw[~r['valid'].numpy()].any(),'Fine-negative weight on an ignored cell'
        if r['row'].get('hard_negative_only'):
            assert not r['semantic'].any() and np.array_equal(bw>0,nw>0)
        r['binary_cell_weights']=torch.from_numpy(bw)
        r['fine_negative_weights']=torch.from_numpy(nw)
    return rows,fp
