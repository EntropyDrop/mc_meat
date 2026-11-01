import io
import unittest
import numpy as np
from PIL import Image
import torch
from prepare_human import validate,digest,groups_for
from uv_annotation_format import encode_mask,skin_parts
from uv_visibility import POLICY,SCHEMA,visible_skin,annotation_mask
from meat_features import local_features,matrix
from train_supervised import weighted_bce,metric


class SupervisedTests(unittest.TestCase):
    def fixture(self):
        a=np.full((64,64,4),255,np.uint8);a[:,:,:3]=120
        parts,_,_=skin_parts(a.tobytes())
        for part in parts:
            if part['layer']:
                for u,v,w,h,*_ in part['faces'].values():a[v:v+h,u:u+w,3]=0
        parts,occupied,_=skin_parts(a.tobytes());_,valid,_=visible_skin(a.tobytes(),parts,occupied)
        out=io.BytesIO();Image.fromarray(a).save(out,format='PNG');skin=out.getvalue();png=annotation_mask([8*64+8],valid)
        meta=dict(schema=SCHEMA,visibility_policy=POLICY,valid_sha256=digest(valid),complete=True,skin_sha256=digest(skin),mask_sha256=digest(png),
                  selected=[8*64+8],selected_count=1,model='classic')
        return a,skin,png,meta
    def test_white_is_meat_padding_ignored(self):
        a,skin,png,m=self.fixture();rgba,valid,mask=validate(skin,png,m)
        self.assertEqual(mask[8,8],255);self.assertFalse(valid[0,0])
        self.assertTrue(valid[8,8]);self.assertEqual(int((mask>0).sum()),1)
    def test_complete_black_mask_valid(self):
        a,skin,png,m=self.fixture();parts,occupied,_=skin_parts(a.tobytes());_,v,_=visible_skin(a.tobytes(),parts,occupied);png=annotation_mask([],v);m.update(mask_sha256=digest(png),selected=[],selected_count=0)
        _,v,y=validate(skin,png,m);self.assertEqual(int(y.sum()),0);self.assertGreater(v.sum(),0)
    def test_stale_hash_and_invalid_selection_rejected(self):
        a,s,p,m=self.fixture();bad=dict(m,skin_sha256='x')
        with self.assertRaises(AssertionError):validate(s,p,bad)
        p=encode_mask([0]);m.update(mask_sha256=digest(p),selected=[0])
        with self.assertRaises(AssertionError):validate(s,p,m)
    def test_local_neighborhood_does_not_cross_uv_faces(self):
        a,_,_,_=self.fixture();parts,v,_=skin_parts(a.tobytes());valid=np.frombuffer(v,np.uint8).astype(bool).reshape(64,64)
        a[~valid]=0;f,_=local_features(a,parts,valid)
        b=a.copy();b[8:16,0:8,:3]=255 # right face immediately beside front
        g,_=local_features(b,parts,valid)
        self.assertTrue(np.array_equal(f.reshape(64,64,-1)[8:16,8:16],g.reshape(64,64,-1)[8:16,8:16]))
    def test_duplicate_grouping_and_class_safety(self):
        a,_,_,_=self.fixture();b=a.copy();b[:,:,:3]=40
        groups,audit=groups_for([a,a.copy(),b],['a','b','c'])
        self.assertEqual(groups[0],groups[1]);self.assertNotEqual(groups[0],groups[2])
        y=torch.tensor([0.,0.,1.,0.]);l=torch.zeros(4,requires_grad=True)
        loss=weighted_bce(l,y,torch.tensor([0,0,1,1]));loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(l.grad).all())
    def test_metrics_ignore_invalid(self):
        m=metric([1,0,0],[.9,.2,.99],np.array([1,1,0],bool))
        self.assertEqual(m['iou'],1);self.assertEqual(m['fp'],0)


if __name__=='__main__':unittest.main()
