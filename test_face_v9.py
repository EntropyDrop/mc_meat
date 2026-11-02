"""Regression tests for seven-class supervision, hidden cells and case isolation."""
import unittest
import numpy as np
import torch
from face_decoder_v9 import FaceDecoder,pack,loss,decode,binary_logits,INTERNAL_IDS
from semantic_schema_v3 import CLASS_IDS,validate_labels
from train_face_v9 import splits

class Logits(torch.nn.Module):
 def __init__(self):super().__init__();self.values=torch.nn.Parameter(torch.zeros(4,8))
 def forward(self,p):return self.values

def small():
 return dict(y=torch.tensor([1.,0.,1.,0.]),head=torch.ones(4,dtype=torch.bool),semantic=torch.zeros(4,dtype=torch.long),fine_weights=torch.zeros(4),reviewed=torch.zeros(4,dtype=torch.bool),region=torch.zeros(4,dtype=torch.long),owner=torch.zeros(4,dtype=torch.long),region_counts=torch.tensor([4]+[0]*11),cases=1)

def row(name='a',body=False):
 valid=np.zeros(4096,bool);local=np.zeros((4096,57),np.float32);regions=np.zeros(4096,np.int64)
 if body:ids=[20*64+20];regions[:]=1;xy=[(0,0)]
 else:ids=[(8+y)*64+8+x for y in range(8) for x in range(8)];xy=[(x,y) for y in range(8) for x in range(8)]
 valid[ids]=True
 for i,(x,y) in zip(ids,xy):local[i,46]=1;local[i,49:51]=(np.array([x,y])+.5)/4-1
 return dict(row={'id':name,'group':name},valid=torch.from_numpy(valid),x=torch.randn(4096,16),y=torch.zeros(4096),semantic=torch.zeros(4096,dtype=torch.long),fine_weights=torch.zeros(4096),reviewed=torch.zeros(4096,dtype=torch.bool),cache={'local':local,'region':regions})

class FaceV9Tests(unittest.TestCase):
 def test_unknown_not_other_and_reviewed_zero_is_other(self):
  net=Logits();p=small();self.assertEqual(float(loss(net,p)[2]),0)
  p['reviewed'][1]=True;fine=loss(net,p)[2];fine.backward()
  self.assertLess(net.values.grad[1,0],0);self.assertEqual(float(net.values.grad[0].abs().sum()),0)
 def test_nose_eyeshadow_are_meat_and_receive_distinct_gradients(self):
  net=Logits();p=small();p['semantic'][:2]=torch.tensor([9,10]);p['fine_weights'][:2]=1
  loss(net,p)[2].backward();self.assertLess(net.values.grad[0,6],0);self.assertLess(net.values.grad[1,7],0)
  z=torch.randn(11,8);self.assertTrue(torch.allclose(binary_logits(z).sigmoid(),z.softmax(-1)[:,[1,2,3,6,7]].sum(-1),atol=1e-6))
 def test_exclusive_output_and_ignore(self):
  q=np.zeros((4096,8));valid=np.zeros(4096,bool);valid[:4]=True;q[0,6]=1;q[1,7]=1;q[2,5]=1;q[3]=1/8;q[4,2]=1
  d=decode(q,valid,valid);self.assertEqual(d['labels'][:5],[9,10,6,0,0]);self.assertEqual(d['selected'],[0,1]);self.assertEqual(d['uncertain'],[3]);self.assertEqual(d['class_ids'],CLASS_IDS)
 def test_retired_ids_rejected(self):
  for c in [4,7,8]:
   labels=[0]*4096;labels[0]=c
   with self.assertRaises(ValueError):validate_labels(labels,[1]*4096,[])
 def test_case_isolation_labels_not_inputs_and_ignored_features(self):
  torch.manual_seed(1);a=row();b=row('b');net=FaceDecoder(16).eval()
  with torch.no_grad():
   first=net(pack([a]));batch=net(pack([a,b]))[:64];self.assertTrue(torch.allclose(first,batch,atol=1e-6))
   a['x'][~a['valid']]=1000;a['y'][:]=1;a['semantic'][:]=10;a['reviewed'][:]=True
   self.assertTrue(torch.equal(first,net(pack([a]))))
 def test_no_head_still_classifies_body(self):
  a=row(body=True)
  for facial in [True,False]:self.assertEqual(tuple(FaceDecoder(16,facial=facial)(pack([a])).shape),(1,8))
 def test_folds_hold_whole_duplicate_groups(self):
  rows=[row(str(i)) for i in range(12)];rows[1]['row']['group']='0';rows[0]['semantic'][8*64+8]=9
  fs=splits(rows);self.assertEqual(sorted(i for f in fs for i in f['held_ids']),sorted(r['row']['id'] for r in rows))
  for f in fs:
   tr={r['row']['group'] for r in rows if r['row']['id'] in f['train_ids']};he={r['row']['group'] for r in rows if r['row']['id'] in f['held_ids']};self.assertFalse(tr&he)

if __name__=='__main__':torch.set_num_threads(4);unittest.main()
