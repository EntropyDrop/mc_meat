"""Tests for partial supervision, rejection, and five-category compatibility."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from face_decoder_v8 import decode,loss,binary_logits
from calibrate_face_v8 import apply,calibrate

class Model(torch.nn.Module):
 def __init__(self):super().__init__();self.logits=torch.nn.Parameter(torch.zeros(4,6))
 def forward(self,p):return self.logits

class FaceV8Test(unittest.TestCase):
 def test_unknown_fine_cells_have_no_fine_loss(self):
  p=dict(y=torch.tensor([1.,0.,1.,0.]),head=torch.ones(4,dtype=torch.bool),semantic=torch.zeros(4,dtype=torch.long),fine_weights=torch.zeros(4),reviewed=torch.zeros(4,dtype=torch.bool),region=torch.zeros(4,dtype=torch.long),owner=torch.zeros(4,dtype=torch.long),region_counts=torch.tensor([4]+[0]*11),cases=1)
  net=Model();total,binary,fine=loss(net,p)
  self.assertEqual(float(fine),0);self.assertEqual(float(total),float(binary))
  p['reviewed'][1]=True
  self.assertGreater(float(loss(net,p)[2]),0)
 def test_joint_binary_is_sum_of_meat_probabilities(self):
  logits=torch.randn(10,6)
  self.assertTrue(torch.allclose(binary_logits(logits).sigmoid(),logits.softmax(-1)[:,1:4].sum(-1),atol=1e-6))
 def test_other_uncertain_invalid_and_exclusive(self):
  q=np.zeros((4096,6));v=np.zeros(4096,bool);v[:3]=True;q[:3]=[[.9,.02,.02,.02,.02,.02],[.1,.1,.1,.1,.3,.3],[.01,.01,.95,.01,.01,.01]]
  out=decode(q,v,v)
  self.assertEqual(out['labels'][:3],[0,0,2]);self.assertEqual(out['selected'],[2]);self.assertEqual(out['uncertain'],[1]);self.assertEqual(out['class_ids'],[1,2,3,5,6])
 def test_body_calibration_cannot_change_head(self):
  q=np.zeros((4096,6));v=np.zeros(4096,bool);v[:2]=True;q[:2]=[.2,.8,0,0,0,0]
  region=np.zeros(4096,int);region[1]=1;thresholds={str(k):dict(threshold=.9) for k in range(1,6)}
  out=apply(q,v,region,thresholds)
  self.assertEqual(out['labels'][:2],[1,0]);self.assertEqual(out['uncertain'],[1])

 def test_positive_support_abstains_without_inventing_labels(self):
  from rejection_face_v8 import build,apply as reject
  values={'region':np.zeros(4096,dtype=int),'local':np.zeros((4096,51),np.float32)}
  values['local'][:,42]=1
  # A beard training point at (2,6), a nearby supported point, and a top point.
  values['local'][:3,49:51]=[(np.array(x)+.5)/4-1 for x in [(2,6),(3,6),(3,0)]]
  valid=torch.zeros(4096,dtype=torch.bool);valid[0]=True
  sem=torch.zeros(4096,dtype=torch.long);sem[0]=6
  domain=build([dict(row={'id':'train'},cache=values,semantic=sem,fine_weights=valid.float(),valid=valid)])
  labels=[0]*4096;labels[1:4]=[6,6,1]
  result=reject({'labels':labels,'uncertain':[]},values,domain)
  self.assertEqual(result['labels'][1:4],[6,0,1])
  self.assertEqual(result['selected'],[3]);self.assertEqual(result['uncertain'],[2])
  self.assertFalse(domain['unknown_used_as_negative']);self.assertEqual(domain['train_ids'],['train'])

if __name__=='__main__':unittest.main()
