import unittest
import torch
from loss_face_v15 import pixel_fine
from test_face_v14 import fixture
class Tests(unittest.TestCase):
 def test_unknown_direct_gradient(self):
  p=fixture();p['semantic'][0]=6;p['fine_weights'][0]=.6;p['fine_negative_weights'][1]=.65
  z=torch.randn(768,8,requires_grad=True);pixel_fine(z,p).backward();self.assertEqual(float(z.grad[2:].abs().sum()),0.);self.assertGreater(float(z.grad[:2].abs().sum()),0.)
 def test_parent_replica_invariance(self):
  p=fixture(2);p['semantic'][[0,768]]=6;p['fine_weights'][[0,768]]=1;p['human_case'][1]=False;z=torch.randn(1536,8);a=pixel_fine(z,p)
  r=fixture(3);r['semantic'][[0,768,1536]]=6;r['fine_weights'][[0,768,1536]]=1;r['human_case'][1:]=False;r['group_weight'][1:]=.5
  self.assertTrue(torch.allclose(a,pixel_fine(torch.cat([z,z[768:]]),r),atol=1e-7))
if __name__=='__main__':unittest.main()
