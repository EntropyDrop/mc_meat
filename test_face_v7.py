"""Confidence weighting must preserve the control and exclude zero-weight rows."""
import unittest
import torch
from train_face_v7 import weighted_loss
from face_decoder_v5b import loss as original_loss

class ConfidenceTest(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(7)
  self.logits=torch.randn(8,requires_grad=True);self.sem=torch.randn(8,5,requires_grad=True)
  self.net=lambda p:(self.logits,self.sem)
  self.p=dict(y=torch.tensor([1.,0.,1.,0.,0.,1.,0.,1.]),owner=torch.tensor([0,0,0,0,1,1,1,1]),semantic=torch.tensor([1,0,2,0,5,1,6,3]),fine_weights=torch.tensor([1.,0,1,0,.6,.75,.7,.5]),front_indices=torch.arange(8).reshape(2,4),front_mask=torch.ones(2,4,dtype=torch.bool))
 def test_all_one_matches_existing_control(self):
  self.assertTrue(torch.allclose(weighted_loss(self.net,self.p,torch.ones(2)),original_loss(self.net,self.p,'face_front_fine'),rtol=0,atol=1e-7))
 def test_zero_weight_case_has_no_gradient(self):
  weighted_loss(self.net,self.p,torch.tensor([1.,0.])).backward()
  self.assertEqual(self.logits.grad[4:].abs().sum().item(),0)
  self.assertEqual(self.sem.grad[4:].abs().sum().item(),0)
  self.assertGreater(self.logits.grad[:4].abs().sum().item(),0)
 def test_unknown_fine_has_no_auxiliary_gradient(self):
  weighted_loss(self.net,self.p,torch.tensor([1.,.5])).backward()
  self.assertEqual(self.sem.grad[self.p['semantic']==0].abs().sum().item(),0)

if __name__=='__main__':unittest.main()
