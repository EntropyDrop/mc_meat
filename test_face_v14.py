import unittest
import torch
from face_decoder_v14 import gate_distribution
from face_decoder_v9 import binary_logits
from loss_face_v14 import fine_loss,rare_loss,presence_loss

def fixture(n=1):
 return dict(cases=n,semantic=torch.zeros(n*768,dtype=torch.long),reviewed=torch.zeros(n*768,dtype=torch.bool),fine_weights=torch.zeros(n*768),fine_negative_weights=torch.zeros(n*768),owner=torch.arange(n).repeat_interleave(768),human_case=torch.ones(n,dtype=torch.bool),group_weight=torch.ones(n),head_indices=torch.arange(n*768).reshape(n,768),head_valid=torch.ones(n,768,dtype=torch.bool),x=torch.ones(n*768,2))

class Tests(unittest.TestCase):
 def test_gate_preserves_binary_mass_and_distribution(self):
  torch.manual_seed(2);z=torch.randn(25,8,requires_grad=True);g=torch.randn(3,3,requires_grad=True);owner=torch.arange(25)%3;head=torch.arange(25)%2==0
  out=gate_distribution(z,g,owner,head)
  self.assertTrue(torch.allclose(binary_logits(z).sigmoid(),binary_logits(out).sigmoid(),atol=2e-7))
  self.assertTrue(torch.allclose(z.softmax(-1)[~head],out.softmax(-1)[~head],atol=1e-7))
  self.assertTrue(torch.allclose(out.exp().sum(1),torch.ones(25),atol=2e-7));out[:,5].sum().backward();self.assertTrue(torch.isfinite(z.grad).all());self.assertGreater(g.grad.abs().sum(),0)
 def test_unknown_cells_have_no_direct_fine_or_rare_gradient(self):
  p=fixture();p['semantic'][0]=6;p['fine_weights'][0]=.6;p['fine_negative_weights'][1]=.65
  z=torch.randn(768,8,requires_grad=True);(fine_loss(z,p)+rare_loss(z,p)).backward()
  self.assertEqual(float(z.grad[2:].abs().sum()),0.);self.assertGreater(float(z.grad[:2].abs().sum()),0.)
 def test_grouped_replica_does_not_increase_supervision(self):
  p=fixture(2);p['semantic'][0]=6;p['semantic'][768]=6;p['fine_weights'][[0,768]]=1;p['human_case'][1]=False
  z=torch.randn(1536,8);a=fine_loss(z,p)
  r=fixture(3);r['semantic'][[0,768,1536]]=6;r['fine_weights'][[0,768,1536]]=1;r['human_case'][1:]=False;r['group_weight'][1:]=.5
  b=fine_loss(torch.cat([z,z[768:]]),r);self.assertTrue(torch.allclose(a,b,atol=1e-7))
 def test_weaker_confidence_reduces_loss(self):
  p=fixture();p['human_case'][0]=False;p['semantic'][0]=9;p['fine_weights'][0]=.6;z=torch.zeros(768,8)
  a=fine_loss(z,p);p['fine_weights'][0]=.3;self.assertTrue(torch.allclose(fine_loss(z,p),a*.5))
 def test_hard_negative_mining_ignores_unknown_difficult_cells(self):
  p=fixture();p['fine_negative_weights'][:20]=.65;z=torch.zeros(768,8,requires_grad=True)
  with torch.no_grad():z[20:,5]=20
  rare_loss(z,p).backward();self.assertEqual(float(z.grad[20:].abs().sum()),0);self.assertGreater(float(z.grad[:20].abs().sum()),0)
 def test_partial_absence_is_not_presence_negative(self):
  p=fixture();p['human_case'][0]=False;z=torch.zeros(1,3,requires_grad=True);presence_loss(z,p).backward();self.assertEqual(float(z.grad.abs().sum()),0)
 def test_human_absence_is_presence_negative(self):
  p=fixture();z=torch.zeros(1,3,requires_grad=True);presence_loss(z,p).backward();self.assertTrue((z.grad>0).all())
if __name__=='__main__':unittest.main()
