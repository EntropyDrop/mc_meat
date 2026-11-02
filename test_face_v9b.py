"""Balanced loss preserves partial supervision and equal class contribution."""
import unittest
import torch
from test_face_v9 import FaceV9Tests,small,Logits
from loss_face_v9b import loss

class BalancedTests(unittest.TestCase):
 def test_equal_per_class_mass_despite_extreme_imbalance(self):
  class Net(torch.nn.Module):
   def __init__(self):super().__init__();self.z=torch.nn.Parameter(torch.zeros(101,8))
   def forward(self,p):return self.z
  net=Net();p=dict(y=torch.zeros(101),semantic=torch.zeros(101,dtype=torch.long),fine_weights=torch.ones(101),reviewed=torch.ones(101,dtype=torch.bool),region=torch.zeros(101,dtype=torch.long),owner=torch.zeros(101,dtype=torch.long),region_counts=torch.tensor([101]+[0]*11),cases=1)
  p['semantic'][-1]=10;p['y'][-1]=1
  loss(net,p)[2].backward()
  self.assertAlmostEqual(float(net.z.grad[:100].abs().sum()),float(net.z.grad[-1].abs().sum()),places=5)
 def test_unknown_fine_is_ignored(self):
  n=Logits();p=small();self.assertEqual(float(loss(n,p)[2].detach()),0)
  p['semantic'][0]=9;p['fine_weights'][0]=1;loss(n,p)[2].backward();self.assertLess(n.values.grad[0,6],0);self.assertEqual(float(n.values.grad[1:].abs().sum()),0)

if __name__=='__main__':torch.set_num_threads(4);unittest.main()
