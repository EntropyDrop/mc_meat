import unittest,torch
from surface_model_v4 import SurfaceHead,loss
from semantic_schema import CLASS_IDS
class FiveClassTests(unittest.TestCase):
 def test_sparse_ids_train_five_outputs_and_reload(self):
  self.assertEqual(CLASS_IDS,[1,2,3,5,6]);torch.manual_seed(3)
  p=dict(x=torch.randn(5,7),neighbors=torch.arange(5)[:,None].repeat(1,4),neighbor_mask=torch.ones(5,4,dtype=torch.bool),region=torch.zeros(5,dtype=torch.long),y=torch.tensor([1.,1.,1.,0.,0.]),owner=torch.zeros(5,dtype=torch.long),semantic=torch.tensor(CLASS_IDS),fine_weights=torch.ones(5))
  n=SurfaceHead(7);self.assertEqual(n.semantic.out_features,5)
  opt=torch.optim.AdamW(n.parameters(),lr=.002)
  for _ in range(3):
   opt.zero_grad();v=loss(n,p,'surface');self.assertTrue(torch.isfinite(v));v.backward();self.assertGreater(float(n.semantic.weight.grad.abs().sum()),0);opt.step()
  for removed in [4,7,8]:
   p['semantic'][0]=removed
   with self.assertRaises(AssertionError):loss(n,p,'surface')
  p['semantic'].zero_();n.zero_grad();loss(n,p,'surface').backward();self.assertIsNone(n.semantic.weight.grad)
if __name__=='__main__':unittest.main()
