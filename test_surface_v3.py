import json,unittest
from pathlib import Path
import numpy as np
import torch
from surface_model_v3b import SurfaceHead,pack,forward,loss
from semantic_schema import validate_labels
class SurfaceTests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(7);torch.set_num_threads(2)
  t=json.loads((Path(__file__).parent/'surface_topology_v3.json').read_text());self.valid=np.array(t['valid'],bool);self.region=np.zeros(4096,np.int64)
  for j,p in enumerate(t['parts']):
   for u,v,w,h,*_ in p['faces'].values():self.region.reshape(64,64)[v:v+h,u:u+w]=j
 def row(self):
  return dict(valid=torch.from_numpy(self.valid.copy()),cache=dict(region=self.region.copy()),x=torch.randn(4096,5),y=torch.zeros(4096),semantic=torch.zeros(4096,dtype=torch.long))
 def test_no_cross_skin_message_or_invalid_texture_influence(self):
  a=self.row();b=self.row();a['valid'][520]=False;net=SurfaceHead(5).eval()
  with torch.no_grad():
   single=forward(net,pack([a]),'surface');batched=forward(net,pack([a,b]),'surface')[:len(single)]
   self.assertTrue(torch.allclose(single,batched,atol=1e-6,rtol=1e-5))
   a['x'][520]=9999;after=forward(net,pack([a]),'surface');self.assertTrue(torch.equal(single,after))
 def test_unknown_fine_ignored_and_known_fine_has_gradient(self):
  a=self.row();a['y'][520]=1;net=SurfaceHead(5);p=pack([a]);loss(net,p,'surface').backward();self.assertIsNone(net.semantic.weight.grad)
  a['semantic'][520]=2;net.zero_grad();loss(net,pack([a]),'surface').backward();self.assertGreater(float(net.semantic.weight.grad.abs().sum()),0)
 def test_surface_adjacency_symmetric_and_no_part_layer_leak(self):
  t=json.loads((Path(__file__).parent/'surface_topology_v3.json').read_text())
  for i,neighbors in enumerate(t['neighbors']):
   if self.valid[i]:self.assertEqual(len(neighbors),4)
   for j in neighbors:self.assertIn(i,t['neighbors'][j]);self.assertEqual(self.region[i],self.region[j])
if __name__=='__main__':unittest.main()
