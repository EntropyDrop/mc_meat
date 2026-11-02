import unittest
import numpy as np
import torch
from test_face_v9 import row
from face_decoder_v9 import FaceDecoder
from rare_face_v11 import initialize,pack,loss,scores,decode,calibrate

class RareTests(unittest.TestCase):
 def test_low_score_thresholds_are_searchable(self):
  score=np.zeros((4096,3));score[0,0]=.003;score[1,0]=.001;truth=np.zeros(4096,int);truth[0]=6
  th,a=calibrate([dict(score=score,valid=np.ones(4096,bool),truth=truth)])
  self.assertLess(th[0],.05);self.assertEqual(a[0]['tp'],1);self.assertEqual(a[0]['fp'],0)
 def test_balanced_vectorization_matches_original_value_and_gradient(self):
  from loss_face_v9b import loss as original
  from balanced_loss_v11 import loss as vector
  from test_face_v9 import small,Logits
  p=small();p['semantic'][:]=torch.tensor([1,0,9,6]);p['reviewed'][:]=True
  a=Logits();b=Logits();x=original(a,p)[0];y=vector(b,p)[0];x.backward();y.backward()
  self.assertTrue(torch.allclose(x,y,atol=1e-6));self.assertTrue(torch.allclose(a.values.grad,b.values.grad,atol=1e-6))
 def test_frozen_backbone_and_rare_gradients(self):
  a=row();a['semantic'][8*64+8]=6;a['reviewed']=a['valid'].clone();base=FaceDecoder(16)
  net=initialize({'input_dim':16,'state_dict':base.state_dict()});p=pack([a]);value,_=loss(net,p);value.backward()
  self.assertTrue(all(x.grad is None for m in [net.input,net.messages,net.norms] for x in m.parameters()))
  self.assertGreater(float(net.head_classes.weight.grad.abs().sum()),0)
 def test_unknown_and_body_cells_never_train_rare(self):
  a=row();a['semantic'][:]=0;a['reviewed'][:]=False
  net=initialize({'input_dim':16,'state_dict':FaceDecoder(16).state_dict()})
  self.assertEqual(float(loss(net,pack([a]))[0].detach()),0)
  b=row(body=True);b['reviewed']=b['valid'].clone();b['semantic'][b['valid']]=6
  self.assertEqual(float(loss(net,pack([b]))[0].detach()),0)
 def test_fallback_exclusivity_and_invalid(self):
  base=np.ones(4096,np.uint8);v=np.ones(4096,bool);v[3]=False;s=np.zeros((4096,3));s[0]=[.8,.9,0];s[3]=[1,1,1]
  d=decode(base,s,v,[.5,.5,.5]);self.assertEqual(d['labels'][:4],[9,1,1,0]);self.assertNotIn(3,d['selected'])
 def test_calibration_abstains_without_positive_support(self):
  x=dict(score=np.ones((4096,3))*.8,valid=np.ones(4096,bool),truth=np.zeros(4096,int))
  t,_=calibrate([x]);self.assertTrue(all(a>1 for a in t))
 def test_label_independence_and_case_isolation(self):
  a=row();b=row('b');net=initialize({'input_dim':16,'state_dict':FaceDecoder(16).state_dict()}).eval()
  with torch.inference_mode():
   x=net(pack([a]));y=net(pack([a,b]));self.assertTrue(torch.allclose(x[0],y[0][:64],atol=1e-6))
   a['semantic'][:]=6;a['reviewed'][:]=True;z=net(pack([a]));self.assertTrue(torch.equal(x[0],z[0]))

if __name__=='__main__':torch.set_num_threads(4);unittest.main()
