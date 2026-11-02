"""Weak-label weighting and fold-isolation regression checks."""
import unittest,torch
from test_face_v9 import small,Logits,row
from face_decoder_v9 import loss as ordinary
from balanced_loss_v11 import loss as balanced
from weighted_face_v12 import loss,pack,fold_training

class WeightedTests(unittest.TestCase):
 def test_human_only_matches_historical_losses(self):
  p=small();p['reviewed'][:]=True;p['semantic'][:]=torch.tensor([9,0,10,6]);n=Logits()
  with torch.no_grad():n.values.copy_(torch.randn_like(n.values))
  for mode,ref in [(False,ordinary),(True,balanced)]:
   for a,b in zip(loss(n,p,30,mode),ref(n,p,30)):self.assertTrue(torch.equal(a,b))
 def test_binary_confidence_scales_gradient(self):
  n=Logits();p=small();p['case_weights']=torch.ones(1);loss(n,p)[1].backward();original=n.values.grad.clone();n.zero_grad();p['case_weights']*=.5;loss(n,p)[1].backward()
  self.assertTrue(torch.allclose(n.values.grad,original*.5))
 def test_partial_unknown_has_no_fine_gradient(self):
  n=Logits();p=small();p['semantic'][0]=6;p['fine_weights'][0]=.6
  for mode in [False,True]:
   n.zero_grad();loss(n,p,balanced=mode)[2].backward();self.assertGreater(float(n.values.grad[0].abs().sum()),0);self.assertEqual(float(n.values.grad[1:].abs().sum()),0)
 def test_fine_confidence_is_not_normalized_away(self):
  n=Logits();p=small();p['semantic'][0]=9;p['fine_weights'][0]=.6
  for mode in [False,True]:
   first=loss(n,p,balanced=mode)[2];p['fine_weights'][0]=.3;second=loss(n,p,balanced=mode)[2];self.assertTrue(torch.allclose(second,first*.5));p['fine_weights'][0]=.6
 def test_metadata_weights_and_validation_purge(self):
  a=row('a');b=row('b');c=row('c');weak=row('w');bad=row('bad')
  weak['row'].update(binary_training_weight=.5,training_eligible=True);bad['row'].update(binary_training_weight=.5,training_eligible=False)
  p=pack([a,weak]);self.assertEqual(p['case_weights'].tolist(),[1,.5])
  f=dict(train_ids=['b','c'],held_ids=['a']);tr,removed=fold_training([a,b,c],[weak,bad],f,[('a','b')],True)
  self.assertEqual([r['row']['id'] for r in tr],['c','w']);self.assertEqual(removed,['b'])
if __name__=='__main__':torch.set_num_threads(4);unittest.main()
