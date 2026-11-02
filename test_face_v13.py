"""Partial-label and reviewed-negative gradient checks."""
import unittest
import torch
from test_face_v9 import small,Logits
from weighted_face_v12 import loss as previous
from weighted_face_v13 import loss

class PartialTests(unittest.TestCase):
    def test_unchanged_labels_match_previous_loss(self):
        p=small();p['reviewed'][:]=True;p['semantic'][:]=torch.tensor([9,0,10,6]);n=Logits()
        with torch.no_grad():n.values.copy_(torch.randn_like(n.values))
        for mode in [False,True]:
            for a,b in zip(loss(n,p,30,mode),previous(n,p,30,mode)):self.assertTrue(torch.equal(a,b))
    def test_unreviewed_binary_cells_have_zero_gradient(self):
        p=small();p['binary_cell_weights']=torch.tensor([1.,0.,0.,0.]);n=Logits();loss(n,p)[1].backward()
        self.assertGreater(float(n.values.grad[0].abs().sum()),0);self.assertEqual(float(n.values.grad[1:].abs().sum()),0)
    def test_explicit_negative_has_fine_gradient(self):
        p=small();p['fine_negative_weights']=torch.tensor([.6,0.,0.,0.]);n=Logits();loss(n,p)[2].backward()
        self.assertLess(float(n.values.grad[0,0]),0);self.assertEqual(float(n.values.grad[1:].abs().sum()),0)
    def test_negative_confidence_not_normalized_away(self):
        p=small();p['fine_negative_weights']=torch.tensor([.6,0.,0.,0.]);n=Logits()
        for mode in [False,True]:
            first=loss(n,p,balanced=mode)[2];p['fine_negative_weights']*=.5;second=loss(n,p,balanced=mode)[2]
            self.assertTrue(torch.allclose(first*.5,second));p['fine_negative_weights']*=2
    def test_unknown_fine_still_ignored(self):
        p=small();n=Logits();loss(n,p)[2].backward();self.assertEqual(float(n.values.grad.abs().sum()),0)
    def test_conflicting_negative_rejected(self):
        p=small();p['semantic'][0]=9;p['fine_negative_weights']=torch.tensor([.6,0.,0.,0.])
        with self.assertRaises(AssertionError):loss(Logits(),p)
if __name__=='__main__':torch.set_num_threads(4);unittest.main()
