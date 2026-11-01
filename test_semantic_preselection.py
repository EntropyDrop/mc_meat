import unittest
import numpy as np
from semantic_preselection import decode_categories

class CategorySuggestionTests(unittest.TestCase):
 def setUp(self):
  self.valid=np.zeros(4096,bool);self.valid[520:529]=True
  self.prob=np.zeros(4096,np.float32);self.prob[520:524]=.9
  self.fine=np.zeros((4096,5),np.float32);self.fine[self.valid]=.2
  self.scope=self.valid.copy()
 def choose(self,i,c):
  self.fine[i]=.025;self.fine[i,c]=.9
 def test_all_five_categories_and_binary_equivalence(self):
  for i,c in [(520,0),(521,1),(522,2),(524,3),(525,4)]:self.choose(i,c)
  out=decode_categories(self.prob,self.fine,self.valid,self.scope)
  self.assertEqual(out['labels'][520:526],[1,2,3,1,5,6])
  self.assertEqual(out['selected'],[520,521,522,523])
  self.assertTrue(out['semantic_available'])
 def test_invalid_scope_confidence_and_parent_conflicts(self):
  self.choose(521,1);self.scope[521]=False
  self.choose(522,4);self.choose(525,1)
  self.fine[0]=[0,0,0,0,1]
  out=decode_categories(self.prob,self.fine,self.valid,self.scope)
  self.assertEqual(out['labels'][0],0)
  self.assertEqual(out['labels'][521],1) # No eyes on body/outside supervised scope.
  self.assertEqual(out['labels'][522],1) # Binary meat cannot become beard.
  self.assertEqual(out['labels'][525],0) # Binary non-meat cannot become eyes.
  self.assertEqual(out['labels'][526],0) # Low-confidence non-meat stays unselected.
 def test_binary_only_checkpoint_is_explicit(self):
  out=decode_categories(self.prob,None,self.valid,self.scope)
  self.assertFalse(out['semantic_available']);self.assertEqual(out['category_counts']['2'],0)
 def test_unseen_facial_region_cannot_become_beard(self):
  self.choose(525,4);support=np.zeros((4096,5),bool);support[:,0]=True
  out=decode_categories(self.prob,self.fine,self.valid,self.scope,class_support=support)
  self.assertEqual(out['labels'][525],0)
  support[525,4]=True
  out=decode_categories(self.prob,self.fine,self.valid,self.scope,class_support=support)
  self.assertEqual(out['labels'][525],6)
 def test_nonfinite_or_bad_distribution_fails(self):
  self.fine[520,0]=np.nan
  with self.assertRaises(AssertionError):decode_categories(self.prob,self.fine,self.valid,self.scope)
  self.fine[520]=0
  with self.assertRaises(AssertionError):decode_categories(self.prob,self.fine,self.valid,self.scope)

if __name__=='__main__':unittest.main()
