import unittest,numpy as np
from calibration_v14 import decode_calibrated,fit_thresholds
class Test(unittest.TestCase):
 def test_selection_and_ignored_cells(self):
  q=np.zeros((4096,8),np.float32);q[:,0]=1;q[0]=[.2,.15,.05,.05,.05,.5,0,0];q[1]=[.1,.4,.1,.1,0,0,.15,.15];v=np.zeros(4096,bool);v[:2]=1;head=v.copy()
  a=decode_calibrated(q,v,head,{'6':.45,'9':.2,'10':.1});self.assertEqual(a['labels'][0],6);self.assertEqual(a['labels'][1],10);self.assertEqual(a['selected'],[1]);self.assertEqual(sum(a['labels'][2:]),0)
 def test_absent_categories_default_to_conservative_grid_end(self):
  q=np.zeros((4096,8));q[:,0]=1;v=np.ones(4096,bool);t,c=fit_thresholds([dict(q=q,valid=v,head=v,labels=np.zeros(4096))]);self.assertEqual(t,{'6':.95,'9':.95,'10':.95})
if __name__=='__main__':unittest.main()
