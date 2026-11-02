import unittest
import numpy as np
from rare_guard import guarded_thresholds, decode


class GuardTests(unittest.TestCase):
    def test_unsupported_class_abstains(self):
        records = [dict(class_id=c, threshold=.01, tp=1, precision=p)
                   for c, p in zip([6, 9, 10], [.004, .5, 0.])]
        threshold, enabled = guarded_thresholds(records)
        self.assertEqual(enabled, [False, True, False])
        self.assertGreater(threshold[0], 1.)
        self.assertEqual(threshold[1], .01)

    def test_fallback_preserves_ordinary_and_invalid_is_transparent(self):
        base = np.array([6, 9, 10, 1, 2, 3, 5, 0])
        valid = np.ones(8, bool); valid[-1] = False
        result = decode(base, np.ones((8, 3)), valid, [1.001]*3, [False]*3)
        self.assertEqual(result['labels'], base.tolist())

    def test_mutually_exclusive_and_disabled_class_cannot_win(self):
        score = np.array([[1., .6, .9], [1., .9, .6], [1., .2, .1]])
        result = decode(np.array([0, 0, 6]), score, np.ones(3, bool), [1.001, .5, .5], [False, True, True])
        self.assertEqual(result['labels'], [10, 9, 6])
        self.assertEqual(result['selected'], [0, 1])


if __name__ == '__main__':
    unittest.main()
