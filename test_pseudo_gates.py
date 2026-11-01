"""Prevent confident acceptance of a label whose view votes favor the other class."""
import unittest
import torch
from probe_no_labels import matching_agreement


class AgreementRegression(unittest.TestCase):
    def test_score_winner_opposes_vote_majority(self):
        # Mean similarity can favor meat due to one extreme observation, despite 9 negative votes.
        emitted = torch.tensor([1, 0, 1, 0, 255, 1], dtype=torch.uint8)
        votes = torch.tensor([[9., 1., 1., 9., 0., 0.], [1., 9., 9., 1., 10., 0.]])
        observations = torch.tensor([10., 10., 10., 10., 10., 0.])
        agreement = matching_agreement(emitted, votes, observations)
        torch.testing.assert_close(agreement, torch.tensor([.1, .1, .9, .9, 0., 0.]))
        self.assertEqual((agreement >= .8).tolist(), [False, False, True, True, False, False])


if __name__ == '__main__':
    unittest.main()
