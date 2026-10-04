import unittest
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from threshold_selection import select_accuracy_threshold


class ThresholdTests(unittest.TestCase):
    def test_exact_accuracy_optimum_and_equal_scores_stay_together(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        p = np.array([0.05, 0.2, 0.3, 0.45, 0.45, 0.7])
        threshold, sweep, report = select_accuracy_threshold(y, p)
        self.assertGreaterEqual(accuracy_score(y, p >= threshold), accuracy_score(y, p >= 0.5))
        self.assertEqual(sweep.selected.sum(), 1)
        self.assertFalse(report['test_used_for_selection'])
        for row in sweep.itertuples():
            self.assertEqual(row.correct, int(np.sum(y == (p >= row.threshold))))
            self.assertAlmostEqual(row.accuracy, accuracy_score(y, p >= row.threshold))
            self.assertAlmostEqual(row.balanced_accuracy, balanced_accuracy_score(y, p >= row.threshold))
        self.assertEqual(sweep[sweep.selected].correct.iloc[0], sweep.correct.max())

    def test_saturated_probabilities_include_all_negative_partition(self):
        threshold, sweep, _ = select_accuracy_threshold([0, 0, 0, 1], [1, 1, 1, 1])
        self.assertGreater(threshold, 1)
        self.assertEqual(sweep[sweep.selected].accuracy.iloc[0], 0.75)

    def test_deterministic_tie_prefers_balanced_accuracy_then_half(self):
        threshold, sweep, _ = select_accuracy_threshold([0, 1], [0.1, 0.9])
        self.assertEqual(threshold, 0.5)
        tied_y, tied_p = [0, 0, 0, 1], [0.1, 0.7, 0.9, 0.8]
        threshold, sweep, _ = select_accuracy_threshold(tied_y, tied_p)
        self.assertEqual(threshold, 0.8)

    def test_rejects_invalid_or_missing_classes(self):
        for y, p in [([0, 0], [0.1, 0.2]), ([0, 1], [0.1, np.nan]),
                     ([0, 1], [-0.1, 0.8]), ([0, 1], [0.1])]:
            with self.assertRaises(ValueError):
                select_accuracy_threshold(y, p)


if __name__ == '__main__':
    unittest.main()
