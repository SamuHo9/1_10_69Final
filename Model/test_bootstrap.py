"""Bootstrap tests cover saved decision rules, degenerate rounds and reproducibility."""
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from bootstrap_all_models import calculate, classification_metrics


class BootstrapTests(unittest.TestCase):
    def test_class_metrics_match_independent_sklearn(self):
        y = np.array([0, 0, 0, 1, 1, 1, 1])
        pred = np.array([0, 1, 0, 0, 1, 1, 1])
        metrics = classification_metrics(y, pred)
        self.assertEqual(metrics['Accuracy'], accuracy_score(y, pred))
        self.assertEqual(metrics['BalancedAccuracy'], balanced_accuracy_score(y, pred))
        self.assertEqual(metrics['F1_macro'], f1_score(y, pred, average='macro'))
        for label in (0, 1):
            self.assertEqual(metrics[f'Sensitivity_Class{label}'], recall_score(y, pred, pos_label=label))
            self.assertEqual(metrics[f'F1_Class{label}'], f1_score(y, pred, pos_label=label))

    def test_saved_predictions_are_used_even_when_probability_threshold_disagrees(self):
        frame = pd.DataFrame({'Subject': ['sub-001', 'sub-002', 'sub-003', 'sub-004'],
            'BinaryClass': [0, 0, 1, 1], 'Probability': [.7, .8, .2, .3], 'Prediction': [0, 0, 1, 1]})
        scores, original, _, _, _ = calculate(frame, 30)
        self.assertEqual(original['Accuracy'], 1)
        self.assertTrue(scores.Accuracy.eq(1).all())

    def test_random_state_matches_legacy_n_of_n_sampling_and_records_skipped_rounds(self):
        frame = pd.DataFrame({'Subject': ['sub-001', 'sub-002'], 'BinaryClass': [0, 1],
            'Probability': [.1, .9], 'Prediction': [0, 1]})
        scores, _, intervals, data, skipped = calculate(frame, 100)
        rng = np.random.RandomState(42)
        expected = np.array([rng.choice(2, 2, replace=True) for _ in range(100)])
        np.testing.assert_array_equal(data['indices'], expected)
        self.assertEqual(len(scores)+len(skipped), 100)
        self.assertGreater(len(skipped), 0)
        self.assertTrue(scores[['TP', 'TN', 'FP', 'FN']].sum(axis=1).eq(2).all())
        self.assertEqual(intervals['AUC']['ci95_lower'], 1)
        self.assertEqual(intervals['AUC']['ci95_upper'], 1)


if __name__ == '__main__':
    unittest.main()
