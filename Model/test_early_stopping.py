import unittest
import numpy as np
import pandas as pd
from early_stopping_training import StopTracker, inner_person_split
from dataset_cross_validation import person_splits, prepare_training
from train_dataset_model import ROOT
import json


class EarlyStoppingTests(unittest.TestCase):
    def test_stops_after_patience_and_keeps_earlier_best_epoch(self):
        tracker = StopTracker(patience=2, min_delta=0.01)
        results = [tracker.update(i, loss) for i, loss in enumerate([0.9, 0.8, 0.85, 0.86], 1)]
        self.assertEqual(results, [(True, False), (True, False), (False, False), (False, True)])
        self.assertEqual(tracker.best_epoch, 2)
        self.assertEqual(tracker.best_loss, 0.8)

    def test_small_improvements_still_save_true_best_but_do_not_reset_patience(self):
        tracker = StopTracker(patience=2, min_delta=0.1)
        tracker.update(1, 1)
        self.assertEqual(tracker.update(2, 0.98), (True, False))
        self.assertEqual(tracker.update(3, 0.97), (True, True))
        self.assertEqual(tracker.best_epoch, 3)

    def test_rejects_nonfinite_loss(self):
        with self.assertRaises(ValueError):
            StopTracker().update(1, np.nan)

    def test_fit_stop_oof_disjoint_and_children_only_from_fit_people(self):
        frame = pd.read_csv(ROOT/'Output_Dataset/coef_raw/left/train_prepared.csv')
        groups, folds = person_splits(frame)
        for fold, (tr, va) in enumerate(folds, 1):
            outer = frame.iloc[tr].reset_index(drop=True)
            fi, si = inner_person_split(outer, seed=42+fold)
            self.assertEqual(len(fi), 216)
            self.assertEqual(len(si), 54)
            self.assertFalse(set(outer.iloc[fi].Subject) & set(outer.iloc[si].Subject))
            self.assertFalse(set(outer.Subject) & set(frame.iloc[va].Subject))
            self.assertEqual(set(outer.iloc[si].BinaryClass), {0, 1})
        tr, va = folds[0]
        outer = frame.iloc[tr].reset_index(drop=True)
        fi, si = inner_person_split(outer, seed=43)
        manifest = json.loads((ROOT/'Output_Dataset/augment_plsda_balanced/left/data_manifest.json').read_text())
        _, pairs, _ = prepare_training(outer.iloc[fi].reset_index(drop=True), 'augment_plsda_balanced', 'coef', manifest, 43)
        self.assertTrue(set(pairs.ParentSubject1).issubset(set(outer.iloc[fi].Subject)))
        self.assertFalse(set(pairs.ParentSubject1) & set(outer.iloc[si].Subject))

    def test_repeated_person_rows_stay_together_in_inner_split(self):
        frame = pd.read_csv(ROOT/'Output_Dataset/coef_raw/left/train_prepared.csv')
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        fi, si = inner_person_split(frame)
        self.assertFalse(set(frame.iloc[fi].Subject) & set(frame.iloc[si].Subject))


if __name__ == '__main__':
    unittest.main()
