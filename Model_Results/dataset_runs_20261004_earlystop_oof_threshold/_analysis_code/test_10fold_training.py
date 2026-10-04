"""Check person-level CV, fold-local augmentation, latent rebuild and entry points."""
import argparse
import ast
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from train_dataset_model import ROOT, load_data
from dataset_cross_validation import person_splits, prepare_training, transformed, original_inputs


class TenFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.test, cls.columns = load_data(ROOT/'Output_Dataset/coef_raw/left/train_prepared.csv',
            ROOT/'Output_Dataset/coef_raw/left/test_prepared.csv', 'coef')

    def test_every_original_is_validated_once_without_person_overlap(self):
        for side in ('left', 'right'):
            frame = pd.read_csv(ROOT/'Output_Dataset/coef_raw'/side/'train_prepared.csv')
            groups, folds = person_splits(frame)
            self.assertEqual(len(folds), 10)
            coverage = np.zeros(len(frame), dtype=int)
            for tr, va in folds:
                coverage[va] += 1
                self.assertFalse(set(groups[tr]) & set(groups[va]))
                self.assertEqual(set(frame.iloc[va].BinaryClass), {0, 1})
            np.testing.assert_array_equal(coverage, np.ones(len(frame)))

    def test_insufficient_people_never_silently_reduce_folds(self):
        too_small = pd.concat([self.train[self.train.BinaryClass == label].head(9) for label in (0, 1)])
        with self.assertRaisesRegex(ValueError, 'at least 10'):
            person_splits(too_small)
        with self.assertRaisesRegex(ValueError, 'exactly 10'):
            person_splits(self.train, folds=5)

    def test_repeated_rows_for_one_person_stay_in_same_fold(self):
        repeated = pd.concat([self.train, self.train.iloc[[0]]], ignore_index=True)
        groups, folds = person_splits(repeated)
        for tr, va in folds:
            self.assertFalse(set(groups[tr]) & set(groups[va]))

    def test_all_augmented_methods_have_only_current_training_parents(self):
        _, folds = person_splits(self.train)
        tr, va = folds[0]
        current = self.train.iloc[tr].reset_index(drop=True)
        for dataset in ('coef_raw_balanced_jitter', 'coef_plsda', 'augment_plsda_balanced'):
            manifest = json.loads((ROOT/'Output_Dataset'/dataset/'left/data_manifest.json').read_text())
            prepared, pairs, info = prepare_training(current, dataset, 'coef', manifest, 43)
            self.assertEqual(prepared.BinaryClass.value_counts()[0], prepared.BinaryClass.value_counts()[1])
            self.assertTrue(set(pairs.ParentSubject1).issubset(set(current.Subject)))
            self.assertTrue(set(pairs.ParentSubject2).issubset(set(current.Subject)))
            self.assertFalse(set(pairs.ParentSubject1) & set(self.train.iloc[va].Subject))
            self.assertTrue(prepared.Subject.is_unique)

    def test_validation_features_and_labels_do_not_change_fitted_latent_pls(self):
        _, folds = person_splits(self.train)
        tr, va = folds[0]
        current, valid = self.train.iloc[tr], self.train.iloc[va]
        args = argparse.Namespace(kind='latent', model='SVM', pls_components=8)
        changed = valid.copy()
        changed.loc[:, self.columns] += 100000
        changed.BinaryClass = 1 - changed.BinaryClass
        _, _, first = transformed(current, valid, self.columns, args)
        _, _, second = transformed(current, changed, self.columns, args)
        np.testing.assert_array_equal(first['latent_pls']['pls'].x_weights_, second['latent_pls']['pls'].x_weights_)
        np.testing.assert_array_equal(first['latent_scaler'].mean_, second['latent_scaler'].mean_)

    def test_latent_reads_original_coefficients_for_outer_cv(self):
        folder = ROOT/'Output_Dataset/plsda_latent_features/left'
        args = argparse.Namespace(train_csv=folder/'train_plsda_latent_features.csv',
            test_csv=folder/'test_plsda_latent_features.csv', kind='latent', dataset='plsda_latent_features', side='left')
        original, _, columns, _, paths = original_inputs(args)
        self.assertEqual(len(columns), 507)
        self.assertIn('original_train', paths)
        np.testing.assert_array_equal(original[self.columns], self.train[self.columns])

    def test_all_fourteen_entry_points_delegate_to_same_protocol(self):
        files = list((ROOT/'Model/All_Augment_tain').rglob('train*.py'))
        self.assertEqual(len(files), 14)
        for path in files:
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            pipeline = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_pipeline')
            calls = [n for n in ast.walk(pipeline) if isinstance(n, ast.Call)]
            self.assertTrue(any(isinstance(n.func, ast.Name) and n.func.id == 'legacy_entrypoint' for n in calls))


if __name__ == '__main__':
    unittest.main()
