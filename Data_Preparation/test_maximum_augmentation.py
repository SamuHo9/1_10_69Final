"""Exercise maximum sizes, unique pairs and streamed CLI exports."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from data_prep_common import (COEF_COLUMNS, XYZ_COLUMNS, augmentation_plan,
                              augment_balanced_jitter, augment_plsda_same_class)
import prepare_coef_raw
import prepare_pointnet_raw
import prepare_coef_plsda
import prepare_pointnet_plsda
from run_all_prepare import validate


def make_frame(n0, n1, kind='coef', prefix='sub-', seed=19):
    columns = COEF_COLUMNS if kind == 'coef' else XYZ_COLUMNS
    frame = pd.DataFrame(np.random.default_rng(seed).normal(size=(n0+n1,len(columns))),columns=columns)
    frame['Subject'] = [prefix+str(i) for i in range(n0+n1)]
    frame['BinaryClass'] = [0]*n0+[1]*n1
    frame['Class'] = frame.BinaryClass
    frame['Group'] = frame.BinaryClass.map({0:'Healthy',1:'TLE'})
    frame['DataType'] = 'Original'
    return frame


class MaximumAugmentationTests(unittest.TestCase):
    def check_invariants(self, frame, prepared, synthetic, pairs, info):
        target = min(n+8*n*(n-1)//2 for n in frame.BinaryClass.value_counts())
        self.assertEqual(prepared.BinaryClass.value_counts().to_dict(),{0:target,1:target})
        unordered = [tuple(sorted((a,b))) for a,b in zip(pairs.ParentSubject1,pairs.ParentSubject2)]
        self.assertEqual(len(unordered),len(set(unordered)))
        self.assertTrue(all(a!=b for a,b in unordered))
        labels = frame.set_index('Subject').BinaryClass.to_dict()
        mapping = pairs.set_index('PairID').to_dict('index')
        sizes = synthetic.groupby('PairID').size().to_dict()
        self.assertEqual(set(sizes),set(mapping))
        for pair,size in sizes.items():
            self.assertLessEqual(size,8)
            self.assertEqual(size,mapping[pair]['Children'])
        for row in synthetic[['PairID','ParentSubject1','ParentSubject2','BinaryClass']].to_dict('records'):
            for field in ['ParentSubject1','ParentSubject2']:
                self.assertEqual(row[field],mapping[row['PairID']][field])
                self.assertEqual(labels[row[field]],row['BinaryClass'])
        pd.testing.assert_frame_equal(prepared.iloc[:len(frame)].reset_index(drop=True),frame)
        self.assertTrue(np.isfinite(synthetic[COEF_COLUMNS].to_numpy()).all())
        self.assertEqual(info['target_per_class'],target)

    def test_jitter_maximum_and_partial_final_pair(self):
        frame = make_frame(5,3)
        result = augment_balanced_jitter(frame,'coef')
        self.check_invariants(frame,*result)
        self.assertEqual(Counter(result[1].groupby('PairID').size()),{8:5,6:1})

    def test_pls_maximum_and_partial_final_pair(self):
        frame = make_frame(5,3)
        prepared,synthetic,pairs,bundle,info = augment_plsda_same_class(frame,'coef',requested_components=2)
        self.check_invariants(frame,prepared,synthetic,pairs,info)
        self.assertEqual(info['pls_fit_scope'],'supplied_training_csv_only')

    def test_equal_original_classes_are_both_augmented(self):
        frame = make_frame(3,3)
        result = augment_balanced_jitter(frame,'coef')
        self.check_invariants(frame,*result)
        self.assertEqual(len(result[1]),48)
        self.assertEqual(len(result[2]),6)

    def test_seed_is_reproducible(self):
        frame = make_frame(5,3)
        first = augment_balanced_jitter(frame,'coef',seed=9)
        again = augment_balanced_jitter(frame,'coef',seed=9)
        different = augment_balanced_jitter(frame,'coef',seed=10)
        pd.testing.assert_frame_equal(first[1],again[1])
        self.assertFalse(first[1].equals(different[1]))

    def test_too_few_unique_pairs_raises(self):
        with self.assertRaisesRegex(ValueError,'too few unique'):
            augment_balanced_jitter(make_frame(19,2),'coef')

    def test_current_dataset_math_without_generating_features(self):
        for n0,n1,target in [(192,108,46332),(219,81,26001)]:
            frame = pd.DataFrame({'Subject':['sub-'+str(i) for i in range(n0+n1)],
                                  'BinaryClass':[0]*n0+[1]*n1})
            plan = augmentation_plan(frame)
            self.assertEqual(plan['target_per_class'],target)
            self.assertEqual(plan['maximum_class_capacities']['1'],target)

    def test_streamed_matches_materialized(self):
        frame = make_frame(12,9)  # Multiple batches.
        result = augment_balanced_jitter(frame,'coef')
        streamed = augment_balanced_jitter(frame,'coef',stream=True)
        pd.testing.assert_frame_equal(result[1],pd.concat(list(streamed[1].iter_chunks()),ignore_index=True))
        self.assertEqual(len(result[0]),len(streamed[0]))

    def test_all_four_process_exports_and_test_unchanged(self):
        cases = [('coef_raw_balanced_jitter','coef',prepare_coef_raw,{'augmentation':'balanced_jitter'}),
                 ('pointnet_raw_balanced_jitter','xyz',prepare_pointnet_raw,{'augmentation':'balanced_jitter'}),
                 ('coef_plsda','coef',prepare_coef_plsda,{'n_components':2}),
                 ('pointnet_plsda','xyz',prepare_pointnet_plsda,{'n_components':2})]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for method,kind,module,extra in cases:
                with self.subTest(method=method):
                    train,test = make_frame(5,3,kind),make_frame(2,2,kind,prefix='sub-test',seed=31)
                    train_path,test_path = root/(method+'_train.csv'),root/(method+'_test.csv')
                    train.to_csv(train_path,index=False)
                    test.to_csv(test_path,index=False)
                    before = test_path.read_bytes()
                    out = root/method
                    module.process(str(train_path),str(test_path),str(out),**extra)
                    checked = validate(method,kind,out,train_path,test_path,True,8)
                    self.assertEqual(checked['class_counts'],{'0':27,'1':27})
                    self.assertTrue(checked['unique_unordered_pairs_verified'])
                    self.assertEqual(test_path.read_bytes(),before)
                    manifest = json.loads((out/'data_manifest.json').read_text())
                    self.assertFalse(manifest['augmentation_applied_to_test'])


if __name__ == '__main__':
    unittest.main()
