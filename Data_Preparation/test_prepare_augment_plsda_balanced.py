"""Validate legacy pair/count semantics and train/test integrity."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from test_maximum_augmentation import make_frame
from prepare_augment_plsda_balanced import augment, balanced_plan, process
from data_prep_common import COEF_COLUMNS


class BalancedPLSDATests(unittest.TestCase):
    def test_current_counts_and_odd_parents(self):
        for n0,n1,target in [(192,108,960),(219,81,1091)]:
            frame = pd.DataFrame({'Subject':list(map(str,range(n0+n1))),
                                  'BinaryClass':[0]*n0+[1]*n1})
            plan = balanced_plan(frame)
            self.assertEqual(plan['target_per_class'],target)
            self.assertEqual(plan['available_disjoint_pairs'],{'0':n0//2,'1':n1//2})
            self.assertEqual(plan['unpaired_originals_per_class'],{'0':n0%2,'1':n1%2})

    def test_original_algorithm_pairing_jitter_and_reconstruction(self):
        frame = make_frame(9,5)
        prepared,synthetic,pairs,bundle,info = augment(frame,n_components=2)
        x = frame[COEF_COLUMNS].to_numpy()
        y = frame.BinaryClass.to_numpy()
        model = PLSRegression(n_components=2,scale=True)
        scores,_ = model.fit_transform(x,np.eye(2)[y])
        rng = np.random.RandomState(42)
        expected_pairs = []
        for label in (0,1):
            pool = list(np.flatnonzero(y==label))
            rng.shuffle(pool)
            while len(pool)>=2:
                first = pool.pop(0)
                distances = [np.linalg.norm(scores[first]-scores[second]) for second in pool]
                second = pool.pop(int(np.argmin(distances)))
                expected_pairs.append((label,first,second))
        self.assertEqual([(row.Class,row.ParentSubject1,row.ParentSubject2) for row in pairs.itertuples()],
                         [(label,frame.iloc[a].Subject,frame.iloc[b].Subject) for label,a,b in expected_pairs])
        actual_sizes = list(pairs.Children)
        self.assertEqual(actual_sizes,[8,8,8,8,18,18])
        offset = 0
        for (_,a,b),n in zip(expected_pairs,actual_sizes):
            alpha = np.clip(np.linspace(0.1,0.9,n)+rng.uniform(-0.02,0.02,n),0.05,0.95)
            expected = np.array([model.inverse_transform(((1-w)*scores[a]+w*scores[b])[None,:])[0] for w in alpha])
            np.testing.assert_allclose(synthetic.Alpha.iloc[offset:offset+n],alpha)
            np.testing.assert_allclose(synthetic[COEF_COLUMNS].iloc[offset:offset+n],expected,rtol=1e-10,atol=1e-10)
            offset += n
        self.assertEqual(prepared.BinaryClass.value_counts().to_dict(),{0:41,1:41})
        self.assertEqual(len(set(pairs.ParentSubject1)|set(pairs.ParentSubject2)),2*len(pairs))
        pd.testing.assert_frame_equal(prepared.iloc[:len(frame)],frame)

    def test_equal_original_counts_use_both_classes(self):
        prepared,synthetic,pairs,bundle,info = augment(make_frame(4,4),n_components=2)
        self.assertEqual(prepared.BinaryClass.value_counts().to_dict(),{0:20,1:20})
        self.assertEqual(len(pairs),4)
        self.assertTrue(pairs.Children.eq(8).all())

    def test_invalid_parameters_and_single_parent(self):
        with self.assertRaises(ValueError):
            balanced_plan(make_frame(5,1))
        with self.assertRaises(ValueError):
            balanced_plan(make_frame(5,3),0)
        with self.assertRaises(ValueError):
            augment(make_frame(5,3),n_components=0)

    def test_stream_repeatability_and_process_export(self):
        frame = make_frame(9,5)
        streamed = augment(frame,n_components=2,stream=True)
        pd.testing.assert_frame_equal(pd.concat(list(streamed[1].iter_chunks()),ignore_index=True),
                                      pd.concat(list(streamed[1].iter_chunks()),ignore_index=True))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path,test_path = root/'train.csv',root/'test.csv'
            frame.to_csv(train_path,index=False)
            make_frame(2,2,prefix='sub-test',seed=70).to_csv(test_path,index=False)
            before = (train_path.read_bytes(),test_path.read_bytes())
            result = process(str(train_path),str(test_path),str(root/'result'),n_components=2)
            self.assertEqual(result['train_rows'],82)
            self.assertTrue(result['same_class_train_only_parents_verified'])
            self.assertEqual(before,(train_path.read_bytes(),test_path.read_bytes()))
            with self.assertRaises(FileExistsError):
                process(str(train_path),str(test_path),str(root/'result'),n_components=2)


if __name__ == '__main__':
    unittest.main()
