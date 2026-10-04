"""Tests for routing and source/person integrity in dataset model runs."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from run_dataset_models import build_jobs,ROOT
from train_dataset_model import safe_batches,load_data,transform_features


class DatasetTrainingTests(unittest.TestCase):
    def test_model_routing_for_all_prepared_sets(self):
        jobs = build_jobs(ROOT/'Output_Dataset',ROOT/'Model_Results/_test')
        self.assertEqual(len(jobs),68)
        self.assertEqual(len({job['output'] for job in jobs}),68)
        self.assertTrue(all((job['kind']=='xyz')==(job['model']=='PointNet') for job in jobs))

    def test_batches_retain_every_original_once(self):
        for size in [32,33,65,97,300,384,438]:
            batches = safe_batches(size,32,np.random.default_rng(42))
            self.assertEqual(sorted(np.concatenate(batches)),list(range(size)))
            self.assertTrue(all(len(batch)>1 for batch in batches))

    def test_scaler_and_pls_only_fit_training_rows(self):
        source = ROOT/'Output_Dataset/coef_raw/left'
        train,test,columns = load_data(source/'train_prepared.csv',source/'test_prepared.csv','coef')
        changed = test.copy();changed.loc[:,columns] += 100000
        _,_,first = transform_features(train,test,columns,'coef',8)
        _,_,second = transform_features(train,changed,columns,'coef',8)
        np.testing.assert_array_equal(first['scaler'].mean_,second['scaler'].mean_)
        np.testing.assert_array_equal(first['pls'].x_weights_,second['pls'].x_weights_)

    def test_test_synthetic_rows_rejected(self):
        source = ROOT/'Output_Dataset/coef_raw/left'
        with tempfile.TemporaryDirectory() as directory:
            test = pd.read_csv(source/'test_prepared.csv');test.loc[0,'DataType']='Synthetic'
            path = Path(directory)/'test.csv';test.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'synthetic'):
                load_data(source/'train_prepared.csv',path,'coef')


if __name__=='__main__':
    unittest.main()
