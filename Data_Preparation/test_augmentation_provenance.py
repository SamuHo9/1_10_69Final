"""Check that augmentation records point to the actual same-class parents."""
import unittest
import numpy as np
import pandas as pd
from data_prep_common import COEF_COLUMNS, augment_balanced_jitter


class AugmentationProvenanceTests(unittest.TestCase):
    def check_case(self, majority, minority, expected_sizes):
        size = majority + minority
        frame = pd.DataFrame(np.arange(size * 507).reshape(size,507),columns=COEF_COLUMNS)
        frame['Subject'] = ['sub-'+str(i) for i in range(size)]
        frame['BinaryClass'] = [0]*majority+[1]*minority
        frame['Class'] = frame['BinaryClass']
        frame['Group'] = ['Healthy']*majority+['TLE']*minority
        frame['DataType'] = 'Original'
        prepared, synthetic, pairs, info = augment_balanced_jitter(frame,'coef',children_per_pair=8,augmentation_size='balance_only')
        mapping = pairs.set_index('PairID').to_dict('index')
        self.assertEqual(list(synthetic.groupby('PairID',sort=True).size()),expected_sizes)
        self.assertEqual(set(synthetic['PairID']),set(mapping))
        for row in synthetic.to_dict('records'):
            pair = mapping[row['PairID']]
            self.assertEqual(row['ParentSubject1'],pair['ParentSubject1'])
            self.assertEqual(row['ParentSubject2'],pair['ParentSubject2'])
            self.assertEqual(row['BinaryClass'],pair['Class'])
        self.assertEqual(len(synthetic),majority-minority)
        self.assertEqual(len(prepared),2*majority)
        self.assertEqual(info['children_per_pair'],8)

    def test_one_complete_pair(self):
        self.check_case(10,2,[8])

    def test_last_partial_pair_keeps_correct_parents(self):
        self.check_case(20,3,[8,8,1])


if __name__=='__main__':
    unittest.main()
