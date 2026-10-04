"""Regression checks for explicit labels and processing-provenance failures."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import feature_metadata as metadata


class FeatureMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def labels(self, headers, rows):
        path = self.folder / 'labels.csv'
        with path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)
        return path

    def source(self, name, contract=None):
        path = self.folder / name
        path.write_text('sample', encoding='utf-8')
        if contract is not None:
            metadata._processing_path(path).write_text(json.dumps({'feature_contract': contract}), encoding='utf-8')
        return path

    def test_no_optional_labels(self):
        self.assertIsNone(metadata.load_labels(None))
        self.assertIsNone(metadata.load_dataset_labels(None))

    def test_contralateral_label_and_explicit_dataset(self):
        path = self.labels(['Subject','BinaryClass','Class','Dataset'], [['left_sub-001',0,2,'ds005602']])
        self.assertEqual(metadata.load_labels(path)['left_sub-001'][1:], (2,0))
        self.assertEqual(metadata.load_dataset_labels(path), {'left_sub-001':'ds005602'})

    def test_conflicting_and_fractional_labels_are_rejected(self):
        for binary, clinical in [(0,1),(.5,0),(1,2)]:
            with self.subTest(binary=binary, clinical=clinical):
                path = self.labels(['Subject','BinaryClass','Class'], [['sub-001',binary,clinical]])
                with self.assertRaises(ValueError):
                    metadata.load_labels(path)

    def test_duplicate_subject_and_blank_dataset_are_rejected(self):
        path = self.labels(['Subject','BinaryClass'], [['sub-001',0],['sub-001',1]])
        with self.assertRaises(ValueError):
            metadata.load_labels(path)
        path = self.labels(['Subject','BinaryClass','Dataset'], [['sub-001',0,'']])
        with self.assertRaises(ValueError):
            metadata.load_dataset_labels(path)

    def test_missing_sidecar_and_mixed_processing_are_rejected(self):
        first = self.source('sub-001_SPHARM_ellalign.coef')
        with self.assertRaisesRegex(ValueError,'Missing processing sidecar'):
            metadata.validate_feature_contracts([first])
        first = self.source(first.name, {'version':'spharm-feature-v1','spharm_degree':12})
        second = self.source('sub-002_SPHARM.coef', {'version':'spharm-feature-v1','spharm_degree':10})
        with self.assertRaisesRegex(ValueError,'Mismatched processing contracts'):
            metadata.validate_feature_contracts([first,second])

    def test_unknown_reference_is_preserved_in_manifest(self):
        contract = {'version':'spharm-feature-v1','icp_mode':'unknown'}
        source = self.source('sub-001_SPHARM_ellalign.coef',contract)
        checked = metadata.validate_feature_contracts([source])
        csv_path = self.labels(['Subject','BinaryClass','Coef_1'], [['sub-001',0,1.5]])
        result = metadata.write_feature_manifest(csv_path,[source],contract=checked)
        self.assertFalse(result['fixed_reference_validated'])
        self.assertEqual(result['feature_contract']['icp_mode'],'unknown')
        self.assertEqual(len(result['source_files'][0]['sha256']),64)
        self.assertTrue(Path(str(csv_path)+'.json').is_file())

    def test_mesh_variants_find_the_same_subject_sidecar(self):
        contract = {'version':'spharm-feature-v1','icp_mode':'unknown'}
        self.source('sub-001_SPHARM.coef',contract)
        for suffix in ['ellalign','realigned','procalign']:
            source = self.source('sub-001_SPHARM_'+suffix+'.vtk')
            self.assertEqual(metadata.validate_mesh_processing_contracts([source]),contract)

    def test_claimed_fixed_reference_requires_hashes(self):
        source = self.source('sub-001_SPHARM.coef',{'version':'spharm-feature-v1','icp_mode':'fixed_reference'})
        with self.assertRaises(ValueError):
            metadata.validate_feature_contracts([source])


if __name__ == '__main__':
    unittest.main()
