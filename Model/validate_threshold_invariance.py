"""Prove that changing Test features and labels cannot change OOF threshold."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from train_dataset_model import ROOT, sha
from audit_10fold_run import audit_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-root', type=Path, default=ROOT/'Model_Results/validation_threshold_20261003')
    args = parser.parse_args()
    baseline = args.validation_root/'augment_plsda_balanced/left/SVM'
    altered = args.validation_root/'test_invariance/SVM'
    if altered.exists():
        raise FileExistsError('Choose a new validation root')
    altered.parent.mkdir(parents=True)
    test_path = ROOT/'Output_Dataset/augment_plsda_balanced/left/test_prepared.csv'
    original_hash = sha(test_path)
    frame = pd.read_csv(test_path)
    frame.BinaryClass = 1-frame.BinaryClass
    columns = [c for c in frame if c.startswith('Coef_')]
    frame.loc[:, columns] = frame[columns].to_numpy()*1.5 + 1000
    altered_input = altered.parent/'altered_test.csv'
    frame.to_csv(altered_input, index=False)
    command = [sys.executable, '-u', str(ROOT/'Model/train_dataset_model.py'),
        '--dataset', 'augment_plsda_balanced', '--side', 'left', '--model', 'SVM', '--kind', 'coef',
        '--train-csv', str(ROOT/'Output_Dataset/augment_plsda_balanced/left/train_prepared.csv'),
        '--test-csv', str(altered_input), '--output-dir', str(altered), '--tune-threshold', '--epochs', '80']
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
    with (altered.parent/'test_invariance.log').open('w', encoding='utf-8') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, check=True)
    first = json.loads((baseline/'threshold_selection.json').read_text())
    second = json.loads((altered/'threshold_selection.json').read_text())
    assert first == second, 'Test changes affected the threshold selection'
    original_oof = pd.read_csv(baseline/'oof_predictions.csv', float_precision='round_trip')
    changed_oof = pd.read_csv(altered/'oof_predictions.csv', float_precision='round_trip')
    pd.testing.assert_frame_equal(original_oof, changed_oof)
    assert sha(test_path) == original_hash
    audit_job(altered)
    report = {'verified': True, 'test_labels_flipped': True, 'test_features_perturbed': True,
              'oof_unchanged': True, 'selected_threshold_unchanged': True, 'original_test_file_unchanged': True,
              'threshold': first['threshold']}
    (altered.parent/'invariance_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
