"""Small integration suite: all model families and data protocols, 1 neural epoch.

This is code validation, not the final 80-epoch experiment.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from train_dataset_model import ROOT
from audit_10fold_run import audit_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT/'Model_Results/validation_10fold_20261003')
    parser.add_argument('--tune-threshold', action='store_true')
    parser.add_argument('--early-stopping', action='store_true')
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError('Choose a new validation output directory')
    args.output_root.mkdir(parents=True)
    cases = [('augment_plsda_balanced', 'left', 'SVM', 'coef'),
        ('pointnet_plsda', 'right', 'PointNet', 'xyz'),
        ('plsda_latent_features', 'left', 'ResNetAE', 'latent'),
        ('coef_raw', 'right', 'PLSDA', 'coef'),
        ('coef_raw_balanced_jitter', 'left', 'SqueezeNet', 'coef'),
        ('coef_plsda', 'right', 'MobileNet', 'coef'),
        ('pointnet_raw_balanced_jitter', 'left', 'PointNet', 'xyz'),
        ('coef_raw', 'left', 'ResNet', 'coef'),
        ('plsda_latent_features', 'right', 'MLP', 'latent')]
    reports = []
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
        PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
    for dataset, side, model, kind in cases:
        output = args.output_root/dataset/side/model
        folder = ROOT/'Output_Dataset'/dataset/side
        if model == 'PLSDA':
            folder = ROOT/'Output_Dataset/coef_raw'/side
        train_name, test_name = ('train_plsda_latent_features.csv', 'test_plsda_latent_features.csv') if kind == 'latent' else ('train_prepared.csv', 'test_prepared.csv')
        command = [sys.executable, '-u', str(ROOT/'Model/train_dataset_model.py'),
            '--dataset', dataset, '--side', side, '--model', model, '--kind', kind,
            '--train-csv', str(folder/train_name), '--test-csv', str(folder/test_name),
            '--output-dir', str(output), '--folds', '10', '--epochs', '1', '--threads', '2']
        # Also check an original All_Augment_tain entry point and its automatic log.
        if model == 'SVM':
            command = [sys.executable, '-u', str(ROOT/'Model/All_Augment_tain/left/SVM/train_svm_pls.py'),
                '--dataset', dataset, '--output-dir', str(output)]
        if args.tune_threshold:
            command.append('--tune-threshold')
        if args.early_stopping:
            command += ['--early-stopping','--epochs','5','--mlp-max-epochs','5','--patience','2','--min-delta','100']
        log = args.output_root/f'{dataset}_{side}_{model}.log'
        print(f'Validating {dataset}/{side}/{model}: 10 folds', flush=True)
        with log.open('w', encoding='utf-8') as stream:
            stream.write('Command: '+json.dumps(command)+'\n'); stream.flush()
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, check=True)
        report = audit_job(output)
        if args.early_stopping and model not in ('SVM','PLSDA'):
            manifest = json.loads((output/'run_manifest.json').read_text())
            for record in manifest['fold_records']:
                assert record['runtime']['stopped_early'] and record['runtime']['epochs_run'] == 3
                if model == 'MLP':
                    import pickle
                    import numpy as np
                    folder = output/f'fold_{record["fold"]:02d}'
                    with (folder/'best_checkpoint.pkl').open('rb') as stream:
                        best = pickle.load(stream)
                    with (folder/'model.pkl').open('rb') as stream:
                        used = pickle.load(stream)
                    for first, second in zip(best.coefs_+best.intercepts_, used.coefs_+used.intercepts_):
                        np.testing.assert_array_equal(first, second)
        reports.append({'dataset': dataset, 'side': side, 'model': model, **report})
        print(f'PASS {model}: {report["oof_rows"]} OOF rows, 10 fold checkpoints', flush=True)
    report = {'purpose': 'integration_validation_not_final_experiment', 'neural_epochs': 5 if args.early_stopping else 1,
        'early_stopping': args.early_stopping,
        'forced_stop_settings': {'patience': 2, 'min_delta': 100, 'max_epochs': 5, 'expected_epochs_run': 3} if args.early_stopping else None,
        'verified_jobs': len(reports), 'reports': reports}
    (args.output_root/'validation_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'All {len(reports)} integration cases passed', flush=True)


if __name__ == '__main__':
    main()
