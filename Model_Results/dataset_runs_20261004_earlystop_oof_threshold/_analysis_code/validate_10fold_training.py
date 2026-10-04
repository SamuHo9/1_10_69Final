"""Small integration suite and checkpoint replay audit for all model families.

The integration mode uses short caps/forced stopping, not the final experiment.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from train_dataset_model import ROOT
from audit_10fold_run import audit_job


def replay_best_oof(output):
    """Recompute real outer OOF from saved best weights and fit-only preprocessing."""
    import pickle
    import numpy as np
    import pandas as pd
    from train_dataset_model import model_class
    from data_prep_common import pls_scores
    manifest = json.loads((output/'run_manifest.json').read_text())
    if not manifest.get('early_stopping_applicable'):
        return 0
    source = manifest['inputs'].get('original_train', manifest['inputs']['requested_train'])
    original = pd.read_csv(source)
    if 'DataType' in original:
        original = original[original.DataType == 'Original']
    original = original.set_index('Subject')
    oof = pd.read_csv(output/'oof_predictions.csv', float_precision='round_trip').set_index('Subject')
    for record in manifest['fold_records']:
        folder = output/f'fold_{record["fold"]:02d}'
        with (folder/'preprocessing.pkl').open('rb') as stream:
            prep = pickle.load(stream)
        data = original.loc[record['validation_subjects'], prep['columns']].to_numpy(dtype=np.float64)
        if prep['kind'] == 'latent_rebuilt':
            data = prep['latent_scaler'].transform(pls_scores(prep['latent_pls'], data)).astype('float32')
        elif prep['kind'] == 'xyz':
            data = data.astype('float32').reshape(-1, 1002, 3).transpose(0, 2, 1)
        else:
            data = prep['scaler'].transform(data)
            if 'pls' in prep:
                data = prep['pls'].transform(data)
            data = data.astype('float32')
        if manifest['model'] == 'MLP':
            with (folder/'best_checkpoint.pkl').open('rb') as stream:
                best = pickle.load(stream)
            with (folder/'model.pkl').open('rb') as stream:
                used = pickle.load(stream)
            for first, second in zip(best.coefs_+best.intercepts_, used.coefs_+used.intercepts_):
                np.testing.assert_array_equal(first, second)
            probability = best.predict_proba(data)[:, 1]
        else:
            import torch
            torch.set_num_threads(2)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            cls, _ = model_class(manifest['model'], manifest['side'])
            model = (cls(seq_length=data.shape[1]) if manifest['model'] == 'ResNetAE' else cls()).to(device)
            checkpoint = torch.load(folder/'best_checkpoint.pt', map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            inputs = data if manifest['model'] == 'PointNet' else data[:, None, :]
            values = []
            with torch.no_grad():
                for start in range(0, len(inputs), manifest['batch_size']):
                    result = model(torch.tensor(inputs[start:start+manifest['batch_size']], device=device))
                    if manifest['model'] == 'ResNetAE':
                        result = result[0]
                    result = (torch.softmax(result, dim=1)[:, 1] if manifest['model'] == 'PointNet' else
                        torch.sigmoid(result).flatten() if manifest['model'] == 'SqueezeNet' else result.flatten())
                    values.extend(result.cpu().numpy())
            probability = np.asarray(values)
            del model, checkpoint
        saved = oof.loc[record['validation_subjects'], 'Probability'].to_numpy()
        np.testing.assert_allclose(probability, saved, atol=2e-6, rtol=1e-5,
            err_msg=f'Best checkpoint does not reproduce outer OOF: {folder}')
    return 10


def audit_existing(root, previous_root=None):
    import numpy as np
    summary = json.loads((root/'run_summary.json').read_text())
    assert summary['completed_training_jobs'] == summary['expected_training_jobs'] == 68
    assert summary['failed_training_jobs'] == 0
    reports, total, canonical_folds = [], 0, {}
    for job in summary['jobs']:
        assert job['status'] == 'completed'
        output = Path(job['output'])
        report = audit_job(output)
        stored = json.loads((output/'metrics.json').read_text())
        for key in ('accuracy','balanced_accuracy','sensitivity','specificity','f1_macro','roc_auc'):
            assert np.isclose(job['metrics'][key], stored['test'][key], atol=1e-12)
            assert np.isclose(job['oof_metrics'][key], stored['cv_oof'][key], atol=1e-12)
        manifest = json.loads((output/'run_manifest.json').read_text())
        assignment = [sorted(r['validation_subjects']) for r in manifest['fold_records']]
        if job['side'] in canonical_folds:
            assert assignment == canonical_folds[job['side']], 'Outer fold assignments differ across models'
        else:
            canonical_folds[job['side']] = assignment
        replayed = replay_best_oof(output)
        total += replayed
        reports.append({'dataset':job['dataset'], 'side':job['side'], 'model':job['model'],
            'best_checkpoint_oof_folds_replayed':replayed, **report})
        print(f'PASS {job["dataset"]}/{job["side"]}/{job["model"]}: audited; best OOF replay {replayed} folds', flush=True)
    assert total == 560
    report = {'verified_jobs':len(reports), 'best_checkpoint_oof_folds_replayed':total,
        'best_checkpoint_oof_rows_replayed':total*30, 'errors':[], 'reports':reports}
    (root/'audit_best_checkpoint_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (root/'audit_report.json').write_text(json.dumps({'verified_jobs':len(reports), 'best_checkpoint_oof_folds_replayed':total,
        'same_outer_folds_across_models':True, 'errors':[]}, indent=2), encoding='utf-8')
    print(f'All {len(reports)} jobs passed; replayed {total} best checkpoints / {total*30} OOF predictions', flush=True)
    if previous_root:
        from compare_threshold_results import compare, plots
        frame, comparison = compare(root, previous_root)
        plots(root, frame)
        print(json.dumps(comparison, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT/'Model_Results/validation_10fold_20261003')
    parser.add_argument('--tune-threshold', action='store_true')
    parser.add_argument('--early-stopping', action='store_true')
    parser.add_argument('--audit-existing', type=Path, help='Audit a completed run and replay OOF from every best checkpoint')
    parser.add_argument('--compare-old-root', type=Path, help='With --audit-existing, compare Test results and save figures')
    args = parser.parse_args()
    if args.audit_existing:
        audit_existing(args.audit_existing.resolve(), args.compare_old_root.resolve() if args.compare_old_root else None)
        return
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
