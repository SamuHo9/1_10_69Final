"""Independently verify all bootstrap counts, resamples, CIs and Test scores."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    recall_score, roc_auc_score, roc_curve)

METRICS = ['Accuracy', 'BalancedAccuracy', 'Sensitivity_Class0', 'Specificity_Class0',
    'F1_Class0', 'Sensitivity_Class1', 'Specificity_Class1', 'F1_Class1', 'F1_macro', 'AUC']


def independent_metrics(y, pred, prob):
    scores = {'Accuracy': accuracy_score(y, pred), 'BalancedAccuracy': balanced_accuracy_score(y, pred),
        'F1_macro': f1_score(y, pred, average='macro', zero_division=0), 'AUC': roc_auc_score(y, prob)}
    for label in (0, 1):
        scores[f'Sensitivity_Class{label}'] = recall_score(y, pred, pos_label=label, zero_division=0)
        scores[f'Specificity_Class{label}'] = recall_score(y, pred, pos_label=1-label, zero_division=0)
        scores[f'F1_Class{label}'] = f1_score(y, pred, pos_label=label, zero_division=0)
    return scores


def close(left, right, name):
    if not np.allclose(left, right, atol=1e-10, rtol=1e-10):
        raise ValueError(f'Mismatch: {name}')


def audit_job(job, rounds, seed):
    output = Path(job['output'])
    manifest = json.loads((output/'bootstrap_manifest.json').read_text())
    source = Path(manifest['input_predictions'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError('Source predictions changed')
    frame = pd.read_csv(source).sort_values('Subject', kind='stable').reset_index(drop=True)
    y, pred, prob = frame.BinaryClass.to_numpy(), frame.Prediction.to_numpy(), frame.Probability.to_numpy()
    rows = pd.read_csv(output/'bootstrap_confusion_matrix.csv')
    stats = json.loads((output/'bootstrap_statistics.json').read_text())
    if manifest['attempted_rounds'] != rounds or manifest['seed'] != seed or manifest['sample_size'] != len(frame):
        raise ValueError('Incorrect sampling settings')
    original = independent_metrics(y, pred, prob)
    with np.load(output/'bootstrap_resamples.npz') as saved:
        expected = np.random.RandomState(seed).choice(len(frame), size=(rounds, len(frame)), replace=True)
        if not np.array_equal(saved['indices'], expected):
            raise ValueError('Resamples differ from declared legacy RNG')
        if not np.array_equal(saved['subjects'], frame.Subject.to_numpy()):
            raise ValueError('Subject ordering differs')
        sampled_y, sampled_pred = y[expected], pred[expected]
        valid_mask = (sampled_y.min(axis=1) == 0) & (sampled_y.max(axis=1) == 1)
        valid_ids = np.flatnonzero(valid_mask) + 1
        if not np.array_equal(rows['round'], valid_ids) or not np.array_equal(saved['valid_rounds'], valid_ids):
            raise ValueError('Missing/repeated or incorrectly skipped rounds')
        skipped_ids = [r['round'] for r in manifest['skipped_rounds']]
        if skipped_ids != list(np.flatnonzero(~valid_mask)+1):
            raise ValueError('Skipped-round record differs')
        if len(rows) != manifest['valid_rounds']:
            raise ValueError('Replicate count differs')
        actual_counts = {
            'TP': ((sampled_y == 1) & (sampled_pred == 1)).sum(axis=1)[valid_mask],
            'TN': ((sampled_y == 0) & (sampled_pred == 0)).sum(axis=1)[valid_mask],
            'FP': ((sampled_y == 0) & (sampled_pred == 1)).sum(axis=1)[valid_mask],
            'FN': ((sampled_y == 1) & (sampled_pred == 0)).sum(axis=1)[valid_mask]}
        for key, values in actual_counts.items():
            if not np.array_equal(rows[key], values):
                raise ValueError(f'Incorrect confusion counts: {key}')
        if not rows[['TP', 'TN', 'FP', 'FN']].sum(axis=1).eq(len(frame)).all():
            raise ValueError('Confusion counts do not equal N')
        for index in np.unique(np.linspace(0, len(rows)-1, 5).astype(int)):
            sample = expected[valid_ids[index]-1]
            scores = independent_metrics(y[sample], pred[sample], prob[sample])
            for key, value in scores.items():
                close(rows.iloc[index][key], value, key)
            fpr, tpr, _ = roc_curve(y[sample], prob[sample])
            curve = np.interp(saved['fpr_grid'], fpr, tpr)
            curve[0], curve[-1] = 0, 1
            close(saved['tpr_curves'][index], curve, 'ROC interpolation')
        close(saved['mean_tpr'], saved['tpr_curves'].mean(axis=0), 'Mean ROC')
        close(saved['lower_tpr'], np.percentile(saved['tpr_curves'], 2.5, axis=0), 'ROC lower band')
        close(saved['upper_tpr'], np.percentile(saved['tpr_curves'], 97.5, axis=0), 'ROC upper band')
    for key in METRICS:
        expected = {'original': original[key], 'bootstrap_mean': rows[key].mean(),
            'bootstrap_std': rows[key].std(ddof=1), 'ci95_lower': np.percentile(rows[key], 2.5),
            'ci95_upper': np.percentile(rows[key], 97.5)}
        for field, value in expected.items():
            close(stats[key][field], value, key+' '+field)
        for field in ('original', 'bootstrap_mean', 'ci95_lower', 'ci95_upper'):
            close(job['scores'][key+'_'+field], expected[field], 'Combined summary: '+key+' '+field)
    for name in ('roc_curve_lines.png', 'roc_curve_ci.png'):
        path = output/name
        if path.stat().st_size < 1000 or not path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError('Missing/invalid PNG: '+name)
    log = Path(job['log']).read_text(encoding='utf-8')
    if not log or 'Traceback (most recent call last)' in log:
        raise ValueError('Missing log or traceback in log')
    report = {'verified': True, 'valid_rounds': len(rows), 'skipped_rounds': len(skipped_ids),
        'source_unchanged': True, 'all_round_counts_recomputed': True, 'metric_and_ROC_spot_checks': 5,
        'all_confidence_intervals_recomputed': True}
    (output/'audit_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', type=Path, default=Path(__file__).resolve().parents[1]/'Model_Results/dataset_runs_20261003_10fold')
    args = parser.parse_args()
    root = args.result_root
    summary = json.loads((root/'bootstrap_run_summary.json').read_text())
    csv = pd.read_csv(root/'bootstrap_summary.csv')
    errors, verified = [], []
    for job in summary['jobs']:
        try:
            if job['status'] != 'completed':
                raise ValueError('Bootstrap job incomplete')
            report = audit_job(job, summary['rounds'], summary['seed'])
            matched = csv[(csv.dataset == job['dataset']) & (csv.side == job['side']) & (csv.model == job['model'])]
            if len(matched) != 1 or matched.iloc[0].status != 'completed':
                raise ValueError('Combined CSV missing/duplicate job')
            for key, value in job['scores'].items():
                close(matched.iloc[0][key], value, 'Combined CSV '+key)
            verified.append({'dataset': job['dataset'], 'side': job['side'], 'model': job['model'], **report})
        except Exception as exc:
            errors.append({'dataset': job['dataset'], 'side': job['side'], 'model': job['model'], 'error': str(exc)})
    if len(verified) != summary['expected_jobs']:
        errors.append({'error': 'Not all jobs verified'})
    report = {'verified_jobs': len(verified), 'valid_bootstrap_rounds': sum(r['valid_rounds'] for r in verified),
        'skipped_rounds': sum(r['skipped_rounds'] for r in verified), 'errors': errors, 'checks': verified}
    (root/'bootstrap_audit_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
