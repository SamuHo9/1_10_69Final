"""Compare frozen OOF thresholds with default predictions from the same refit."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, f1_score, roc_auc_score
from train_dataset_model import ROOT, sha

METRICS = ('accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1_macro', 'roc_auc')


def metrics(frame, prediction):
    y = frame.BinaryClass.to_numpy()
    return {'accuracy': float(accuracy_score(y, prediction)),
        'balanced_accuracy': float(balanced_accuracy_score(y, prediction)),
        'sensitivity': float(recall_score(y, prediction, pos_label=1, zero_division=0)),
        'specificity': float(recall_score(y, prediction, pos_label=0, zero_division=0)),
        'f1_macro': float(f1_score(y, prediction, average='macro', zero_division=0)),
        'roc_auc': float(roc_auc_score(y, frame.Probability))}


def compare(new_root, old_root):
    new_run = json.loads((new_root/'run_summary.json').read_text())
    old_run = json.loads((old_root/'run_summary.json').read_text())
    assert new_run['protocol'] in ('person_10fold_oof_accuracy_threshold_v1',
        'person_10fold_inner_early_stop_oof_accuracy_threshold_v1')
    assert new_run['completed_training_jobs'] == new_run['expected_training_jobs'] == 68
    assert new_run['failed_training_jobs'] == 0
    for key in ('seed', 'pls_components', 'folds'):
        assert new_run[key] == old_run[key], key
    old_jobs = {(j['dataset'], j['side'], j['model']): j for j in old_run['jobs']}
    rows, snapshots, old_hashes, epoch_rows = [], [], {}, []
    for entry in json.loads((new_root/'source_code_manifest.json').read_text()):
        assert sha(entry['source']) == sha(entry['snapshot']) == entry['sha256']
        snapshots.append(entry['source'])
    canonical_subjects = {}
    for job in sorted(new_run['jobs'], key=lambda j: (j['side'], j['dataset'], j['model'])):
        assert job['status'] == 'completed'
        key = (job['dataset'], job['side'], job['model'])
        old_folder, folder = Path(old_jobs[key]['output']), Path(job['output'])
        old_manifest = json.loads((old_folder/'run_manifest.json').read_text())
        new_manifest = json.loads((folder/'run_manifest.json').read_text())
        assert old_manifest['input_sha256'] == new_manifest['input_sha256'], 'Dataset changed since the baseline run'
        previous_path, current_path = old_folder/'test_predictions.csv', folder/'test_predictions.csv'
        old_hashes[str(previous_path)] = sha(previous_path)
        previous = pd.read_csv(previous_path, float_precision='round_trip').sort_values('Subject').reset_index(drop=True)
        current = pd.read_csv(current_path, float_precision='round_trip').sort_values('Subject').reset_index(drop=True)
        assert list(current.Subject) == list(previous.Subject)
        assert np.array_equal(current.BinaryClass, previous.BinaryClass)
        assert np.isfinite(current.Probability).all()
        if job['side'] in canonical_subjects:
            assert list(current.Subject) == canonical_subjects[job['side']]
        else:
            canonical_subjects[job['side']] = list(current.Subject)
        default = metrics(current, current.PredictionDefault)
        tuned = metrics(current, current.Prediction)
        old = metrics(previous, previous.Prediction)
        stored = json.loads((folder/'metrics.json').read_text())
        for name in METRICS:
            assert np.isclose(default[name], stored['test_default'][name], atol=1e-12)
            assert np.isclose(tuned[name], stored['test'][name], atol=1e-12)
            assert np.isclose(old[name], old_jobs[key]['metrics'][name], atol=1e-12)
        default_correct = (current.PredictionDefault == current.BinaryClass).to_numpy()
        tuned_correct = (current.Prediction == current.BinaryClass).to_numpy()
        previous_correct = (previous.Prediction == previous.BinaryClass).to_numpy()
        fixed = int(np.sum(~default_correct & tuned_correct))
        regressed = int(np.sum(default_correct & ~tuned_correct))
        gain = 100 * (tuned_correct.astype(int)-default_correct.astype(int))
        samples = np.random.RandomState(42).choice(len(current), size=(1000, len(current)), replace=True)
        ci = np.percentile(gain[samples].mean(axis=1), [2.5, 97.5])
        previous_gain = 100 * (tuned_correct.astype(int)-previous_correct.astype(int))
        previous_ci = np.percentile(previous_gain[samples].mean(axis=1), [2.5, 97.5])
        row = {'dataset': job['dataset'], 'side': job['side'], 'model': job['model'],
            'test_rows': len(current), 'threshold': stored['decision_threshold'],
            'previous_test_accuracy': old['accuracy'], 'refit_default_test_accuracy': default['accuracy'],
            'tuned_test_accuracy': tuned['accuracy'], 'threshold_gain_pp': float(gain.mean()),
            'gain_ci95_lower_pp': float(ci[0]), 'gain_ci95_upper_pp': float(ci[1]),
            'change_from_previous_pp': 100*(tuned['accuracy']-old['accuracy']),
            'previous_change_ci95_lower_pp': float(previous_ci[0]),
            'previous_change_ci95_upper_pp': float(previous_ci[1]),
            'result_vs_previous': 'improved' if tuned_correct.sum()>previous_correct.sum() else 'decreased' if tuned_correct.sum()<previous_correct.sum() else 'unchanged',
            'fixed_previous_errors': int(np.sum(~previous_correct & tuned_correct)),
            'new_errors_vs_previous': int(np.sum(previous_correct & ~tuned_correct)),
            'final_epochs': stored.get('final_epochs'),
            'refit_default_change_pp': 100*(default['accuracy']-old['accuracy']),
            'refit_default_predictions_changed': int(np.sum(current.PredictionDefault.to_numpy() != previous.Prediction.to_numpy())),
            'fixed_test_errors': fixed, 'new_test_errors': regressed,
            'default_correct': int(default_correct.sum()), 'tuned_correct': int(tuned_correct.sum()),
            'result': 'improved' if fixed>regressed else 'decreased' if fixed<regressed else 'unchanged',
            'max_probability_change_from_previous': float(np.max(np.abs(current.Probability-previous.Probability))),
            'oof_default_accuracy': stored['cv_oof']['accuracy'],
            'oof_threshold_selection_accuracy': stored['oof_threshold_selection']['accuracy'],
            'output': str(folder), 'log': job['log']}
        for name in METRICS:
            if name != 'accuracy':
                row['previous_'+name] = old[name]
                row['default_'+name] = default[name]
                row['tuned_'+name] = tuned[name]
                row[name+'_change'] = tuned[name]-default[name]
                row[name+'_change_from_previous'] = tuned[name]-old[name]
        assert row['roc_auc_change'] == 0
        rows.append(row)
        if new_manifest.get('early_stopping_applicable'):
            for fold in new_manifest['fold_records']:
                runtime = fold['runtime']
                epoch_rows.append({'dataset':job['dataset'], 'side':job['side'], 'model':job['model'],
                    'fold':fold['fold'], 'epochs_run':runtime['epochs_run'], 'best_epoch':runtime['best_epoch'],
                    'best_validation_loss':runtime['best_validation_loss'], 'max_epochs':runtime['max_epochs'],
                    'stopped_early':runtime['stopped_early'], 'best_checkpoint_restored':runtime['best_checkpoint_restored'],
                    'final_epochs':stored['final_epochs'], 'fit_train_people':len(fold['fit_train_people']),
                    'stop_people':len(fold['stop_people']), 'outer_oof_people':len(fold['validation_people'])})
    frame = pd.DataFrame(rows)
    frame.to_csv(new_root/'threshold_comparison.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
    if epoch_rows:
        assert len(epoch_rows) == 560
        pd.DataFrame(epoch_rows).to_csv(new_root/'early_stopping_epochs.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
    by_side = []
    for side, group in frame.groupby('side'):
        by_side.append({'side': side, 'jobs': len(group), 'test_rows_per_job': int(group.test_rows.iloc[0]),
            'improved': int((group.result == 'improved').sum()), 'decreased': int((group.result == 'decreased').sum()),
            'unchanged': int((group.result == 'unchanged').sum()),
            'mean_default_accuracy': float(group.refit_default_test_accuracy.mean()),
            'mean_tuned_accuracy': float(group.tuned_test_accuracy.mean()),
            'mean_threshold_gain_pp': float(group.threshold_gain_pp.mean())})
        by_side[-1].update({'improved_vs_previous':int((group.result_vs_previous == 'improved').sum()),
            'decreased_vs_previous':int((group.result_vs_previous == 'decreased').sum()),
            'unchanged_vs_previous':int((group.result_vs_previous == 'unchanged').sum()),
            'mean_previous_accuracy':float(group.previous_test_accuracy.mean()),
            'mean_change_from_previous_pp':float(group.change_from_previous_pp.mean())})
    pd.DataFrame(by_side).to_csv(new_root/'threshold_comparison_by_side.csv', index=False, encoding='utf-8-sig')
    for path, value in old_hashes.items():
        assert sha(path) == value, 'Previous predictions changed'
    report = {'verified_jobs': len(rows), 'old_result_root': str(old_root), 'new_result_root': str(new_root),
        'objective': 'Train OOF accuracy', 'test_used_for_threshold_selection': False,
        'comparison': 'default and tuned decisions use identical probabilities from the same refitted model',
        'improved': int((frame.result == 'improved').sum()), 'decreased': int((frame.result == 'decreased').sum()),
        'unchanged': int((frame.result == 'unchanged').sum()), 'by_side': by_side,
        'improved_vs_previous':int((frame.result_vs_previous == 'improved').sum()),
        'decreased_vs_previous':int((frame.result_vs_previous == 'decreased').sum()),
        'unchanged_vs_previous':int((frame.result_vs_previous == 'unchanged').sum()),
        'early_stopping_fold_records':len(epoch_rows),
        'training_settings_previous':{k:old_run.get(k) for k in ('epochs','mlp_max_epochs','early_stopping','patience','min_delta','stop_fraction')},
        'training_settings_current':{k:new_run.get(k) for k in ('epochs','mlp_max_epochs','early_stopping','patience','min_delta','stop_fraction')},
        'gain_ci_method': 'paired ordinary Test bootstrap, 1000 resamples, RandomState seed 42, percentile CI95; descriptive per-model intervals without multiplicity correction',
        'source_files_verified': len(snapshots), 'previous_predictions_unchanged': True,
        'known_limitations': 'Tuned OOF is a selection score. Existing Test has been examined across experiments; this is exploratory, not new external validation. Source ICP reference includes Test and hemisphere class-0 labels include contralateral patient hemispheres.',
        'errors': []}
    (new_root/'threshold_comparison_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (new_root/'previous_predictions_sha256.json').write_text(json.dumps(old_hashes, indent=2), encoding='utf-8')
    return frame, report


def plots(root, frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for side, group in frame.groupby('side'):
        ordered = group.sort_values(['threshold_gain_pp', 'dataset', 'model'])
        fig, ax = plt.subplots(figsize=(11, 13))
        values = ordered.threshold_gain_pp.to_numpy()
        ax.barh(np.arange(len(ordered)), values,
                color=['#2f855a' if v>0 else '#c53030' if v<0 else '#718096' for v in values])
        ax.set_yticks(np.arange(len(ordered)))
        ax.set_yticklabels([f'{r.dataset} / {r.model}' for r in ordered.itertuples()], fontsize=9)
        ax.axvline(0, color='black', linewidth=1)
        ax.set(xlabel='Test accuracy change (percentage points)',
               title=f'{side.capitalize()}: frozen OOF threshold vs default rule\nSame refitted model, same Test subjects')
        ax.grid(axis='x', alpha=0.2)
        fig.tight_layout()
        fig.savefig(root/f'threshold_accuracy_change_{side}.png', dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-root', type=Path, default=ROOT/'Model_Results/dataset_runs_20261003_oof_threshold')
    parser.add_argument('--old-root', type=Path, default=ROOT/'Model_Results/dataset_runs_20261003_10fold')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    frame, report = compare(args.new_root.resolve(), args.old_root.resolve())
    if not args.no_plots:
        plots(args.new_root, frame)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
