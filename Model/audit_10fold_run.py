"""Independently validate saved 10-fold OOF coverage, parents and metrics."""
import argparse
import json
import zipfile
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from train_dataset_model import sha, ROOT
sys.path.insert(0, str(ROOT/'Data_Preparation'))


def scores(frame):
    y, predicted = frame.BinaryClass, frame.Prediction
    return {'accuracy': accuracy_score(y, predicted), 'balanced_accuracy': balanced_accuracy_score(y, predicted),
        'sensitivity': recall_score(y, predicted, pos_label=1, zero_division=0),
        'specificity': recall_score(y, predicted, pos_label=0, zero_division=0),
        'f1_macro': f1_score(y, predicted, average='macro', zero_division=0),
        'roc_auc': roc_auc_score(y, frame.Probability)}


def audit_job(output):
    output = Path(output)
    manifest = json.loads((output/'run_manifest.json').read_text())
    early_mode = manifest['protocol'] == 'person_10fold_inner_early_stop_oof_accuracy_threshold_v1'
    threshold_mode = manifest['protocol'] in ('person_10fold_oof_accuracy_threshold_v1', 'person_10fold_inner_early_stop_oof_accuracy_threshold_v1')
    assert manifest['protocol'] in ('person_10fold_oof_v1', 'person_10fold_oof_accuracy_threshold_v1', 'person_10fold_inner_early_stop_oof_accuracy_threshold_v1')
    epoch_model = early_mode and manifest['early_stopping_applicable']
    assert manifest['folds'] == 10 and len(manifest['fold_records']) == 10
    assert not manifest['test_used_for_selection']
    assert manifest['outer_oof_used_for_selection'] == threshold_mode
    for name, path in manifest['inputs'].items():
        assert sha(path) == manifest['input_sha256'][name], f'Changed input: {name}'
    train = pd.read_csv(manifest['inputs']['requested_train'])
    if 'DataType' in train:
        train = train[train.DataType == 'Original'].reset_index(drop=True)
    test = pd.read_csv(manifest['inputs']['requested_test'])
    oof, predictions = pd.read_csv(output/'oof_predictions.csv', float_precision='round_trip'), pd.read_csv(output/'test_predictions.csv', float_precision='round_trip')
    assert list(oof.Subject) == list(train.Subject)
    assert np.array_equal(oof.BinaryClass, train.BinaryClass)
    assert oof.Subject.is_unique and set(oof.Fold) == set(range(1, 11))
    assert list(predictions.Subject) == list(test.Subject)
    assert np.array_equal(predictions.BinaryClass, test.BinaryClass)
    assert not set(oof.Subject) & set(predictions.Subject)
    stored = json.loads((output/'metrics.json').read_text())
    if threshold_mode:
        selection = json.loads((output/'threshold_selection.json').read_text())
        assert manifest['threshold_optimization_performed'] and selection['objective'] == 'accuracy'
        assert not selection['test_used_for_selection']
        assert selection['selection_source'] == 'original_train_10fold_oof'
        assert selection['selection_rows'] == len(oof)
        assert selection['oof_predictions_sha256'] == sha(output/'oof_predictions.csv')
        sweep = pd.read_csv(output/'threshold_sweep.csv', float_precision='round_trip')
        candidates = np.unique(np.r_[0.0, 0.5, 1.0, oof.Probability.to_numpy(), np.nextafter(oof.Probability.max(), np.inf)])
        assert np.array_equal(sweep.threshold, candidates)
        y, p = oof.BinaryClass.to_numpy(), oof.Probability.to_numpy()
        independently_scored = []
        for row in sweep.itertuples():
            pred = (p >= row.threshold).astype(int)
            correct = int(np.sum(y == pred))
            balanced = balanced_accuracy_score(y, pred)
            assert row.correct == correct and row.rows == len(y)
            assert np.isclose(row.accuracy, correct/len(y), atol=1e-12)
            assert np.isclose(row.balanced_accuracy, balanced, atol=1e-12)
            independently_scored.append((-correct, -balanced, abs(row.threshold-0.5), row.threshold))
        best_index = min(range(len(sweep)), key=lambda i: independently_scored[i])
        chosen = float(sweep.threshold.iloc[best_index])
        assert int(sweep.selected.sum()) == 1 and bool(sweep.selected.iloc[best_index])
        assert chosen == selection['threshold'] == manifest['decision_threshold'] == stored['decision_threshold']
        assert np.all(predictions.DecisionThreshold == chosen)
        assert np.array_equal(predictions.Prediction, (predictions.Probability.to_numpy() >= chosen).astype(int))
        assert stored['test']['roc_auc'] == stored['test_default']['roc_auc']
        assert np.isclose(stored['test_accuracy_change'], stored['test']['accuracy']-stored['test_default']['accuracy'])
        baseline = predictions.copy()
        baseline.Prediction = predictions.PredictionDefault
        tuned_oof = oof.copy()
        tuned_oof.Prediction = (p >= chosen).astype(int)
        for key, frame in [('test_default', baseline), ('oof_threshold_selection', tuned_oof)]:
            for metric, value in scores(frame).items():
                assert np.isclose(stored[key][metric], value, atol=1e-10, rtol=1e-10), metric
    for key, frame in [('cv_oof', oof), ('test', predictions)]:
        assert np.isfinite(frame.Probability).all() and frame.Probability.between(0, 1).all()
        assert set(frame.Prediction).issubset({0, 1})
        for metric, value in scores(frame).items():
            assert np.isclose(stored[key][metric], value, atol=1e-10, rtol=1e-10), metric
    all_people = set(oof.PatientID)
    from data_prep_common import patient_group_id
    test_people = {patient_group_id(s) for s in predictions.Subject}
    model_name = 'direct_plsda_model.pkl' if manifest['model'] == 'PLSDA' else 'model.pkl' if manifest['model'] in ('SVM', 'MLP') else 'model.pt'
    for record in manifest['fold_records']:
        fold = record['fold']
        valid = oof[oof.Fold == fold]
        assert list(valid.Subject) == record['validation_subjects']
        assert set(valid.PatientID) == set(record['validation_people'])
        assert not set(record['train_people']) & set(record['validation_people'])
        assert set(record['train_people']) | set(record['validation_people']) == all_people
        assert not set(record['train_subjects']) & set(record['validation_subjects'])
        folder = output/f'fold_{fold:02d}'
        if early_mode:
            fit_people, stop_people = set(record['fit_train_people']), set(record['stop_people'])
            assert not fit_people & stop_people
            assert fit_people | stop_people == set(record['train_people'])
            assert not (fit_people | stop_people) & (set(record['validation_people']) | test_people)
            assert fit_people == {patient_group_id(s) for s in record['fit_train_subjects']}
            assert stop_people == {patient_group_id(s) for s in record['stop_subjects']}
            assert record['early_stopping_applicable'] == epoch_model
            assert bool(stop_people) == epoch_model
        assert (folder/model_name).stat().st_size > 0
        assert (folder/'preprocessing.pkl').stat().st_size > 0
        for metric, value in scores(valid).items():
            assert np.isclose(record['metrics'][metric], value, atol=1e-10, rtol=1e-10), metric
        pair_path = folder/'pair_manifest.csv'
        if pair_path.is_file():
            pairs = pd.read_csv(pair_path)
            allowed = set(record['fit_train_subjects'] if early_mode else record['train_subjects'])
            assert set(pairs.ParentSubject1).issubset(allowed)
            assert set(pairs.ParentSubject2).issubset(allowed)
        if epoch_model:
            history = pd.read_csv(folder/'training_history.csv', float_precision='round_trip')
            stopping = json.loads((folder/'early_stopping.json').read_text())
            cap = manifest['mlp_max_epochs'] if manifest['model'] == 'MLP' else manifest['epochs']
            assert 1 <= len(history) <= cap and len(history) == stopping['epochs_run']
            assert np.isfinite(history.loss).all() and np.isfinite(history.validation_loss).all()
            best = int(np.argmin(history.validation_loss.to_numpy()))+1
            assert best == stopping['best_epoch'] == record['runtime']['best_epoch']
            assert stopping['best_validation_loss'] == float(history.validation_loss.iloc[best-1])
            assert stopping['best_checkpoint_restored']
            assert not stopping['oof_used_for_epoch_selection'] and not stopping['test_used_for_epoch_selection']
            assert stopping['stopped_early'] == (len(history) < cap)
            # Independently replay the patience rule and prove training ended
            # at its first stop signal (or the configured maximum).
            reference, stale, first_stop = np.inf, 0, None
            for epoch, loss in enumerate(history.validation_loss, 1):
                if loss < reference-manifest['min_delta']:
                    reference, stale = loss, 0
                else:
                    stale += 1
                if stale >= manifest['patience']:
                    first_stop = epoch
                    break
            assert first_stop == len(history) if first_stop is not None else len(history) == cap
            checkpoint_name = 'best_checkpoint.pkl' if manifest['model'] == 'MLP' else 'best_checkpoint.pt'
            assert (folder/checkpoint_name).stat().st_size > 0
            if manifest['model'] != 'MLP':
                def storages(path):
                    with zipfile.ZipFile(path) as archive:
                        return {name.split('/', 1)[1]: archive.read(name) for name in archive.namelist()
                                if '/data/' in name}
                best_weights, saved_weights = storages(folder/checkpoint_name), storages(folder/model_name)
                assert best_weights and best_weights == saved_weights, 'Saved OOF model differs from best tensor weights'
        elif manifest['model'] not in ('SVM', 'MLP', 'PLSDA'):
            history = pd.read_csv(folder/'training_history.csv')
            assert len(history) == manifest['epochs'] and np.isfinite(history.loss).all()
    assert (output/model_name).stat().st_size > 0
    if threshold_mode:
        assert (output/'threshold_selection.json').stat().st_mtime_ns <= (output/model_name).stat().st_mtime_ns, 'Threshold must be frozen before final model save'
    assert (output/'preprocessing.pkl').stat().st_size > 0
    if epoch_model:
        history = pd.read_csv(output/'training_history.csv')
        chosen_epochs = int(np.ceil(np.median([r['runtime']['best_epoch'] for r in manifest['fold_records']])))
        policy = json.loads((output/'final_epoch_selection.json').read_text())
        assert chosen_epochs == manifest['final_epochs'] == stored['final_epochs'] == policy['final_epochs']
        assert len(history) == chosen_epochs and np.isfinite(history.loss).all()
        assert not policy['test_used_for_selection'] and not manifest['outer_oof_used_for_epoch_selection']
        if manifest['model'] != 'MLP':
            assert manifest['runtime']['device'] == 'cuda'
    elif manifest['model'] not in ('SVM', 'MLP', 'PLSDA'):
        history = pd.read_csv(output/'training_history.csv')
        assert len(history) == manifest['epochs'] and np.isfinite(history.loss).all()
        assert manifest['runtime']['device'] == 'cuda', 'Expected the available CUDA GPU for full neural training'
    with np.load(output/'oof_predictions.npz') as saved:
        assert np.array_equal(saved['fold'], oof.Fold)
        assert np.array_equal(saved['y_true'], oof.BinaryClass)
        assert np.array_equal(saved['y_pred'], oof.Prediction)
        assert np.allclose(saved['y_prob'], oof.Probability, atol=1e-7)
    report = {'verified': True, 'folds': 10, 'oof_rows': len(oof), 'test_rows': len(predictions),
        'person_disjoint_folds': True, 'fold_local_synthetic_parents': True, 'metrics_recomputed': True,
        'threshold_selection_independently_verified': threshold_mode,
        'inner_stop_outer_oof_disjoint_verified': early_mode,
        'early_stopping_and_best_epochs_verified': epoch_model}
    (output/'audit_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    summary_path = args.output/'run_summary.json'
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        reports = []
        for job in summary['jobs']:
            assert job['status'] == 'completed', job
            reports.append(audit_job(job['output']))
            stored = json.loads((Path(job['output'])/'metrics.json').read_text())
            for key, value in scores(pd.read_csv(Path(job['output'])/'test_predictions.csv')).items():
                assert np.isclose(job['metrics'][key], value, atol=1e-10, rtol=1e-10)
            for key, value in stored['cv_oof'].items():
                if isinstance(value, (int, float)):
                    assert np.isclose(job['oof_metrics'][key], value, atol=1e-10, rtol=1e-10)
        assert len(reports) == summary['expected_training_jobs']
        report = {'verified_jobs': len(reports), 'errors': []}
        (args.output/'audit_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    else:
        report = audit_job(args.output)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
