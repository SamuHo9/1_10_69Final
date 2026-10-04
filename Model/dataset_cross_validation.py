"""Shared 10-fold person-level CV with fold-local preparation and genuine OOF."""
from copy import copy
from pathlib import Path
import argparse
import json
import pickle
import random
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import train_dataset_model as training
from threshold_selection import select_accuracy_threshold, plot_threshold_sweep

sys.path.insert(0, str(training.ROOT / 'Data_Preparation'))
from data_prep_common import (COEF_COLUMNS, augment_balanced_jitter,
    augment_plsda_same_class, fit_plsda, patient_group_id, pls_scores)
from prepare_augment_plsda_balanced import augment as augment_legacy_balanced

FOLDS = 10
PROTOCOL = 'person_10fold_oof_v1'
THRESHOLD_PROTOCOL = 'person_10fold_oof_accuracy_threshold_v1'


def person_splits(frame, folds=FOLDS, seed=42):
    if folds != FOLDS:
        raise ValueError('This protocol requires exactly 10 folds')
    groups = np.array([patient_group_id(s) for s in frame.Subject])
    unique, first = np.unique(groups, return_index=True)
    labels = frame.BinaryClass.to_numpy(dtype=int)
    group_y = labels[first]
    if any(len(np.unique(labels[groups == g])) != 1 for g in unique):
        raise ValueError('A person has conflicting labels')
    if np.bincount(group_y, minlength=2).min() < FOLDS:
        raise ValueError('Exactly 10-fold CV requires at least 10 original people per class; folds will not be reduced')
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    result = []
    for tr, va in splitter.split(unique, group_y):
        result.append((np.flatnonzero(np.isin(groups, unique[tr])),
                       np.flatnonzero(np.isin(groups, unique[va]))))
    return groups, result


def original_inputs(args):
    requested_train, requested_test, _ = training.load_data(args.train_csv, args.test_csv, args.kind)
    paths = {'requested_train': args.train_csv, 'requested_test': args.test_csv}
    manifest_path = args.train_csv.parent / 'data_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
    if args.kind == 'latent':
        # Supervised latent CSVs fitted on all Train cannot be reused in outer CV.
        folder = args.train_csv.resolve().parents[2] / 'coef_raw' / args.side
        train_path, test_path = folder / 'train_prepared.csv', folder / 'test_prepared.csv'
        original, tested, columns = training.load_data(train_path, test_path, 'coef')
        for supplied, raw in ((requested_train, original), (requested_test, tested)):
            if list(supplied.Subject) != list(raw.Subject) or not np.array_equal(supplied.BinaryClass, raw.BinaryClass):
                raise ValueError('Latent input people/labels differ from the original coefficient source')
        paths.update(original_train=train_path, original_test=test_path)
    else:
        original = requested_train[requested_train.DataType == 'Original'].copy() if 'DataType' in requested_train else requested_train.copy()
        tested = requested_test.copy()
        columns = [c for c in original if c.startswith('Coef_')] if args.kind == 'coef' else [f'{a}_{p}' for p in range(1002) for a in ('x', 'y', 'z')]
    original = original.reset_index(drop=True)
    if not len(original) or not original.Subject.is_unique:
        raise ValueError('Original training rows are missing or repeated')
    known = {'coef_raw', 'coef_raw_balanced_jitter', 'coef_plsda', 'augment_plsda_balanced',
             'plsda_latent_features', 'pointnet_raw', 'pointnet_raw_balanced_jitter', 'pointnet_plsda', 'plsda_direct'}
    if args.dataset not in known:
        raise ValueError('Unknown dataset preparation protocol')
    if args.dataset.endswith('balanced_jitter') or args.dataset in ('coef_plsda', 'pointnet_plsda', 'augment_plsda_balanced'):
        if not manifest:
            raise ValueError('Augmented datasets require data_manifest.json to rebuild preparation inside each fold')
        if manifest.get('info', {}).get('augmentation_size') == 'maximum':
            raise ValueError('This runner targets the current moderate datasets; maximum-pair generation needs an explicit separate experiment')
        paths['preparation_manifest'] = manifest_path
    return original, tested, columns, manifest, paths


def prepare_training(frame, dataset, kind, manifest, seed):
    """Generate children only from the currently supplied training people."""
    info = manifest.get('info', {})
    children = info.get('children_per_pair', 8)
    size = info.get('augmentation_size', 'balance_only')
    pairs = pd.DataFrame()
    if dataset == 'augment_plsda_balanced':
        prepared, _, pairs, _, result = augment_legacy_balanced(frame, children,
            info.get('pls_components_requested', 10), seed)
    elif dataset.endswith('balanced_jitter'):
        prepared, _, pairs, result = augment_balanced_jitter(frame, kind, seed,
            info.get('noise_scale', 0.02), children, dataset, augmentation_size=size)
    elif dataset in ('coef_plsda', 'pointnet_plsda'):
        prepared, _, pairs, _, result = augment_plsda_same_class(frame, kind, seed,
            info.get('pls_components_requested', 8), children, dataset,
            info.get('normalize_xyz_after_reconstruction', False), augmentation_size=size)
    else:
        if 'DataType' in frame and not frame.DataType.eq('Original').all():
            raise ValueError('Pre-augmented training rows must not enter fold preparation')
        prepared, result = frame.copy(), {'augmentation_kind': 'none', 'synthetic_train_n': 0}
    allowed = set(frame.Subject)
    for name in ('ParentSubject1', 'ParentSubject2'):
        if name in pairs and not set(pairs[name]).issubset(allowed):
            raise ValueError('A synthetic parent is outside the current training fold')
    return prepared, pairs, result


def transformed(train, evaluate, columns, args):
    if args.kind == 'latent':
        # Rebuild the same one-hot PLS latent method from original coefficients.
        bundle = fit_plsda(train[columns].to_numpy(), train.BinaryClass.to_numpy(), args.pls_components)
        x = pls_scores(bundle, train[columns].to_numpy())
        xt = pls_scores(bundle, evaluate[columns].to_numpy())
        scaler = StandardScaler().fit(x)
        return scaler.transform(x).astype('float32'), scaler.transform(xt).astype('float32'), {
            'kind': 'latent_rebuilt', 'columns': columns, 'latent_pls': bundle, 'latent_scaler': scaler}
    if args.model == 'PLSDA':
        bundle = fit_plsda(train[columns].to_numpy(), train.BinaryClass.to_numpy(), args.pls_components)
        return train[columns].to_numpy(), evaluate[columns].to_numpy(), bundle
    return training.transform_features(train, evaluate, columns, args.kind, args.pls_components)


def fit_predict(args, x, y, xt, preprocessing):
    np.random.seed(args.seed)
    random.seed(args.seed)
    history, runtime = [], {}
    if args.model == 'PLSDA':
        raw = preprocessing['pls'].predict(preprocessing['scaler'].transform(xt))
        raw -= raw.max(axis=1, keepdims=True)
        exponential = np.exp(raw)
        probability = exponential[:, 1] / exponential.sum(axis=1)
        prediction = (probability >= 0.5).astype(int)
        with (args.output_dir / 'direct_plsda_model.pkl').open('wb') as stream:
            pickle.dump(preprocessing, stream)
    elif args.model in ('SVM', 'MLP'):
        estimator = (SVC(C=1.0, kernel='rbf', probability=True, random_state=args.seed) if args.model == 'SVM' else
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, alpha=0.001,
                solver='adam', random_state=args.seed, early_stopping=False, verbose=True))
        estimator.fit(x, y)
        probability, prediction = estimator.predict_proba(xt)[:, 1], estimator.predict(xt)
        history = [{'epoch': i+1, 'loss': float(loss)} for i, loss in enumerate(getattr(estimator, 'loss_curve_', []))]
        with (args.output_dir / 'model.pkl').open('wb') as stream:
            pickle.dump(estimator, stream)
    else:
        probability, history, runtime = training.train_torch(args, x, y, xt)
        prediction = (probability >= 0.5).astype(int)
    if not np.isfinite(probability).all() or not ((0 <= probability) & (probability <= 1)).all():
        raise ValueError('Invalid predicted probabilities')
    with (args.output_dir / 'preprocessing.pkl').open('wb') as stream:
        pickle.dump(preprocessing, stream)
    pd.DataFrame(history, columns=['epoch', 'loss', 'seconds']).to_csv(args.output_dir / 'training_history.csv', index=False)
    return probability, prediction, history, runtime


def run(args):
    if getattr(args, 'early_stopping', False):
        from early_stopping_training import run as run_early_stopping
        return run_early_stopping(args)
    tune_threshold = getattr(args, 'tune_threshold', False)
    if args.folds != FOLDS:
        raise ValueError('Exactly 10 folds are required')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Choose a new or empty output directory; previous results will be preserved')
    if (args.kind == 'xyz') != (args.model == 'PointNet'):
        raise ValueError('PointNet requires XYZ; tabular models require coefficient or latent datasets')
    original, tested, columns, source_manifest, paths = original_inputs(args)
    hashes = {name: training.sha(path) for name, path in paths.items()}
    groups, splits = person_splits(original, args.folds, args.seed)
    test_groups = {patient_group_id(s) for s in tested.Subject}
    if test_groups & set(groups):
        raise ValueError('Train/Test person overlap')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probability = np.full(len(original), np.nan)
    prediction = np.full(len(original), -1, dtype=int)
    fold_id = np.zeros(len(original), dtype=int)
    records = []
    start = time.monotonic()
    print(f'10-fold CV: dataset={args.dataset} side={args.side} model={args.model}; original_train={len(original)} test={len(tested)}', flush=True)
    for fold, (tr, va) in enumerate(splits, 1):
        local = copy(args)
        local.output_dir = args.output_dir / f'fold_{fold:02d}'
        local.output_dir.mkdir()
        # Identical fold assignments and augmentation seed across comparable models.
        train, pairs, info = prepare_training(original.iloc[tr].reset_index(drop=True), args.dataset,
            'coef' if args.kind == 'latent' else args.kind, source_manifest, args.seed + fold)
        valid = original.iloc[va].reset_index(drop=True)
        x, xv, preprocessing = transformed(train, valid, columns, args)
        prob, pred, _, _ = fit_predict(local, x, train.BinaryClass.to_numpy(dtype=int), xv, preprocessing)
        probability[va], prediction[va], fold_id[va] = prob, pred, fold
        score = training.metrics(valid.BinaryClass.to_numpy(), prob, pred)
        record = {'fold': fold, 'train_subjects': list(original.iloc[tr].Subject),
            'validation_subjects': list(valid.Subject), 'train_people': list(np.unique(groups[tr])),
            'validation_people': list(np.unique(groups[va])), 'training_rows_after_augmentation': len(train),
            'original_training_rows': len(tr), 'validation_rows': len(va),
            'augmentation': info, 'metrics': score}
        training.write_json(local.output_dir / 'fold_manifest.json', record)
        if len(pairs):
            pairs.to_csv(local.output_dir / 'pair_manifest.csv', index=False)
        training.write_json(local.output_dir / 'metrics.json', score)
        records.append(record)
        print(f'fold={fold}/10 OOF_accuracy={score["accuracy"]:.6f} OOF_AUC={score["roc_auc"]:.6f}', flush=True)
    if np.any(fold_id == 0) or not np.isfinite(probability).all():
        raise ValueError('Incomplete OOF coverage')
    oof = pd.DataFrame({'Subject': original.Subject, 'PatientID': groups, 'Fold': fold_id,
        'BinaryClass': original.BinaryClass, 'Probability': probability, 'Prediction': prediction})
    oof.to_csv(args.output_dir / 'oof_predictions.csv', index=False)
    np.savez(args.output_dir / 'oof_predictions.npz', y_true=oof.BinaryClass.to_numpy(),
        y_pred=prediction, y_prob=probability, fold=fold_id, subject=original.Subject.to_numpy(dtype=str))
    oof_metrics = training.metrics(original.BinaryClass.to_numpy(), probability, prediction)
    training.write_json(args.output_dir / 'oof_metrics.json', oof_metrics)
    threshold, selection_report, selection_metrics = None, None, None
    if tune_threshold:
        # Freeze before fitting/predicting the final Test model. Test labels/scores
        # are never supplied to the selector. Keep genuine default-rule CV separate.
        threshold, sweep, selection_report = select_accuracy_threshold(original.BinaryClass.to_numpy(), probability)
        sweep.to_csv(args.output_dir / 'threshold_sweep.csv', index=False, float_format='%.17g')
        selection_report['oof_predictions_sha256'] = training.sha(args.output_dir / 'oof_predictions.csv')
        training.write_json(args.output_dir / 'threshold_selection.json', selection_report)
        selection_metrics = training.metrics(original.BinaryClass.to_numpy(), probability, (probability >= threshold).astype(int))
        training.write_json(args.output_dir / 'oof_threshold_selection_metrics.json', selection_metrics)
        plot_threshold_sweep(args.output_dir, sweep, threshold)
        print(f'Frozen Train OOF threshold={threshold:.17g}; objective=accuracy; OOF selection accuracy={selection_metrics["accuracy"]:.6f}', flush=True)
    pd.DataFrame([{'fold': row['fold'], **{k: row['metrics'][k] for k in
        ('accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1_macro', 'roc_auc')}} for row in records]).to_csv(args.output_dir / 'fold_metrics.csv', index=False)
    # Refit independently on all original Train; outer OOF/Test do not select settings.
    train, pairs, info = prepare_training(original, args.dataset,
        'coef' if args.kind == 'latent' else args.kind, source_manifest, args.seed)
    x, xt, preprocessing = transformed(train, tested, columns, args)
    prob, pred, history, runtime = fit_predict(args, x, train.BinaryClass.to_numpy(dtype=int), xt, preprocessing)
    default_pred = pred.copy()
    default_test_metrics = training.metrics(tested.BinaryClass.to_numpy(), prob, default_pred)
    if tune_threshold:
        pred = (np.asarray(prob, dtype=float) >= threshold).astype(int)
    test_metrics = training.metrics(tested.BinaryClass.to_numpy(), prob, pred)
    result = {'cv_oof': oof_metrics, 'test': test_metrics, 'folds': FOLDS,
        'original_train_rows': len(original), 'train_rows': len(train), 'test_rows': len(tested),
        'elapsed_seconds': time.monotonic()-start}
    if tune_threshold:
        result.update(test_default=default_test_metrics, decision_threshold=threshold,
            oof_threshold_selection=selection_metrics,
            test_accuracy_change=test_metrics['accuracy']-default_test_metrics['accuracy'])
    training.write_json(args.output_dir / 'metrics.json', result)
    test_frame = pd.DataFrame({'Subject': tested.Subject, 'BinaryClass': tested.BinaryClass,
        'Probability': prob, 'Prediction': pred})
    if tune_threshold:
        test_frame['PredictionDefault'] = default_pred
        test_frame['DecisionThreshold'] = threshold
    test_frame.to_csv(args.output_dir / 'test_predictions.csv', index=False, float_format='%.17g')
    np.savez(args.output_dir / 'test_predictions.npz', y_test=tested.BinaryClass.to_numpy(), y_pred=pred, y_prob=prob)
    if len(pairs):
        pairs.to_csv(args.output_dir / 'final_pair_manifest.csv', index=False)
    manifest = {'protocol': THRESHOLD_PROTOCOL if tune_threshold else PROTOCOL, 'dataset': args.dataset, 'kind': args.kind, 'side': args.side,
        'model': args.model, 'folds': FOLDS, 'seed': args.seed, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'pls_components': args.pls_components,
        'cross_validation_performed': True, 'test_used_for_selection': False,
        'outer_oof_used_for_selection': tune_threshold, 'fold_preparation_fit_scope': 'current_training_people_only',
        'threshold_optimization_performed': tune_threshold,
        'prediction_rule': 'probability >= OOF-selected threshold' if tune_threshold else 'estimator.predict' if args.model in ('SVM', 'MLP') else 'probability >= 0.5',
        'decision_threshold': threshold if tune_threshold else None if args.model in ('SVM', 'MLP') else 0.5,
        'threshold_selection': selection_report,
        'default_prediction_rule': 'estimator.predict' if args.model in ('SVM', 'MLP') else 'probability >= 0.5',
        'oof_metric_scope': 'default-rule genuine OOF; tuned OOF is reported separately as threshold-selection score',
        'oof_scope': 'original_training_rows_only_once_each', 'final_test_model': 'refit_on_full_train',
        'inputs': {k: str(Path(p).resolve()) for k, p in paths.items()}, 'input_sha256': hashes,
        'final_augmentation': info, 'runtime': runtime,
        'known_input_limitations': 'Existing hemisphere labels and source ICP reference including Test remain unchanged.',
        'fold_records': records}
    training.write_json(args.output_dir / 'run_manifest.json', manifest)
    training.plots(args.output_dir, test_metrics, tested.BinaryClass.to_numpy(), prob, history)
    if tune_threshold:
        default_plots = args.output_dir / 'default_test_plots'
        default_plots.mkdir()
        training.plots(default_plots, default_test_metrics, tested.BinaryClass.to_numpy(), prob, [])
    oof_plots = args.output_dir / 'oof_plots'
    oof_plots.mkdir()
    training.plots(oof_plots, oof_metrics, original.BinaryClass.to_numpy(), probability, [])
    if hashes != {name: training.sha(path) for name, path in paths.items()}:
        raise ValueError('Input files changed during training')
    training.write_json(args.output_dir / 'validation_report.json', {'input_hashes_unchanged': True,
        'folds': FOLDS, 'oof_rows': len(oof), 'oof_complete_once_per_original': True,
        'prediction_rows': len(tested), 'train_validation_test_people_disjoint': True,
        'synthetic_parents_fold_local': True})
    print(json.dumps(result), flush=True)
    return result


def legacy_entrypoint(side, model):
    """Shared CLI used by all original All_Augment_tain entry points."""
    dataset = 'pointnet_plsda' if model == 'PointNet' else 'augment_plsda_balanced'
    kind = 'xyz' if model == 'PointNet' else 'coef'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=dataset)
    parser.add_argument('--dataset-root', type=Path, default=training.ROOT / 'Output_Dataset')
    parser.add_argument('--kind', choices=['coef', 'latent', 'xyz'], default=kind)
    parser.add_argument('--train-csv', type=Path)
    parser.add_argument('--test-csv', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--folds', type=int, choices=[FOLDS], default=FOLDS)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--pls-components', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--tune-threshold', action='store_true', help='Select accuracy threshold on Train OOF; freeze before final Test')
    parser.add_argument('--early-stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min-delta', type=float, default=1e-4)
    parser.add_argument('--stop-fraction', type=float, default=0.2)
    parser.add_argument('--mlp-max-epochs', type=int, default=None, help='Defaults to the same maximum as --epochs')
    args = parser.parse_args()
    args.side, args.model = side, model
    folder = args.dataset_root / args.dataset / side
    args.train_csv = args.train_csv or folder / ('train_plsda_latent_features.csv' if args.kind == 'latent' else 'train_prepared.csv')
    args.test_csv = args.test_csv or folder / ('test_plsda_latent_features.csv' if args.kind == 'latent' else 'test_prepared.csv')
    args.output_dir = args.output_dir or training.ROOT / 'Model_Results/dataset_runs_20261003_10fold' / args.dataset / side / model
    if min(args.epochs, args.pls_components, args.batch_size, args.threads) < 1:
        parser.error('Training settings must be positive')
    if args.patience < 1 or args.mlp_max_epochs < 1 or args.min_delta < 0 or not 0 < args.stop_fraction < 1:
        parser.error('Invalid early stopping settings')
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    # A standalone script also retains its own complete stdout/stderr log.
    log_path = args.output_dir.parent / (args.output_dir.name + '.log')
    if log_path.exists():
        raise FileExistsError('Choose a new output directory/log path')
    class Tee:
        def __init__(self, terminal, log):
            self.terminal, self.log = terminal, log
        def write(self, value):
            self.terminal.write(value); self.log.write(value); self.log.flush()
        def flush(self):
            self.terminal.flush(); self.log.flush()
    with log_path.open('w', encoding='utf-8') as stream:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, stream), Tee(stderr, stream)
        try:
            run(args)
        except Exception:
            import traceback
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr
