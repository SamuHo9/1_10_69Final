"""Early stopping on inner people; outer OOF is never used for epoch selection."""
from copy import copy
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier

import train_dataset_model as training

PROTOCOL = 'person_10fold_inner_early_stop_oof_accuracy_threshold_v1'
EPOCH_MODELS = {'MLP', *training.ARCHITECTURES}


class StopTracker:
    """Save the absolute best checkpoint; patience uses min_delta separately."""
    def __init__(self, patience=10, min_delta=1e-4):
        if patience < 1 or min_delta < 0:
            raise ValueError('Patience must be positive and min_delta nonnegative')
        self.patience, self.min_delta = patience, min_delta
        self.best_loss, self.reference_loss = np.inf, np.inf
        self.best_epoch, self.stale = 0, 0

    def update(self, epoch, loss):
        if not np.isfinite(loss):
            raise ValueError('Non-finite validation loss')
        save = loss < self.best_loss
        if save:
            self.best_loss, self.best_epoch = float(loss), int(epoch)
        if loss < self.reference_loss-self.min_delta:
            self.reference_loss, self.stale = float(loss), 0
        else:
            self.stale += 1
        return save, self.stale >= self.patience


def inner_person_split(frame, fraction=0.2, seed=42):
    from data_prep_common import patient_group_id
    groups = np.array([patient_group_id(s) for s in frame.Subject])
    unique, first = np.unique(groups, return_index=True)
    labels = frame.BinaryClass.to_numpy(dtype=int)
    if not 0 < fraction < 1 or any(len(np.unique(labels[groups == g])) != 1 for g in unique):
        raise ValueError('Invalid inner split fraction or conflicting person labels')
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    fit, stop = next(splitter.split(unique, labels[first]))
    return (np.flatnonzero(np.isin(groups, unique[fit])),
            np.flatnonzero(np.isin(groups, unique[stop])))


def train_mlp(args, x, y, xt, xs=None, ys=None):
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.001, solver='adam',
        random_state=np.random.RandomState(args.seed), max_iter=1, early_stopping=False)
    tracker = StopTracker(args.patience, args.min_delta) if xs is not None else None
    cap = args.mlp_max_epochs if tracker else args.epochs
    history = []
    for epoch in range(1, cap+1):
        started = time.monotonic()
        model.partial_fit(x, y, classes=np.array([0, 1]))
        val = float(log_loss(ys, model.predict_proba(xs), labels=[0, 1])) if tracker else None
        save, stop = tracker.update(epoch, val) if tracker else (False, False)
        if save:
            with (args.output_dir/'best_checkpoint.pkl').open('wb') as stream:
                pickle.dump(model, stream)
        history.append({'epoch': epoch, 'loss': float(model.loss_), 'validation_loss': val,
                        'is_best_checkpoint': save, 'seconds': time.monotonic()-started})
        print(f'epoch={epoch}/{cap} loss={model.loss_:.6f} validation_loss={val} best={save}', flush=True)
        if stop:
            break
    if tracker:
        with (args.output_dir/'best_checkpoint.pkl').open('rb') as stream:
            model = pickle.load(stream)
    with (args.output_dir/'model.pkl').open('wb') as stream:
        pickle.dump(model, stream)
    probability, prediction = model.predict_proba(xt)[:, 1], model.predict(xt)
    runtime = {'backend': 'sklearn_partial_fit', 'early_stopping': bool(tracker),
        'epochs_run': len(history), 'best_epoch': tracker.best_epoch if tracker else len(history),
        'best_validation_loss': tracker.best_loss if tracker else None,
        'stopped_early': bool(tracker and len(history) < cap), 'max_epochs': cap,
        'best_checkpoint_restored': bool(tracker)}
    return probability, prediction, history, runtime


def train_neural(args, x, y, xt, xs, ys):
    import torch
    import torch.nn as nn
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cls, source = training.model_class(args.model, args.side)
    model = (cls(seq_length=x.shape[1]) if args.model == 'ResNetAE' else cls()).to(device)
    pointnet = args.model == 'PointNet'
    def tensor_x(array):
        return torch.tensor(array if pointnet else array[:, None, :], device=device)
    def tensor_y(array):
        result = torch.tensor(array, dtype=torch.long if pointnet else torch.float32, device=device)
        return result if pointnet else result[:, None]
    tx, ty, sx, sy, ex = tensor_x(x), tensor_y(y), tensor_x(xs), tensor_y(ys), tensor_x(xt)
    counts = np.bincount(y, minlength=2)
    if pointnet:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(len(y)/(2*counts), dtype=torch.float32, device=device))
    elif args.model == 'SqueezeNet':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([counts[0]/counts[1]], dtype=torch.float32, device=device))
    else:
        criterion = nn.BCELoss()
    def loss_for(output, bx, by):
        if args.model == 'ResNetAE':
            prediction, reconstruction = output
            return criterion(prediction, by)+0.5*nn.functional.mse_loss(reconstruction, bx)
        return criterion(output, by)
    optimizer_cls = torch.optim.AdamW if args.model in ('PointNet', 'SqueezeNet') else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=0.001,
        **({'weight_decay': 1e-3} if args.model in ('PointNet', 'SqueezeNet') else {}))
    tracker, rng, history = StopTracker(args.patience, args.min_delta), np.random.default_rng(args.seed), []
    # Weighted CrossEntropyLoss is aggregated by summed target weights, rather
    # than batch sizes, so stop loss is independent of validation batch boundaries.
    def stop_loss():
        total, denominator = 0.0, 0.0
        with torch.no_grad():
            for start in range(0, len(ys), args.batch_size):
                bx, by = sx[start:start+args.batch_size], sy[start:start+args.batch_size]
                value = loss_for(model(bx), bx, by)
                weight = float(criterion.weight[by].sum()) if pointnet else len(by)
                total += float(value)*weight
                denominator += weight
        return total/denominator
    print(f'Device={device}; inner stopping people={len(ys)}; max_epochs={args.epochs}; patience={args.patience}', flush=True)
    for epoch in range(1, args.epochs+1):
        started, total_loss = time.monotonic(), 0.0
        model.train()
        for indices in training.safe_batches(len(y), args.batch_size, rng):
            bx, by = tx[indices], ty[indices]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model(bx), bx, by)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())*len(indices)
        model.eval()
        val = stop_loss()
        save, stop = tracker.update(epoch, val)
        if save:
            torch.save({'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'architecture': cls.__name__, 'source': str(source), 'source_sha256': training.sha(source),
                'best_epoch': epoch, 'validation_loss': val, 'seed': args.seed,
                'input_features': x.shape[1:] if pointnet else x.shape[1]}, args.output_dir/'best_checkpoint.pt')
        history.append({'epoch': epoch, 'loss': total_loss/len(y), 'validation_loss': val,
                        'is_best_checkpoint': save, 'seconds': time.monotonic()-started})
        print(f'epoch={epoch}/{args.epochs} loss={history[-1]["loss"]:.6f} validation_loss={val:.6f} best={save}', flush=True)
        if stop:
            break
    checkpoint = torch.load(args.output_dir/'best_checkpoint.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(xt), args.batch_size):
            output = model(ex[start:start+args.batch_size])
            if args.model == 'ResNetAE':
                output = output[0]
            probability = (torch.softmax(output, dim=1)[:, 1] if pointnet else
                torch.sigmoid(output).flatten() if args.model == 'SqueezeNet' else output.flatten())
            probabilities.extend(probability.cpu().numpy())
    torch.save(checkpoint, args.output_dir/'model.pt')
    return np.asarray(probabilities), (np.asarray(probabilities) >= 0.5).astype(int), history, {
        'device': str(device), 'torch_version': torch.__version__, 'early_stopping': True,
        'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        'architecture_source': str(source), 'architecture_sha256': training.sha(source),
        'epochs_run': len(history), 'best_epoch': tracker.best_epoch, 'best_validation_loss': tracker.best_loss,
        'stopped_early': len(history) < args.epochs, 'max_epochs': args.epochs, 'best_checkpoint_restored': True}


def fit_epoch_model(args, x, y, xt, preprocessing, xs=None, ys=None):
    if args.model == 'MLP':
        prob, pred, history, runtime = train_mlp(args, x, y, xt, xs, ys)
    elif xs is not None:
        prob, pred, history, runtime = train_neural(args, x, y, xt, xs, ys)
    else:
        prob, history, runtime = training.train_torch(args, x, y, xt)
        pred = (prob >= 0.5).astype(int)
        runtime.update(early_stopping=False, epochs_run=args.epochs, max_epochs=args.epochs,
                       epoch_policy='ceil_median_inner_best_epochs')
    if not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
        raise ValueError('Invalid probabilities')
    pd.DataFrame(history).to_csv(args.output_dir/'training_history.csv', index=False, float_format='%.17g')
    with (args.output_dir/'preprocessing.pkl').open('wb') as stream:
        pickle.dump(preprocessing, stream)
    if xs is not None:
        training.write_json(args.output_dir/'early_stopping.json', {
            **runtime, 'patience': args.patience, 'min_delta': args.min_delta,
            'monitor': 'inner_validation_loss', 'oof_used_for_epoch_selection': False,
            'test_used_for_epoch_selection': False})
    return prob, pred, history, runtime


def run(args):
    import dataset_cross_validation as cv
    from threshold_selection import select_accuracy_threshold, plot_threshold_sweep
    from data_prep_common import patient_group_id
    if args.folds != 10 or (args.kind == 'xyz') != (args.model == 'PointNet'):
        raise ValueError('Require 10 folds and valid model/data kind')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    original, tested, columns, source_manifest, paths = cv.original_inputs(args)
    hashes = {k: training.sha(p) for k, p in paths.items()}
    groups, splits = cv.person_splits(original, args.folds, args.seed)
    if set(groups) & {patient_group_id(s) for s in tested.Subject}:
        raise ValueError('Train/Test overlap')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probability, prediction, fold_id = np.full(len(original), np.nan), np.full(len(original), -1, dtype=int), np.zeros(len(original), dtype=int)
    records, started = [], time.monotonic()
    uses_epochs = args.model in EPOCH_MODELS
    print(f'Inner early stopping: {args.dataset}/{args.side}/{args.model}; original Train={len(original)} Test={len(tested)}', flush=True)
    for fold, (tr, va) in enumerate(splits, 1):
        local = copy(args)
        local.output_dir = args.output_dir/f'fold_{fold:02d}'
        local.output_dir.mkdir()
        outer_train = original.iloc[tr].reset_index(drop=True)
        valid = original.iloc[va].reset_index(drop=True)
        if uses_epochs:
            fi, si = inner_person_split(outer_train, args.stop_fraction, args.seed+fold)
            fit, stop_frame = outer_train.iloc[fi].reset_index(drop=True), outer_train.iloc[si].reset_index(drop=True)
        else:
            fit, stop_frame = outer_train, outer_train.iloc[:0].copy()
        train, pairs, info = cv.prepare_training(fit, args.dataset, 'coef' if args.kind == 'latent' else args.kind, source_manifest, args.seed+fold)
        evaluate = pd.concat([stop_frame, valid], ignore_index=True)
        x, xe, preprocessing = cv.transformed(train, evaluate, columns, args)
        if uses_epochs:
            xs, xv = xe[:len(stop_frame)], xe[len(stop_frame):]
            prob, pred, _, runtime = fit_epoch_model(local, x, train.BinaryClass.to_numpy(dtype=int), xv,
                preprocessing, xs, stop_frame.BinaryClass.to_numpy(dtype=int))
        else:
            prob, pred, _, runtime = cv.fit_predict(local, x, train.BinaryClass.to_numpy(dtype=int), xe, preprocessing)
        probability[va], prediction[va], fold_id[va] = prob, pred, fold
        score = training.metrics(valid.BinaryClass.to_numpy(), prob, pred)
        record = {'fold': fold, 'train_subjects': list(outer_train.Subject), 'validation_subjects': list(valid.Subject),
            'train_people': list(np.unique(groups[tr])), 'validation_people': list(np.unique(groups[va])),
            'fit_train_subjects': list(fit.Subject), 'stop_subjects': list(stop_frame.Subject),
            'fit_train_people': sorted({patient_group_id(s) for s in fit.Subject}),
            'stop_people': sorted({patient_group_id(s) for s in stop_frame.Subject}),
            'original_training_rows': len(fit), 'outer_training_rows': len(outer_train),
            'stop_rows': len(stop_frame), 'validation_rows': len(valid),
            'training_rows_after_augmentation': len(train), 'augmentation': info,
            'early_stopping_applicable': uses_epochs, 'runtime': runtime, 'metrics': score}
        training.write_json(local.output_dir/'fold_manifest.json', record)
        training.write_json(local.output_dir/'metrics.json', score)
        if len(pairs):
            pairs.to_csv(local.output_dir/'pair_manifest.csv', index=False)
        records.append(record)
        print(f'fold={fold}/10 OOF_accuracy={score["accuracy"]:.6f} best_epoch={runtime.get("best_epoch")}', flush=True)
    if np.any(fold_id == 0) or not np.isfinite(probability).all():
        raise ValueError('Incomplete OOF')
    oof = pd.DataFrame({'Subject': original.Subject, 'PatientID': groups, 'Fold': fold_id,
        'BinaryClass': original.BinaryClass, 'Probability': probability, 'Prediction': prediction})
    oof.to_csv(args.output_dir/'oof_predictions.csv', index=False, float_format='%.17g')
    np.savez(args.output_dir/'oof_predictions.npz', y_true=oof.BinaryClass.to_numpy(), y_pred=prediction,
             y_prob=probability, fold=fold_id, subject=original.Subject.to_numpy(dtype=str))
    oof_metrics = training.metrics(original.BinaryClass.to_numpy(), probability, prediction)
    threshold, sweep, selection = select_accuracy_threshold(original.BinaryClass.to_numpy(), probability)
    sweep.to_csv(args.output_dir/'threshold_sweep.csv', index=False, float_format='%.17g')
    selection['oof_predictions_sha256'] = training.sha(args.output_dir/'oof_predictions.csv')
    selection['epoch_selection_source'] = 'inner_validation_stop_people_only'
    training.write_json(args.output_dir/'threshold_selection.json', selection)
    selected_oof = training.metrics(original.BinaryClass.to_numpy(), probability, (probability >= threshold).astype(int))
    training.write_json(args.output_dir/'oof_metrics.json', oof_metrics)
    training.write_json(args.output_dir/'oof_threshold_selection_metrics.json', selected_oof)
    plot_threshold_sweep(args.output_dir, sweep, threshold)
    final_epochs = int(np.ceil(np.median([r['runtime']['best_epoch'] for r in records]))) if uses_epochs else None
    pd.DataFrame([{'fold': r['fold'], 'best_epoch': r['runtime'].get('best_epoch'),
        'epochs_run': r['runtime'].get('epochs_run'), 'best_validation_loss': r['runtime'].get('best_validation_loss'),
        **{k: r['metrics'][k] for k in ('accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1_macro', 'roc_auc')}}
        for r in records]).to_csv(args.output_dir/'fold_metrics.csv', index=False)
    training.write_json(args.output_dir/'final_epoch_selection.json', {'applicable': uses_epochs,
        'policy': 'ceil_median_inner_best_epochs' if uses_epochs else 'not_applicable_solver_convergence',
        'best_epochs': [r['runtime']['best_epoch'] for r in records] if uses_epochs else [],
        'final_epochs': final_epochs, 'test_used_for_selection': False})
    print(f'Frozen OOF threshold={threshold:.17g}; final refit epochs={final_epochs}', flush=True)
    train, pairs, info = cv.prepare_training(original, args.dataset, 'coef' if args.kind == 'latent' else args.kind, source_manifest, args.seed)
    x, xt, preprocessing = cv.transformed(train, tested, columns, args)
    final = copy(args)
    if uses_epochs:
        final.epochs = final_epochs
        prob, default_pred, history, runtime = fit_epoch_model(final, x, train.BinaryClass.to_numpy(dtype=int), xt, preprocessing)
    else:
        prob, default_pred, history, runtime = cv.fit_predict(final, x, train.BinaryClass.to_numpy(dtype=int), xt, preprocessing)
    pred = (np.asarray(prob, dtype=float) >= threshold).astype(int)
    default_metrics, test_metrics = training.metrics(tested.BinaryClass.to_numpy(), prob, default_pred), training.metrics(tested.BinaryClass.to_numpy(), prob, pred)
    result = {'cv_oof': oof_metrics, 'test': test_metrics, 'test_default': default_metrics,
        'oof_threshold_selection': selected_oof, 'decision_threshold': threshold, 'folds': 10,
        'original_train_rows': len(original), 'train_rows': len(train), 'test_rows': len(tested),
        'final_epochs': final_epochs, 'elapsed_seconds': time.monotonic()-started,
        'test_accuracy_change': test_metrics['accuracy']-default_metrics['accuracy']}
    training.write_json(args.output_dir/'metrics.json', result)
    pd.DataFrame({'Subject': tested.Subject, 'BinaryClass': tested.BinaryClass, 'Probability': prob,
        'Prediction': pred, 'PredictionDefault': default_pred, 'DecisionThreshold': threshold}).to_csv(
        args.output_dir/'test_predictions.csv', index=False, float_format='%.17g')
    np.savez(args.output_dir/'test_predictions.npz', y_test=tested.BinaryClass.to_numpy(), y_pred=pred, y_prob=prob)
    if len(pairs):
        pairs.to_csv(args.output_dir/'final_pair_manifest.csv', index=False)
    manifest = {'protocol': PROTOCOL, 'dataset': args.dataset, 'kind': args.kind, 'side': args.side,
        'model': args.model, 'folds': 10, 'seed': args.seed, 'epochs': args.epochs,
        'mlp_max_epochs': args.mlp_max_epochs, 'batch_size': args.batch_size, 'pls_components': args.pls_components,
        'patience': args.patience, 'min_delta': args.min_delta, 'stop_fraction': args.stop_fraction,
        'early_stopping_applicable': uses_epochs, 'early_stopping_selection_scope': 'inner_stop_people_only',
        'outer_oof_used_for_epoch_selection': False, 'test_used_for_selection': False,
        'outer_oof_used_for_selection': True, 'cross_validation_performed': True,
        'threshold_optimization_performed': True, 'threshold_selection': selection,
        'prediction_rule': 'probability >= OOF-selected threshold', 'decision_threshold': threshold,
        'default_prediction_rule': 'estimator.predict' if args.model in ('SVM', 'MLP') else 'probability >= 0.5',
        'oof_metric_scope': 'default-rule outer OOF; tuned OOF is a threshold-selection score, not unbiased tuned CV',
        'fold_preparation_fit_scope': 'inner_fit_people_only_for_epoch_models',
        'final_test_model': 'refit_on_full_train', 'final_epochs': final_epochs,
        'final_augmentation': info, 'runtime': runtime, 'fold_records': records,
        'inputs': {k: str(Path(p).resolve()) for k, p in paths.items()}, 'input_sha256': hashes,
        'known_input_limitations': 'Existing hemisphere labels and source ICP reference including Test remain unchanged.'}
    training.write_json(args.output_dir/'run_manifest.json', manifest)
    training.plots(args.output_dir, test_metrics, tested.BinaryClass.to_numpy(), prob, history)
    for folder_name, score, labels, probabilities in [('default_test_plots', default_metrics, tested.BinaryClass.to_numpy(), prob),
        ('oof_plots', oof_metrics, original.BinaryClass.to_numpy(), probability)]:
        folder = args.output_dir/folder_name
        folder.mkdir()
        training.plots(folder, score, labels, probabilities, [])
    if hashes != {k: training.sha(p) for k, p in paths.items()}:
        raise ValueError('Inputs changed during training')
    training.write_json(args.output_dir/'validation_report.json', {'input_hashes_unchanged': True,
        'folds': 10, 'oof_rows': len(oof), 'oof_complete_once_per_original': True,
        'prediction_rows': len(tested), 'fit_stop_oof_test_people_disjoint': True,
        'synthetic_parents_inner_fit_only': True, 'best_checkpoints_restored_before_oof': uses_epochs})
    print(json.dumps(result), flush=True)
    return result
