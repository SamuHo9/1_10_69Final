"""Choose one probability threshold using original Train OOF rows only."""
import numpy as np
import pandas as pd


def select_accuracy_threshold(y_true, probability):
    """Exhaust all distinct >= decision partitions; never receives Test data.

    Ties: maximum balanced accuracy, closest to 0.5, then lower threshold.
    A threshold just above the maximum score represents predicting all negative.
    OOF scores used for selection are descriptive, not unbiased validation.
    """
    y = np.asarray(y_true)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError('Labels and probabilities must be nonempty matching vectors')
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Threshold selection requires both binary classes')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Probabilities must be finite and within [0, 1]')
    candidates = np.unique(np.r_[0.0, 0.5, 1.0, p, np.nextafter(p.max(), np.inf)])
    predicted = p[None, :] >= candidates[:, None]
    positive = y == 1
    tp = np.sum(predicted & positive, axis=1)
    fp = np.sum(predicted & ~positive, axis=1)
    fn, tn = positive.sum() - tp, (~positive).sum() - fp
    sensitivity, specificity = tp / positive.sum(), tn / (~positive).sum()
    f1_pos = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(len(tp)), where=2 * tp + fp + fn != 0)
    f1_neg = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros(len(tn)), where=2 * tn + fp + fn != 0)
    sweep = pd.DataFrame({'threshold': candidates, 'correct': tp + tn, 'rows': len(y),
        'accuracy': (tp + tn) / len(y), 'balanced_accuracy': (sensitivity + specificity) / 2,
        'sensitivity': sensitivity, 'specificity': specificity, 'f1_macro': (f1_pos + f1_neg) / 2,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn})
    best = min(range(len(sweep)), key=lambda i: (-int(tp[i] + tn[i]),
        -float(sweep.balanced_accuracy.iloc[i]), abs(candidates[i] - 0.5), candidates[i]))
    threshold = float(candidates[best])
    sweep['selected'] = False
    sweep.loc[best, 'selected'] = True
    report = {'objective': 'accuracy', 'threshold': threshold, 'comparison': 'probability >= threshold',
        'tie_break': ['highest_balanced_accuracy', 'closest_to_0.5', 'lower_threshold'],
        'selection_source': 'original_train_10fold_oof', 'selection_rows': len(y),
        'test_used_for_selection': False, 'candidates': len(candidates),
        'oof_selection_accuracy': float(sweep.accuracy.iloc[best]),
        'oof_selection_balanced_accuracy': float(sweep.balanced_accuracy.iloc[best]),
        'score_interpretation': 'OOF labels selected this threshold; tuned OOF scores are selection scores, not unbiased CV estimates'}
    return threshold, sweep, report


def plot_threshold_sweep(folder, sweep, threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for name, label in [('accuracy', 'OOF accuracy'), ('balanced_accuracy', 'OOF balanced accuracy'),
                        ('sensitivity', 'OOF sensitivity (class 1)')]:
        ax.plot(sweep.threshold, sweep[name], label=label)
    ax.axvline(threshold, color='black', linestyle='--', label=f'Selected threshold {threshold:.4f}')
    ax.set(xlabel='Probability threshold (>=)', ylabel='Score', ylim=(0, 1.02),
           title='Train OOF threshold selection')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(folder / 'threshold_selection.png', dpi=160)
    plt.close(fig)
