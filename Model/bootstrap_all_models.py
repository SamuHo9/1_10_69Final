"""Bootstrap fixed Test predictions for every completed 10-fold model job.

Legacy sampling: 1,000 ordinary N-of-N resamples with replacement, seed 42.
Saved Prediction values preserve each model's existing decision rule.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['Accuracy', 'BalancedAccuracy', 'Sensitivity_Class0', 'Specificity_Class0',
           'F1_Class0', 'Sensitivity_Class1', 'Specificity_Class1', 'F1_Class1', 'F1_macro', 'AUC']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def validate_predictions(frame):
    if not {'Subject', 'BinaryClass', 'Probability', 'Prediction'}.issubset(frame.columns):
        raise ValueError('Missing prediction columns')
    if frame.Subject.isna().any() or not frame.Subject.is_unique:
        raise ValueError('Test subjects must be present and unique')
    if set(frame.BinaryClass) != {0, 1} or not set(frame.Prediction).issubset({0, 1}):
        raise ValueError('Invalid binary labels/predictions')
    if not np.isfinite(frame.Probability).all() or not frame.Probability.between(0, 1).all():
        raise ValueError('Invalid probabilities')
    # Each observation must represent one person in the chosen hemisphere.
    sys.path.insert(0, str(ROOT/'Data_Preparation'))
    from data_prep_common import patient_group_id
    if len({patient_group_id(s) for s in frame.Subject}) != len(frame):
        raise ValueError('Repeated people require a cluster bootstrap instead')


def classification_metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    tp = int(np.sum((y == 1) & (prediction == 1)))
    divide = lambda a, b: float(a/b) if b else 0.0
    sens0, sens1 = divide(tn, tn+fp), divide(tp, tp+fn)
    f0, f1 = divide(2*tn, 2*tn+fp+fn), divide(2*tp, 2*tp+fp+fn)
    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Accuracy': divide(tp+tn, len(y)), 'BalancedAccuracy': (sens0+sens1)/2,
        'Sensitivity_Class0': sens0, 'Specificity_Class0': sens1, 'F1_Class0': f0,
        'Sensitivity_Class1': sens1, 'Specificity_Class1': sens0, 'F1_Class1': f1,
        'F1_macro': (f0+f1)/2}


def calculate(frame, rounds=1000, seed=42):
    """Return full-precision replicate scores and reproducible sampling indices."""
    validate_predictions(frame)
    if rounds < 2:
        raise ValueError('At least two bootstrap attempts are required')
    y = frame.BinaryClass.to_numpy(dtype=int)
    predicted = frame.Prediction.to_numpy(dtype=int)
    probability = frame.Probability.to_numpy(dtype=float)
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(frame), size=(rounds, len(frame)), replace=True)
    grid = np.linspace(0, 1, 101)
    rows, curves, valid_rounds, skipped = [], [], [], []
    for attempt, sample in enumerate(indices, 1):
        sy = y[sample]
        if np.unique(sy).size < 2:
            skipped.append({'round': attempt, 'reason': 'Only one true class in resample'})
            continue
        fpr, tpr, _ = roc_curve(sy, probability[sample])
        row = {'round': attempt, **classification_metrics(sy, predicted[sample]), 'AUC': float(auc(fpr, tpr))}
        rows.append(row)
        interpolated = np.interp(grid, fpr, tpr)
        interpolated[0], interpolated[-1] = 0.0, 1.0
        curves.append(interpolated)
        valid_rounds.append(attempt)
    if len(rows) < 2:
        raise ValueError('Too few valid two-class resamples to estimate intervals')
    scores = pd.DataFrame(rows)
    fpr, tpr, _ = roc_curve(y, probability)
    original = {**classification_metrics(y, predicted), 'AUC': float(auc(fpr, tpr))}
    intervals = {key: {'original': original[key], 'bootstrap_mean': float(scores[key].mean()),
        'bootstrap_std': float(scores[key].std(ddof=1)),
        'ci95_lower': float(np.percentile(scores[key], 2.5)),
        'ci95_upper': float(np.percentile(scores[key], 97.5))} for key in METRICS}
    curves = np.array(curves)
    data = {'indices': indices, 'valid_rounds': np.array(valid_rounds), 'fpr_grid': grid,
        'tpr_curves': curves, 'mean_tpr': curves.mean(axis=0),
        'lower_tpr': np.percentile(curves, 2.5, axis=0),
        'upper_tpr': np.percentile(curves, 97.5, axis=0),
        'original_fpr': fpr, 'original_tpr': tpr,
        'plot_curve_indices': rng.choice(len(curves), min(100, len(curves)), replace=False)}
    return scores, original, intervals, data, skipped


def verify_baseline(original, stored):
    mapping = {'Accuracy': 'accuracy', 'BalancedAccuracy': 'balanced_accuracy',
        'Sensitivity_Class1': 'sensitivity', 'Specificity_Class1': 'specificity',
        'F1_macro': 'f1_macro', 'AUC': 'roc_auc'}
    for key, target in mapping.items():
        if not np.isclose(original[key], stored[target], atol=1e-10, rtol=1e-10):
            raise ValueError(f'Test score differs from the completed training run: {target}')


def plots(output, original, intervals, data, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    grid, mean = data['fpr_grid'], data['mean_tpr']
    for name in ('roc_curve_lines.png', 'roc_curve_ci.png'):
        fig, ax = plt.subplots(figsize=(7.5, 6.2))
        if name == 'roc_curve_lines.png':
            for number, index in enumerate(data['plot_curve_indices']):
                ax.plot(grid, data['tpr_curves'][index], color='steelblue', alpha=0.2, lw=0.8,
                        label='Bootstrap samples (100)' if number == 0 else None)
        else:
            ax.fill_between(grid, data['lower_tpr'], data['upper_tpr'], color='grey', alpha=0.25,
                            label='Pointwise 95% bootstrap interval')
        ax.plot(grid, mean, color='darkorange', lw=2, linestyle='--', label='Mean bootstrap ROC')
        interval = intervals['AUC']
        ax.plot(data['original_fpr'], data['original_tpr'], color='navy', lw=2.2,
                label=f"Test ROC: AUC {original['AUC']:.3f}\n95% CI [{interval['ci95_lower']:.3f}, {interval['ci95_upper']:.3f}]")
        ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle=':')
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='False positive rate', ylabel='True positive rate', title=title)
        ax.grid(alpha=0.2)
        ax.legend(loc='lower right', fontsize=9)
        fig.tight_layout()
        fig.savefig(output/name, dpi=180)
        plt.close(fig)


def process(job, rounds, seed):
    source = Path(job['output'])/'test_predictions.csv'
    output = Path(job['output'])/'bootstrap'
    source_hash = sha(source)
    frame = pd.read_csv(source).sort_values('Subject', kind='stable').reset_index(drop=True)
    scores, original, intervals, data, skipped = calculate(frame, rounds, seed)
    verify_baseline(original, job['metrics'])
    scores.to_csv(output/'bootstrap_confusion_matrix.csv', index=False, encoding='utf-8-sig')
    np.savez_compressed(output/'bootstrap_resamples.npz', **data,
        subjects=frame.Subject.to_numpy(dtype=str), labels=frame.BinaryClass.to_numpy(),
        probability=frame.Probability.to_numpy(), prediction=frame.Prediction.to_numpy())
    write_json(output/'bootstrap_statistics.json', intervals)
    training_manifest = json.loads((Path(job['output'])/'run_manifest.json').read_text())
    manifest = {'protocol': 'fixed_test_predictions_ordinary_bootstrap_v1',
        'dataset': job['dataset'], 'side': job['side'], 'model': job['model'],
        'input_predictions': str(source), 'input_sha256': source_hash,
        'resampling': 'ordinary bootstrap of people with replacement',
        'sample_size': len(frame), 'sample_fraction': 1.0, 'seed': seed, 'attempted_rounds': rounds,
        'valid_rounds': len(scores), 'skipped_rounds': skipped,
        'canonical_subject_order': list(frame.Subject), 'same_subject_resamples_shared_across_models': True,
        'decision_rule': 'saved Prediction column', 'original_model_prediction_rule': training_manifest['prediction_rule'],
        'threshold_optimization_performed': False, 'model_retrained': False,
        'confidence_interval': 'percentile 2.5/97.5 on unrounded replicate scores',
        'roc_band': 'pointwise percentiles on 101 FPR grid points',
        'uncertainty_scope': 'Test resampling conditional on the saved trained model',
        'numpy_version': np.__version__, 'pandas_version': pd.__version__, 'python_version': sys.version,
        'original_test_scores': original}
    write_json(output/'bootstrap_manifest.json', manifest)
    print(f"Dataset={job['dataset']} side={job['side']} model={job['model']}")
    print(f'Test people={len(frame)}; sampling=N of N with replacement; seed={seed}; saved Prediction unchanged')
    print(f'Attempted={rounds}; valid={len(scores)}; skipped={len(skipped)}')
    print(json.dumps(intervals, indent=2))
    plots(output, original, intervals, data, f"{job['model']}, {job['dataset']}, {job['side']}\nTest bootstrap, {rounds:,} resamples")
    if sha(source) != source_hash:
        raise ValueError('Source prediction CSV changed during bootstrap')
    write_json(output/'validation_report.json', {'input_unchanged': True,
        'original_metrics_match_training_results': True, 'sample_size': len(frame), 'valid_rounds': len(scores),
        'all_confusion_totals_equal_sample_size': bool((scores[['TP', 'TN', 'FP', 'FN']].sum(axis=1) == len(frame)).all())})
    return {'test_rows': len(frame), 'valid_rounds': len(scores), 'skipped_rounds': len(skipped),
        **{f'{key}_{field}': value[field] for key, value in intervals.items()
           for field in ('original', 'bootstrap_mean', 'ci95_lower', 'ci95_upper')}}


def write_reports(root, summary):
    write_json(root/'bootstrap_run_summary.json', summary)
    rows = [{**{key: row.get(key) for key in ('dataset', 'side', 'model', 'status', 'output', 'log')},
             **row.get('scores', {})} for row in summary['jobs']]
    pd.DataFrame(rows).to_csv(root/'bootstrap_summary.csv', index=False, encoding='utf-8-sig')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', type=Path, default=ROOT/'Model_Results/dataset_runs_20261003_10fold')
    parser.add_argument('--rounds', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    root = args.result_root.resolve()
    training = json.loads((root/'run_summary.json').read_text())
    if training.get('completed_training_jobs') != training['expected_training_jobs'] or training.get('failed_training_jobs'):
        raise ValueError('Training must complete successfully before bootstrapping')
    if (root/'bootstrap_run_summary.json').exists():
        raise FileExistsError('Bootstrap results already exist; existing results will not be replaced')
    # Preflight all jobs before creating result artifacts; align people within a hemisphere.
    people_by_side = {}
    for job in training['jobs']:
        if job['status'] != 'completed':
            raise ValueError('Incomplete training job')
        output = Path(job['output'])/'bootstrap'
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f'Existing bootstrap artifacts: {output}')
        frame = pd.read_csv(Path(job['output'])/'test_predictions.csv').sort_values('Subject').reset_index(drop=True)
        validate_predictions(frame)
        identity = list(zip(frame.Subject, frame.BinaryClass))
        if job['side'] in people_by_side and identity != people_by_side[job['side']]:
            raise ValueError('Models have different Test people/labels in the same hemisphere')
        people_by_side[job['side']] = identity
    summary = {'protocol': 'fixed_test_predictions_ordinary_bootstrap_v1', 'expected_jobs': len(training['jobs']),
        'rounds': args.rounds, 'seed': args.seed, 'started_utc': datetime.now(timezone.utc).isoformat(), 'jobs': []}
    logs = root/'_logs'
    logs.mkdir(exist_ok=True)
    master = logs/'bootstrap_all.log'
    with master.open('w', encoding='utf-8') as stream:
        def announce(value):
            print(value, flush=True); stream.write(value+'\n'); stream.flush()
        announce('Command: '+json.dumps(sys.argv))
        for number, job in enumerate(training['jobs'], 1):
            output = Path(job['output'])/'bootstrap'
            output.mkdir(exist_ok=True)
            log = output/'bootstrap.log'
            record = {key: job[key] for key in ('dataset', 'side', 'model')}
            record.update(status='running', output=str(output), log=str(log))
            summary['jobs'].append(record)
            write_reports(root, summary)
            announce(f"[{number}/{len(training['jobs'])}] START {job['dataset']}/{job['side']}/{job['model']}")
            start = time.monotonic()
            with log.open('w', encoding='utf-8') as detail, redirect_stdout(detail), redirect_stderr(detail):
                try:
                    record['scores'] = process(job, args.rounds, args.seed)
                    record['status'] = 'completed'
                except Exception:
                    traceback.print_exc()
                    record['status'] = 'failed'
            record['elapsed_seconds'] = round(time.monotonic()-start, 3)
            write_reports(root, summary)
            announce(f"[{number}/{len(training['jobs'])}] {record['status']} seconds={record['elapsed_seconds']}")
        summary.update(completed_jobs=sum(r['status'] == 'completed' for r in summary['jobs']),
            failed_jobs=sum(r['status'] == 'failed' for r in summary['jobs']), finished_utc=datetime.now(timezone.utc).isoformat())
        write_reports(root, summary)
        announce(f"Completed={summary['completed_jobs']} Failed={summary['failed_jobs']}")
    (root/'BOOTSTRAP_README_TH.md').write_text('''# Bootstrap ของผล Test

ทำบนผลทำนาย Test ของทั้ง 68 งาน ใช้คนต้นฉบับเท่านั้น ซ้าย 73 คน ขวา 77 คน
สุ่ม 1,000 รอบ ขนาด N คน แบบคืนตัวอย่าง seed 42 เรียง Subject ก่อนสุ่ม
ชุดสุ่มเดียวกันใช้กับโมเดลที่ประเมินคนชุดเดียวกันในข้างเดียวกัน
รอบที่มีคลาสเดียวจะข้ามและบันทึกเหตุผลใน manifest

ใช้ Prediction ที่บันทึกไว้ตามเกณฑ์ของแต่ละโมเดล จึงคงคะแนน Test รอบล่าสุด
ไม่ได้ฝึกโมเดลใหม่หรือเลือก threshold ใหม่ ความแปรปรวนนี้อ้างอิงโมเดลที่ฝึกไว้
และการสุ่มคนใน Test ไม่ครอบคลุมการฝึกโมเดลใหม่

bootstrap_summary.csv สรุปทั้ง 68 งาน โดย *_original เป็นคะแนน Test จริง
*_bootstrap_mean เป็นค่าเฉลี่ยจากการสุ่ม และ *_ci95_lower/upper เป็น percentile
2.5/97.5 คำนวณจากค่าที่ไม่ปัดเศษ แถบ ROC เป็นช่วงรายจุด FPR บนกริด 101 จุด
กราฟแสดงเส้น Test จริง เส้นเฉลี่ย bootstrap และ AUC/CI ของ Test แยกชัดเจน

แต่ละโฟลเดอร์ dataset/side/model/bootstrap มี:
- bootstrap_confusion_matrix.csv: คะแนนและ confusion counts ของทุกรอบที่ valid
- bootstrap_statistics.json: คะแนนจริง ค่าเฉลี่ย SD และ CI 95% ของทุกตัวชี้วัด
- bootstrap_resamples.npz: ดัชนีสุ่ม คน/label/prediction และข้อมูล ROC สำหรับตรวจซ้ำ
- roc_curve_lines.png: เส้นสุ่ม 100 เส้น เส้นเฉลี่ย และ ROC จาก Test จริง
- roc_curve_ci.png: ROC พร้อมแถบ percentile 95% รายจุด
- bootstrap_manifest.json, validation_report.json, bootstrap.log

log รวมอยู่ที่ _logs/bootstrap_all.log และสถานะทุกงานอยู่ใน bootstrap_run_summary.json
BinaryClass เป็น label ของ hippocampus แต่ละข้างตามข้อมูลต้นทาง
''', encoding='utf-8')
    if summary['failed_jobs']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
