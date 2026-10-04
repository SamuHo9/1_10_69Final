"""Independently check completed model artifacts, metrics and source hashes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,confusion_matrix,
                             f1_score,recall_score,roc_auc_score)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_root',type=Path)
    args = parser.parse_args()
    summary = json.loads((args.result_root/'run_summary.json').read_text())
    errors = []
    verified = []
    hash_cache = {}
    for job in summary['jobs']:
        label = '/'.join(job[key] for key in ['dataset','side','model'])
        try:
            if job['status']!='completed':
                raise ValueError(f"Job is {job['status']}")
            output = Path(job['output'])
            test = pd.read_csv(job['test'])
            prediction = pd.read_csv(output/'test_predictions.csv')
            if list(prediction.Subject)!=list(test.Subject) or not np.array_equal(prediction.BinaryClass,test.BinaryClass):
                raise ValueError('Test IDs or labels differ')
            probability = prediction.Probability.to_numpy()
            if not np.isfinite(probability).all() or not ((probability>=0)&(probability<=1)).all():
                raise ValueError('Invalid probabilities')
            y,predicted = prediction.BinaryClass.to_numpy(),prediction.Prediction.to_numpy()
            if not set(predicted).issubset({0,1}):
                raise ValueError('Invalid predicted classes')
            expected = {
                'accuracy':accuracy_score(y,predicted),
                'balanced_accuracy':balanced_accuracy_score(y,predicted),
                'sensitivity':recall_score(y,predicted,pos_label=1,zero_division=0),
                'specificity':recall_score(y,predicted,pos_label=0,zero_division=0),
                'f1_macro':f1_score(y,predicted,average='macro',zero_division=0),
                'roc_auc':roc_auc_score(y,probability),
            }
            for metric,value in expected.items():
                if not np.isclose(job['metrics'][metric],value,rtol=1e-10,atol=1e-10):
                    raise ValueError(f'Metric mismatch: {metric}')
            manifest = json.loads((output/'run_manifest.json').read_text())
            if job['model']=='PLSDA':
                if not (output/'direct_plsda_model.pkl').is_file():
                    raise ValueError('Missing PLS-DA model')
                if manifest['test_sha256']!=sha(job['test']):
                    raise ValueError('Test source hash mismatch')
            else:
                for part in ['train','test']:
                    source = job[part]
                    if source not in hash_cache:
                        hash_cache[source] = sha(source)
                    if manifest['input_sha256'][part]!=hash_cache[source]:
                        raise ValueError(f'Changed {part} input')
                model_file = output/('model.pkl' if job['model'] in ['SVM','MLP'] else 'model.pt')
                if not model_file.is_file() or model_file.stat().st_size==0:
                    raise ValueError('Missing/empty model')
                if not (output/'preprocessing.pkl').is_file():
                    raise ValueError('Missing preprocessing')
                saved_metrics = json.loads((output/'metrics.json').read_text())
                if saved_metrics['confusion_matrix']!=confusion_matrix(y,predicted,labels=[0,1]).tolist():
                    raise ValueError('Confusion matrix mismatch')
                if manifest['test_used_for_selection'] or manifest['cross_validation_performed']:
                    raise ValueError('Unexpected model selection protocol')
                report = json.loads((output/'validation_report.json').read_text())
                if not report['input_hashes_unchanged'] or report['prediction_rows']!=len(test):
                    raise ValueError('Invalid worker validation report')
                if job['model'] not in ['SVM','MLP']:
                    history = pd.read_csv(output/'training_history.csv')
                    if len(history)!=summary['epochs'] or not np.isfinite(history.loss).all():
                        raise ValueError('Incorrect epoch count or non-finite loss')
                    if manifest['runtime']['device']!='cuda':
                        raise ValueError('Neural model did not use expected GPU')
            log = Path(job['log'])
            if not log.is_file() or log.stat().st_size==0:
                raise ValueError('Missing/empty log')
            if 'Traceback (most recent call last)' in log.read_text(encoding='utf-8'):
                raise ValueError('Traceback in completed log')
            verified.append(dict(job=label,test_rows=len(test),metrics_verified=True))
        except Exception as exc:
            errors.append(dict(job=label,error=str(exc)))
    if len(verified)!=summary['expected_training_jobs']:
        errors.append({'error':f"Expected {summary['expected_training_jobs']} completed jobs; verified {len(verified)}"})
    report = {'verified_jobs':len(verified),'expected_jobs':summary['expected_training_jobs'],
              'errors':errors,'checks':verified}
    (args.result_root/'audit_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'verified_jobs':len(verified),'errors':errors},indent=2))
    if errors:
        raise SystemExit(1)


if __name__=='__main__':
    main()
