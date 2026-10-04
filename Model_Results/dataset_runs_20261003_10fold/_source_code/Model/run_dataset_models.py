"""Run models separately on every prepared dataset and save per-job logs."""
import argparse
import csv
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {'coef_raw':'coef','coef_raw_balanced_jitter':'coef','coef_plsda':'coef',
            'augment_plsda_balanced':'coef','plsda_latent_features':'latent',
            'pointnet_raw':'xyz','pointnet_raw_balanced_jitter':'xyz','pointnet_plsda':'xyz'}
TABULAR_MODELS = ['SVM','MLP','ResNet','ResNetAE','MobileNet','SqueezeNet']


class ConsoleLog:
    """Retain the runner's stdout/stderr in addition to every worker log."""
    def __init__(self, terminal, stream):
        self.terminal, self.stream = terminal, stream

    def write(self, value):
        self.terminal.write(value)
        self.stream.write(value)
        self.stream.flush()

    def flush(self):
        self.terminal.flush()
        self.stream.flush()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def build_jobs(dataset_root,output_root,scope='all'):
    jobs = []
    for dataset,kind in DATASETS.items():
        for side in ['left','right']:
            folder = dataset_root/dataset/side
            if kind=='latent':
                train,test = folder/'train_plsda_latent_features.csv',folder/'test_plsda_latent_features.csv'
            else:
                train,test = folder/'train_prepared.csv',folder/'test_prepared.csv'
            if not train.is_file() or not test.is_file():
                raise FileNotFoundError(f'Missing prepared inputs: {folder}')
            for model in ['PointNet'] if kind=='xyz' else TABULAR_MODELS if scope=='all' else ['SVM']:
                jobs.append(dict(dataset=dataset,kind=kind,side=side,model=model,
                                 train=str(train.resolve()),test=str(test.resolve()),
                                 output=str((output_root/dataset/side/model).resolve())))
    for side in ['left','right']:
        source = dataset_root/'coef_raw'/side
        jobs.append(dict(dataset='plsda_direct',kind='coef',side=side,model='PLSDA',
                         train=str((source/'train_prepared.csv').resolve()),
                         test=str((source/'test_prepared.csv').resolve()),
                         output=str((output_root/'plsda_direct'/side/'PLSDA').resolve())))
    return jobs


def update_reports(root,summary):
    write_json(root/'run_summary.json',summary)
    columns = ['dataset','side','model','status','train_rows','test_rows','accuracy',
               'balanced_accuracy','sensitivity','specificity','f1_macro','roc_auc',
               'oof_accuracy','oof_balanced_accuracy','oof_sensitivity','oof_specificity','oof_f1_macro','oof_roc_auc','log','output']
    with (root/'results_summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore')
        writer.writeheader()
        for row in summary['jobs']:
            writer.writerow({**row,**row.get('metrics',{}),**{'oof_'+key:value for key,value in row.get('oof_metrics',{}).items()}})
    lines = ['# Dataset model runs','',
             f"Scope: {summary['scope']}. Seed: {summary['seed']}. Neural epochs: {summary['epochs']}. Coefficient model PLS components: {summary['pls_components']}.",
             'Each prepared dataset and hemisphere is trained independently. Existing repository architecture classes are loaded without legacy hardcoded dataset paths.',
             'Coefficient models use train-fitted StandardScaler and binary-target PLS. Latent data are rebuilt using one-hot PLS from the corresponding original coefficients. PointNet uses the supplied 1002-point XYZ clouds.',
             'Exactly 10 stratified folds of original Train people for every model. Each original row has one OOF prediction; synthetic rows are not evaluated as OOF. No silent fold reduction.',
             'Augmentation, scaling and supervised PLS are regenerated/fitted inside each training fold. Latent inputs are rebuilt from original coefficients inside each fold.',
             'Fixed settings (PLS components and epochs), no outer OOF/Test-based model or epoch selection. After CV, refit on full Train and predict the held-out Test once.',
             'Prepared synthetic CSV rows and pre-fitted latent coordinates are not reused in outer folds. The dataset preparation method is rebuilt from its original training people and data_manifest.json.',
             'Input limitations: current labels are hemisphere labels; existing ICP reference included test. These results are an exploratory comparison on the supplied datasets.','',
             '| Dataset | Side | Model | Status | Test accuracy | Balanced accuracy | ROC-AUC | OOF accuracy | OOF ROC-AUC |',
             '|---|---|---|---|---:|---:|---:|---:|---:|']
    for row in summary['jobs']:
        m = row.get('metrics',{})
        values = [f'{m[key]:.4f}' if key in m else '' for key in ['accuracy','balanced_accuracy','roc_auc']]
        values += [f"{row['oof_metrics'][key]:.4f}" if key in row.get('oof_metrics',{}) else '' for key in ['accuracy','roc_auc']]
        lines.append(f"| {row['dataset']} | {row['side']} | {row['model']} | {row['status']} | {' | '.join(values)} |")
    lines += ['', 'Each dataset/side/model folder contains model/preprocessing files, metrics.json, predictions, training_history.csv, plots, run_manifest.json and validation_report.json.',
              'Logs are in _logs/<dataset>_<side>_<model>.log. results_summary.csv contains all metrics; run_summary.json records exact commands, timings and failures.',
              'Every model, including Direct PLS-DA, uses the same 10-fold person assignments within each hemisphere. oof_predictions.csv/.npz, oof_metrics.json and fold_metrics.csv are saved per job; fold_01..fold_10 contain checkpoints and provenance.','']
    (root/'README.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,default=ROOT/'Output_Dataset')
    parser.add_argument('--output-root',type=Path,default=ROOT/'Model_Results/dataset_runs_20261003_10fold')
    parser.add_argument('--scope',choices=['all','baseline'],default='all')
    parser.add_argument('--folds',type=int,choices=[10],default=10)
    parser.add_argument('--epochs',type=int,default=80)
    parser.add_argument('--pls-components',type=int,default=8)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    jobs = build_jobs(args.dataset_root,args.output_root,args.scope)
    if args.dry_run:
        print(json.dumps({'jobs':len(jobs),'plan':jobs},indent=2));return
    if min(args.epochs,args.pls_components,args.threads)<1:
        parser.error('Epochs, components and threads must be positive')
    path = args.output_root/'run_summary.json'
    if args.output_root.exists():
        if not path.is_file():
            raise FileExistsError('Existing result directory has no compatible run summary')
        summary = json.loads(path.read_text())
        if summary.get('protocol') != 'person_10fold_oof_v1':
            raise ValueError('Previous holdout results cannot be resumed as 10-fold; choose a new result directory')
        for key in ['scope','epochs','seed','pls_components','folds']:
            if summary[key]!=getattr(args,key):
                raise ValueError('Use a new result directory for different settings')
    else:
        args.output_root.mkdir(parents=True)
        summary = dict(started_utc=datetime.now(timezone.utc).isoformat(),scope=args.scope,
                       epochs=args.epochs,seed=args.seed,pls_components=args.pls_components,folds=args.folds,protocol='person_10fold_oof_v1',
                       expected_training_jobs=len(jobs),jobs=[],python=sys.executable)
    logs = args.output_root/'_logs';logs.mkdir(exist_ok=True)
    master_stream = (logs/'run_all.log').open('a',encoding='utf-8')
    sys.stdout = ConsoleLog(sys.stdout,master_stream)
    sys.stderr = ConsoleLog(sys.stderr,master_stream)
    print('Run invocation: '+json.dumps(sys.argv),flush=True)
    print(f'Python: {sys.executable}; jobs={len(jobs)}; folds={args.folds}; neural_epochs={args.epochs}',flush=True)
    update_reports(args.output_root,summary)
    completed = {(row['dataset'],row['side'],row['model']) for row in summary['jobs'] if row['status']=='completed'}
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads),PYTHONIOENCODING='utf-8',MPLBACKEND='Agg')
    worker = ROOT/'Model/train_dataset_model.py'
    for index,job in enumerate(jobs,1):
        key = (job['dataset'],job['side'],job['model'])
        if key in completed:
            continue
        log = logs/('_'.join(key)+'.log')
        command = [sys.executable,'-u',str(worker),'--dataset',job['dataset'],'--side',job['side'],
                   '--model',job['model'],'--kind',job['kind'],'--train-csv',job['train'],
                   '--test-csv',job['test'],'--output-dir',job['output'],'--epochs',str(args.epochs),
                   '--folds',str(args.folds),'--pls-components',str(args.pls_components),'--seed',str(args.seed),'--threads',str(args.threads)]
        record = {**job,'command':command,'log':str(log),'status':'running'}
        summary['jobs'] = [row for row in summary['jobs'] if (row['dataset'],row['side'],row['model'])!=key]+[record]
        update_reports(args.output_root,summary)
        print(f'[{index}/{len(jobs)}] START {key} -> {log}',flush=True)
        start = time.monotonic()
        with log.open('w',encoding='utf-8') as stream:
            stream.write('Command: '+json.dumps(command)+'\n');stream.flush()
            result = subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        record['elapsed_seconds'] = round(time.monotonic()-start,2)
        if result.returncode==0:
            record['status'] = 'completed'
            saved_metrics = json.loads((Path(job['output'])/'metrics.json').read_text())
            record['metrics'] = {**saved_metrics['test'],
                'train_rows':saved_metrics['train_rows'],'test_rows':saved_metrics['test_rows'],
                'original_train_rows':saved_metrics['original_train_rows']}
            record['oof_metrics'] = saved_metrics['cv_oof']
        else:
            record.update(status='failed',exit_code=result.returncode)
        update_reports(args.output_root,summary)
        m = record.get('metrics',{})
        print(f"[{index}/{len(jobs)}] {record['status']} {key} seconds={record['elapsed_seconds']} test_accuracy={m.get('accuracy')} AUC={m.get('roc_auc')}",flush=True)
    summary['completed_training_jobs'] = sum(row['status']=='completed' for row in summary['jobs'])
    summary['failed_training_jobs'] = sum(row['status']=='failed' for row in summary['jobs'])
    summary['reused_jobs'] = sum(row['status']=='reused' for row in summary['jobs'])
    summary['finished_utc'] = datetime.now(timezone.utc).isoformat()
    update_reports(args.output_root,summary)
    print(f"Completed={summary['completed_training_jobs']} Failed={summary['failed_training_jobs']} Reused={summary['reused_jobs']}",flush=True)
    if summary['failed_training_jobs']:
        sys.exit(1)


if __name__=='__main__':
    main()
