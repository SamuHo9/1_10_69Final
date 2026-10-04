"""Run every preparation method for both hemispheres, with verified outputs."""
import argparse
from collections import Counter
from itertools import zip_longest
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
from data_prep_common import COEF_COLUMNS, XYZ_COLUMNS, load_pair, normalize_xyz_frame, patient_group_id, augmentation_plan

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
CONFIGS = [
    ('coef_raw','prepare_coef_raw.py','coef',['--augmentation','none'],False),
    ('pointnet_raw','prepare_pointnet_raw.py','xyz',['--augmentation','none','--normalize','per_cloud'],False),
    ('plsda_latent_features','prepare_plsda_latent_features.py','coef',[],False),
    ('plsda_direct','prepare_plsda_direct.py','coef',[],False),
    ('coef_raw_balanced_jitter','prepare_coef_raw.py','coef',['--augmentation','balanced_jitter'],True),
    ('pointnet_raw_balanced_jitter','prepare_pointnet_raw.py','xyz',['--augmentation','balanced_jitter','--normalize','per_cloud'],True),
    ('coef_plsda','prepare_coef_plsda.py','coef',[],True),
    ('pointnet_plsda','prepare_pointnet_plsda.py','xyz',['--normalize','per_cloud'],True),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def validate(method,kind,out,train_path,test_path,augmented,children):
    train,test = load_pair(train_path,test_path,kind)
    columns = COEF_COLUMNS if kind=='coef' else XYZ_COLUMNS
    if method=='plsda_direct':
        manifest = json.loads((out/'run_manifest.json').read_text())
        oof = pd.read_csv(out/'oof_predictions.csv')
        prediction = pd.read_csv(out/'test_predictions.csv')
        assert list(oof.Subject)==list(train.Subject) and list(prediction.Subject)==list(test.Subject)
        for expected,actual in [(train,oof),(test,prediction)]:
            assert np.array_equal(expected.BinaryClass,actual.BinaryClass)
            assert np.isfinite(actual.Probability).all()
            assert actual.Probability.between(0,1).all()
        assert (out/'direct_plsda_model.pkl').is_file()
        assert manifest['test_sha256']==sha(test_path)
        return dict(train_rows=len(train),test_rows=len(test),synthetic_rows=0,features=507,
                    grouped_cv_folds=manifest['actual_folds'],metrics=json.loads((out/'metrics.json').read_text()))
    manifest = json.loads((out/'data_manifest.json').read_text())
    assert manifest['test_sha256']==sha(test_path)
    assert not manifest['augmentation_applied_to_test']
    if method=='plsda_latent_features':
        prepared = pd.read_csv(out/'train_plsda_latent_features.csv')
        tested = pd.read_csv(out/'test_plsda_latent_features.csv')
        features = [c for c in prepared if c.startswith('PLS_')]
        assert len(features)==8 and len(prepared)==len(train) and len(tested)==len(test)
        assert list(prepared.Subject)==list(train.Subject) and list(tested.Subject)==list(test.Subject)
        assert np.array_equal(prepared.BinaryClass,train.BinaryClass) and np.array_equal(tested.BinaryClass,test.BinaryClass)
        assert np.isfinite(prepared[features].to_numpy()).all() and np.isfinite(tested[features].to_numpy()).all()
        assert manifest['pls_fit_scope']=='supplied_training_csv_only'
        return dict(train_rows=len(train),test_rows=len(test),synthetic_rows=0,features=8)
    tested = pd.read_csv(out/'test_prepared.csv',encoding='utf-8-sig')
    pairs = pd.read_csv(out/'pair_manifest.csv',encoding='utf-8-sig')
    normalized_train = normalize_xyz_frame(train,'per_cloud') if kind=='xyz' else train
    normalized_test = normalize_xyz_frame(test,'per_cloud') if kind=='xyz' else test
    assert len(tested)==len(test) and list(tested.Subject)==list(test.Subject)
    assert np.array_equal(tested.BinaryClass,test.BinaryClass)
    assert tested.DataType.eq('Original').all()
    assert np.allclose(tested[columns],normalized_test[columns],rtol=1e-10,atol=1e-10)
    original = pd.read_csv(out/'train_prepared.csv',nrows=len(train),encoding='utf-8-sig')
    assert list(original.Subject)==list(train.Subject)
    assert np.allclose(original[columns],normalized_train[columns],rtol=1e-10,atol=1e-10)
    assert np.array_equal(original.BinaryClass,train.BinaryClass)
    assert original.DataType.eq('Original').all()
    seen = set(train.Subject)
    counts = Counter(map(int, train.BinaryClass))
    sizes = Counter()
    synthetic_n = 0
    parents = train.set_index('Subject').BinaryClass.to_dict()
    mapping = pairs.set_index('PairID').to_dict('index')
    test_ids = {patient_group_id(x) for x in test.Subject}
    if augmented:
        assert manifest['info']['children_per_pair']==children
        assert manifest['same_class_pairing'] and not manifest['cross_class_pairs']
        assert pairs.PairID.is_unique
        unordered = [tuple(sorted((a,b))) for a,b in zip(pairs.ParentSubject1,pairs.ParentSubject2)]
        assert len(unordered)==len(set(unordered))
        assert all(a!=b for a,b in unordered)
        assert manifest['info']['unique_unordered_pairs']
    synth_reader = pd.read_csv(out/'synthetic_rows.csv',encoding='utf-8-sig',chunksize=256)
    tail_reader = pd.read_csv(out/'train_prepared.csv',encoding='utf-8-sig',
                             skiprows=range(1,len(train)+1),chunksize=256)
    for synthetic,tail in zip_longest(synth_reader,tail_reader):
        assert synthetic is not None and tail is not None
        assert len(synthetic)==len(tail)
        assert list(synthetic.Subject)==list(tail.Subject)
        assert np.array_equal(synthetic.BinaryClass,tail.BinaryClass)
        assert synthetic.DataType.eq('Synthetic').all() and tail.DataType.eq('Synthetic').all()
        assert np.isfinite(synthetic[columns].to_numpy()).all()
        assert np.allclose(synthetic[columns],tail[columns],rtol=1e-10,atol=1e-10)
        assert synthetic.Subject.is_unique and not seen.intersection(synthetic.Subject)
        seen.update(synthetic.Subject)
        synthetic_n += len(synthetic)
        counts.update(map(int, synthetic.BinaryClass))
        for row in synthetic[['PairID','ParentSubject1','ParentSubject2','BinaryClass']].to_dict('records'):
            pair = mapping[row['PairID']]
            for field in ['ParentSubject1','ParentSubject2']:
                assert row[field]==pair[field]
                assert row[field] in parents and int(parents[row[field]])==int(row['BinaryClass'])
                assert patient_group_id(row[field]) not in test_ids
            sizes[row['PairID']] += 1
    assert manifest['prepared_train_rows']==len(train)+synthetic_n
    assert manifest['synthetic_rows']==synthetic_n
    pair_sizes = {}
    if augmented:
        assert set(sizes)==set(mapping)
        assert all(n<=children and n==int(mapping[pair]['Children']) for pair,n in sizes.items())
        pair_sizes = {str(n):int(count) for n,count in Counter(sizes.values()).items()}
        assert counts[0]==counts[1]
        plan = augmentation_plan(train,children,manifest['info']['augmentation_size'])
        assert counts[0]==plan['target_per_class']
    else:
        assert synthetic_n==0 and len(pairs)==0
    return dict(train_rows=len(train)+synthetic_n,original_train_rows=len(train),test_rows=len(test),
                synthetic_rows=synthetic_n,features=len(columns),
                class_counts={str(k):int(v) for k,v in counts.items()},
                children_per_pair=children if augmented else None,pairs_by_child_count=pair_sizes,
                test_features_verified=True,same_class_train_only_parents_verified=augmented,
                unique_unordered_pairs_verified=augmented)


def run_job(job,args):
    method,script,kind,extra,augmented,side = job
    filename = 'spharm_results_coef_features.csv' if kind=='coef' else 'spharm_xyz_coords.csv'
    train = ROOT/'SplitData/train'/('SPHARM_'+side.upper())/'ml_features'/filename
    test = ROOT/'SplitData/test'/('SPHARM_'+side.upper())/'ml_features'/filename
    out = args.output_dir/method/side
    command = [sys.executable,str(SCRIPTS/script),'--train-csv',str(train),'--test-csv',str(test),
               '--output-dir',str(out),'--label-column','BinaryClass']+extra
    if augmented:
        command += ['--children-per-pair',str(args.children_per_pair),
                    '--augmentation-size',args.augmentation_size]
    if method!='plsda_latent_features':
        command += ['--seed',str(args.seed)]
    if 'plsda' in method:
        command += ['--n-components',str(args.n_components)]
    if method=='plsda_direct':
        command += ['--folds','5']
    record = dict(method=method,side=side,kind=kind,command=command,output=str(out),
                  input_train=str(train),input_test=str(test),input_train_sha256=sha(train),input_test_sha256=sha(test))
    start = time.monotonic()
    log = args.output_dir/'_logs'/(method+'_'+side+'.log')
    try:
        assert not out.exists(), 'Refusing to overwrite an existing job output'
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
        with log.open('w',encoding='utf-8') as stream:
            result = subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Prepare exited with code {result.returncode}; see {log}')
        record['validation'] = validate(method,kind,out,train,test,augmented,args.children_per_pair)
        assert sha(train)==record['input_train_sha256'] and sha(test)==record['input_test_sha256'], 'Input CSV changed during job'
        record['status'] = 'completed'
    except Exception as exc:
        record.update(status='failed',error=str(exc),traceback=traceback.format_exc())
    record.update(elapsed_seconds=round(time.monotonic()-start,2),log=str(log))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--augmentation-size',choices=['maximum','balance_only'],default='maximum')
    parser.add_argument('--dry-run',action='store_true',help='Show counts without generating datasets')
    parser.add_argument('--phase',choices=['baseline','augmentation','all'],default='all')
    parser.add_argument('--children-per-pair',type=int,default=8)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--n-components',type=int,default=8)
    parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    args.output_dir = (args.output_dir or ROOT/'Output_Dataset'/(args.augmentation_size+'_unique_pairs')).resolve()
    if args.dry_run:
        for side in ['left','right']:
            path = ROOT/'SplitData/train'/('SPHARM_'+side.upper())/'ml_features/spharm_results_coef_features.csv'
            frame = pd.read_csv(path,usecols=['Subject','BinaryClass'])
            print(side,json.dumps(augmentation_plan(frame,args.children_per_pair,args.augmentation_size)),flush=True)
        print('Output directory:',args.output_dir)
        return
    summary_path = args.output_dir/'run_all_summary.json'
    if args.output_dir.exists():
        assert summary_path.is_file(), 'Existing output folder has no compatible run summary'
        summary = json.loads(summary_path.read_text())
        assert (summary['seed']==args.seed and summary['children_per_pair']==args.children_per_pair
                and summary['n_components']==args.n_components
                and summary.get('augmentation_size')==args.augmentation_size), 'Choose a new output directory for different settings'
    else:
        args.output_dir.mkdir(parents=True)
        summary = dict(date='2026-10-03',augmentation_size=args.augmentation_size,seed=args.seed,children_per_pair=args.children_per_pair,n_components=args.n_components,
                       label_column='BinaryClass',label_definition='Original hemisphere filename labels; not person-level clinical labels.',
                       children_note='Eight is the per-pair parameter. The final pair may have fewer children to balance class counts.',
                       known_input_limitations='Existing ICP reference used all 381 persons, including test. Source SPHARM reference provenance is unknown.',
                       jobs=[])
    (args.output_dir/'_logs').mkdir(exist_ok=True)
    completed = {(r['method'],r['side']) for r in summary['jobs'] if r['status']=='completed'}
    jobs = [config+(side,) for config in CONFIGS for side in ['left','right']
            if (args.phase=='all' or config[4]==(args.phase=='augmentation')) and (config[0],side) not in completed]
    write_json(summary_path,summary)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job,job,args):job for job in jobs}
        for future in as_completed(futures):
            record = future.result()
            summary['jobs'] = [r for r in summary['jobs'] if (r['method'],r['side'])!=(record['method'],record['side'])]+[record]
            write_json(summary_path,summary)
            print(record['method'],record['side'],record['status'],record.get('validation',record.get('error')),flush=True)
    summary['completed_jobs'] = sum(r['status']=='completed' for r in summary['jobs'])
    summary['failed_jobs'] = sum(r['status']=='failed' for r in summary['jobs'])
    write_json(summary_path,summary)
    lines = ['# Prepared datasets','', 'Input: current extracted coefficient and XYZ train/test CSVs in SplitData.',
             'Labels: original BinaryClass (hemisphere target). Seed: 42. PLS components: 8. Augmentation: train only, same-class, children-per-pair parameter 8.',
             f'Augmentation size: {args.augmentation_size}. Each unordered same-class parent pair is used once. The final pair may have fewer children for equal class totals.','',
             '| Method | Side | Status | Train rows | Synthetic rows | Test rows | Features |',
             '|---|---|---|---:|---:|---:|---:|']
    for r in sorted(summary['jobs'],key=lambda r:(r['method'],r['side'])):
        v = r.get('validation',{})
        lines.append(f"| {r['method']} | {r['side']} | {r['status']} | {v.get('train_rows','')} | {v.get('synthetic_rows','')} | {v.get('test_rows','')} | {v.get('features','')} |")
    lines += ['', 'Each method contains left/ and right/ output folders. Raw and augmented methods export train_prepared.csv and test_prepared.csv. Latent-feature outputs use train_plsda_latent_features.csv and test_plsda_latent_features.csv. Direct PLS-DA exports models, metrics and predictions.', '',
              'prepare_plsda_direct.py aliases run_plsda_direct.py, so this classifier is executed once per side. Raw baselines and optional raw balanced-jitter augmentation are stored separately.', '',
              'Validation checked feature schemas, finite values, row counts, unchanged source CSV hashes, test preprocessing, same-class train parents and PairID provenance. run_all_summary.json records exact commands and validation details; _logs/ contains execution logs.', '',
              'Known input limitations: labels are hemisphere labels; contralateral hemispheres of inferred TLE persons are labelled 0. Existing ICP references were fit on all 381 inputs including test; these outputs are not an independently held-out preprocessing evaluation.','']
    (args.output_dir/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f"Completed jobs: {summary['completed_jobs']}; failed jobs: {summary['failed_jobs']}",flush=True)
    if summary['failed_jobs']:
        sys.exit(1)


if __name__=='__main__':
    main()
