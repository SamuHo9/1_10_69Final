"""Prepare coefficient CSVs with the legacy balanced PLS-DA pairing method.

With no arguments, process both hemispheres from SplitData into
Output_Dataset/augment_plsda_balanced/{left,right}. Only original training
rows are fitted/paired. Eight is the INITIAL children/pair, not a maximum:
minority pairs receive extra children to reach the majority augmented size.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from data_prep_common import (
    COEF_COLUMNS, SYNTH_PROVENANCE_COLUMNS, _batch_factory,
    _augmentation_result, class_counts, load_pair, patient_group_id,
    sha256_file, write_json, write_preparation_outputs,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "augment_plsda_balanced"


def balanced_plan(frame, children_per_pair=8):
    if children_per_pair < 1:
        raise ValueError("children_per_pair must be positive")
    if set(frame.BinaryClass.unique()) != {0, 1}:
        raise ValueError("Both classes 0 and 1 are required")
    if not frame.Subject.is_unique:
        raise ValueError("Original Subject values must be unique")
    counts = class_counts(frame)
    majority = 0 if counts['0'] >= counts['1'] else 1
    minority = 1 - majority
    pairs = {key: count // 2 for key, count in counts.items()}
    if min(pairs.values()) < 1:
        raise ValueError("At least two original rows per class are required for distinct-parent pairs")
    target = counts[str(majority)] + pairs[str(majority)] * children_per_pair
    needed = {key: target - count for key, count in counts.items()}
    return {
        "augmentation_size": "augment_both_then_balance",
        "children_per_pair": int(children_per_pair),
        "children_per_pair_is_initial_not_cap": True,
        "original_class_counts": counts,
        "majority_class": majority, "minority_class": minority,
        "target_per_class": target,
        "required_synthetic_per_class": needed,
        "final_class_counts": {key: target for key in counts},
        "available_disjoint_pairs": pairs,
        "unpaired_originals_per_class": {key: count % 2 for key, count in counts.items()},
        "unique_unordered_pairs": True, "self_pairing": False,
        "parent_reuse_across_different_pairs": False,
    }


def _disjoint_nearest_pairs(indices, scores, rng):
    """Exactly the legacy greedy rule: shuffle, select nearest, remove both."""
    pool = list(map(int, indices))
    rng.shuffle(pool)
    result = []
    while len(pool) >= 2:
        first = pool.pop(0)
        distance = np.linalg.norm(scores[pool] - scores[first], axis=1)
        second = pool.pop(int(np.argmin(distance)))
        result.append((first, second))
    return result


def augment(frame, children_per_pair=8, n_components=10, seed=42, stream=False):
    plan = balanced_plan(frame, children_per_pair)
    if n_components < 1:
        raise ValueError("n_components must be positive")
    x = frame[COEF_COLUMNS].to_numpy(dtype=np.float64)
    y = frame.BinaryClass.to_numpy(dtype=int)
    target_y = np.eye(2)[y]
    effective = min(n_components, len(frame)-1, len(COEF_COLUMNS))
    model = PLSRegression(n_components=effective, scale=True)
    scores, _ = model.fit_transform(x, target_y)
    rng = np.random.RandomState(seed)
    schedule = []
    for label in [plan['majority_class'], plan['minority_class']]:
        pairs = _disjoint_nearest_pairs(np.flatnonzero(y == label), scores, rng)
        extra = plan['required_synthetic_per_class'][str(label)] - len(pairs)*children_per_pair
        base, remainder = divmod(extra, len(pairs))
        for index,(first,second) in enumerate(pairs):
            schedule.append({
                'PairID': f'{PROTOCOL}_class{label}_pair{index:04d}',
                'Class': label,
                'ParentSubject1': str(frame.iloc[first].Subject),
                'ParentSubject2': str(frame.iloc[second].Subject),
                'Children': children_per_pair + base + (index < remainder),
                'PLSScoreDistance': float(np.linalg.norm(scores[first]-scores[second])),
                'Method': PROTOCOL, 'FirstIndex': first, 'SecondIndex': second,
            })
    # Match the original RNG sequence: all pair shuffles precede alpha jitter.
    state_after_pairs = rng.get_state()

    def factory():
        child_rng = np.random.RandomState()
        child_rng.set_state(state_after_pairs)

        def values(first,second,n,unused_rng):
            if n == 1:
                alpha = np.array([0.5])
            else:
                alpha = np.clip(np.linspace(0.1,0.9,n) + child_rng.uniform(-0.02,0.02,n),0.05,0.95)
            score_new = (1.0-alpha[:,None])*scores[first] + alpha[:,None]*scores[second]
            return alpha, model.inverse_transform(score_new)

        yield from _batch_factory(frame,COEF_COLUMNS,schedule,values,seed,PROTOCOL)()

    info = {
        'augmentation_protocol': PROTOCOL,
        'augmentation_kind': 'same_class_plsda_score_interpolation',
        'same_class_pairing': True, 'pls_fit_scope': 'supplied_training_csv_only',
        'pls_components_requested': n_components, 'pls_components_effective': effective,
        'pls_scale': True, 'pairing': 'greedy_nearest_neighbor_disjoint_parents',
        'alpha_range_before_jitter': [0.1,0.9], 'alpha_jitter_range': [-0.02,0.02],
        'alpha_clip_range': [0.05,0.95],
    }
    prepared,synthetic,pairs,info = _augmentation_result(
        frame,COEF_COLUMNS,plan,schedule,factory,info,stream)
    bundle = {'pls': model, 'requested_components': n_components,
              'effective_components': effective, 'scale_inside_pls': True}
    return prepared,synthetic,pairs,bundle,info


def verify_outputs(root, train, test, info):
    """Verify counts, source/test integrity, disjoint parents and finite features."""
    prepared = pd.read_csv(root/'train_prepared.csv')
    tested = pd.read_csv(root/'test_prepared.csv')
    synthetic = pd.read_csv(root/'synthetic_rows.csv')
    pairs = pd.read_csv(root/'pair_manifest.csv')
    if not prepared.Subject.is_unique or not pairs.PairID.is_unique:
        raise ValueError('Duplicate Subject or PairID in output')
    pd.testing.assert_frame_equal(prepared.iloc[:len(train)].reset_index(drop=True),train,
                                  check_dtype=False,rtol=1e-10,atol=1e-10)
    pd.testing.assert_frame_equal(tested,test,check_dtype=False,rtol=1e-10,atol=1e-10)
    if class_counts(prepared) != info['final_class_counts']:
        raise ValueError('Class counts differ from the balanced plan')
    if len(synthetic) != sum(info['required_synthetic_per_class'].values()):
        raise ValueError('Synthetic count differs from plan')
    pd.testing.assert_frame_equal(prepared.iloc[len(train):].reset_index(drop=True),
                                  synthetic[list(train.columns)],check_dtype=False,
                                  rtol=1e-10,atol=1e-10)
    for frame in [prepared,tested,synthetic]:
        if not np.isfinite(frame[COEF_COLUMNS].to_numpy()).all():
            raise ValueError('Non-finite feature values')
    original_labels = train.set_index('Subject').BinaryClass.to_dict()
    test_ids = {patient_group_id(value) for value in test.Subject}
    parent_values = list(pairs.ParentSubject1)+list(pairs.ParentSubject2)
    if len(parent_values) != len(set(parent_values)):
        raise ValueError('A parent was used in more than one pair')
    mapping = pairs.set_index('PairID').to_dict('index')
    sizes = synthetic.groupby('PairID').size().to_dict()
    if set(sizes) != set(mapping):
        raise ValueError('Pair provenance does not match synthetic rows')
    for pair_id,row in mapping.items():
        if sizes[pair_id] != row['Children']:
            raise ValueError('Incorrect children count for pair')
    for row in synthetic[['PairID','ParentSubject1','ParentSubject2','BinaryClass','Alpha']].to_dict('records'):
        pair = mapping[row['PairID']]
        for field in ['ParentSubject1','ParentSubject2']:
            parent = row[field]
            if (parent != pair[field] or original_labels.get(parent) != row['BinaryClass']
                    or patient_group_id(parent) in test_ids):
                raise ValueError('Invalid same-class training parent')
        if not 0.05 <= row['Alpha'] <= 0.95:
            raise ValueError('Alpha outside interpolation range')
    if not synthetic.DataType.eq('Synthetic').all() or not tested.DataType.eq('Original').all():
        raise ValueError('Incorrect original/synthetic designation')
    result = {
        'train_rows':len(prepared), 'synthetic_rows':len(synthetic),'test_rows':len(tested),
        'class_counts':class_counts(prepared), 'features':len(COEF_COLUMNS),
        'unique_pairs_and_disjoint_parents_verified':True,
        'same_class_train_only_parents_verified':True,
        'original_train_and_test_values_verified':True,
        'finite_features_verified':True,
        'children_distribution_by_class': {
            str(label): {str(n):int(count) for n,count in
                         Counter(pairs.loc[pairs.Class == label,'Children']).items()}
            for label in (0,1)},
    }
    write_json(root/'validation_report.json',result)
    return result


def process(train_csv,test_csv,output_dir,label_column='BinaryClass',
            n_components=10,seed=42,children_per_pair=8):
    train,test = load_pair(train_csv,test_csv,'coef',label_column)
    train_hash,test_hash = sha256_file(Path(train_csv)),sha256_file(Path(test_csv))
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Output directory contains existing files: {root}. Choose a new output directory.')
    prepared,synthetic,pairs,bundle,info = augment(
        train,children_per_pair,n_components,seed,stream=True)
    write_preparation_outputs(root,PROTOCOL,'coef',train_csv,test_csv,train,
                              prepared,test,synthetic,pairs,info,pls_bundle=bundle,
                              extra_manifest={
                                  'seed':seed,'train_sha256':train_hash,
                                  'model_input':'Coef_1..Coef_507',
                                  'test_preprocessing':'none',
                                  'label_definition':'Existing hemisphere BinaryClass, not person-level clinical labels',
                                  'known_input_limitations':'Existing ICP reference included test; original SPHARM reference provenance is unknown',
                                  'reference_method':'Data_Preparation/augment_plsda_balanced.py',
                              })
    result = verify_outputs(root,train,test,info)
    if sha256_file(Path(train_csv)) != train_hash or sha256_file(Path(test_csv)) != test_hash:
        raise ValueError('Input CSV changed during preparation')
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side',choices=['left','right','both'],default='both')
    parser.add_argument('--train-csv')
    parser.add_argument('--test-csv')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--split-dir',type=Path,default=ROOT/'SplitData')
    parser.add_argument('--label-column',default='BinaryClass')
    parser.add_argument('--n-components','--n_components',type=int,default=10)
    parser.add_argument('--children-per-pair','--num_per_pair',type=int,default=8)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--dry-run',action='store_true')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if bool(args.train_csv) != bool(args.test_csv):
        parser.error('Supply both --train-csv and --test-csv')
    if args.n_components < 1 or args.children_per_pair < 1 or args.seed < 0:
        parser.error('Components/children must be positive; seed must be non-negative')
    output = (args.output_dir or ROOT/'Output_Dataset'/PROTOCOL).resolve()
    jobs = []
    if args.train_csv:
        if not args.output_dir:
            parser.error('Custom CSV inputs require --output-dir')
        jobs.append(('custom',Path(args.train_csv),Path(args.test_csv),output))
    else:
        for side in ['left','right'] if args.side == 'both' else [args.side]:
            filename = 'ml_features/spharm_results_coef_features.csv'
            jobs.append((side,args.split_dir/'train'/('SPHARM_'+side.upper())/filename,
                         args.split_dir/'test'/('SPHARM_'+side.upper())/filename,output/side))
    # Preflight every input and output before writing either side.
    plans = []
    for side,train_path,test_path,job_output in jobs:
        train,test = load_pair(train_path,test_path,'coef',args.label_column)
        plan = balanced_plan(train,args.children_per_pair)
        plans.append(dict(side=side,output=str(job_output),test_rows=len(test),plan=plan))
        if not args.dry_run and job_output.exists() and any(job_output.iterdir()):
            raise FileExistsError(f'Output directory contains existing files: {job_output}')
    if args.dry_run:
        print(json.dumps(plans,indent=2))
        return
    records = []
    for side,train_path,test_path,job_output in jobs:
        print(f'Preparing {side}: {job_output}',flush=True)
        result = process(str(train_path),str(test_path),str(job_output),args.label_column,
                         args.n_components,args.seed,args.children_per_pair)
        records.append(dict(side=side,output=str(job_output),validation=result))
        print(json.dumps(records[-1]),flush=True)
    summary_root = output
    write_json(summary_root/'run_summary.json',dict(seed=args.seed,n_components=args.n_components,
               initial_children_per_pair=args.children_per_pair,jobs=records))
    lines = ['# Balanced PLS-DA preparation','',
             'Method: Data_Preparation/augment_plsda_balanced.py. Coefficient CSV input/output; no VTK meshes are generated by this Prepare script.',
             'Fit PLSRegression(scale=True) on original train only. Greedy nearest-neighbor pairs within each class; each original parent is used in at most one pair. Odd unpaired originals are retained.',
             f'Initial children per pair: {args.children_per_pair}. The minority receives extra children on its existing pairs. PLS components: {args.n_components}. Seed: {args.seed}.',
             'Target per class = majority originals + floor(majority originals / 2) * initial children per pair.',
             'Test is unchanged and never used to fit PLS-DA, select pairs or create children.','',
             '| Side | Train rows | Synthetic rows | Test rows | Class 0 | Class 1 |',
             '|---|---:|---:|---:|---:|---:|']
    for row in records:
        v = row['validation']
        lines.append(f"| {row['side']} | {v['train_rows']} | {v['synthetic_rows']} | {v['test_rows']} | {v['class_counts']['0']} | {v['class_counts']['1']} |")
    lines += ['', 'Outputs: train_prepared.csv, test_prepared.csv, synthetic_rows.csv (features and parent/alpha provenance), pair_manifest.csv (actual children per pair), plsda_model.pkl, data_manifest.json and validation_report.json.',
              'Labels retain the existing hemisphere BinaryClass. Existing ICP reference included test; these inputs are not an independently held-out preprocessing evaluation.',
              'For cross-validation, call process with original train/validation CSVs within each fold. Do not split a pre-augmented train_prepared.csv into folds.','']
    (summary_root/'README.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__ == '__main__':
    main()
