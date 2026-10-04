"""Copy existing SPHARM files into a reproducible, subject-disjoint 80/20 split."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import random
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'SplitData'
SEED = 42


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    if DEST.exists():
        raise FileExistsError(f'Refusing to overwrite an existing split: {DEST}')
    inputs, labels = {}, {}
    for side in ['left','right']:
        folder = ROOT/'SPHARM'/('SPHARM_'+side.upper())
        inputs[side], labels[side] = {}, {}
        for path in sorted(folder.glob('spharm_status_shard*.json')):
            for item in json.loads(path.read_text())['subjects']:
                name = Path(item['input']).name.removesuffix('.nii.gz')
                sid = re.search(r'sub-([^_]+)',name).group(1)
                assert sid not in inputs[side], f'Duplicate ID in {side}: {sid}'
                inputs[side][sid] = dict(basename=name,success=item['success'])
                labels[side][sid] = re.search(r'_(Healthy|TLE)_',name).group(1)
    assert labels['left'].keys() == labels['right'].keys()
    cohorts = {}
    for sid in sorted(labels['left']):
        pair = labels['left'][sid],labels['right'][sid]
        assert pair in {('Healthy','Healthy'),('TLE','Healthy'),('Healthy','TLE')}
        cohorts[sid] = 'Control' if pair==('Healthy','Healthy') else ('Left_TLE' if pair[0]=='TLE' else 'Right_TLE')
        assert inputs['left'][sid]['success'] or inputs['right'][sid]['success'], 'Subject has no usable surface'
    cohort_counts = Counter(cohorts.values())
    n_test = math.ceil(len(cohorts)*.2)
    targets = {g:n_test*n/len(cohorts) for g,n in cohort_counts.items()}
    allocations = {g:math.floor(n) for g,n in targets.items()}
    for g in sorted(targets,key=lambda g:(-(targets[g]-allocations[g]),g))[:n_test-sum(allocations.values())]:
        allocations[g] += 1
    rng = random.Random(SEED)
    assignments = {}
    for g in sorted(cohort_counts):
        ids = sorted(sid for sid in cohorts if cohorts[sid]==g)
        rng.shuffle(ids)
        for i,sid in enumerate(ids):
            assignments[sid] = 'test' if i<allocations[g] else 'train'
    train_ids = {sid for sid,s in assignments.items() if s=='train'}
    test_ids = {sid for sid,s in assignments.items() if s=='test'}
    assert not train_ids & test_ids
    assert len(test_ids)==n_test and train_ids|test_ids==cohorts.keys()
    # Inventory is built before copying, so unexpected filenames or missing outputs stop the run.
    inventory, excluded = [], []
    for side in ['left','right']:
        folder = ROOT/'SPHARM'/('SPHARM_'+side.upper())/'spharm_results'
        for src in sorted(folder.iterdir()):
            if not src.is_file():
                continue
            match = re.search(r'sub-([^_]+)',src.name)
            if not match:
                excluded.append(dict(source=str(src.relative_to(ROOT)),reason='No subject ID'))
                continue
            sid = match.group(1)
            assert sid in inputs[side] and src.name.startswith(inputs[side][sid]['basename']+'_'), src
            if not inputs[side][sid]['success']:
                excluded.append(dict(source=str(src.relative_to(ROOT)),reason='Failed SPHARM processing'))
                continue
            split = assignments[sid]
            target = DEST/split/('SPHARM_'+side.upper())/'spharm_results'/src.name
            inventory.append((src,target,sid,side,split))
        for sid,item in inputs[side].items():
            if item['success']:
                for suffix in ['_SPHARM.vtk','_SPHARM.coef','_SPHARM_grid.vtk','_processing.json']:
                    assert (folder/(item['basename']+suffix)).is_file(), (side,sid,suffix)
    DEST.mkdir()
    for split in ['train','test']:
        for side in ['left','right']:
            (DEST/split/('SPHARM_'+side.upper())/'spharm_results').mkdir(parents=True)
    copied = []
    for index,(src,target,sid,side,split) in enumerate(inventory,1):
        shutil.copy2(src,target)
        original = src.read_bytes()
        duplicate = target.read_bytes()
        assert original==duplicate, f'Copy verification failed: {target}'
        copied.append(dict(source=str(src.relative_to(ROOT)),destination=str(target.relative_to(DEST)),
                           subject_id=sid,side=side,split=split,bytes=len(original),
                           sha256=hashlib.sha256(original).hexdigest()))
        if index % 1000 == 0:
            print(f'Copied and verified {index}/{len(inventory)} files',flush=True)
    summary = {}
    subject_records = []
    for sid in sorted(assignments):
        subject_records.append(dict(subject_id=sid,split=assignments[sid],inferred_cohort=cohorts[sid],
                                    inferred_person_group='Control' if cohorts[sid]=='Control' else 'TLE',
                                    left_filename_label=labels['left'][sid],right_filename_label=labels['right'][sid],
                                    left_available=inputs['left'][sid]['success'],right_available=inputs['right'][sid]['success']))
    for split in ['train','test']:
        records = [r for r in subject_records if r['split']==split]
        (DEST/split/'subject_ids.txt').write_text('\n'.join(r['subject_id'] for r in records)+'\n',encoding='utf-8')
        write_json(DEST/split/'subjects.json',records)
        summary[split] = dict(subjects=len(records),inferred_cohorts=dict(Counter(r['inferred_cohort'] for r in records)),
                              inferred_person_groups=dict(Counter(r['inferred_person_group'] for r in records)),
                              sides={})
        for side in ['left','right']:
            eligible = [r for r in records if r[side+'_available']]
            summary[split]['sides'][side] = dict(valid_surfaces=len(eligible),
                filename_labels=dict(Counter(r[side+'_filename_label'] for r in eligible)),
                inferred_person_groups=dict(Counter(r['inferred_person_group'] for r in eligible)))
            folder = DEST/split/('SPHARM_'+side.upper())/'spharm_results'
            assert len(list(folder.glob('*_SPHARM.vtk')))==len(eligible)
            assert len(list(folder.glob('*_SPHARM.coef')))==len(eligible)
    assert sum(summary[s]['sides']['left']['valid_surfaces'] for s in summary)==373
    assert sum(summary[s]['sides']['right']['valid_surfaces'] for s in summary)==377
    preprocessing = {}
    for side in ['left','right']:
        status = ROOT/'ICP'/('output_'+side+'_hippocampus')/'icp_status.json'
        contract = ROOT/'ICP'/('output_'+side+'_hippocampus')/'mean_shape.ply.json'
        s, c = json.loads(status.read_text()),json.loads(contract.read_text())
        preprocessing[side] = dict(mode=s.get('mode'),reference_sha256=s.get('reference_sha256'),
                                   reference_input_count=len(c.get('training_inputs',[])))
    metadata = dict(seed=SEED,requested_train_fraction=.8,requested_test_fraction=.2,
                    actual_train_fraction=len(train_ids)/len(cohorts),actual_test_fraction=len(test_ids)/len(cohorts),
                    method='Subject-level random split, stratified by inferred Control/Left_TLE/Right_TLE cohorts; test size rounded up.',
                    identity_assumption='Exact sub-ID matches refer to the same person across sides.',
                    grouping_assumption='Control = Healthy on both sides; TLE = TLE on either side; not clinically independently confirmed.',
                    source='SPHARM/SPHARM_LEFT and SPHARM/SPHARM_RIGHT',summary=summary,
                    copied_file_count=len(copied),copied_bytes=sum(f['bytes'] for f in copied),excluded_files=excluded,
                    verification=dict(subject_overlap=0,all_copied_files_byte_identical=True,
                                      all_successful_surfaces_included=True,original_sources_preserved=True),
                    existing_preprocessing=preprocessing,
                    evaluation_limitation='This splits existing processed data only. The existing ICP references were fit on all 381 subjects. For fully held-out evaluation, fit data-dependent preprocessing on train only and transform test with the frozen train reference.')
    write_json(DEST/'split_summary.json',metadata)
    write_json(DEST/'split_manifest.json',subject_records)
    write_json(DEST/'file_manifest.json',copied)
    lines = ['# SPHARM train/test split','',
             'Seed: 42. Split unit: person ID, shared across both hemispheres. Stratified by the inferred Control, Left_TLE and Right_TLE cohorts.',
             'Train: 304 persons (79.79%). Test: 77 persons (20.21%). Rounding is necessary for 381 persons.','',
             '| Split | Persons | Control | Left TLE | Right TLE | Left surfaces | Right surfaces |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for split in ['train','test']:
        s = summary[split]
        lines.append(f"| {split} | {s['subjects']} | {s['inferred_cohorts']['Control']} | {s['inferred_cohorts']['Left_TLE']} | {s['inferred_cohorts']['Right_TLE']} | {s['sides']['left']['valid_surfaces']} | {s['sides']['right']['valid_surfaces']} |")
    lines += ['', '## Folder structure','', '```text', 'SplitData/', '  train/', '    SPHARM_LEFT/spharm_results/',
              '    SPHARM_RIGHT/spharm_results/', '    subjects.json', '    subject_ids.txt',
              '  test/', '    SPHARM_LEFT/spharm_results/', '    SPHARM_RIGHT/spharm_results/',
              '    subjects.json', '    subject_ids.txt', '  split_manifest.json', '  file_manifest.json', '  split_summary.json','```','',
              'All subject-specific files from successful SPHARM runs are copied, retaining their original names. Failed-run residual files are excluded and listed in split_summary.json. Missing hemispheres are recorded in subjects.json.', '',
              '## Labels and validation','',
              'Control/TLE person groups and TLE laterality are inferred from paired filename labels. A Healthy hemisphere may belong to a person whose opposite hemisphere is labelled TLE. Both the original hemisphere labels and inferred person labels are recorded in the manifest.', '',
              f"Copied and verified {len(copied)} files by byte comparison; SHA-256 checksums are recorded. All 750 successful subject-side surfaces are included exactly once. Train/test ID overlap is zero.", '',
              '## Existing preprocessing','',
              'This split uses existing ICP/SPHARM outputs. The existing ICP references were fit using all 381 subjects, including IDs now assigned to test. This split prevents subject overlap but does not undo reference-fitting leakage. For a fully held-out evaluation, fit data-dependent preprocessing on train only, then transform test using the frozen training reference.', '',
              'Creation script: Analysis/spharm_volume/split_spharm_data.py. The script refuses to overwrite an existing SplitData folder.', '']
    (DEST/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':
    main()
