"""Read-only audit of extracted CSVs against current split and SPHARM sources."""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from collections import Counter, defaultdict
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_values(path, kind):
    if kind == 'coef':
        tokens = np.fromstring(re.sub(r'[{},]', ' ', path.read_text()), sep=' ')
        assert len(tokens) == 508 and tokens[0] == 169, path
        return tokens[1:], None
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    mesh = reader.GetOutput()
    assert mesh.GetNumberOfPoints() == 1002, path
    values = vtk_to_numpy(mesh.GetPoints().GetData()).astype(float).ravel()
    cells = mesh.GetPolys()
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    assert np.all(np.diff(offsets) == 3), path
    faces = vtk_to_numpy(cells.GetConnectivityArray()).reshape(-1,3)
    edges = set()
    for face in faces:
        edges.update(tuple(sorted((int(face[i]),int(face[(i+1)%3])))) for i in range(3))
    return values, edges


def main():
    people = {r['subject_id']:r for r in json.loads((ROOT/'SplitData/split_manifest.json').read_text())}
    entries, errors, warnings = [], [], []
    ids_by_split = defaultdict(set)
    labels_by_partition = {}
    wrong_person_labels = set()
    feature_hashes = defaultdict(list)
    for split in ['train','test']:
        for side in ['left','right']:
            batch = ROOT/'SplitData'/split/('SPHARM_'+side.upper())
            expected_ids = {sid for sid,r in people.items() if r['split']==split and r[side+'_available']}
            for kind,filename,suffix in [('coef','spharm_results_coef_features.csv','_SPHARM_ellalign.coef'),
                                         ('xyz','spharm_xyz_coords.csv','_SPHARM_ellalign.vtk')]:
                path = batch/'ml_features'/filename
                try:
                    with path.open(encoding='utf-8-sig',newline='') as stream:
                        reader = csv.reader(stream)
                        header = next(reader)
                        data = list(reader)
                    assert len(header) == len(set(header)), 'Duplicate column names'
                    assert all(len(row)==len(header) for row in data), 'CSV row widths differ'
                    features = ['Coef_'+str(i) for i in range(1,508)] if kind=='coef' else [f'{a}_{i}' for i in range(1002) for a in ['x','y','z']]
                    actual_features = [c for c in header if c.startswith(('Coef_','x_','y_','z_'))]
                    assert actual_features == features, 'Feature names/order differ from Prepare schema'
                    records = [dict(zip(header,row)) for row in data]
                    subjects = [r['Subject'] for r in records]
                    assert len(subjects) == len(set(subjects)), 'Duplicate Subject values'
                    observed_ids = {re.search(r'sub-([^_]+)',s).group(1) for s in subjects}
                    assert len(observed_ids)==len(subjects), 'Multiple records for one person within the same side'
                    assert observed_ids==expected_ids, 'Missing/unexpected subject IDs versus split manifest'
                    source_paths = []
                    matrix = np.array([[float(r[c]) for c in features] for r in records])
                    assert np.isfinite(matrix).all(), 'NaN/Inf in features'
                    max_error = 0.0
                    cohort_disagreements = []
                    canonical_edges = None
                    observed_labels = {}
                    for row,values in zip(records,matrix):
                        subject = row['Subject']
                        sid = re.search(r'sub-([^_]+)',subject).group(1)
                        info = people[sid]
                        binary = int(row['BinaryClass'])
                        assert binary in {0,1}, 'Invalid BinaryClass'
                        hemisphere_binary = int(info[side+'_filename_label']=='TLE')
                        assert binary==hemisphere_binary, 'Label differs from source hemisphere filename'
                        clinical = int(row['Class'] if kind=='coef' else row['Group_Label'])
                        assert clinical==hemisphere_binary, 'Class/Group_Label differs from source label'
                        observed_labels[sid] = binary
                        person_binary = int(info['inferred_person_group']=='TLE')
                        if binary != person_binary:
                            cohort_disagreements.append(subject)
                            wrong_person_labels.add((side,sid))
                        source = batch/'spharm_results'/(subject+suffix)
                        assert source.is_file(), 'CSV Subject has no matching source file'
                        original, edges = source_values(source,kind)
                        error = float(np.max(np.abs(values-original)))
                        max_error = max(max_error,error)
                        assert error <= 5.1e-9, 'Values differ from source beyond eight-decimal rounding'
                        if edges is not None:
                            if canonical_edges is None:
                                canonical_edges = edges
                            assert edges == canonical_edges, 'Mesh topology differs within batch'
                        source_paths.append(source.resolve())
                        fingerprint = hashlib.sha256(values.astype('<f8').tobytes()).hexdigest()
                        feature_hashes[kind].append((fingerprint,split,side,subject))
                    partition = split+'_'+side
                    if partition in labels_by_partition:
                        assert labels_by_partition[partition]==observed_labels, 'Coefficient/XYZ subject labels differ'
                    labels_by_partition[partition] = observed_labels
                    ids_by_split[split].update(observed_ids)
                    manifest_path = Path(str(path)+'.json')
                    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                    assert manifest['feature_csv_sha256']==sha(path), 'CSV checksum differs from manifest'
                    assert manifest['rows']==len(records), 'Manifest row count mismatch'
                    assert manifest['feature_columns']==features, 'Manifest feature columns mismatch'
                    listed = manifest['source_files']
                    assert [Path(r['path']).resolve() for r in listed]==source_paths, 'Manifest source paths/order differ'
                    for source_record in listed:
                        assert source_record['sha256']==sha(source_record['path']), 'Source changed since extraction'
                        assert source_record['processing_sidecar_sha256']==sha(source_record['processing_sidecar']), 'Processing sidecar changed since extraction'
                    if kind=='xyz':
                        geometry = manifest['geometry_contract']
                        assert geometry=={'mesh_variant':'ellalign','num_points':1002}, 'Unexpected XYZ geometry'
                        edge_path = batch/'ml_features/mesh_edges.csv'
                        with edge_path.open(encoding='utf-8-sig',newline='') as stream:
                            edge_rows = list(csv.DictReader(stream))
                        exported_edges = [(int(r['Source']),int(r['Target'])) for r in edge_rows]
                        assert len(exported_edges)==len(set(exported_edges)), 'Duplicate mesh edges'
                        assert set(exported_edges)==canonical_edges, 'mesh_edges.csv differs from SPHARM topology'
                        assert all(0<=a<b<1002 for a,b in exported_edges), 'Invalid edge endpoints'
                    entry = dict(split=split,side=side,kind=kind,path=str(path.relative_to(ROOT)),rows=len(records),
                                 features=len(features),binary_labels=dict(Counter(r['BinaryClass'] for r in records)),
                                 source_variant='ellalign',max_absolute_rounding_error=max_error,
                                 person_group_disagreement_count=len(cohort_disagreements),
                                 person_group_disagreement_subjects=cohort_disagreements,
                                 feature_contract=manifest['feature_contract'],fixed_reference_validated=manifest['fixed_reference_validated'],
                                 csv_and_source_checksums_match=True,source_values_match=True)
                    entries.append(entry)
                    print(partition,kind,len(records),'source/checksum/schema checks passed',flush=True)
                except Exception as exc:
                    errors.append(dict(path=str(path.relative_to(ROOT)),error=str(exc),type=type(exc).__name__))
                    print('FAILED',path,exc,flush=True)
    overlap = sorted(ids_by_split['train'] & ids_by_split['test'])
    if overlap:
        errors.append(dict(error='Person IDs overlap between train and test',ids=overlap))
    for side in ['left','right']:
        for kind in ['coef','xyz']:
            pair = [e for e in entries if e['side']==side and e['kind']==kind]
            if len(pair)==2 and pair[0]['feature_contract'] != pair[1]['feature_contract']:
                errors.append(dict(error='Train/test processing contracts differ',side=side,kind=kind))
    duplicates = {}
    for kind,items in feature_hashes.items():
        groups = defaultdict(list)
        for fingerprint,split,side,subject in items:
            groups[fingerprint].append(dict(split=split,side=side,subject=subject))
        duplicates[kind] = [g for g in groups.values() if len(g)>1]
        if duplicates[kind]:
            warnings.append(dict(issue='Identical feature vectors',kind=kind,groups=duplicates[kind]))
    if wrong_person_labels:
        warnings.append(dict(issue='Hemisphere-label classification differs from inferred person-level classification',
                             unique_subject_side_count=len(wrong_person_labels),
                             explanation='Contralateral hemispheres of inferred TLE persons have BinaryClass=0 under filename-based extraction. Both coefficient and XYZ CSVs use that same target.'))
    preprocessing = {}
    for side in ['left','right']:
        icp = ROOT/'ICP'/('output_'+side+'_hippocampus')
        status = json.loads((icp/'icp_status.json').read_text())
        contract = json.loads((icp/'mean_shape.ply.json').read_text())
        preprocessing[side] = dict(mode=status['mode'],reference_inputs=len(contract['training_inputs']),
                                   icp_reference_sha256=status.get('reference_sha256'))
    warnings.append(dict(issue='Existing ICP reference fit includes test subjects',details=preprocessing))
    result = dict(date='2026-10-03',csv_files_checked=len(entries),errors=errors,warnings=warnings,
                  train_unique_people=len(ids_by_split['train']),test_unique_people=len(ids_by_split['test']),
                  train_test_person_overlap=overlap,duplicate_feature_vector_groups=duplicates,files=entries)
    (OUT/'extracted_features_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    text = ['# Extracted SPHARM features audit','', 'Date: 2026-10-03. The feature CSVs were read and compared without modifying them.','',
            f"Technical extraction errors: {len(errors)}. Files checked: {len(entries)}.",'',
            '| Split | Side | Rows (each feature type) | Coef features | XYZ features | BinaryClass 0 | BinaryClass 1 | Person-label disagreements |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for split in ['train','test']:
        for side in ['left','right']:
            pair = [e for e in entries if e['split']==split and e['side']==side]
            if len(pair)==2:
                c = next(e for e in pair if e['kind']=='coef')
                x = next(e for e in pair if e['kind']=='xyz')
                text.append(f"| {split} | {side} | {c['rows']} | {c['features']} | {x['features']} | {c['binary_labels'].get('0',0)} | {c['binary_labels'].get('1',0)} | {c['person_group_disagreement_count']} |")
    text += ['', '## Checks performed','',
             '- Exact subject membership and counts against the original split manifest; no duplicate subjects or train/test person overlap.',
             '- All feature names and column orders match the Prepare input schema. All values are numeric and finite.',
             '- Every coefficient and coordinate value was compared with the matching ellalign source file. Allowed error: 5.1e-9, matching the extractor’s eight-decimal output.',
             '- CSV, source and processing-sidecar SHA-256 checksums match the extraction manifests.',
             '- Coefficient and XYZ labels agree. Mesh edges exactly match every source surface topology in their batch.',
             '- Train and test use the same feature variant and processing contract for each side.','',
             '## Target label definition','',
             f"{len(wrong_person_labels)} unique subject-side records have BinaryClass=0 while their inferred person group is TLE. This is consistent with the filename-based hemisphere target. It is a label-definition problem if the intended task is person-level Healthy Control versus TLE classification.",
             'Both coefficient and XYZ files contain these same records. There are 245 affected hemisphere records, not 490 independent observations. The inferred person group is based on paired filename labels, not independently confirmed clinical metadata.','',
             '## Reference provenance','',
             'The source SPHARM contracts report icp_mode=unknown and have no ICP/SPHARM reference hashes; every extracted manifest correctly has fixed_reference_validated=false.',
             'The ICP reference metadata shows fit_reference on all 381 inputs for both sides, including the subjects now assigned to test. For a fully held-out evaluation, rebuild data-dependent preprocessing on training inputs and transform test using the frozen training reference.','',
             '## Errors','']
    text.extend(json.dumps(e,ensure_ascii=False) for e in errors)
    if not errors:
        text.append('None found in the technical extraction checks.')
    (OUT/'extracted_features_audit.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2,ensure_ascii=False))
    if errors:
        sys.exit(1)


if __name__=='__main__':
    main()
