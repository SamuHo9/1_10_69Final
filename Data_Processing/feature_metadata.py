"""Labels and processing provenance shared by the SPHARM feature extractors.

This is a project module, not an external pip dependency. Unknown ICP provenance
is retained as unknown; matching sidecars do not establish a fixed reference.
"""
import csv
import hashlib
import json
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_labels(label_path):
    if label_path is None:
        return None, None
    with Path(label_path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not {'Subject', 'BinaryClass'}.issubset(fields):
            raise ValueError('Labels CSV requires Subject and BinaryClass columns')
        rows = list(reader)
    seen = set()
    for row in rows:
        subject = (row.get('Subject') or '').strip()
        if not subject or subject in seen:
            raise ValueError('Labels CSV contains an empty or duplicate Subject: ' + repr(subject))
        row['Subject'] = subject
        seen.add(subject)
    if not rows:
        raise ValueError('Labels CSV contains no subjects')
    return fields, rows


def _integer_label(value, field, subject, allowed):
    try:
        number = float(value)
        result = int(number)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'Invalid {field} for {subject}: {value!r}') from None
    if number != result or result not in allowed:
        raise ValueError(f'Invalid {field} for {subject}: {value!r}')
    return result


def load_labels(label_path):
    """Return Subject -> (Group, Class, BinaryClass), or None without an override."""
    fields, rows = _read_labels(label_path)
    if rows is None:
        return None
    labels = {}
    for row in rows:
        subject = row['Subject']
        binary = _integer_label(row.get('BinaryClass'), 'BinaryClass', subject, {0, 1})
        class_value = row.get('Class')
        clinical_class = binary if class_value is None or not class_value.strip() else _integer_label(
            class_value, 'Class', subject, {0, 1, 2})
        # Class 2 denotes a contralateral hemisphere, represented by binary 0.
        if int(clinical_class == 1) != binary:
            raise ValueError(f'Class and BinaryClass disagree for {subject}')
        group = (row.get('Group') or '').strip()
        if not group:
            group = {0: 'Healthy', 1: 'TLE', 2: 'Contralateral TLE (Healthy-side)'}[clinical_class]
        labels[subject] = (group, clinical_class, binary)
    return labels


def load_dataset_labels(label_path):
    """Read optional, explicit Dataset membership without inferring it from an ID."""
    fields, rows = _read_labels(label_path)
    if rows is None or 'Dataset' not in fields:
        return None
    datasets = {}
    for row in rows:
        dataset = (row.get('Dataset') or '').strip()
        if not dataset:
            raise ValueError(f'Missing Dataset for {row["Subject"]}')
        datasets[row['Subject']] = dataset
    return datasets


def _processing_path(source):
    source = Path(source)
    for suffix in ('_SPHARM_ellalign.coef', '_SPHARM.coef',
                   '_SPHARM_ellalign.vtk', '_SPHARM_realigned.vtk',
                   '_SPHARM_procalign.vtk', '_SPHARM.vtk'):
        if source.name.endswith(suffix):
            return source.with_name(source.name[:-len(suffix)] + '_processing.json')
    raise ValueError(f'Unsupported SPHARM source name for processing provenance: {source}')


def _validate_contracts(source_files):
    sources = list(source_files)
    if not sources:
        raise ValueError('Cannot validate processing provenance for an empty batch')
    expected = None
    for source in sources:
        sidecar = _processing_path(source)
        if not sidecar.is_file():
            raise ValueError(f'Missing processing sidecar: {sidecar}')
        try:
            payload = json.loads(sidecar.read_text(encoding='utf-8-sig'))
            contract = payload.get('feature_contract') if isinstance(payload, dict) else None
        except (OSError, ValueError) as exc:
            raise ValueError(f'Invalid processing sidecar {sidecar}: {exc}') from exc
        if not isinstance(contract, dict) or contract.get('version') != 'spharm-feature-v1':
            raise ValueError(f'Missing or unsupported feature_contract in {sidecar}')
        if contract.get('icp_mode') == 'fixed_reference':
            if not contract.get('icp_geometry_version'):
                raise ValueError(f'Missing fixed-reference geometry version in {sidecar}')
            for field in ('icp_reference_sha256', 'spharm_template_sha256'):
                value = contract.get(field)
                if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdefABCDEF' for c in value):
                    raise ValueError(f'Missing or invalid {field} in {sidecar}')
        if expected is None:
            expected = contract
        elif contract != expected:
            differing = sorted(k for k in set(expected) | set(contract) if expected.get(k) != contract.get(k))
            raise ValueError(f'Mismatched processing contracts in {sidecar}; fields: {", ".join(differing)}')
    return dict(expected)


def validate_feature_contracts(coef_files):
    """Require present, compatible processing sidecars for a coefficient batch."""
    return _validate_contracts(coef_files)


def validate_mesh_processing_contracts(mesh_files):
    """Use the same subject sidecars for the supported mesh variants."""
    return _validate_contracts(mesh_files)


def _write_manifest(csv_path, source_files, label_path, contract, kind, geometry=None):
    csv_path = Path(csv_path)
    sources = list(source_files)
    with csv_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        row_count = sum(1 for _ in reader)
    if row_count != len(sources):
        raise ValueError(f'Feature CSV/source count mismatch: {row_count} rows versus {len(sources)} files')
    entries = []
    for source in sources:
        source = Path(source)
        entry = {'path': str(source.resolve()), 'sha256': _sha256(source)}
        sidecar = _processing_path(source)
        if sidecar.is_file():
            entry.update(processing_sidecar=str(sidecar.resolve()), processing_sidecar_sha256=_sha256(sidecar))
        entries.append(entry)
    feature_contract = dict(contract) if contract is not None else None
    fixed = bool(feature_contract and feature_contract.get('icp_mode') == 'fixed_reference'
                 and feature_contract.get('icp_geometry_version')
                 and feature_contract.get('icp_reference_sha256')
                 and feature_contract.get('spharm_template_sha256')
                 and feature_contract.get('provenance_status') != 'missing')
    payload = {
        'version': 'spharm-feature-manifest-v1', 'feature_kind': kind,
        'feature_csv': str(csv_path.resolve()), 'feature_csv_sha256': _sha256(csv_path),
        'rows': row_count, 'feature_columns': [c for c in columns if c.startswith(('Coef_', 'x_', 'y_', 'z_'))],
        'feature_contract': feature_contract, 'source_files': entries,
        'labels_csv': str(Path(label_path).resolve()) if label_path is not None else None,
        'labels_sha256': _sha256(label_path) if label_path is not None else None,
        'processing_provenance_status': 'sidecars_checked' if feature_contract and feature_contract.get('provenance_status') != 'missing' else 'missing',
        'fixed_reference_validated': fixed,
    }
    if geometry is not None:
        payload['geometry_contract'] = geometry
    output = Path(str(csv_path) + '.json')
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return payload


def write_feature_manifest(csv_path, coef_files, label_path=None, contract=None):
    return _write_manifest(csv_path, coef_files, label_path, contract, 'coef')


def write_geometry_manifest(csv_path, mesh_files, mesh_variant, num_points,
                            label_path=None, processing_contract=None):
    return _write_manifest(csv_path, mesh_files, label_path, processing_contract, 'xyz',
                           {'mesh_variant': mesh_variant, 'num_points': int(num_points)})
