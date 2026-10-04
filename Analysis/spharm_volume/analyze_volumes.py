"""Explore existing SPHARM surfaces; preserve all source data and pipeline code."""
from pathlib import Path
import gzip
import hashlib
import json
import re
import struct
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'merged_ds005602_ds004469_spharm_ready'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_volume(path):
    with gzip.open(path, 'rb') as stream:
        raw = stream.read()
    endian = '<' if struct.unpack('<i', raw[:4])[0] == 348 else '>'
    assert struct.unpack(endian + 'i', raw[:4])[0] == 348, path
    dims = struct.unpack(endian + '8h', raw[40:56])
    datatype = struct.unpack(endian + 'h', raw[70:72])[0]
    dtype = {2:'u1',4:'i2',8:'i4',16:'f4',64:'f8',256:'i1',512:'u2',768:'u4'}[datatype]
    pixdim = struct.unpack(endian + '8f', raw[76:108])
    offset = int(struct.unpack(endian + 'f', raw[108:112])[0])
    units_code = raw[123] & 7
    assert units_code in {0,2}, 'Unexpected explicit spatial units; conversion required'
    data = np.frombuffer(raw, dtype=np.dtype(endian + dtype), count=int(np.prod(dims[1:1+dims[0]])), offset=offset)
    assert np.all((data == 0) | (data == 1)), 'Expected binary hippocampus mask'
    voxel_mm3 = float(np.prod(np.abs(pixdim[1:4])))
    if struct.unpack(endian + 'h', raw[254:256])[0] > 0:
        affine = np.array(struct.unpack(endian + '12f', raw[280:328])).reshape(3,4)
        assert np.isclose(abs(np.linalg.det(affine[:,:3])), voxel_mm3, rtol=1e-5)
    return float(np.count_nonzero(data) * voxel_mm3), units_code


def mesh_volume(path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(reader.GetOutput())
    tri.Update()
    poly = tri.GetOutput()
    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    assert np.all(np.isfinite(points)) and poly.GetNumberOfPolys() > 0, path
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(poly)
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    bad_edges = edges.GetOutput().GetNumberOfCells()
    cells = poly.GetPolys()
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    connectivity = vtk_to_numpy(cells.GetConnectivityArray())
    assert np.all(np.diff(offsets) == 3)
    faces = connectivity.reshape(-1,3)
    centered = points - points.mean(axis=0)
    tetra = abs(float(np.einsum('ij,ij->i', centered[faces[:,0]], np.cross(centered[faces[:,1]], centered[faces[:,2]])).sum() / 6))
    mass = vtk.vtkMassProperties()
    mass.SetInputData(poly)
    mass.Update()
    volume = float(mass.GetVolume())
    assert volume > 0 and np.isclose(volume, tetra, rtol=1e-5, atol=1e-10), path
    assert bad_edges == 0, f'Open/nonmanifold surface: {path}'
    return volume, len(points), len(faces)


def describe(values):
    values = np.asarray(values)
    q1, median, q3 = np.quantile(values, [.25,.5,.75])
    return dict(n=len(values), mean=float(values.mean()), sd=float(values.std(ddof=1)),
                median=float(median), q1=float(q1), q3=float(q3), min=float(values.min()), max=float(values.max()),
                iqr_outliers=int(((values < q1-1.5*(q3-q1)) | (values > q3+1.5*(q3-q1))).sum()))


def compare(healthy, tle):
    h, t = np.asarray(healthy), np.asarray(tle)
    difference = float(t.mean()-h.mean())
    vh, vt = h.var(ddof=1)/len(h), t.var(ddof=1)/len(t)
    se = np.sqrt(vh+vt)
    df = (vh+vt)**2 / (vh**2/(len(h)-1)+vt**2/(len(t)-1))
    ci = stats.t.ppf(.975,df)*se
    pooled = np.sqrt(((len(h)-1)*h.var(ddof=1)+(len(t)-1)*t.var(ddof=1))/(len(h)+len(t)-2))
    return dict(mean_difference_tle_minus_healthy_mm3=difference,
                percent_difference=float(100*difference/h.mean()), ci95=[float(difference-ci),float(difference+ci)],
                welch_p=float(stats.ttest_ind(t,h,equal_var=False).pvalue),
                mann_whitney_p=float(stats.mannwhitneyu(t,h,alternative='two-sided').pvalue),
                hedges_g=float((1-3/(4*(len(h)+len(t))-9))*difference/pooled))


def main():
    rows, audit, summary, comparisons = [], {}, {}, {}
    for side in ['left','right']:
        folder = ROOT / 'SPHARM' / ('SPHARM_'+side.upper())
        icp = ROOT / 'ICP' / ('output_'+side+'_hippocampus')
        contract = json.loads((icp / 'mean_shape.ply.json').read_text())
        assert sha(icp/'mean_shape.ply') == contract['template_sha256']
        scale = contract['physical_to_normalized_scale']
        transforms = np.load(icp/'T_matrices.npy')
        singular = np.linalg.svd(transforms[:,:3,:3], compute_uv=False)
        assert np.allclose(singular, scale, rtol=1e-5), 'Nonuniform/different scale'
        assert np.all(np.linalg.det(transforms[:,:3,:3]) > 0)
        inputs = {}
        for status_path in sorted(folder.glob('spharm_status_shard*.json')):
            for item in json.loads(status_path.read_text())['subjects']:
                name = Path(item['input']).name.replace('.nii.gz','')
                assert name not in inputs, name
                inputs[name] = item
        side_rows, failures, missing = [], [], []
        for name, item in sorted(inputs.items()):
            group = re.search(r'_(Healthy|TLE)_', name).group(1)
            surface = folder/'spharm_results'/(name+'_SPHARM.vtk')
            if not item['success']:
                failures.append(dict(input=name,group=group,surface_exists=surface.exists()))
                continue
            if not surface.exists():
                missing.append(name)
                continue
            provenance = json.loads((folder/'spharm_results'/(name+'_processing.json')).read_text())
            aligned = icp/'aligned_nifti'/(name+'.nii.gz')
            assert sha(aligned) == provenance['input_sha256'], 'Source provenance mismatch'
            original = SOURCE/(side+'_hippocampus')/(name.replace('_aligned','')+'.nii.gz')
            normalized, points, triangles = mesh_volume(surface)
            mm3 = normalized / scale**3
            raw_mm3, units_code = mask_volume(original)
            row = dict(side=side,group=group,subject_id=re.search(r'sub-([^_]+)',name).group(1),
                       spharm_volume_mm3=mm3,original_mask_volume_mm3=raw_mm3,
                       normalized_mesh_volume=normalized,physical_to_normalized_scale=scale,
                       mesh_change_percent=100*(mm3/raw_mm3-1),points=points,triangles=triangles,
                       nifti_spatial_units_code=units_code,
                       surface=str(surface.relative_to(ROOT)),original_mask=str(original.relative_to(ROOT.parent)))
            rows.append(row)
            side_rows.append(row)
            if len(side_rows) % 50 == 0:
                print(side, len(side_rows), 'surfaces verified', flush=True)
        expected = {p.name for p in (folder/'spharm_results').glob('*_SPHARM.vtk')}
        analyzed = {Path(r['surface']).name for r in side_rows}
        audit[side] = dict(input_count=len(inputs),eligible_count=len(side_rows),failure_count=len(failures),
                           failures=failures,missing_successful_surfaces=missing,
                           unanalysed_surfaces=sorted(expected-analyzed),scale=scale,
                           all_transforms_share_scale=True,all_source_hashes_match=True,
                           all_surfaces_closed_and_tetra_volume_verified=True)
        summary[side] = {}
        for group in ['Healthy','TLE']:
            selected = [r for r in side_rows if r['group']==group]
            summary[side][group] = dict(spharm=describe([r['spharm_volume_mm3'] for r in selected]),
                                       original_mask=describe([r['original_mask_volume_mm3'] for r in selected]),
                                       mesh_change_percent=describe([r['mesh_change_percent'] for r in selected]),
                                       input_count=sum(('_'+group+'_') in n for n in inputs))
        comparisons[side] = {}
        for metric in ['spharm_volume_mm3','original_mask_volume_mm3']:
            comparisons[side][metric] = compare([r[metric] for r in side_rows if r['group']=='Healthy'],
                                                [r[metric] for r in side_rows if r['group']=='TLE'])
        print(side, len(side_rows), 'completed', flush=True)
    for metric in ['spharm_volume_mm3','original_mask_volume_mm3']:
        ordered = sorted(['left','right'], key=lambda s:comparisons[s][metric]['welch_p'])
        last = 0
        for rank,side in enumerate(ordered):
            last = max(last,min(1,(2-rank)*comparisons[side][metric]['welch_p']))
            comparisons[side][metric]['welch_p_holm_two_sides'] = last
    left = {r['subject_id']:r['group'] for r in rows if r['side']=='left'}
    right = {r['subject_id']:r['group'] for r in rows if r['side']=='right'}
    shared = left.keys() & right.keys()
    overlap = dict(shared_ids=len(shared),same_group=sum(left[s]==right[s] for s in shared),
                   different_group_ids=sorted(s for s in shared if left[s]!=right[s]),
                   note='Exact filename IDs only; no cross-dataset identity or clinical laterality metadata supplied.')
    result = dict(date='2026-10-03',summary=summary,comparisons=comparisons,audit=audit,id_overlap=overlap)
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (OUT/'subject_volumes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    report = ['# SPHARM volume exploration', '', 'Analysis date: 2026-10-03. Groups follow the filename labels Healthy and TLE.', '',
              'Unit assumption: original NIfTI masks omit spatial units (code 0). Values below assume the coordinate units are millimetres, following the ICP pipeline convention. Percent differences do not require this assumption.', '',
              '## SPHARM surface volume (mm³, assuming mm coordinates)', '', '| Side | Group | Inputs | Valid surfaces | Mean ± SD | Median | Q1–Q3 | Min–max |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for side in ['left','right']:
        for group in ['Healthy','TLE']:
            s = summary[side][group]['spharm']
            report.append(f"| {side} | {group} | {summary[side][group]['input_count']} | {s['n']} | {s['mean']:.2f} ± {s['sd']:.2f} | {s['median']:.2f} | {s['q1']:.2f}–{s['q3']:.2f} | {s['min']:.2f}–{s['max']:.2f} |")
    report += ['', '## Group comparisons', '', '| Side | TLE − Healthy (mm³) | Change vs Healthy | 95% CI of difference | Welch p | Holm p (2 sides) | Hedges g |',
               '|---|---:|---:|---:|---:|---:|---:|']
    for side in ['left','right']:
        c = comparisons[side]['spharm_volume_mm3']
        report.append(f"| {side} | {c['mean_difference_tle_minus_healthy_mm3']:.2f} | {c['percent_difference']:.2f}% | {c['ci95'][0]:.2f} to {c['ci95'][1]:.2f} | {c['welch_p']:.4g} | {c['welch_p_holm_two_sides']:.4g} | {c['hedges_g']:.3f} |")
    report += ['', '## Cross-check against original binary masks', '', '| Side | Group | Original mask mean ± SD (mm³) | Mean individual SPHARM change |', '|---|---|---:|---:|']
    for side in ['left','right']:
        for group in ['Healthy','TLE']:
            s = summary[side][group]
            report.append(f"| {side} | {group} | {s['original_mask']['mean']:.2f} ± {s['original_mask']['sd']:.2f} | {s['mesh_change_percent']['mean']:.2f}% |")
    report += ['', '## Method and limitations', '',
               '- One native `_SPHARM.vtk` per successful subject/side; grid, para, ellalign, medial-axis and template surfaces are not additional subjects.',
               '- All included surfaces are finite, closed, and have no boundary/nonmanifold edges. VTK mass-properties volume agrees with an independent signed-tetrahedron sum.',
               '- Physical volume = normalized surface volume / scale³. The ICP metadata scale matches every saved transform; template hashes and every SPHARM input hash were checked.',
               '- Original-mask volume = count of label-1 voxels × voxel volume; affine voxel volume agrees with pixdim. Original headers have unspecified spatial units (code 0), so reported mm³ assumes millimetres as used by the ICP pipeline.',
               '- Surface processing includes resampling, label cleanup, and harmonic approximation; SPHARM volume is a processed-surface estimate, distinct from original segmentation volume.',
               '- All valid samples and IQR outliers are retained. Summary and tests use the available sample for each side separately.',
               '- Welch tests and confidence intervals compare independent filename groups within each side. Holm correction is across the two sides; original-mask comparisons are a separate sensitivity analysis.',
               '- No adjustment for age, sex, intracranial volume, scan site, or dataset. This is a descriptive group comparison, not a diagnostic threshold or causal inference.',
               f"- Exact IDs shared across valid sides: {overlap['shared_ids']}; matching group labels: {overlap['same_group']}; different labels: {len(overlap['different_group_ids'])}. Side-specific labels must not be treated as a person-level diagnosis without metadata.", '',
               '## Failed processing inputs', '']
    for side in ['left','right']:
        report.append(f"{side}: {audit[side]['failure_count']} failed of {audit[side]['input_count']} inputs.")
        report.extend('- '+r['input'] for r in audit[side]['failures'])
        if audit[side]['missing_successful_surfaces'] or audit[side]['unanalysed_surfaces']:
            report.append('See results.json for missing/unanalysed surfaces.')
        report.append('')
    (OUT/'report.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
