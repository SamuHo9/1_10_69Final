"""Separate filename hemisphere labels from inferred person-level cohorts."""
from pathlib import Path
import json
import re
from analyze_volumes import describe, compare, ROOT, OUT


def main():
    rows = json.loads((OUT/'subject_volumes.json').read_text())
    labels = {}
    for side in ['left','right']:
        labels[side] = {}
        for path in (ROOT/'SPHARM'/('SPHARM_'+side.upper())).glob('spharm_status_shard*.json'):
            for item in json.loads(path.read_text())['subjects']:
                sid = re.search(r'sub-([^_]+)', item['input']).group(1)
                labels[side][sid] = re.search(r'_(Healthy|TLE)_', item['input']).group(1)
    assert labels['left'].keys() == labels['right'].keys()
    cohorts = {}
    for sid in labels['left']:
        pair = (labels['left'][sid],labels['right'][sid])
        assert pair in {('Healthy','Healthy'),('TLE','Healthy'),('Healthy','TLE')}
        cohorts[sid] = 'Control' if pair == ('Healthy','Healthy') else ('Left_TLE' if pair[0]=='TLE' else 'Right_TLE')
    summary, comparisons = {}, {}
    for row in rows:
        row['inferred_cohort'] = cohorts[row['subject_id']]
        row['inferred_person_group'] = 'Control' if row['inferred_cohort']=='Control' else 'TLE'
    for side in ['left','right']:
        summary[side], comparisons[side] = {}, {}
        for group in ['Control','TLE','Left_TLE','Right_TLE']:
            selected = [r for r in rows if r['side']==side and (r['inferred_person_group']==group or r['inferred_cohort']==group)]
            summary[side][group] = dict(spharm=describe([r['spharm_volume_mm3'] for r in selected]),
                                        original_mask=describe([r['original_mask_volume_mm3'] for r in selected]))
        for metric in ['spharm_volume_mm3','original_mask_volume_mm3']:
            comparisons[side][metric] = compare([r[metric] for r in rows if r['side']==side and r['inferred_person_group']=='Control'],
                                                [r[metric] for r in rows if r['side']==side and r['inferred_person_group']=='TLE'])
    for metric in ['spharm_volume_mm3','original_mask_volume_mm3']:
        ordered = sorted(['left','right'],key=lambda s:comparisons[s][metric]['welch_p'])
        last = 0
        for rank,side in enumerate(ordered):
            last = max(last,min(1,(2-rank)*comparisons[side][metric]['welch_p']))
            comparisons[side][metric]['welch_p_holm_two_sides'] = last
    counts = {g:sum(c==g for c in cohorts.values()) for g in ['Control','Left_TLE','Right_TLE']}
    result = dict(cohort_definition='INFERRED from exact matching filename IDs: Control = Healthy both sides; TLE = at least one TLE side. Clinical diagnosis and laterality are not independently confirmed.',
                  input_person_counts=counts,summary=summary,comparisons=comparisons)
    (OUT/'person_group_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (OUT/'subject_volumes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    report = ['# SPHARM: inferred person-level volume comparison','',
              'Date: 2026-10-03. Input IDs: 381; inferred controls: 135; inferred TLE: 246 (141 left-label TLE and 105 right-label TLE).', '',
              '**Grouping assumption:** Control means Healthy in both filename labels. TLE means TLE in at least one side. The same ID is assumed to refer to the same person across sides. This is inferred from file naming; clinical metadata has not confirmed it.', '',
              'Healthy on one side alone does not establish that the person is a healthy control: 235 of 369 IDs with valid bilateral surfaces have different labels between sides. See [the separate filename-label report](report.md).', '',
              '**Units:** mm³ assumes source coordinates are millimetres, following the ICP convention. All source NIfTI headers omit spatial units (code 0). Percent differences remain valid in the shared coordinate unit.', '',
              '## Control versus all inferred TLE participants','',
              '| Side | Person group | Valid surfaces | SPHARM mean ± SD (mm³) | Median | Q1–Q3 | Original mask mean ± SD (mm³) |',
              '|---|---|---:|---:|---:|---:|---:|']
    for side in ['left','right']:
        for group in ['Control','TLE']:
            s = summary[side][group]['spharm']
            original = summary[side][group]['original_mask']
            report.append(f"| {side} | {group} | {s['n']} | {s['mean']:.2f} ± {s['sd']:.2f} | {s['median']:.2f} | {s['q1']:.2f}–{s['q3']:.2f} | {original['mean']:.2f} ± {original['sd']:.2f} |")
    report += ['', '| Side | TLE − Control (mm³) | Change vs Control | 95% CI | Welch p | Holm p (2 sides) | Hedges g |',
               '|---|---:|---:|---:|---:|---:|---:|']
    for side in ['left','right']:
        c = comparisons[side]['spharm_volume_mm3']
        report.append(f"| {side} | {c['mean_difference_tle_minus_healthy_mm3']:.2f} | {c['percent_difference']:.2f}% | {c['ci95'][0]:.2f} to {c['ci95'][1]:.2f} | {c['welch_p']:.4g} | {c['welch_p_holm_two_sides']:.4g} | {c['hedges_g']:.3f} |")
    report += ['', '## Inferred laterality subgroups','',
               'All-patient means combine the side labelled TLE with the opposite side. The subgroups below keep the inferred laterality visible.', '',
               '| Surface side | Inferred cohort | Valid surfaces | SPHARM mean ± SD (mm³) |','|---|---|---:|---:|']
    for side in ['left','right']:
        for group in ['Control','Left_TLE','Right_TLE']:
            s = summary[side][group]['spharm']
            report.append(f"| {side} | {group} | {s['n']} | {s['mean']:.2f} ± {s['sd']:.2f} |")
    report += ['', '## Verification and scope','',
               '- Included 373 left and 377 right surfaces; 8 left and 4 right processing failures are excluded. Each available hemisphere is analysed without requiring a complete bilateral pair.',
               '- Every included native SPHARM surface was closed, finite and without boundary/nonmanifold edges. Volume was checked with two independent calculations.',
               '- The ICP physical-to-normalized scale was checked against every transformation matrix, and SPHARM input provenance hashes matched the current aligned inputs.',
               '- Values are processed SPHARM surface estimates. Original mask volumes are shown separately as a sensitivity check.',
               '- Every valid surface is retained, including IQR outliers. Statistical comparisons use Welch tests and Holm correction over two sides.',
               '- These are unadjusted comparisons: age, sex, intracranial volume, acquisition site and dataset effects are not controlled. No diagnostic threshold or causal inference is established.',
               '- Detailed statistics, including original-mask comparisons and Mann–Whitney tests, are in person_group_results.json. Per-subject values and both grouping definitions are in subject_volumes.json.', '']
    (OUT/'person_group_report.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
