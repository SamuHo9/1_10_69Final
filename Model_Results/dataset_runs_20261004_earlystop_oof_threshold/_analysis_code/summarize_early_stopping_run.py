# -*- coding: utf-8 -*-
"""Write the Thai result report and inventory after successful training/audit."""
import argparse
import json
from collections import Counter
from pathlib import Path
import shutil
import pandas as pd
from train_dataset_model import ROOT, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    summary = json.loads((root/'run_summary.json').read_text())
    comparison = json.loads((root/'threshold_comparison_report.json').read_text())
    audit = json.loads((root/'audit_best_checkpoint_report.json').read_text())
    assert summary['completed_training_jobs'] == comparison['verified_jobs'] == audit['verified_jobs'] == 68
    assert summary['failed_training_jobs'] == 0 and not audit['errors']
    frame = pd.read_csv(root/'threshold_comparison.csv')
    epochs = pd.read_csv(root/'early_stopping_epochs.csv')
    assert len(epochs) == audit['best_checkpoint_oof_folds_replayed'] == 560
    assert (epochs.max_epochs == 100).all() and (epochs.epochs_run <= 100).all()
    assert (epochs.best_epoch >= 1).all() and (epochs.best_epoch <= epochs.epochs_run).all()
    assert epochs.best_checkpoint_restored.all()
    stopped = int(epochs.stopped_early.sum())
    final = epochs.drop_duplicates(['dataset','side','model'])
    lines = ['# ผลการเทรนใหม่: 100 epochs + Early stopping + OOF Accuracy threshold', '',
        'รันสำเร็จครบ 68 งาน: แยกทุกดาต้าเซ็ตและทั้งซ้าย/ขวา ทุกงานใช้ 10-fold ตามคน และ OOF 300 แถว',
        'MLP และโมเดล Neural Network ใช้เพดาน 100 epochs เท่ากัน; patience 10, min_delta 0.0001; SVM/PLS-DA ไม่มี epoch',
        'แยก inner fit 216 / inner stop 54 / outer OOF 30 ต่อ fold เลือก checkpoint ด้วย inner validation loss แล้วโหลด best ก่อนทำนาย OOF',
        'เลือก threshold ที่ Accuracy ของ Train OOF สูงสุดและตรึงก่อน final refit/Test', '',
        f"ตรวจ best checkpoint ครบ {len(epochs)} folds และโหลดกลับมาทำนาย OOF ซ้ำครบ {audit['best_checkpoint_oof_rows_replayed']:,} แถว ได้ตรงกับผลที่บันทึกไว้",
        f"Early stopping หยุดก่อนเพดาน {stopped}/{len(epochs)} folds; เทรนจริง {int(epochs.epochs_run.min())}–{int(epochs.epochs_run.max())} epochs; best epoch {int(epochs.best_epoch.min())}–{int(epochs.best_epoch.max())}",
        f"Final refit ใช้ ceil(median(best epochs ของแต่ละโมเดล)): {int(final.final_epochs.min())}–{int(final.final_epochs.max())} epochs ใช้ original Train ทั้ง 300 คนและ augment ของวิธีนั้น", '',
        '## เปรียบเทียบกับรอบ threshold ก่อนหน้า', '',
        f"Accuracy สูงขึ้น {comparison['improved_vs_previous']} งาน ลดลง {comparison['decreased_vs_previous']} งาน เท่าเดิม {comparison['unchanged_vs_previous']} งาน", '',
        '| ข้าง | งาน | รอบก่อน | รอบใหม่ | เปลี่ยน (จุดเปอร์เซ็นต์) | เพิ่ม / ลด / เท่าเดิม |',
        '|---|---:|---:|---:|---:|---|']
    for row in comparison['by_side']:
        lines.append(f"| {row['side']} | {row['jobs']} | {100*row['mean_previous_accuracy']:.2f}% | {100*row['mean_tuned_accuracy']:.2f}% | {row['mean_change_from_previous_pp']:+.2f} | {row['improved_vs_previous']} / {row['decreased_vs_previous']} / {row['unchanged_vs_previous']} |")
    lines += ['', 'ค่าเฉลี่ยข้างละ 34 งานเป็นค่าเฉลี่ยของโมเดลที่ใช้ Test กลุ่มเดียวกัน ไม่ใช่ผลการทดลองอิสระ 34 ชุด',
        'เพดาน epoch เปลี่ยนจาก neural 80 / MLP 500 เป็น 100 พร้อมเปลี่ยน checkpoint และการเลือกจำนวน epoch จึงไม่สามารถแยกผลของ Early stopping เพียงอย่างเดียวได้', '',
        '## ตัวอย่างการเปลี่ยนแปลง', '',
        '| Dataset | ข้าง | โมเดล | Accuracy ก่อน → หลัง | Sensitivity ก่อน → หลัง | เปลี่ยน Accuracy (จุดเปอร์เซ็นต์) |',
        '|---|---|---|---|---|---:|']
    examples = pd.concat([frame.nlargest(3,'change_from_previous_pp'), frame.nsmallest(3,'change_from_previous_pp')]).drop_duplicates(['dataset','side','model'])
    for row in examples.itertuples():
        lines.append(f'| {row.dataset} | {row.side} | {row.model} | {100*row.previous_test_accuracy:.2f}% → {100*row.tuned_test_accuracy:.2f}% | {100*row.previous_sensitivity:.2f}% → {100*row.tuned_sensitivity:.2f}% | {row.change_from_previous_pp:+.2f} |')
    lines += ['', '## ผลของ threshold ภายในรอบใหม่', '',
        f"เมื่อใช้ probabilities ของ final model เดียวกัน เทียบ threshold จาก OOF กับเกณฑ์เดิม: เพิ่ม {comparison['improved']} / ลด {comparison['decreased']} / เท่าเดิม {comparison['unchanged']} งาน", '',
        '- `threshold_comparison.csv`: ครบ 68 งาน พร้อม Accuracy, Balanced Accuracy, Sensitivity, Specificity, F1, AUC ก่อน/หลัง, error changes และ paired bootstrap CI95 จำนวน 1,000 รอบ',
        '- `threshold_comparison_by_side.csv`: ผลรวมแยกซ้าย/ขวา',
        '- `early_stopping_epochs.csv`: ครบ 560 folds พร้อม best epoch, epochs ที่รันจริง, validation loss และ final refit epochs',
        '- `results_summary.csv`: ผลทุกโมเดล; `_logs/run_all.log`: log รวม; `_logs/`: log แยกทุกงานและ log ตรวจผล',
        '- `audit_report.json` และ `audit_best_checkpoint_report.json`: ผลตรวจชุดคน, augment parents, threshold, metrics และการทำนายซ้ำจาก best checkpoint',
        '- `_source_code/`: code ที่ใช้เทรนจริง; `_analysis_code/`: code ตรวจและสรุปผล; `README_EARLY_STOPPING_TH.md`: วิธีและคำสั่งรัน', '',
        'OOF ที่ปรับ threshold เป็นคะแนนสำหรับเลือก threshold จึงไม่ใช่การประเมิน threshold แบบอิสระ ผล Test เป็นผลหลักหลังตรึง threshold',
        'Test ชุดนี้ถูกตรวจในหลายการทดลองแล้ว และ ICP reference เดิมรวม Test อีกทั้ง class 0 รวมด้านตรงข้ามของผู้ป่วย จึงรายงานเป็นการเปรียบเทียบเชิงสำรวจ ไม่ควรเลือกโมเดลใหม่จาก Test แล้วอ้างเป็นผลยืนยันอิสระ',
        'Bootstrap CI เป็นช่วงเชิงพรรณนารายโมเดล ไม่ได้แก้สำหรับการเปรียบเทียบหลายโมเดล และไม่ใช่การเทรน bootstrap ใหม่', '']
    zero_sensitivity = frame[frame.tuned_sensitivity == 0]
    if len(zero_sensitivity):
        lines += ['## กรณีที่ตรวจผู้ป่วยไม่พบใน Test', '',
            f'มี {len(zero_sensitivity)} งานที่ Sensitivity = 0% แม้บางงานมี Accuracy สูงขึ้น ดังนั้น Accuracy ที่เพิ่มไม่เพียงพอสำหรับสรุปว่าโมเดลตรวจผู้ป่วยดีขึ้น', '',
            '| Dataset | ข้าง | โมเดล | Accuracy | ROC-AUC |', '|---|---|---|---:|---:|']
        for row in zero_sensitivity.itertuples():
            lines.append(f'| {row.dataset} | {row.side} | {row.model} | {100*row.tuned_test_accuracy:.2f}% | {row.tuned_roc_auc:.4f} |')
        lines += ['']
    (root/'RESULTS_EARLY_STOPPING_TH.md').write_text('\n'.join(lines), encoding='utf-8')
    analysis = root/'_analysis_code'
    analysis.mkdir(exist_ok=True)
    sources = []
    for name in ('audit_10fold_run.py','compare_threshold_results.py','validate_10fold_training.py',
            'test_early_stopping.py','test_dataset_training.py','test_10fold_training.py','test_threshold_selection.py',
            'summarize_early_stopping_run.py'):
        source = ROOT/'Model'/name
        target = analysis/name
        shutil.copy2(source,target)
        sources.append({'source':str(source),'snapshot':str(target),'sha256':sha(target)})
    (root/'analysis_code_manifest.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'artifact_inventory.json')
    inventory = {'files':len(files), 'bytes':sum(p.stat().st_size for p in files),
        'extensions':dict(Counter(p.suffix for p in files)), 'training_jobs':68, 'outer_cv_folds':680,
        'best_checkpoints':560, 'original_oof_rows_per_job':300, 'training_source_snapshots':21,
        'top_level_report_sha256':{p.name:sha(p) for p in files if p.parent==root and p.suffix in ('.csv','.json','.md')},
        'log_files':[str(p.relative_to(root)) for p in files if p.suffix=='.log']}
    (root/'artifact_inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    print(json.dumps({k:inventory[k] for k in ('files','bytes','extensions','training_jobs','best_checkpoints')}, indent=2))


if __name__ == '__main__':
    main()
