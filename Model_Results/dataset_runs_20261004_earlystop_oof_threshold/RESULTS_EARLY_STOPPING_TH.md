# ผลการเทรนใหม่: 100 epochs + Early stopping + OOF Accuracy threshold

รันสำเร็จครบ 68 งาน: แยกทุกดาต้าเซ็ตและทั้งซ้าย/ขวา ทุกงานใช้ 10-fold ตามคน และ OOF 300 แถว
MLP และโมเดล Neural Network ใช้เพดาน 100 epochs เท่ากัน; patience 10, min_delta 0.0001; SVM/PLS-DA ไม่มี epoch
แยก inner fit 216 / inner stop 54 / outer OOF 30 ต่อ fold เลือก checkpoint ด้วย inner validation loss แล้วโหลด best ก่อนทำนาย OOF
เลือก threshold ที่ Accuracy ของ Train OOF สูงสุดและตรึงก่อน final refit/Test

ตรวจ best checkpoint ครบ 560 folds และโหลดกลับมาทำนาย OOF ซ้ำครบ 16,800 แถว ได้ตรงกับผลที่บันทึกไว้
Early stopping หยุดก่อนเพดาน 560/560 folds; เทรนจริง 11–54 epochs; best epoch 1–44
Final refit ใช้ ceil(median(best epochs ของแต่ละโมเดล)): 1–24 epochs ใช้ original Train ทั้ง 300 คนและ augment ของวิธีนั้น

## เปรียบเทียบกับรอบ threshold ก่อนหน้า

Accuracy สูงขึ้น 20 งาน ลดลง 24 งาน เท่าเดิม 24 งาน

| ข้าง | งาน | รอบก่อน | รอบใหม่ | เปลี่ยน (จุดเปอร์เซ็นต์) | เพิ่ม / ลด / เท่าเดิม |
|---|---:|---:|---:|---:|---|
| left | 34 | 78.44% | 76.59% | -1.85 | 7 / 14 / 13 |
| right | 34 | 81.21% | 81.17% | -0.04 | 13 / 10 / 11 |

ค่าเฉลี่ยข้างละ 34 งานเป็นค่าเฉลี่ยของโมเดลที่ใช้ Test กลุ่มเดียวกัน ไม่ใช่ผลการทดลองอิสระ 34 ชุด
เพดาน epoch เปลี่ยนจาก neural 80 / MLP 500 เป็น 100 พร้อมเปลี่ยน checkpoint และการเลือกจำนวน epoch จึงไม่สามารถแยกผลของ Early stopping เพียงอย่างเดียวได้

## ตัวอย่างการเปลี่ยนแปลง

| Dataset | ข้าง | โมเดล | Accuracy ก่อน → หลัง | Sensitivity ก่อน → หลัง | เปลี่ยน Accuracy (จุดเปอร์เซ็นต์) |
|---|---|---|---|---|---:|
| pointnet_plsda | left | PointNet | 68.49% → 75.34% | 11.54% → 38.46% | +6.85 |
| coef_raw_balanced_jitter | left | MobileNet | 76.71% → 82.19% | 69.23% → 69.23% | +5.48 |
| augment_plsda_balanced | right | ResNetAE | 77.92% → 83.12% | 33.33% → 52.38% | +5.19 |
| pointnet_raw_balanced_jitter | left | PointNet | 72.60% → 47.95% | 34.62% → 84.62% | -24.66 |
| coef_raw_balanced_jitter | left | SqueezeNet | 79.45% → 67.12% | 53.85% → 50.00% | -12.33 |
| pointnet_raw | left | PointNet | 76.71% → 64.38% | 61.54% → 0.00% | -12.33 |

## ผลของ threshold ภายในรอบใหม่

เมื่อใช้ probabilities ของ final model เดียวกัน เทียบ threshold จาก OOF กับเกณฑ์เดิม: เพิ่ม 25 / ลด 16 / เท่าเดิม 27 งาน

- `threshold_comparison.csv`: ครบ 68 งาน พร้อม Accuracy, Balanced Accuracy, Sensitivity, Specificity, F1, AUC ก่อน/หลัง, error changes และ paired bootstrap CI95 จำนวน 1,000 รอบ
- `threshold_comparison_by_side.csv`: ผลรวมแยกซ้าย/ขวา
- `early_stopping_epochs.csv`: ครบ 560 folds พร้อม best epoch, epochs ที่รันจริง, validation loss และ final refit epochs
- `results_summary.csv`: ผลทุกโมเดล; `_logs/run_all.log`: log รวม; `_logs/`: log แยกทุกงานและ log ตรวจผล
- `audit_report.json` และ `audit_best_checkpoint_report.json`: ผลตรวจชุดคน, augment parents, threshold, metrics และการทำนายซ้ำจาก best checkpoint
- `_source_code/`: code ที่ใช้เทรนจริง; `_analysis_code/`: code ตรวจและสรุปผล; `README_EARLY_STOPPING_TH.md`: วิธีและคำสั่งรัน

OOF ที่ปรับ threshold เป็นคะแนนสำหรับเลือก threshold จึงไม่ใช่การประเมิน threshold แบบอิสระ ผล Test เป็นผลหลักหลังตรึง threshold
Test ชุดนี้ถูกตรวจในหลายการทดลองแล้ว และ ICP reference เดิมรวม Test อีกทั้ง class 0 รวมด้านตรงข้ามของผู้ป่วย จึงรายงานเป็นการเปรียบเทียบเชิงสำรวจ ไม่ควรเลือกโมเดลใหม่จาก Test แล้วอ้างเป็นผลยืนยันอิสระ
Bootstrap CI เป็นช่วงเชิงพรรณนารายโมเดล ไม่ได้แก้สำหรับการเปรียบเทียบหลายโมเดล และไม่ใช่การเทรน bootstrap ใหม่

## กรณีที่ตรวจผู้ป่วยไม่พบใน Test

มี 2 งานที่ Sensitivity = 0% แม้บางงานมี Accuracy สูงขึ้น ดังนั้น Accuracy ที่เพิ่มไม่เพียงพอสำหรับสรุปว่าโมเดลตรวจผู้ป่วยดีขึ้น

| Dataset | ข้าง | โมเดล | Accuracy | ROC-AUC |
|---|---|---|---:|---:|
| pointnet_raw | left | PointNet | 64.38% | 0.4959 |
| pointnet_raw | right | PointNet | 70.13% | 0.6726 |
