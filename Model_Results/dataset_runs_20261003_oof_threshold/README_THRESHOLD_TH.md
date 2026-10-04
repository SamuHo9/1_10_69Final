# การทดลองเลือก threshold จาก Train OOF

เทรนใหม่ทุกดาต้าเซ็ตและโมเดล ทั้งซ้ายและขวา รวม 68 งาน โดยใช้ seed 42,
10-fold CV ระดับคน, PLS 8 components และ Neural Network 80 epochs เหมือนรอบเดิม
ผลใหม่แยกจากผลเดิมที่ `Model_Results/dataset_runs_20261003_oof_threshold`

## วิธีเลือก threshold

1. แต่ละคนใน Train ได้ probability จากโมเดลของ fold ที่ไม่เคยเทรนกับคนนั้น
   Augmentation, scaler และ PLS ใช้เฉพาะ Train ของ fold
2. รวม OOF ของคนต้นฉบับ 300 คนต่อข้าง แล้วลองทุกจุดตัดที่ให้ชุดคำทำนายต่างกัน
   ใช้กฎ `probability >= threshold` ค่าที่เกิน probability สูงสุดเล็กน้อยแทนกรณีทำนายทุกคนเป็น class 0
3. เลือก Accuracy สูงสุด หากเสมอกันเลือก Balanced Accuracy สูงกว่า
   จากนั้นเลือกค่าที่ใกล้ 0.5 และค่าต่ำกว่า ตามลำดับ
4. บันทึกและตรึง threshold ก่อนเทรนโมเดลสุดท้ายด้วย Train ทั้งหมด
   ไม่ใช้ label หรือ probability ของ Test เลือกค่า
5. ทำนาย Test ด้วยโมเดลสุดท้ายครั้งเดียว แล้วประเมินทั้งกฎเดิมและ threshold ใหม่

`cv_oof` และ `oof_predictions.csv` ยังคงเป็นผล OOF ของกฎเดิม
`oof_threshold_selection_metrics.json` เป็นคะแนนบน OOF ที่ใช้เลือก threshold
จึงไม่ใช่คะแนน CV ที่เป็นอิสระจากการเลือก threshold หากต้องการคะแนน CV ของ
กระบวนการเลือก threshold อย่างอิสระ ต้องเพิ่ม nested CV

## ผลเปรียบเทียบ

- `threshold_comparison.csv`: เปรียบเทียบกฎเดิมและ threshold ใหม่จากโมเดลที่เทรนรอบเดียวกัน
  พร้อมผลรอบก่อน, จำนวนคนที่แก้ทำนายถูก/ทำนายผิดเพิ่ม, Accuracy, Balanced Accuracy,
  Sensitivity, Specificity, F1 และ ROC-AUC
- `threshold_gain_pp`: การเปลี่ยน Accuracy เป็นจุดเปอร์เซ็นต์ เช่น 5 หมายถึงเพิ่มจาก 75% เป็น 80%
- คอลัมน์ Accuracy, Balanced Accuracy, Sensitivity, Specificity และ F1 ใช้ค่า 0–1
  เช่น 0.8493 หมายถึง 84.93% ส่วนคอลัมน์ลงท้าย `_pp` ใช้จุดเปอร์เซ็นต์
- `threshold_comparison_by_side.csv`: จำนวนงานที่เพิ่ม ลด หรือเท่าเดิมในแต่ละข้าง
- CI ของผลต่าง Accuracy ใช้ paired bootstrap ของ Test เดิม 1,000 รอบ seed 42
  และ percentile 2.5/97.5 เป็นช่วงรายโมเดล ไม่ได้ปรับสำหรับการเปรียบเทียบหลายโมเดล
- `_logs`: log ครบทุกงานและ log รวม
- แต่ละงานมี `threshold_selection.json`, `threshold_sweep.csv`, กราฟเลือก threshold,
  `test_predictions.csv` ที่แยก `Prediction` ใหม่และ `PredictionDefault` เดิม
- ROC-AUC ไม่เปลี่ยนจาก threshold เพราะใช้ probability ชุดเดียวกัน

การเพิ่ม Accuracy อาจทำให้ Sensitivity ของผู้ป่วยลดลง ต้องอ่านคู่กัน
ผลนี้เป็นการทดลองบน Test เดิมที่เคยดูคะแนนแล้ว ไม่ใช่การยืนยันบนข้อมูลภายนอกใหม่
นิยาม class 0 ของแต่ละข้างและ reference ICP เดิมที่รวม Test ยังเป็นข้อจำกัดของข้อมูลต้นทาง
เอกสารหลักการ: https://scikit-learn.org/stable/modules/classification_threshold.html

## คำสั่ง

```bash
python Model/run_dataset_models.py --tune-threshold --workers 2 --output-root Model_Results/dataset_runs_20261003_oof_threshold --folds 10 --epochs 80 --pls-components 8 --seed 42
python Model/audit_10fold_run.py Model_Results/dataset_runs_20261003_oof_threshold
python Model/compare_threshold_results.py
```

ใช้ output-root ใหม่สำหรับการทดลองแต่ละชุด สคริปต์ไม่เขียนทับผลโมเดลที่มีอยู่
