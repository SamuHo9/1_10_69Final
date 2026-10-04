# การรัน 10-fold และ OOF

สคริปต์ทั้งหมดใน `All_Augment_tain/left` และ `right` รวมถึง
`run_dataset_models.py` ใช้ขั้นตอนร่วมใน `dataset_cross_validation.py`
และบังคับ 10-fold หากคลาสใดมีคนต้นฉบับน้อยกว่า 10 คนจะแจ้ง error
แทนการลด fold โดยไม่แจ้ง

แบ่งเฉพาะคนต้นฉบับใน Train เป็น 10 ส่วน ใช้ 9 ส่วนฝึกและทำนาย 1 ส่วน
หมุนจนครบ ทุกแถวต้นฉบับมี OOF prediction หนึ่งครั้ง คนเดียวกันอยู่ fold
เดียวกันเสมอ และใช้การแบ่งเดียวกันระหว่างดาต้าเซ็ต/โมเดลของข้างเดียวกัน
ไม่สร้าง OOF สำหรับข้อมูลสังเคราะห์

ทำ augmentation, scaling และ supervised PLS ใหม่ในส่วนฝึกของแต่ละ fold
ไม่ใช้ลูก augment ที่เตรียมจาก Train ทั้งชุดมาแบ่ง CV สำหรับ PLS latent
จะสร้างพิกัดใหม่จาก coefficients ต้นฉบับภายในแต่ละ fold ไม่ใช้ไฟล์ latent
ที่ fit จาก Train ทั้งชุด ตัว augmentation ใช้จำนวนและวิธีตาม manifest
ของดาต้าเซ็ตปัจจุบัน ได้แก่ balance-only สำหรับชุด jitter/coef_plsda/pointnet_plsda
และวิธี augment ทั้งสองคลาสแล้ว balance สำหรับ augment_plsda_balanced

คงโครงสร้างโมเดลเดิม ใช้ PLS 8 components และ Neural Network 80 epochs
เป็นค่าตั้งต้น ไม่มีการเลือกจำนวน components หรือ epoch จาก outer OOF/Test
MLP หยุดจาก training loss; SVM ใช้ probability calibration ภายในส่วนฝึก
หลังได้ OOF แล้วฝึกโมเดลสุดท้ายบน Train ทั้งชุด และทำนาย Test แยกหนึ่งครั้ง
OOF จึงเป็นคะแนนของโมเดลที่ฝึกบน 90% ของ Train ไม่ใช่คะแนน Test
ยังไม่มีขั้นตอนหา threshold ที่ดีที่สุด ใช้ 0.5 สำหรับ Neural Network/Direct
PLS-DA และ `estimator.predict()` สำหรับ SVM/MLP บันทึกสถานะนี้ใน manifest
หากต้องการเลือก threshold ภายหลัง ให้ใช้ OOF ของ Train และตรึงค่าก่อนประเมิน Test

รันจากโฟลเดอร์โปรเจกต์:

```bash
python Model/run_dataset_models.py --scope all --folds 10 --epochs 80 --pls-components 8 --seed 42
```

รันครบ 68 งาน: 5 ดาต้าเซ็ตตาราง × 2 ข้าง × 6 โมเดล,
3 ดาต้าเซ็ต XYZ × 2 ข้าง × PointNet และ Direct PLS-DA อีก 2 ข้าง
ผลชุดใหม่อยู่ใน `Model_Results/dataset_runs_20261003_10fold`
ส่วนผล `dataset_runs_20261003` เป็นผลเก่าก่อนปรับ และจะไม่ถูกเขียนทับ
โฟลเดอร์ผลใหม่จะสร้างเมื่อรันจริง ไม่มีการแปลงผลเก่าให้เป็น OOF

รันสคริปต์เดิมรายตัวได้ เช่น:

```bash
python Model/All_Augment_tain/left/SVM/train_svm_pls.py --dataset coef_raw --output-dir Model_Results/my_10fold/coef_raw/left/SVM
python Model/All_Augment_tain/right/PointNet/train_pointnet.py --dataset pointnet_raw --output-dir Model_Results/my_10fold/pointnet_raw/right/PointNet
```

หากไม่ระบุ dataset สคริปต์ตารางเลือก `augment_plsda_balanced` และ PointNet
เลือก `pointnet_plsda` หากใช้ `plsda_latent_features` ต้องเพิ่ม `--kind latent`
และมี `coef_raw` คู่กันใต้ dataset-root เดียวกัน

แต่ละงานบันทึก `oof_predictions.csv` และ `.npz`, `oof_metrics.json`,
`fold_metrics.csv`, `metrics.json` (แยก `cv_oof` กับ `test`),
`test_predictions.csv` และโมเดลสุดท้าย `fold_01` ถึง `fold_10`
เก็บโมเดล, preprocessing, training history, รายชื่อ train/validation
และคู่พ่อแม่ augment เพื่อให้ตรวจสอบการแยกข้อมูลได้
CSV OOF มี Subject, PatientID, Fold, BinaryClass, Probability และ Prediction
สคริปต์รวมบันทึก log ที่ `_logs` และคะแนนทั้ง OOF/Test ใน `results_summary.csv`
สคริปต์รายตัวบันทึก log ข้างโฟลเดอร์โมเดล เช่น `SVM.log`

ตรวจสอบผลที่รันครบแล้ว:

```bash
python Model/audit_10fold_run.py Model_Results/dataset_runs_20261003_10fold
```

`Model_Results/validation_10fold_20261003` เป็นผลทดสอบโค้ด 9 กรณี
ที่ใช้ Neural Network เพียง 1 epoch ไม่ใช่ผลการทดลองเต็ม 80 epochs

ข้อจำกัดของข้อมูลต้นทางยังเหมือนเดิม: BinaryClass เป็นสถานะของ hippocampus
แต่ละข้าง และ reference ของ ICP เดิมรวม Test ขั้นตอน CV ใหม่นี้ไม่สามารถ
ย้อนแก้การเตรียมรูปทรงต้นทางนั้นได้
