# การเทรน 10-fold + Early stopping + Threshold จาก Accuracy ของ OOF

รอบนี้ใช้ข้อมูลและคนใน Train/Test ชุดเดียวกับรอบก่อน ทุกวิธีและทั้งซ้าย/ขวา รวม 68 งาน โมเดลที่เรียนราย epoch (MLP, ResNet, ResNetAE, MobileNet, SqueezeNet, PointNet) มีเพดาน **100 epochs เท่ากัน** SVM/PLS-DA ใช้ solver ของตนเอง ไม่มี epoch สำหรับ Early stopping

## ขั้นตอน

1. แบ่ง original Train 300 คนเป็น 10 outer folds แบบ stratified และแยกตามคน: Train 270 / OOF validation 30 ต่อ fold
2. สำหรับโมเดลราย epoch แบ่ง outer Train 270 คนเป็น inner fit 216 / inner stopping validation 54 โดยไม่ซ้ำคนกับ outer OOF และ Test
3. สร้าง augment จาก inner fit เท่านั้น และ fit scaler/PLS จากข้อมูลฝั่ง inner fit เท่านั้น ชุด inner stop และ OOF เป็นคนจริง ไม่ augment
4. ติดตาม loss ของ inner stop ทุก epoch: patience 10, min_delta 0.0001 บันทึก checkpoint ที่ loss ต่ำที่สุดจริง รวมกรณีดีขึ้นน้อยกว่า min_delta แต่ความดีขึ้นเล็กน้อยจะไม่รีเซ็ต patience
5. โหลด best checkpoint กลับมา แล้ว predict outer OOF 30 คน แต่ละคนมี OOF prediction หนึ่งครั้ง ครบ 10 folds ได้ 300 แถว
6. เลือก threshold ที่ Accuracy ของ OOF สูงสุด กรณีเสมอใช้ Balanced Accuracy สูงสุด ตามด้วยใกล้ 0.5 ที่สุด แล้ว threshold ต่ำกว่า ตรึง threshold ก่อน refit และประเมิน Test
7. โมเดลสุดท้าย refit ด้วย original Train ทั้ง 300 คนและ augment ของวิธีนั้น จำนวน epoch ใช้ `ceil(median(best_epoch ของ 10 folds))` ไม่เกิน 100 รอบนี้ไม่ทำ Early stopping ใน final refit เพราะไม่มีการกันคนออกจาก full Train
8. ประเมิน Test ด้วยทั้งเกณฑ์เดิมและ threshold ที่ตรึง โดยใช้ probabilities ชุดเดียวกัน เปรียบเทียบอีกชุดกับผล threshold ของรอบก่อน

สำหรับ SVM/PLS-DA ใช้ outer Train ทั้ง 270 คน ไม่แบ่ง inner stop แต่ยังทำ 10-fold OOF และเลือก threshold ตามขั้นตอนเดียวกัน

## Loss และไฟล์ที่ตรวจสอบได้

- MLP: log loss; โมเดล PyTorch: classification loss ตาม architecture เดิม โดย ResNetAE รวม reconstruction MSE น้ำหนัก 0.5 และ PointNet ใช้ weighted cross-entropy ที่เฉลี่ยด้วยผลรวม class weights ของ target
- `fold_01` ถึง `fold_10`: `training_history.csv`, `best_checkpoint.pt` หรือ `.pkl`, `model.pt` หรือ `.pkl` ที่โหลด best แล้ว, `early_stopping.json`, `fold_manifest.json` และ pair manifest ถ้ามี augment
- แต่ละงาน: `oof_predictions.csv`, `threshold_sweep.csv`, `threshold_selection.json`, `final_epoch_selection.json`, `test_predictions.csv`, `metrics.json`, preprocessing และกราฟ
- รวมทุกงาน: `results_summary.csv`, `threshold_comparison.csv`, `threshold_comparison_by_side.csv`, `early_stopping_epochs.csv`, `_logs/`, `audit_report.json` และสำเนา source code
- `cv_oof` เป็นผล OOF ที่ใช้เกณฑ์เดิม ส่วน `oof_threshold_selection` เป็นคะแนนที่ใช้เลือก threshold จึงไม่ใช่คะแนนประเมิน threshold ที่เป็นอิสระ ผลหลักหลังตรึง threshold อยู่ใน `test`

## รันซ้ำ

รันจาก root ของโครงการ โดยเลือก output directory ใหม่:

```powershell
python -u Model/run_dataset_models.py --early-stopping --workers 2 --epochs 100 --mlp-max-epochs 100 --folds 10 --patience 10 --min-delta 0.0001 --stop-fraction 0.2 --pls-components 8 --seed 42 --output-root Model_Results/dataset_runs_20261004_earlystop_oof_threshold
```

เพดาน epoch เปลี่ยนจาก neural 80 / MLP 500 ในรอบก่อนเป็น 100 เท่ากัน พร้อมเปลี่ยนวิธีเลือก checkpoint และ epoch จึงเป็นการเปรียบเทียบทั้ง protocol ไม่สามารถแยกผลที่เกิดจาก Early stopping เพียงอย่างเดียวได้

ข้อมูลเดิมมีข้อจำกัด: Test ถูกตรวจในหลายการทดลองแล้ว, ICP reference เดิมรวม Test และ class 0 ของ hemisphere รวมด้านตรงข้ามของผู้ป่วย ผลจึงเป็นการเปรียบเทียบเชิงสำรวจของข้อมูลชุดนี้

สคริปต์หลัก: `early_stopping_training.py`; ตรวจผล: `audit_10fold_run.py`; เปรียบเทียบ: `compare_threshold_results.py`
