# ผลการเทรนใหม่และเลือก threshold จาก OOF

เทรนใหม่ครบ 68 งาน ใช้ 10-fold CV ระดับคน, seed 42, PLS 8 components
และ Neural Network 80 epochs ต่อการเทรน ผ่านการตรวจผลทั้ง 68 งาน ไม่มีข้อผิดพลาด
ผลทำนายด้วยกฎเดิมจากโมเดลที่เทรนใหม่ตรงกับรอบก่อนทั้ง 68 งาน
จึงเปรียบเทียบผลของ threshold ใหม่กับกฎเดิมได้โดยตรง

| ข้าง | จำนวนงาน | Accuracy เพิ่ม | ลด | เท่าเดิม |
|---|---:|---:|---:|---:|
| ซ้าย | 34 | 25 | 4 | 5 |
| ขวา | 34 | 11 | 9 | 14 |
| รวม | 68 | 36 | 13 | 19 |

ตัวอย่างผลบน Test ชุดเดิม:

| Dataset / Model / ข้าง | Threshold | Accuracy เดิม | Accuracy ใหม่ | Sensitivity เดิม | Sensitivity ใหม่ |
|---|---:|---:|---:|---:|---:|
| coef_plsda / MLP / ซ้าย | 0.983811 | 72.60% | 84.93% | 73.08% | 73.08% |
| coef_raw / SVM / ซ้าย | 0.960380 | 78.08% | 83.56% | 73.08% | 61.54% |
| coef_raw / SqueezeNet / ขวา | 0.947223 | 87.01% | 88.31% | 71.43% | 66.67% |

Threshold แต่ละงานเลือกจาก OOF ของ Train 300 คนให้ Accuracy สูงสุด
ก่อนเทรนโมเดลสุดท้ายและทำนาย Test ไม่ใช้ Test เลือก threshold
ROC-AUC ของกฎเดิมกับกฎใหม่เท่ากัน เพราะใช้ probability ชุดเดียวกัน
Accuracy ที่เพิ่มขึ้นอาจมาพร้อม Sensitivity ของผู้ป่วยที่ลดลง

- ผลละเอียดทุกตัว: `threshold_comparison.csv`
- ผลสรุปแต่ละข้าง: `threshold_comparison_by_side.csv`
- รายงานตรวจผล: `audit_report.json` และ `threshold_comparison_report.json`
- Log รวม: `_logs/run_all.log`
- Log ตรวจผล: `_logs/audit_threshold.log` และ `_logs/compare_threshold_results.log`
- แต่ละ dataset/side/model มี threshold, ตารางจุดตัด, กราฟ, ผล Test และโมเดล 10 folds กับโมเดลสุดท้าย

ช่วงความเชื่อมั่นของผลต่าง Accuracy ใช้ paired Test bootstrap 1,000 รอบ seed 42
เป็นช่วงรายโมเดล ไม่ได้ปรับการเปรียบเทียบหลายโมเดล
คะแนน OOF หลังเลือก threshold เป็นคะแนนที่ใช้เลือกค่า ไม่ใช่ CV ที่เป็นอิสระจากการเลือก
ผลนี้ใช้ Test เดิมที่เคยดูผลแล้ว เป็นผลทดลองและไม่ใช่การยืนยันบนข้อมูลภายนอกใหม่
ข้อจำกัดนิยามคลาสของแต่ละข้างและ reference ICP เดิมยังอยู่ ดูรายละเอียดใน `README_THRESHOLD_TH.md`
