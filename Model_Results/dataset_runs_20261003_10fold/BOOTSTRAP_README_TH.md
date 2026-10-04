# Bootstrap ของผล Test

ทำบนผลทำนาย Test ของทั้ง 68 งาน ใช้คนต้นฉบับเท่านั้น ซ้าย 73 คน ขวา 77 คน
สุ่ม 1,000 รอบ ขนาด N คน แบบคืนตัวอย่าง seed 42 เรียง Subject ก่อนสุ่ม
ชุดสุ่มเดียวกันใช้กับโมเดลที่ประเมินคนชุดเดียวกันในข้างเดียวกัน
รอบที่มีคลาสเดียวจะข้ามและบันทึกเหตุผลใน manifest

ใช้ Prediction ที่บันทึกไว้ตามเกณฑ์ของแต่ละโมเดล จึงคงคะแนน Test รอบล่าสุด
ไม่ได้ฝึกโมเดลใหม่หรือเลือก threshold ใหม่ ความแปรปรวนนี้อ้างอิงโมเดลที่ฝึกไว้
และการสุ่มคนใน Test ไม่ครอบคลุมการฝึกโมเดลใหม่

bootstrap_summary.csv สรุปทั้ง 68 งาน โดย *_original เป็นคะแนน Test จริง
*_bootstrap_mean เป็นค่าเฉลี่ยจากการสุ่ม และ *_ci95_lower/upper เป็น percentile
2.5/97.5 คำนวณจากค่าที่ไม่ปัดเศษ แถบ ROC เป็นช่วงรายจุด FPR บนกริด 101 จุด
กราฟแสดงเส้น Test จริง เส้นเฉลี่ย bootstrap และ AUC/CI ของ Test แยกชัดเจน

แต่ละโฟลเดอร์ dataset/side/model/bootstrap มี:
- bootstrap_confusion_matrix.csv: คะแนนและ confusion counts ของทุกรอบที่ valid
- bootstrap_statistics.json: คะแนนจริง ค่าเฉลี่ย SD และ CI 95% ของทุกตัวชี้วัด
- bootstrap_resamples.npz: ดัชนีสุ่ม คน/label/prediction และข้อมูล ROC สำหรับตรวจซ้ำ
- roc_curve_lines.png: เส้นสุ่ม 100 เส้น เส้นเฉลี่ย และ ROC จาก Test จริง
- roc_curve_ci.png: ROC พร้อมแถบ percentile 95% รายจุด
- bootstrap_manifest.json, validation_report.json, bootstrap.log

log รวมอยู่ที่ _logs/bootstrap_all.log และสถานะทุกงานอยู่ใน bootstrap_run_summary.json
BinaryClass เป็น label ของ hippocampus แต่ละข้างตามข้อมูลต้นทาง
