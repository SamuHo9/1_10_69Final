# Data Preparation GUI

ชุดนี้แยกการเตรียมข้อมูลออกเป็น 5 วิธี และมี Direct PLS-DA เป็น runner แยกต่างหาก
ทุก entry point รองรับสองรูปแบบ:

~~~powershell
# เปิด GUI ใน SlicerSALT ซึ่งมี Qt widgets
& "C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe" --python-script .\prepare_coef_raw.py

# ใช้ command line แบบไม่เปิด GUI
& "C:\Program Files\SlicerSALT 6.0.0\bin\PythonSlicer.exe" .\prepare_coef_raw.py --train-csv "D:\data\train.csv" --test-csv "D:\data\test.csv" --output-dir "D:\outputs\coef_raw"
~~~

ถ้ามี Python ปกติที่ติดตั้ง tkinter แล้ว สามารถเปิด GUI ด้วยคำสั่ง python prepare_coef_raw.py
ได้เช่นกัน ส่วน PythonSlicer แบบ standalone ใช้สำหรับ command line เพราะไม่มี tkinter
และไม่โหลด Qt widgets จนกว่าจะรันอยู่ใน SlicerSALT

## ไฟล์

| ไฟล์ | หน้าที่ | Synthetic rows |
|---|---|---|
| prepare_coef_raw.py | coefficient 507 มิติแบบ raw; เลือก none หรือ balanced_jitter ได้ | optional, same-class |
| prepare_coef_plsda.py | ใช้ PLS-DA สร้าง coefficient synthetic rows | yes, same-class |
| prepare_pointnet_raw.py | XYZ 1,002 จุดแบบ raw; เลือก none หรือ balanced_jitter ได้ | optional, same-class |
| prepare_pointnet_plsda.py | ใช้ PLS-DA สร้าง XYZ synthetic clouds | yes, same-class |
| prepare_plsda_latent_features.py | แปลง coefficient เป็น PLS_1 ... PLS_k | no |
| run_plsda_direct.py | ใช้ PLS-DA เป็น classifier โดยตรง | no |
| prepare_plsda_direct.py | alias ของ run_plsda_direct.py | no |

between_class ไม่ถูกใช้ในชุดนี้ ทุก augmentation จับคู่ภายในคลาสเดียวกันเท่านั้น

## ไฟล์ผลลัพธ์

วิธีเตรียมข้อมูลจะเขียน:

~~~text
train_prepared.csv
test_prepared.csv
synthetic_rows.csv
pair_manifest.csv
scaler_stats.csv
plsda_model.pkl
data_manifest.json
~~~

run_plsda_direct.py จะเขียน direct_plsda_model.pkl, OOF/test predictions,
metrics และ run_manifest.json แทน

## ข้อควรระวัง

สคริปต์แบบ explicit train/test เป็น fixed-split preparation tool หากทำ grouped
cross-validation ต้องเรียก augmentation ภายใน training fold ของแต่ละ fold ไม่ควร
สร้าง synthetic rows จาก train ทั้งชุดแล้วนำไปแบ่ง fold ภายหลัง

Scaler และ PLS-DA จะ fit จาก train เท่านั้น ส่วน test จะไม่ถูกใช้สร้างคู่หรือ
สร้างข้อมูลสังเคราะห์
