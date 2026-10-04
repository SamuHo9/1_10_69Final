"""
config.py
=========
ตั้งค่า Path และ Parameters สำหรับ Hippocampus Preprocessing Pipeline
ผู้ใช้แก้ไขเฉพาะไฟล์นี้ก่อนรัน
"""

import os
import sys

def _get_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = _get_base_dir()

# =============================================================================
# [จำเป็น] ตั้งค่า Path ไปยัง FastSurfer
# =============================================================================
# ชี้ไปยังโฟลเดอร์ที่มี FastSurferCNN/ อยู่ข้างใน
FASTSURFER_DIR = os.path.join(BASE_DIR, "FastSurfer")

# =============================================================================
# [ไม่ต้องแก้] Output Directory
# =============================================================================
# ผลลัพธ์สุดท้าย (binary mask พร้อมเข้า SPHARM-PDM)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputpreprocess")

# ผลลัพธ์ระหว่างทาง (FastSurfer segmentation output)
FASTSURFER_OUTPUT_DIR = os.path.join(BASE_DIR, "fastsurfer_temp")

# =============================================================================
# [ไม่ต้องแก้] Preprocessing Parameters
# =============================================================================
# โหมดการสกัด: strict, moderate, generous, smooth
# แนะนำ "moderate" สำหรับ SPHARM-PDM
EXTRACTION_MODE = "moderate"

# จำนวนรอบ closing (สำหรับ moderate mode)
CLOSE_ITERATIONS = 1

# =============================================================================
# [ไม่ต้องแก้] FastSurfer Parameters
# =============================================================================
# ขนาด voxel: "min" = ใช้ความละเอียดต่ำสุดของภาพต้นฉบับ
VOX_SIZE = "min"

# Batch size สำหรับ CNN inference
BATCH_SIZE = 1

# Device: "auto" = ใช้ GPU ถ้ามี, "cpu" = บังคับใช้ CPU
DEVICE = "auto"

# จำนวน threads
THREADS = 1
