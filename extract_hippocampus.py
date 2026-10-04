#!/usr/bin/env python3
"""
extract_hippocampus.py
=====================
สกัด Left & Right Hippocampus จาก FastSurfer segmentation แยกเป็น 2 ไฟล์
พร้อมตัวเลือกปรับความเข้มข้น (intensity) ของการสกัด

Label IDs (FreeSurfer / FastSurfer):
  - 17 = Left-Hippocampus
  - 53 = Right-Hippocampus

ระดับความเข้มข้นในการสกัด (Extraction Intensity):
  1. strict   — ใช้เฉพาะ voxel ที่ตรงกับ label เป๊ะ ๆ (ไม่ขยาย)
  2. moderate — เพิ่ม morphological closing เพื่ออุดรูเล็ก ๆ ในเนื้อ hippocampus
  3. generous — เพิ่ม dilation ขยายขอบเขตออก 1-N voxels
  4. smooth   — ใช้ Gaussian smoothing + threshold เพื่อให้ขอบนุ่มขึ้น

Usage:
    python extract_hippocampus.py --input <seg.mgz or seg.nii.gz> --output_dir <dir>
    python extract_hippocampus.py --input <seg.mgz> --output_dir <dir> --mode generous --dilate 2
    python extract_hippocampus.py --input <seg.mgz> --output_dir <dir> --mode smooth --sigma 1.5 --threshold 0.3
"""

import os
import argparse
import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("ERROR: nibabel not installed. Run: pip install nibabel")
    exit(1)

try:
    from scipy import ndimage
except ImportError:
    print("ERROR: scipy not installed. Run: pip install scipy")
    exit(1)


# === FastSurfer / FreeSurfer Hippocampus Labels ===
LABEL_LEFT_HIPPO = 17
LABEL_RIGHT_HIPPO = 53

# Optional: include neighboring structures for "extended" extraction
EXTENDED_LABELS = {
    "left": {
        17: "Left-Hippocampus",
        18: "Left-Amygdala",          # ติดกับ hippocampus
    },
    "right": {
        53: "Right-Hippocampus",
        54: "Right-Amygdala",
    },
}


def load_segmentation(filepath: str):
    """โหลด segmentation file (.mgz, .nii, .nii.gz)"""
    print(f"[INFO] Loading segmentation: {filepath}")
    img = nib.load(filepath)
    data = np.asarray(img.dataobj, dtype=np.int32)
    print(f"  Shape: {data.shape}, Voxel size: {img.header.get_zooms()[:3]}")
    return img, data


def extract_largest_component(mask: np.ndarray) -> np.ndarray:
    """ลบเศษ Voxel ที่ลอยแยกออกไป เก็บไว้แค่ก้อนที่ใหญ่ที่สุดก้อนเดียว"""
    labeled_array, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
    
    sizes = ndimage.sum(mask, labeled_array, range(1, num_features + 1))
    max_label = np.argmax(sizes) + 1
    
    lcc_mask = (labeled_array == max_label).astype(np.uint8)
    return lcc_mask



def extract_binary_mask(data: np.ndarray, label: int) -> np.ndarray:
    """สร้าง binary mask จาก label ที่ระบุ"""
    mask = (data == label).astype(np.uint8)
    voxel_count = mask.sum()
    print(f"  Label {label}: {voxel_count} voxels")
    if voxel_count == 0:
        print(f"  [WARNING] Label {label} not found in segmentation!")
    return mask


def extract_multi_label_mask(data: np.ndarray, labels: dict) -> np.ndarray:
    """สร้าง binary mask จากหลาย labels (extended mode)"""
    mask = np.zeros_like(data, dtype=np.uint8)
    for label_id, label_name in labels.items():
        count = (data == label_id).sum()
        print(f"  Label {label_id} ({label_name}): {count} voxels")
        mask[data == label_id] = 1
    print(f"  Total combined: {mask.sum()} voxels")
    return mask


def apply_mode(mask: np.ndarray, mode: str, **kwargs) -> np.ndarray:
    """
    ปรับความเข้มข้นของ mask ตามโหมดที่เลือก

    Modes:
      strict   — ไม่ทำอะไรเพิ่ม ใช้ mask ดิบ
      moderate — morphological closing (อุดรู + ปิดช่องว่าง)
      generous — dilation ขยายขอบเขต
      smooth   — Gaussian smooth + re-threshold
    """
    if mode == "strict":
        print("  [MODE] strict — ใช้ mask ดิบ ไม่ปรับแต่ง")
        return mask

    elif mode == "moderate":
        iterations = kwargs.get("close_iter", 1)
        struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
        print(f"  [MODE] moderate — closing iterations={iterations}")
        result = ndimage.binary_closing(mask, structure=struct, iterations=iterations)
        # fill holes inside the structure
        result = ndimage.binary_fill_holes(result)
        return result.astype(np.uint8)

    elif mode == "generous":
        dilate_voxels = kwargs.get("dilate", 1)
        struct = ndimage.generate_binary_structure(3, 1)
        print(f"  [MODE] generous — dilation={dilate_voxels} voxels")
        result = ndimage.binary_dilation(mask, structure=struct, iterations=dilate_voxels)
        return result.astype(np.uint8)

    elif mode == "smooth":
        sigma = kwargs.get("sigma", 1.0)
        threshold = kwargs.get("threshold", 0.5)
        print(f"  [MODE] smooth — sigma={sigma}, threshold={threshold}")
        smoothed = ndimage.gaussian_filter(mask.astype(np.float32), sigma=sigma)
        result = (smoothed >= threshold).astype(np.uint8)
        return result

    elif mode == "topology_fix":
        iterations = kwargs.get("close_iter", 1)
        struct = ndimage.generate_binary_structure(3, 1)
        print(f"  [MODE] topology_fix — forcing spherical topology")
        result = extract_largest_component(mask)
        result = ndimage.binary_closing(result, structure=struct, iterations=iterations)
        result = ndimage.binary_opening(result, structure=struct, iterations=1)
        result = ndimage.binary_fill_holes(result)
        result = extract_largest_component(result)
        return result.astype(np.uint8)

    else:
        raise ValueError(f"Unknown mode: {mode}. Choose: strict, moderate, generous, smooth")


def crop_to_roi(data: np.ndarray, mask: np.ndarray, padding: int = 5):
    """
    Crop data & mask ให้เหลือเฉพาะ ROI + padding
    ลดขนาดไฟล์และเร็วขึ้นสำหรับ downstream processing
    """
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return data, mask, None

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    # Add padding
    mins = np.maximum(mins - padding, 0)
    maxs = np.minimum(maxs + padding + 1, np.array(data.shape))

    slices = tuple(slice(mn, mx) for mn, mx in zip(mins, maxs))
    return data[slices], mask[slices], (mins, maxs)


def save_nifti(data: np.ndarray, reference_img, output_path: str,
               crop_origin=None):
    """บันทึก NIfTI file โดยรักษา header/affine จากต้นฉบับ"""
    affine = reference_img.affine.copy()

    # ถ้า crop แล้ว ต้องปรับ affine origin
    if crop_origin is not None:
        mins, _ = crop_origin
        # Shift origin in world coordinates correctly
        vox_offset = np.zeros(4)
        vox_offset[:3] = mins
        vox_offset[3] = 1.0  # Homogeneous coordinate must be 1
        world_offset = affine @ vox_offset
        affine[:3, 3] = world_offset[:3]

    new_img = nib.Nifti1Image(data, affine, reference_img.header)
    new_img.header.set_data_dtype(np.uint8)
    nib.save(new_img, output_path)
    print(f"  [SAVED] {output_path} — shape: {data.shape}, voxels: {data.sum()}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Left & Right Hippocampus from FastSurfer segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ตัวอย่างการใช้งาน:
  # แบบพื้นฐาน (strict mode)
  python extract_hippocampus.py --input ./subjectX/mri/aparc.DKTatlas+aseg.deep.mgz --output_dir ./hippo_output

  # ปิดรูเล็ก ๆ (moderate)
  python extract_hippocampus.py --input ./subjectX/mri/aparc.DKTatlas+aseg.deep.mgz --output_dir ./hippo_output --mode moderate

  # ขยายขอบเขต 2 voxels (generous)
  python extract_hippocampus.py --input ./subjectX/mri/aparc.DKTatlas+aseg.deep.mgz --output_dir ./hippo_output --mode generous --dilate 2

  # ขอบนุ่ม (smooth)
  python extract_hippocampus.py --input ./subjectX/mri/aparc.DKTatlas+aseg.deep.mgz --output_dir ./hippo_output --mode smooth --sigma 1.5 --threshold 0.3

  # รวม Amygdala (extended)
  python extract_hippocampus.py --input ./subjectX/mri/aparc.DKTatlas+aseg.deep.mgz --output_dir ./hippo_output --extended

  # Crop เฉพาะ ROI + ไม่ต้อง full volume
  python extract_hippocampus.py --input ./subjectX/mri/aparc.DKTatlas+aseg.deep.mgz --output_dir ./hippo_output --crop --padding 10
        """,
    )

    # Required
    parser.add_argument("--input", "-i", required=True,
                        help="Path to segmentation file (.mgz, .nii, .nii.gz)")
    parser.add_argument("--output_dir", "-o", required=True,
                        help="Output directory for extracted hippocampus files")

    # Extraction mode
    parser.add_argument("--mode", "-m", default="strict",
                        choices=["strict", "moderate", "generous", "smooth", "topology_fix"],
                        help="Extraction intensity mode (default: strict)")


    # Mode-specific parameters
    parser.add_argument("--dilate", type=int, default=1,
                        help="Dilation voxels for 'generous' mode (default: 1)")
    parser.add_argument("--close_iter", type=int, default=1,
                        help="Closing iterations for 'moderate' mode (default: 1)")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Gaussian sigma for 'smooth' mode (default: 1.0)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for 'smooth' mode (default: 0.5)")

    # Options
    parser.add_argument("--extended", action="store_true",
                        help="Include Amygdala in extraction (hippocampus + amygdala)")
    parser.add_argument("--crop", action="store_true",
                        help="Crop output to ROI bounding box (smaller file)")
    parser.add_argument("--padding", type=int, default=5,
                        help="Padding voxels around ROI when cropping (default: 5)")
    parser.add_argument("--prefix", default="",
                        help="Prefix for output filenames (e.g. subject ID)")
    parser.add_argument("--intensity_image", default=None,
                        help="Optional: T1w image to extract intensity values within mask")

    args = parser.parse_args()

    # === Validate input ===
    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # === Load segmentation ===
    img, seg_data = load_segmentation(args.input)

    # === Build filename prefix ===
    prefix = f"{args.prefix}_" if args.prefix else ""

    # === Process each hemisphere ===
    for side, label in [("lh", LABEL_LEFT_HIPPO), ("rh", LABEL_RIGHT_HIPPO)]:
        side_name = "Left" if side == "lh" else "Right"
        print(f"\n{'='*50}")
        print(f"Processing {side_name} Hippocampus (label={label})")
        print(f"{'='*50}")

        # Extract mask
        if args.extended:
            ext_labels = EXTENDED_LABELS["left" if side == "lh" else "right"]
            print(f"  [EXTENDED] Including: {list(ext_labels.values())}")
            mask = extract_multi_label_mask(seg_data, ext_labels)
        else:
            mask = extract_binary_mask(seg_data, label)

        if mask.sum() == 0:
            print(f"  [SKIP] No voxels found for {side_name} Hippocampus")
            continue

        # Apply extraction mode
        mask = apply_mode(
            mask, args.mode,
            dilate=args.dilate,
            close_iter=args.close_iter,
            sigma=args.sigma,
            threshold=args.threshold,
        )

        print(f"  Final mask voxels: {mask.sum()}")

        # Calculate volume
        voxel_sizes = img.header.get_zooms()[:3]
        voxel_vol = np.prod(voxel_sizes)
        volume_mm3 = mask.sum() * voxel_vol
        print(f"  Volume: {volume_mm3:.2f} mm^3 ({volume_mm3/1000:.3f} cm^3)")

        # Crop if requested
        crop_origin = None
        save_data = mask
        if args.crop:
            _, save_data, crop_origin = crop_to_roi(seg_data, mask, args.padding)
            save_data = mask
            if crop_origin is not None:
                mins, maxs = crop_origin
                slices = tuple(slice(mn, mx) for mn, mx in zip(mins, maxs))
                save_data = mask[slices]
                print(f"  Cropped: {mask.shape} → {save_data.shape}")

        # Save binary mask
        mask_filename = f"{side}_{prefix}hippocampus.nii.gz"
        mask_path = os.path.join(args.output_dir, mask_filename)
        save_nifti(save_data, img, mask_path, crop_origin)

        # Optionally extract intensity values from T1w
        if args.intensity_image and os.path.isfile(args.intensity_image):
            print(f"  [INTENSITY] Extracting from: {args.intensity_image}")
            t1_img = nib.load(args.intensity_image)
            t1_data = np.asarray(t1_img.dataobj)

            # Masked intensity
            intensity_data = (t1_data * mask).astype(np.float32)

            if args.crop and crop_origin is not None:
                mins, maxs = crop_origin
                slices = tuple(slice(mn, mx) for mn, mx in zip(mins, maxs))
                intensity_data = intensity_data[slices]

            int_filename = f"{side}_{prefix}hippocampus_intensity.nii.gz"
            int_path = os.path.join(args.output_dir, int_filename)
            int_img = nib.Nifti1Image(intensity_data, img.affine, img.header)
            nib.save(int_img, int_path)
            print(f"  [SAVED] {int_path}")

            # Stats
            vals = t1_data[mask > 0]
            print(f"  Intensity stats: mean={vals.mean():.1f}, std={vals.std():.1f}, "
                  f"min={vals.min():.1f}, max={vals.max():.1f}")

    # === Summary ===
    print(f"\n{'='*50}")
    print(f"Done! Output files saved to: {args.output_dir}")
    print(f"Mode: {args.mode}")
    if args.extended:
        print(f"Extended: Yes (includes Amygdala)")
    if args.crop:
        print(f"Cropped: Yes (padding={args.padding})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
