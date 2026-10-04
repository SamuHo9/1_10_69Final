# Hippocampal Shape Analysis for Epilepsy Detection

A comprehensive, end-to-end medical imaging pipeline for the automated extraction, 3D shape parameterization, statistical morphological analysis, and machine-learning-based classification of the hippocampus from structural MRI scans. 

This project aims to detect and lateralize **Temporal Lobe Epilepsy (TLE)** by identifying subtle morphometric (shape) changes and atrophy patterns in the hippocampus that are often invisible to the naked eye but statistically significant across populations.

---

## 🏛️ Project Architecture & Directory Structure

The repository is modularly designed, separating preprocessing, shape analysis, data augmentation, visualization, and machine learning into distinct components.

```text
Hippocampal-Shape-Analysis-for-Epilepsy-Detection/
│
├── DesktopApp/         # Graphical User Interface (PyQt/PySide) for end-users
├── FastSurfer/         # Deep Learning brain segmentation engine (external dependency integration)
├── ICP/                # Group-wise Rigid Alignment via Iterative Closest Point
├── SPHARM/             # Spherical Harmonics Parameterization (SPHARM-PDM) & Mesh Re-alignment
├── Data_Processing/    # Feature Extraction (ML preparation) & Synthetic Data Augmentation
├── Visualize/          # 3D Mesh Viewers and Statistical Data Plots (PLS-DA)
├── Model/              # Machine Learning classification, Cross-Dataset Validation, and Reporting
│
├── run_pipeline.py     # Entry script: MRI Preprocessing (FastSurfer -> Extraction)
├── run_everything.bat  # Master Batch Script: Full Shape Analysis Pipeline automation
└── config.py           # Global configuration variables (Paths, Threads, Checkpoints)
```

---

## ⚙️ Detailed Pipeline Workflow

### Phase 1: Data Preprocessing & Segmentation (`run_pipeline.py`)
The pipeline begins with raw patient MRI scans.
1. **Automated Segmentation (FastSurfer):** Raw T1-weighted `.nii.gz` files are fed into FastSurferCNN. It utilizes a deep learning network to rapidly segment the entire brain into standard anatomical labels (DKT Atlas) in under a minute per subject.
2. **Hippocampus Extraction (`extract_hippocampus.py`):** The pipeline isolates Label `17` (Left Hippocampus) and Label `53` (Right Hippocampus).
3. **Morphological Cleaning:** Extracted regions undergo a 3D binary closing operation (mathematical morphology) to fill internal holes and smooth rough boundaries caused by voxelization.
4. **Organized Output:** Masks are saved neatly into `left_hippocampus` and `right_hippocampus` folders, ready for shape analysis.

### Phase 2: Shape Analysis via SlicerSALT (`run_everything.bat`)
This phase relies on the **SlicerSALT** (Shape Analysis Toolbox) environment to perform robust geometric operations.

#### Step 2.1: Group-wise ICP Alignment (`ICP/ICP.py`)
* **Problem:** Patients' heads are scanned at different angles and positions in the MRI machine.
* **Solution:** The Iterative Closest Point (ICP) algorithm aligns all hippocampal masks in the dataset into a common, normalized physical coordinate space.
* **Mechanism:** It uses a group-wise approach. It repeatedly aligns all subjects to a continuously updating "mean shape" (average template) until convergence is reached. It also checks for "pole flips" (ensuring the head and tail of the hippocampus point in the correct direction).

#### Step 2.2: SPHARM-PDM Parameterization (`SPHARM/run_spharm_batch.py`)
* **Problem:** Voxel-based masks cannot be compared point-by-point across different subjects because they have different surface topographies.
* **Solution:** Spherical Harmonics Point Distribution Models (SPHARM-PDM).
* **Mechanism:** The voxel masks are mapped onto a sphere and parameterized using spherical harmonic basis functions (similar to a Fourier transform, but for 3D surfaces). This generates a uniform 3D triangular mesh (VTK file) for every subject where Vertex `N` on Patient A corresponds exactly to the anatomical location of Vertex `N` on Patient B.

#### Step 2.3: Anatomical Re-alignment (`SPHARM/realign_spharm.py`)
Even after ICP and SPHARM, meshes might have rotational phase shifts along the spherical parameterization. This script enforces strict anatomical alignment (anterior-posterior, superior-inferior, medial-lateral) using geometric landmarks (e.g., finding the furthest vertex to define the "tail").

### Phase 3: Statistical Analysis & Visualization (`Visualize/`)
* **PLS-DA (`visualize_plsda.py`):** Partial Least Squares Discriminant Analysis is applied to the high-dimensional vertex data. It identifies the specific directions of shape variance that maximize the separation between healthy controls and epilepsy patients. 
* **3D Visualizations:** Includes Python scripts using `pyvista` and `vtk` to render the 3D average shapes and overlay color maps showing areas of significant inward/outward deformation (atrophy/hypertrophy) associated with the disease.

### Phase 4: Data Augmentation & Machine Learning (`Data_Processing/` & `Model/`)
* **Feature Extraction (`extract_ml_features.py`):** Converts the complex 3D SPHARM VTK coordinates and SPHARM coefficients into structured 1D CSV vectors suitable for standard machine learning classifiers.
* **Data Augmentation (`augment_plsda_interpolation.py`):** Medical datasets are often small. This script utilizes the latent space generated by PLS-DA to interpolate and generate *synthetic yet anatomically realistic* hippocampus shapes to balance the dataset and prevent ML overfitting.
* **Model Training & Cross-Dataset Validation:** Trains classical ML models (e.g., SVM, Random Forest, Logistic Regression) to classify the disease. It strongly emphasizes robustness by training on one dataset (e.g., `Ds005602`) and validating on a completely unseen dataset (e.g., `Ds004469`).
* **Statistical Reporting (`generate_report.py`):** Automatically compiles performance metrics, ROC curves, and P-value summaries into professional PDF reports (`Model_Test_Report.pdf`).

---

## 💻 The Desktop Application (`DesktopApp/`)

To make this complex pipeline accessible to clinicians and researchers without programming knowledge, a complete Desktop UI is provided.

- **Built with:** PyQt/PySide
- **Features:** 
  - One-click dataset selection and execution.
  - Interactive parameter tuning (e.g., adjusting ICP iterations, SPHARM degrees).
  - Integrated 3D mesh viewer directly inside the application window.
  - Real-time logging console to track pipeline progress.
- **Execution:** Simply run `python DesktopApp/main.py`

---

## 🚀 Getting Started & Execution

### Prerequisites
- **OS:** Windows (Pipeline heavily utilizes `.bat` scripts and Powershell)
- **Python:** Python 3.8+ 
- **Dependencies:** `pip install -r requirements.txt` (includes `numpy`, `scipy`, `pandas`, `vtk`, `matplotlib`, `pyvista`, `scikit-learn`, `PyQt5`)
- **External Software:** 
  - **SlicerSALT (v6.0+):** Must be installed and configured in `run_everything.bat` (e.g., `C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe`).
  - **FastSurfer:** Must be cloned and checkpoints downloaded. Update the path in `config.py`.

### Running the Full Pipeline

**Step 1: Extract Hippocampus from MRI**
```bash
python run_pipeline.py --input_dir /path/to/raw/mri/
```
*Output will be organized into `/output/left_hippocampus` and `/output/right_hippocampus`.*

**Step 2: Shape Analysis (ICP + SPHARM + PLS-DA)**
Double-click `run_everything.bat` or run it from the command line:
```bash
run_everything.bat
```
A folder-picker will prompt you to select the directory containing your extracted hippocampi (e.g., the `left_hippocampus` folder). SlicerSALT will launch headlessly and process the entire batch.

**Step 3: Machine Learning & Reporting**
Navigate to the `Model/` directory and execute the Powershell script to train models and generate the PDF report:
```powershell
cd Model
./run_everything.ps1
```

---

## 🔬 Scientific Context
This project leverages structural MRI to quantify **Hippocampal Sclerosis**, the most common neuropathological finding in Drug-Resistant Temporal Lobe Epilepsy. By moving beyond simple volumetric measurements (measuring just the size/volume) and utilizing SPHARM-PDM, we can localize exactly *where* on the surface the atrophy is occurring, providing a much higher diagnostic yield and offering powerful predictive features for machine learning pipelines.
## 🤝 Acknowledgements & Credits
- **FastSurfer:** Deep learning based neuroimaging pipeline for fast and robust brain segmentation. [https://github.com/Deep-MI/FastSurfer](https://github.com/Deep-MI/FastSurfer).
- **SlicerSALT:** Open-source platform for conducting 3D shape analysis of anatomical structures. [https://salt.slicer.org/](https://salt.slicer.org/).
