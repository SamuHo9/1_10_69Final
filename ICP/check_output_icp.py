#!/usr/bin/env python3
"""GUI viewer for aligned ICP NIfTI outputs.

The viewer runs inside SlicerSALT so it does not require nibabel or
matplotlib.  It can inspect one aligned mask at a time and can build a
voxel-wise consensus from every file in an ``aligned_nifti`` folder.  In
overlay mode the selected subject is shown over the consensus in Slicer slice
views and as two 3-D surfaces.
"""

from __future__ import print_function

import argparse
import os
import re
import subprocess
import sys
import traceback


def _bootstrap_slicer():
    try:
        import slicer  # noqa: F401
        return
    except ImportError:
        pass

    slicer_exe = os.environ.get(
        "SLICER_EXE",
        r"C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe",
    )
    if not os.path.isfile(slicer_exe):
        print("[ERROR] SlicerSALT not found: {}".format(slicer_exe))
        print("        Set SLICER_EXE to the SlicerSALT executable path.")
        sys.exit(1)

    script_path = os.path.abspath(__file__)
    command = [
        slicer_exe,
        "--no-splash",
        "--python-script",
        script_path,
    ] + sys.argv[1:]
    print("[INFO] No 'slicer' module in this Python. Re-launching via SlicerSALT:")
    print("       {}".format(slicer_exe))
    try:
        process = subprocess.Popen(
            command,
            cwd=os.path.dirname(script_path),
            close_fds=True,
        )
    except Exception as error:
        print("[ERROR] Could not start SlicerSALT: {}".format(error))
        sys.exit(1)
    print("[INFO] SlicerSALT started (PID={}); the viewer window will open when startup completes.".format(process.pid))
    # The viewer is interactive.  Do not keep the console blocked while the
    # Slicer application remains open.
    sys.exit(0)


_bootstrap_slicer()

import numpy as np
import qt
import slicer
import vtk
from vtk.util.numpy_support import vtk_to_numpy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORTED_EXTENSIONS = (".nii", ".nii.gz", ".nrrd", ".hdr")


def _qt_value(widget, name):
    """Read a Qt/PythonQt value exposed as either a property or a method.

    SlicerSALT's older PythonQt bindings expose some Qt properties (for
    example ``currentIndex`` and ``value``) as plain ``int`` objects, while
    newer bindings may expose the corresponding getter as a callable.  The
    viewer must work with both forms.
    """
    value = getattr(widget, name)
    return value() if callable(value) else value


def _natural_key(path):
    name = os.path.basename(path).casefold()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def _is_supported_volume(path):
    return str(path).lower().endswith(SUPPORTED_EXTENSIONS)


def find_volume_files(path):
    """Return supported NIfTI/NRRD files from a file or folder."""
    path = os.path.abspath(os.path.expanduser(str(path)))
    if os.path.isfile(path):
        if not _is_supported_volume(path):
            raise ValueError("Unsupported volume file: {}".format(path))
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError("Input path not found: {}".format(path))

    files = []
    for root, _dirs, names in os.walk(path):
        for name in names:
            candidate = os.path.join(root, name)
            if _is_supported_volume(candidate):
                files.append(candidate)
    return sorted(files, key=_natural_key)


def _show_error(parent, title, message):
    qt.QMessageBox.critical(parent, title, str(message))


def _show_info(parent, title, message):
    qt.QMessageBox.information(parent, title, str(message))


def _load_label_volume(path):
    """Load one mask as a labelmap node for slice and surface display."""
    node = slicer.util.loadVolume(path, properties={"labelmap": True})
    if node is None:
        raise RuntimeError("Slicer could not read volume: {}".format(path))
    node.SetName("ICP_subject_" + os.path.basename(path))
    node.CreateDefaultDisplayNodes()
    display = node.GetDisplayNode()
    if display is not None and hasattr(display, "SetInterpolate"):
        display.SetInterpolate(False)
    return node


def _read_volume_array(path):
    """Read a mask and return a copied [z, y, x] array plus its IJK-to-RAS matrix."""
    node = _load_label_volume(path)
    try:
        array = np.array(slicer.util.arrayFromVolume(node), dtype=np.float32, copy=True)
        matrix = vtk.vtkMatrix4x4()
        node.GetIJKToRASMatrix(matrix)
    finally:
        slicer.mrmlScene.RemoveNode(node)
    return array, matrix


def _new_scalar_volume(name, array, matrix):
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
    slicer.util.updateVolumeFromArray(node, np.asarray(array, dtype=np.float32))
    node.SetIJKToRASMatrix(matrix)
    node.CreateDefaultDisplayNodes()
    display = node.GetDisplayNode()
    if display is not None and hasattr(display, "SetInterpolate"):
        display.SetInterpolate(False)
    if display is not None and hasattr(display, "SetWindowLevel"):
        display.SetWindowLevel(1.0, 0.5)
    return node


def _extract_surface(volume_node, model_node, color, opacity=1.0, threshold=0.5):
    """Create a RAS-space surface from a binary/mean volume."""
    image = volume_node.GetImageData() if volume_node is not None else None
    if image is None or image.GetPointData() is None or image.GetPointData().GetScalars() is None:
        model_node.SetAndObservePolyData(vtk.vtkPolyData())
        model_node.GetDisplayNode().SetVisibility(False)
        return

    if hasattr(vtk, "vtkFlyingEdges3D"):
        extractor = vtk.vtkFlyingEdges3D()
    else:
        extractor = vtk.vtkMarchingCubes()
    extractor.SetInputData(image)
    extractor.SetValue(0, float(threshold))
    extractor.Update()

    matrix = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(matrix)
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    transformer = vtk.vtkTransformPolyDataFilter()
    transformer.SetInputData(extractor.GetOutput())
    transformer.SetTransform(transform)
    transformer.Update()

    model_node.SetAndObservePolyData(transformer.GetOutput())
    if model_node.GetDisplayNode() is None:
        model_node.CreateDefaultDisplayNodes()
    display = model_node.GetDisplayNode()
    display.SetColor(float(color[0]), float(color[1]), float(color[2]))
    display.SetOpacity(float(opacity))
    display.SetVisibility(True)
    if hasattr(display, "SetRepresentationToSurface"):
        display.SetRepresentationToSurface()


def _ensure_slice_layout():
    """Show Slicer's slice/3-D viewers instead of the Home/Help module."""
    try:
        layout_manager = slicer.app.layoutManager()
        if layout_manager is None:
            return
        layout_node_class = getattr(slicer, "vtkMRMLLayoutNode", None)
        four_up = getattr(layout_node_class, "SlicerLayoutFourUpView", 501)
        layout_manager.setLayout(four_up)
        main_window = slicer.util.mainWindow() if hasattr(slicer.util, "mainWindow") else None
        if main_window is not None:
            main_window.show()
    except Exception:
        # The viewer still works with the current Slicer layout if a specific
        # SlicerSALT build does not expose the layout manager API.
        pass


def _set_slice_layers(background=None, foreground=None, opacity=0.5):
    _ensure_slice_layout()
    kwargs = {}
    if background is not None:
        kwargs["background"] = background
    kwargs["foreground"] = foreground
    if foreground is not None:
        kwargs["foregroundOpacity"] = float(opacity)
    slicer.util.setSliceViewerLayers(**kwargs)
    try:
        slicer.util.resetSliceViews()
        slicer.util.resetThreeDViews()
    except Exception:
        pass


def launch_viewer(initial_path=None):
    """Open the Slicer Qt viewer and keep all state in one dialog."""
    parent = slicer.util.mainWindow() if hasattr(slicer.util, "mainWindow") else None
    dialog = qt.QDialog(parent)
    dialog.setWindowTitle("Check ICP aligned NIfTI")
    dialog.resize(900, 650)

    layout = qt.QVBoxLayout(dialog)
    title = qt.QLabel("ตรวจรูปร่าง hippocampus หลังผ่าน ICP")
    title.setStyleSheet("font-size: 18px; font-weight: bold;")
    layout.addWidget(title)
    description = qt.QLabel(
        "เลือกโฟลเดอร์ aligned_nifti เพื่อเลื่อนดูทีละ subject "
        "หรือสร้าง consensus overlay จากทุกไฟล์ที่จัดแนวแล้ว"
    )
    description.setWordWrap(True)
    layout.addWidget(description)

    path_row = qt.QHBoxLayout()
    path_edit = qt.QLineEdit(str(initial_path or ""))
    folder_button = qt.QPushButton("เลือกโฟลเดอร์")
    file_button = qt.QPushButton("เลือกไฟล์เดี่ยว")
    path_row.addWidget(path_edit)
    path_row.addWidget(folder_button)
    path_row.addWidget(file_button)
    layout.addLayout(path_row)

    control_grid = qt.QGridLayout()
    layout.addLayout(control_grid)
    mode_label = qt.QLabel("โหมดแสดงผล:")
    mode_box = qt.QComboBox()
    mode_box.addItems([
        "ทีละไฟล์",
        "ซ้อนกับ consensus ของทุกไฟล์",
    ])
    control_grid.addWidget(mode_label, 0, 0)
    control_grid.addWidget(mode_box, 0, 1)

    subject_label = qt.QLabel("ยังไม่ได้เลือก input")
    control_grid.addWidget(subject_label, 0, 2, 1, 3)

    subject_slider = qt.QSlider(qt.Qt.Horizontal)
    subject_slider.setMinimum(0)
    subject_slider.setMaximum(0)
    subject_slider.setEnabled(False)
    subject_slider.setTracking(False)
    control_grid.addWidget(qt.QLabel("Subject:"), 1, 0)
    control_grid.addWidget(subject_slider, 1, 1, 1, 4)

    previous_button = qt.QPushButton("ก่อนหน้า")
    next_button = qt.QPushButton("ถัดไป")
    overlay_button = qt.QPushButton("สร้าง consensus overlay")
    opacity_slider = qt.QSlider(qt.Qt.Horizontal)
    opacity_slider.setRange(0, 100)
    opacity_slider.setValue(50)
    control_grid.addWidget(previous_button, 2, 0)
    control_grid.addWidget(next_button, 2, 1)
    control_grid.addWidget(overlay_button, 2, 2)
    control_grid.addWidget(qt.QLabel("ความทึบ overlay:"), 2, 3)
    control_grid.addWidget(opacity_slider, 2, 4)

    status_label = qt.QLabel("พร้อมใช้งาน")
    status_label.setWordWrap(True)
    layout.addWidget(status_label)
    close_button = qt.QPushButton("ปิด")
    close_row = qt.QHBoxLayout()
    close_row.addStretch(1)
    close_row.addWidget(close_button)
    layout.addLayout(close_row)

    state = {
        "files": [],
        "index": 0,
        "subject_node": None,
        "consensus_node": None,
        "subject_model": None,
        "consensus_model": None,
        "consensus_ready": False,
    }

    def process_events():
        try:
            qt.QApplication.processEvents()
        except Exception:
            pass

    def remove_node(key):
        node = state.get(key)
        if node is not None:
            try:
                slicer.mrmlScene.RemoveNode(node)
            except Exception:
                pass
            state[key] = None

    def update_subject_label():
        files = state["files"]
        if not files:
            subject_label.setText("ไม่พบไฟล์")
            return
        index = state["index"]
        subject_label.setText(
            "{}/{}: {}".format(index + 1, len(files), os.path.basename(files[index]))
        )

    def update_display():
        files = state["files"]
        if not files:
            return
        index = max(0, min(state["index"], len(files) - 1))
        state["index"] = index
        remove_node("subject_node")
        remove_node("subject_model")
        try:
            subject = _load_label_volume(files[index])
            subject_model = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLModelNode", "ICP_subject_surface"
            )
            subject_model.CreateDefaultDisplayNodes()
            state["subject_node"] = subject
            state["subject_model"] = subject_model

            overlay_mode = _qt_value(mode_box, "currentIndex") == 1 and state["consensus_ready"]
            if overlay_mode:
                _set_slice_layers(
                    background=state["consensus_node"],
                    foreground=subject,
                    opacity=float(_qt_value(opacity_slider, "value")) / 100.0,
                )
                _extract_surface(subject, subject_model, (0.95, 0.15, 0.10), 0.75)
                if state["consensus_model"] is not None:
                    state["consensus_model"].GetDisplayNode().SetVisibility(True)
                status_label.setText(
                    "Overlay: subject สีแดง ซ้อนกับ consensus สีเขียว | {}".format(
                        os.path.basename(files[index])
                    )
                )
            else:
                _set_slice_layers(background=subject)
                _extract_surface(subject, subject_model, (0.20, 0.65, 0.95), 1.0)
                if state["consensus_model"] is not None:
                    state["consensus_model"].GetDisplayNode().SetVisibility(False)
                status_label.setText("แสดง subject: {}".format(os.path.basename(files[index])))
            update_subject_label()
        except Exception as error:
            status_label.setText("โหลดไม่สำเร็จ: {}".format(error))
            _show_error(dialog, "เปิดไฟล์ไม่สำเร็จ", error)

    def load_path(path):
        try:
            files = find_volume_files(path)
        except Exception as error:
            _show_error(dialog, "อ่าน input ไม่สำเร็จ", error)
            return
        if not files:
            _show_error(dialog, "ไม่พบไฟล์", "ไม่พบ .nii/.nii.gz/.nrrd/.hdr ใน input ที่เลือก")
            return
        state["files"] = files
        state["index"] = 0
        state["consensus_ready"] = False
        remove_node("consensus_node")
        remove_node("consensus_model")
        subject_slider.setRange(0, max(0, len(files) - 1))
        subject_slider.setEnabled(len(files) > 1)
        subject_slider.setValue(0)
        update_subject_label()
        status_label.setText("พบ {} volume(s); กำลังแสดงไฟล์แรก".format(len(files)))
        update_display()

    def choose_folder():
        selected = qt.QFileDialog.getExistingDirectory(dialog, "เลือกโฟลเดอร์ aligned_nifti")
        if selected:
            path_edit.setText(str(selected))
            load_path(str(selected))

    def choose_file():
        selected = qt.QFileDialog.getOpenFileName(
            dialog,
            "เลือก aligned NIfTI",
            "",
            "Aligned volumes (*.nii *.nii.gz *.nrrd *.hdr);;All files (*)",
        )
        if isinstance(selected, (tuple, list)):
            selected = selected[0]
        if selected:
            path_edit.setText(str(selected))
            load_path(str(selected))

    def on_subject_changed(value):
        state["index"] = int(value)
        update_subject_label()
        update_display()

    def previous_subject():
        subject_slider.setValue(max(
            int(_qt_value(subject_slider, "minimum")),
            int(_qt_value(subject_slider, "value")) - 1,
        ))

    def next_subject():
        subject_slider.setValue(min(
            int(_qt_value(subject_slider, "maximum")),
            int(_qt_value(subject_slider, "value")) + 1,
        ))

    def update_overlay_opacity(value):
        if state["subject_node"] is not None and state["consensus_node"] is not None:
            _set_slice_layers(
                background=state["consensus_node"],
                foreground=state["subject_node"],
                opacity=float(value) / 100.0,
            )

    def build_consensus():
        files = state["files"]
        if not files:
            _show_error(dialog, "ยังไม่มี input", "กรุณาเลือกโฟลเดอร์หรือไฟล์ก่อน")
            return
        overlay_button.setEnabled(False)
        try:
            status_label.setText("กำลังสร้าง consensus จาก {} ไฟล์...".format(len(files)))
            process_events()
            accumulator = None
            matrix = None
            expected_shape = None
            for index, path in enumerate(files):
                array, current_matrix = _read_volume_array(path)
                if expected_shape is None:
                    expected_shape = array.shape
                    accumulator = np.zeros(expected_shape, dtype=np.float32)
                    matrix = current_matrix
                if array.shape != expected_shape:
                    raise ValueError(
                        "Volume shape mismatch: {} has {}, expected {}".format(
                            os.path.basename(path), array.shape, expected_shape
                        )
                    )
                accumulator += (array > 0).astype(np.float32)
                if index == 0 or (index + 1) % 10 == 0:
                    status_label.setText(
                        "กำลังสร้าง consensus: {}/{}".format(index + 1, len(files))
                    )
                    process_events()

            consensus = accumulator / float(len(files))
            remove_node("consensus_node")
            remove_node("consensus_model")
            state["consensus_node"] = _new_scalar_volume(
                "ICP_consensus_occupancy", consensus, matrix
            )
            consensus_model = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLModelNode", "ICP_consensus_surface"
            )
            consensus_model.CreateDefaultDisplayNodes()
            state["consensus_model"] = consensus_model
            _extract_surface(state["consensus_node"], consensus_model, (0.15, 0.85, 0.25), 0.38)
            state["consensus_ready"] = True
            mode_box.setCurrentIndex(1)
            update_display()
            status_label.setText(
                "สร้าง consensus สำเร็จ: ค่า voxel คือสัดส่วนไฟล์ที่มี hippocampus ในตำแหน่งนั้น"
            )
        except Exception as error:
            state["consensus_ready"] = False
            status_label.setText("สร้าง consensus ไม่สำเร็จ: {}".format(error))
            _show_error(dialog, "สร้าง consensus ไม่สำเร็จ", traceback.format_exc())
        finally:
            overlay_button.setEnabled(True)

    def update_mode(*_args):
        if _qt_value(mode_box, "currentIndex") == 1 and not state["consensus_ready"]:
            status_label.setText("กด 'สร้าง consensus overlay' เพื่อรวมทุกไฟล์")
        else:
            update_display()

    folder_button.clicked.connect(choose_folder)
    file_button.clicked.connect(choose_file)
    subject_slider.valueChanged.connect(on_subject_changed)
    previous_button.clicked.connect(previous_subject)
    next_button.clicked.connect(next_subject)
    overlay_button.clicked.connect(build_consensus)
    opacity_slider.valueChanged.connect(update_overlay_opacity)
    mode_box.currentIndexChanged.connect(update_mode)
    # QPushButton.clicked carries a boolean argument in this Qt binding,
    # while QDialog.close() takes no arguments.  Use a wrapper so PythonQt
    # does not try to call close(False).
    close_button.clicked.connect(lambda *_args: dialog.close())

    if initial_path and os.path.exists(str(initial_path)):
        load_path(str(initial_path))

    exec_method = getattr(dialog, "exec_", None) or getattr(dialog, "exec")
    return exec_method()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Inspect aligned ICP NIfTI volumes in a Slicer GUI.")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--input_dir", default=None, help="aligned_nifti folder to open")
    input_group.add_argument("--input_file", default=None, help="one aligned NIfTI/NRRD file to open")
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    initial_path = args.input_dir or args.input_file
    if initial_path is None:
        left_default = os.path.join(SCRIPT_DIR, "output_left_hippocampus", "aligned_nifti")
        right_default = os.path.join(SCRIPT_DIR, "output_right_hippocampus", "aligned_nifti")
        if os.path.isdir(left_default):
            initial_path = left_default
        elif os.path.isdir(right_default):
            initial_path = right_default
    return launch_viewer(initial_path)


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = main()
    except Exception as error:
        print("[ERROR] {}".format(error))
        traceback.print_exc()
        exit_code = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        try:
            if getattr(slicer, "app", None) is not None:
                slicer.app.quit()
        except Exception:
            pass
        try:
            os._exit(exit_code)
        except Exception:
            pass
