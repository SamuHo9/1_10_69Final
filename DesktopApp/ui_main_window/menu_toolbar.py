import webbrowser
from PyQt6.QtWidgets import QLabel, QComboBox
from PyQt6.QtGui import QAction, QKeySequence

def setup_menus_and_toolbar(window):
    """
    Constructs the application MenuBar and Main Toolbar, binding all actions
    and keyboard shortcuts to the MainWindow instance.
    """
    menubar = window.menuBar()

    # ====================================================
    # 1. FILE MENU
    # ====================================================
    file_menu = menubar.addMenu("&File")

    open_folder_act = QAction("📂 &Open Dataset Folder...", window)
    open_folder_act.setShortcut(QKeySequence("Ctrl+O"))
    open_folder_act.setStatusTip("Select MRI patient dataset folder")
    open_folder_act.triggered.connect(window.open_dataset_folder)
    file_menu.addAction(open_folder_act)

    choose_out_act = QAction("📁 Choose &Output Folder...", window)
    choose_out_act.setShortcut(QKeySequence("Ctrl+Shift+O"))
    choose_out_act.setStatusTip("Select directory for pipeline output results")
    choose_out_act.triggered.connect(window.choose_output_folder)
    file_menu.addAction(choose_out_act)

    open_mesh_act = QAction("🧊 Open Single &3D Mesh...", window)
    open_mesh_act.setShortcut(QKeySequence("Ctrl+M"))
    open_mesh_act.setStatusTip("Directly load and view a .vtk, .stl, or .ply mesh")
    open_mesh_act.triggered.connect(window.open_single_mesh)
    file_menu.addAction(open_mesh_act)

    file_menu.addSeparator()

    snap_act = QAction("📷 &Capture 3D Viewport (PNG)...", window)
    snap_act.setShortcut(QKeySequence("Ctrl+P"))
    snap_act.setStatusTip("Save current 3D viewport rendering as a high-resolution PNG")
    snap_act.triggered.connect(window.capture_3d_screenshot)
    file_menu.addAction(snap_act)

    file_menu.addSeparator()

    clear_view_act = QAction("🧹 Clear 3D View", window)
    clear_view_act.setShortcut(QKeySequence("Ctrl+W"))
    clear_view_act.setStatusTip("Clear current 3D mesh and overlay rendering")
    clear_view_act.triggered.connect(window.clear_3d_view)
    file_menu.addAction(clear_view_act)

    exit_act = QAction("❌ E&xit", window)
    exit_act.setShortcut(QKeySequence("Alt+F4"))
    exit_act.setStatusTip("Close the application")
    exit_act.triggered.connect(window.close)
    file_menu.addAction(exit_act)

    # ====================================================
    # 2. EDIT MENU
    # ====================================================
    edit_menu = menubar.addMenu("&Edit")

    copy_log_act = QAction("📋 Copy Terminal Logs", window)
    copy_log_act.setShortcut(QKeySequence("Ctrl+Shift+C"))
    copy_log_act.setStatusTip("Copy console log messages to clipboard")
    copy_log_act.triggered.connect(window.copy_console_logs)
    edit_menu.addAction(copy_log_act)

    clear_log_act = QAction("🗑️ Clear Terminal Logs", window)
    clear_log_act.setShortcut(QKeySequence("Ctrl+L"))
    clear_log_act.setStatusTip("Clear all messages from execution terminal")
    clear_log_act.triggered.connect(window.clear_console_logs)
    edit_menu.addAction(clear_log_act)

    edit_menu.addSeparator()

    pref_act = QAction("⚙️ &Preferences...", window)
    pref_act.setShortcut(QKeySequence("Ctrl+,"))
    pref_act.setStatusTip("Configure SlicerSALT path, compute device, and 3D theme")
    pref_act.triggered.connect(window.show_preferences)
    edit_menu.addAction(pref_act)

    # ====================================================
    # 3. VIEW MENU
    # ====================================================
    view_menu = menubar.addMenu("&View")

    window.toggle_left_action = QAction("Toggle Left Sidebar", window, checkable=True)
    window.toggle_left_action.setChecked(True)
    window.toggle_left_action.setShortcut(QKeySequence("Ctrl+B"))
    window.toggle_left_action.setStatusTip("Hide or show the left parameter and module panel")
    window.toggle_left_action.triggered.connect(window.toggle_left_sidebar)
    view_menu.addAction(window.toggle_left_action)

    window.toggle_terminal_action = QAction("Terminal / Execution Console", window, checkable=True)
    window.toggle_terminal_action.setChecked(True)
    window.toggle_terminal_action.setShortcut(QKeySequence("Ctrl+`"))
    window.toggle_terminal_action.setStatusTip("Hide or show the bottom execution terminal")
    window.toggle_terminal_action.triggered.connect(window.set_terminal_visible)
    view_menu.addAction(window.toggle_terminal_action)

    toggle_toolbar_act = QAction("Main Toolbar", window, checkable=True)
    toggle_toolbar_act.setChecked(True)
    toggle_toolbar_act.setStatusTip("Hide or show the top main toolbar")
    toggle_toolbar_act.triggered.connect(lambda vis: window.main_toolbar.setVisible(vis))
    view_menu.addAction(toggle_toolbar_act)

    view_menu.addSeparator()

    window.fullscreen_action = QAction("Fullscreen Mode", window, checkable=True)
    window.fullscreen_action.setShortcut(QKeySequence("F11"))
    window.fullscreen_action.setStatusTip("Enter or exit fullscreen mode")
    window.fullscreen_action.triggered.connect(window.toggle_fullscreen)
    view_menu.addAction(window.fullscreen_action)

    # ====================================================
    # 4. TOOLS MENU
    # ====================================================
    tools_menu = menubar.addMenu("&Tools")

    run_all_act = QAction("▶ &Run Full Pipeline", window)
    run_all_act.setShortcut(QKeySequence("Ctrl+R"))
    run_all_act.setStatusTip("Execute complete batch pipeline (FastSurfer -> ICP -> SPHARM -> Result)")
    run_all_act.triggered.connect(window.run_full_pipeline)
    tools_menu.addAction(run_all_act)

    tools_menu.addSeparator()

    module_submenu = tools_menu.addMenu("🧭 Switch Module")
    modules = [
        ("Main Dashboard", "Alt+1", "Main Panel"),
        ("1. Data Importer", "Alt+2", "Data Importer"),
        ("2. FastSurfer Segmentation", "Alt+3", "FastSurfer Segmentation"),
        ("3. ICP Rigid Registration", "Alt+4", "ICP Registration"),
        ("4. SPHARM Processing", "Alt+5", "SPHARM Processing"),
        ("5. Result & Prediction Panel", "Alt+6", "Result Panel")
    ]
    for name, shortcut, target in modules:
        m_act = QAction(f"{name} ({shortcut})", window)
        m_act.setShortcut(QKeySequence(shortcut))
        m_act.triggered.connect(lambda checked=False, t=target: window.module_combo.setCurrentText(t))
        module_submenu.addAction(m_act)

    tools_menu.addSeparator()

    diag_act = QAction("🩺 System & Hardware Diagnostics...", window)
    diag_act.setStatusTip("Check GPU/CUDA acceleration, SlicerSALT path, and environment")
    diag_act.triggered.connect(window.show_diagnostics)
    tools_menu.addAction(diag_act)

    # ====================================================
    # 5. HELP MENU
    # ====================================================
    help_menu = menubar.addMenu("&Help")

    salt_web_act = QAction("🌐 SlicerSALT Official Website", window)
    salt_web_act.setStatusTip("Open SlicerSALT download and documentation portal")
    salt_web_act.triggered.connect(lambda: webbrowser.open("https://salt.slicer.org/"))
    help_menu.addAction(salt_web_act)

    help_menu.addSeparator()

    about_act = QAction("ℹ️ About Hippocampal Shape Analysis Toolbox", window)
    about_act.setStatusTip("View software version, authors, and methodology info")
    about_act.triggered.connect(window.show_about)
    help_menu.addAction(about_act)

    # ====================================================
    # 6. MAIN TOOLBAR
    # ====================================================
    toolbar = window.addToolBar("Main Toolbar")
    toolbar.setMovable(False)
    window.main_toolbar = toolbar

    tb_mesh = QAction("🧊 Open Mesh", window)
    tb_mesh.setToolTip("Open single 3D mesh (.vtk, .ply, .stl) (Ctrl+M)")
    tb_mesh.triggered.connect(window.open_single_mesh)
    toolbar.addAction(tb_mesh)

    tb_snap = QAction("📷 Snapshot", window)
    tb_snap.setToolTip("Capture 3D Viewport Image (Ctrl+P)")
    tb_snap.triggered.connect(window.capture_3d_screenshot)
    toolbar.addAction(tb_snap)

    toolbar.addSeparator()

    toolbar.addWidget(QLabel("  Modules: "))
    module_combo = QComboBox()
    module_combo.addItems([
        "Main Panel",
        "Data Importer",
        "FastSurfer Segmentation",
        "ICP Registration",
        "SPHARM Processing",
        "Result Panel"
    ])
    module_combo.setMinimumWidth(190)
    toolbar.addWidget(module_combo)
    window.module_combo = module_combo

    toolbar.addSeparator()

    tb_cam = QAction("⌖ Center Camera", window)
    tb_cam.setToolTip("Reset and Center 3D Camera (Space)")
    tb_cam.triggered.connect(window.reset_3d_camera)
    toolbar.addAction(tb_cam)

    return module_combo, toolbar
