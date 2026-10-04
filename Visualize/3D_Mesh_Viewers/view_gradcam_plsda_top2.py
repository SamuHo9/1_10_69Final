#!/usr/bin/env python3
"""
================================================================================
Interactive 3D Viewer: Top 2 PLS-DA Components (PLS1 & PLS2) Distance Mapping
================================================================================
Features:
- Side-by-side synchronized viewports comparing PLS1 and PLS2 side-by-side (50% - 50% split).
- Pure White background (#FFFFFF) by default for publication clarity, toggleable via [B].
- Camera synchronization across both viewports (rotate/pan one, both follow).
- Continuous scrubbing across -3.0 to +3.0 SD (step 0.1) or key SD milestones.
- Multiple visualization modes:
  * [M1] Distance Mapping (Absolute displacement magnitude in mm)
  * [M2] Signed Deformation (Inward atrophy [Blue] vs Outward expansion [Red])
  * [M3] Grad-CAM Heatmap (ResNet class activation attention on subfields)
- Hotkeys:
  * Left/Right arrow or [ / ]: Step -0.1 / +0.1 SD
  * Space: Play / Pause continuous deformation animation
  * M: Cycle scalar mode (DistanceMapping -> SignedDistance -> GradCAM)
  * B: Toggle White / Dark Navy background theme
  * A: Toggle dynamic Auto-Scale
  * F: Front view, T: Top view, S: Side view, R: Reset view
  * W: Toggle wireframe
  * Q / Esc: Exit viewer
================================================================================
"""

import os
import sys

# Import the core viewer class from view_gradcam_plsda_top3
viewer_module_dir = os.path.dirname(os.path.abspath(__file__))
if viewer_module_dir not in sys.path:
    sys.path.insert(0, viewer_module_dir)

from view_gradcam_plsda_top3 import main as top_main

if __name__ == "__main__":
    top_main()
