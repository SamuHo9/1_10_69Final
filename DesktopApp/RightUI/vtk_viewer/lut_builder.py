import vtk
import numpy as np

def build_lut_gradcam():
    """Jet colormap matching view_gradcam_plsda_top3.py (0.0 Blue to 1.0 Red)."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(0.0, 1.0)
    lut.Build()
    for i in range(256):
        t = i / 255.0
        r = np.clip(1.5 - abs(4.0 * t - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - abs(4.0 * t - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - abs(4.0 * t - 1.0), 0.0, 1.0)
        lut.SetTableValue(i, float(r), float(g), float(b), 1.0)
    return lut

def build_lut_signed_distance(max_val=0.16):
    """Diverging Blue-White-Red colormap (-max to +max mm) matching view_gradcam_plsda_top3.py."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(-max_val, max_val)
    lut.Build()
    for i in range(256):
        t = (i / 255.0) * 2.0 - 1.0
        if t < 0:
            frac = 1.0 + t
            r, g, b = frac, frac, 1.0
        else:
            frac = 1.0 - t
            r, g, b = 1.0, frac, frac
        lut.SetTableValue(i, float(r), float(g), float(b), 1.0)
    return lut

def build_lut_distance_mapping(max_val=0.16):
    """Jet colormap matching view_gradcam_plsda_top3.py (0.0 Blue to max mm Red)."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(0.0, max_val)
    lut.Build()
    for i in range(256):
        t = i / 255.0
        r = np.clip(1.5 - abs(4.0 * t - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - abs(4.0 * t - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - abs(4.0 * t - 1.0), 0.0, 1.0)
        lut.SetTableValue(i, float(r), float(g), float(b), 1.0)
    return lut
