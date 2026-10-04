import numpy as np
import scipy.special as sp
import vtk
import os
import re
import sys

def parse_coef(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    pattern = re.compile(r"\{([-+]?[\d\.eE+-]+),\s*([-+]?[\d\.eE+-]+),\s*([-+]?[\d\.eE+-]+)\}")
    matches = pattern.findall(content)
    
    coeffs = []
    for m in matches:
        coeffs.append([float(x) for x in m])
    
    num_match = re.search(r"\{\s*(\d+)", content)
    if num_match:
        num_coeffs = int(num_match.group(1))
        return coeffs[:num_coeffs]
    
    return coeffs

def evaluate_spharm(coeffs_list, L, theta_grid, phi_grid):
    THETA, PHI = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    
    X = np.zeros_like(THETA)
    Y = np.zeros_like(THETA)
    Z = np.zeros_like(THETA)
    
    idx = 0
    for l in range(L + 1):
        c = coeffs_list[idx]
        y = sp.sph_harm(0, l, PHI, THETA).real
        X += c[0] * y
        Y += c[1] * y
        Z += c[2] * y
        idx += 1
        
        for m in range(1, l + 1):
            cr = coeffs_list[idx]
            idx += 1
            ci = coeffs_list[idx]
            idx += 1
            
            y_comp = sp.sph_harm(m, l, PHI, THETA)
            
            factor = np.sqrt(2)
            
            X += factor * (cr[0] * y_comp.real + ci[0] * y_comp.imag)
            Y += factor * (cr[1] * y_comp.real + ci[1] * y_comp.imag)
            Z += factor * (cr[2] * y_comp.real + ci[2] * y_comp.imag)
            
    return X, Y, Z

def save_grid_vtk(X, Y, Z, theta_deg, phi_deg, output_path):
    num_theta = X.shape[0]
    num_phi = X.shape[1]
    
    points = vtk.vtkPoints()
    theta_arr = vtk.vtkFloatArray()
    theta_arr.SetName("_paraTheta")
    phi_arr = vtk.vtkFloatArray()
    phi_arr.SetName("_paraPhi")
    
    for i in range(num_theta):
        for j in range(num_phi):
            points.InsertNextPoint(X[i, j], Y[i, j], Z[i, j])
            theta_arr.InsertNextValue(np.radians(theta_deg[i]))
            phi_arr.InsertNextValue(np.radians(phi_deg[j]))
            
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.GetPointData().AddArray(theta_arr)
    poly.GetPointData().AddArray(phi_arr)
    
    cells = vtk.vtkCellArray()
    for i in range(num_theta - 1):
        for j in range(num_phi):
            j_next = (j + 1) % num_phi
            p1 = i * num_phi + j
            p2 = i * num_phi + j_next
            p3 = (i + 1) * num_phi + j_next
            p4 = (i + 1) * num_phi + j
            
            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, p1)
            quad.GetPointIds().SetId(1, p2)
            quad.GetPointIds().SetId(2, p3)
            quad.GetPointIds().SetId(3, p4)
            cells.InsertNextCell(quad)
            
    poly.SetPolys(cells)
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(poly)
    writer.Write()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python resample_spharm_grid.py <input.coef> <output.vtk> [theta_step=9] [phi_step=9]")
        sys.exit(1)
        
    input_coef = sys.argv[1]
    output_vtk = sys.argv[2]
    theta_step = float(sys.argv[3]) if len(sys.argv) > 3 else 9.0
    phi_step = float(sys.argv[4]) if len(sys.argv) > 4 else 9.0
    
    if not os.path.exists(input_coef):
        print(f"Error: {input_coef} not found.")
        sys.exit(1)
        
    coeffs = parse_coef(input_coef)
    L = int(np.sqrt(len(coeffs))) - 1
    
    theta_deg = np.linspace(0, 180, int(180/theta_step) + 1)
    phi_deg = np.linspace(0, 360, int(360/phi_step) + 1)[:-1]
    
    if not any(np.isclose(phi_deg, 270)):
        pass

    X, Y, Z = evaluate_spharm(coeffs, L, np.radians(theta_deg), np.radians(phi_deg))
    save_grid_vtk(X, Y, Z, theta_deg, phi_deg, output_vtk)
    print(f"Successfully resampled to {output_vtk} ({len(theta_deg)}x{len(phi_deg)} grid)")
