import pandas as pd
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
import sys

def select_file():
    """Opens a file dialog to select a score CSV file."""
    root = tk.Tk()
    root.withdraw() # Hide the main window
    root.attributes('-topmost', True) # Bring to front
    
    file_path = filedialog.askopenfilename(
        title="Select Score CSV File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialdir=os.getcwd()
    )
    root.destroy()
    return file_path

def main():
    print("--- 2D Data Plot Visualizer ---")
    
    # 1. Select File
    csv_path = select_file()
    
    if not csv_path:
        print("No file selected. Exiting.")
        return

    print(f"Loading: {csv_path}")
    
    # 2. Setup Paths
    output_root = os.path.dirname(csv_path)
    
    # 3. Load Data
    try:
        df = pd.read_csv(csv_path)
        
        # Detect dimensions
        x_col, y_col = None, None
        method_name = ""
        if 'PLS-DA 1' in df.columns and 'PLS-DA 2' in df.columns:
            x_col, y_col = 'PLS-DA 1', 'PLS-DA 2'
            method_name = "PLS-DA"
        elif 'Comp 1' in df.columns and 'Comp 2' in df.columns:
            x_col, y_col = 'Comp 1', 'Comp 2'
            method_name = "Component"
        elif len(df.columns) >= 3:
            # Fallback to 2nd and 3rd columns (1st assumed Subject)
            x_col, y_col = df.columns[1], df.columns[2]
            method_name = "Score"
            
        if not x_col or not y_col:
            print("Error: Selected CSV must contain 2 score columns (e.g. 'PLS-DA 1', 'PLS-DA 2').")
            return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Loaded {len(df)} subjects using {method_name}.")

    # 4. Create Visualization
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    
    # Set window title
    fig.canvas.manager.set_window_title(f"{method_name} Results - {os.path.basename(csv_path)}")

    # Classify subjects into categories
    def classify_subject(subject_name):
        is_left_side = subject_name.startswith("left_") or subject_name.startswith("lh_") or "_lh" in subject_name.lower()
        name_upper = subject_name.upper()
        
        # 1. Healthy Control / Normal
        if "_HEALTHY" in name_upper or "HEALTHY" in name_upper or "HFH_" in name_upper or "NORMAL" in name_upper:
            return "Healthy Control", "royalblue"
        
        # 2. Ipsilateral TLE (Diseased)
        elif (is_left_side and "LEFT-TLE" in name_upper) or (not is_left_side and "RIGHT-TLE" in name_upper):
            return "Ipsilateral TLE (Diseased)", "crimson"
            
        # 3. Contralateral TLE (Healthy-side)
        elif (is_left_side and "RIGHT-TLE" in name_upper) or (not is_left_side and "LEFT-TLE" in name_upper):
            return "Contralateral TLE (Healthy-side)", "royalblue"
            
        # 4. General TLE
        elif "TLE" in name_upper:
            return "Ipsilateral TLE (Diseased)", "crimson"
            
        return "Unknown", "gray"

    # Map classifications
    classes_and_colors = df['Subject'].apply(classify_subject)
    df['Group'] = [c[0] for c in classes_and_colors]
    df['Color'] = [c[1] for c in classes_and_colors]

    # Plot each group separately to get clear labels in the legend
    groups = df.groupby(['Group', 'Color'])
    has_groups = False
    for (group_name, color), group_df in groups:
        ax1.scatter(group_df[x_col], group_df[y_col], c=color, alpha=0.7, edgecolors='w', s=100, label=group_name)
        has_groups = True
        
    if has_groups:
        ax1.legend(loc='best', fontsize=10)
    
    ax1.set_xlabel(x_col, fontsize=12, fontweight='bold')
    ax1.set_ylabel(y_col, fontsize=12, fontweight='bold')
    ax1.set_title(f'{method_name} Distribution\n({os.path.basename(output_root)})', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    print("Opening plot window...")
    plt.show()

if __name__ == "__main__":
    main()

