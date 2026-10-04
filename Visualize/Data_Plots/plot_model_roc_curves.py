"""
================================================================================
Model ROC Curve & Bootstrap Band Visualizer: All Models in a Single Graph
================================================================================
Plots:
1. Single Graph: Mean ROC Curves of all 7 models plotted together for comparison
   with legend showing [Mean AUC ± SD] ranked from highest to lowest.
2. Grid Graph: Individual ROC curves with ±1 SD bootstrap confidence bands per model.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Set scientific visualization aesthetic
sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})

# Curated palette and line styles for distinct readability in a single graph
MODEL_STYLES = {
    "ResNet":     {"color": "#e74c3c", "lw": 2.8, "ls": "-"},     # Bold Red (Grad-CAM Backbone)
    "PointNet":   {"color": "#9b59b6", "lw": 2.4, "ls": "-"},     # Purple
    "SqueezeNet": {"color": "#16a085", "lw": 2.4, "ls": "-"},     # Teal
    "ResNet+AE":  {"color": "#e67e22", "lw": 2.0, "ls": "--"},    # Orange dashed
    "SVM":        {"color": "#27ae60", "lw": 2.0, "ls": "-."},    # Green dash-dot
    "MobileNet":  {"color": "#2980b9", "lw": 2.0, "ls": ":"},     # Blue dotted
    "MLP":        {"color": "#7f8c8d", "lw": 1.8, "ls": "--"},    # Gray dashed
}

MODELS_CONFIG = [
    ("MLP", "MLP", "test_predictions.npz"),
    ("MobileNet", "MobileNet", "test_predictions.npz"),
    ("PointNet", "PointNet", "test_predictions.npz"),
    ("ResNet+AE", "ResNet", "test_predictions_ae.npz"),
    ("ResNet", "ResNet", "test_predictions.npz"),
    ("SqueezeNet", "SqueezeNet", "test_predictions.npz"),
    ("SVM", "SVM", "test_predictions.npz"),
]

COHORTS = [
    ("Ds005602", "right"),
    ("Ds005602", "left"),
    ("Ds004469", "right"),
    ("Ds004469", "left"),
    ("All_Augment_tain", "right"),
    ("All_Augment_tain", "left"),
]

def compute_bootstrap_roc(y_true, y_prob, n_bootstraps=1000, seed=42):
    """
    Computes mean ROC curve and std bands across bootstrap resamplings.
    """
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob.flatten()
        
    rng = np.random.RandomState(seed)
    mean_fpr = np.linspace(0, 1, 101)
    tprs = []
    aucs = []
    
    n_samples = len(y_true)
    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n_samples, n_samples, replace=True)
        yt_b = y_true[boot_idx]
        yp_b = y_prob[boot_idx]
        
        # Check if both classes are present
        if len(np.unique(yt_b)) < 2:
            continue
            
        fpr, tpr, _ = roc_curve(yt_b, yp_b)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))
        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    
    # Clip bounds to [0, 1]
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1.0)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0.0)
    
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    return mean_fpr, mean_tpr, tpr_lower, tpr_upper, mean_auc, std_auc

def plot_cohort_roc(dataset, side, repo_root, base_output_dir):
    cohort_title = f"{dataset} [{side.capitalize()} Hippocampus]"
    print(f"\nProcessing ROC curves for: {cohort_title}...")
    
    out_dir = os.path.join(base_output_dir, dataset, side)
    os.makedirs(out_dir, exist_ok=True)
    
    # Check if target output exists for Grad-CAM plots
    gradcam_plots_dir = os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", dataset, side, "plots")
    if os.path.exists(os.path.dirname(gradcam_plots_dir)):
        os.makedirs(gradcam_plots_dir, exist_ok=True)
    else:
        gradcam_plots_dir = None

    roc_results = []
    for m_name, subfolder, npz_name in MODELS_CONFIG:
        npz_path = os.path.join(repo_root, "Model", dataset, side, subfolder, "results", npz_name)
        if not os.path.exists(npz_path):
            print(f"  [WARN] Missing prediction file for {m_name}: {npz_path}")
            continue
            
        data = np.load(npz_path)
        y_true = data["y_test"]
        y_prob = data["y_prob"]
        
        fpr, tpr_mean, tpr_low, tpr_high, mean_auc, std_auc = compute_bootstrap_roc(y_true, y_prob)
        
        roc_results.append({
            "model": m_name,
            "fpr": fpr,
            "tpr_mean": tpr_mean,
            "tpr_low": tpr_low,
            "tpr_high": tpr_high,
            "mean_auc": mean_auc,
            "std_auc": std_auc
        })
        
    if not roc_results:
        print(f"  [ERROR] No valid ROC results for {cohort_title}")
        return

    # Sort models by Mean AUC descending
    roc_results = sorted(roc_results, key=lambda x: x["mean_auc"], reverse=True)

    # =========================================================================
    # 1. THE REQUESTED SINGLE GRAPH: Mean ROC Curves of All Models in One Plot
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9, 7.5))
    
    # Plot diagonal reference line (Random Chance)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#95a5a6", lw=1.5, label="Chance (AUC = 0.500)", zorder=1)
    
    # Plot Mean ROC line for each model
    for item in roc_results:
        m_name = item["model"]
        style = MODEL_STYLES.get(m_name, {"color": "black", "lw": 2.0, "ls": "-"})
        label = f"{m_name:<11} : AUC = {item['mean_auc']:.3f} ± {item['std_auc']:.3f}"
        
        ax.plot(
            item["fpr"],
            item["tpr_mean"],
            color=style["color"],
            lw=style["lw"],
            linestyle=style["ls"],
            label=label,
            zorder=3 if m_name == "ResNet" else 2
        )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.03])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Mean ROC Curves Comparison (1,000 Bootstraps)\n{cohort_title}",
        fontsize=13, fontweight="bold", pad=12
    )
    
    # Clean professional legend ranked by AUC
    legend = ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#bdc3c7", title="Model Architectures (Mean AUC ± SD)")
    legend.get_title().set_fontweight("bold")
    
    # Highlight box note
    best_m = roc_results[0]
    ax.annotate(
        f"★ Best Performer: {best_m['model']} (AUC = {best_m['mean_auc']:.3f})\n"
        f"Plotted using 1,000 bootstrap mean curves per architecture.",
        xy=(0.03, 0.04), xycoords="axes fraction",
        fontsize=9.5, fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fdfefe", edgecolor="#bdc3c7", alpha=0.9)
    )

    plt.tight_layout()
    single_plot_path = os.path.join(out_dir, f"mean_roc_curves_single_graph_{dataset}_{side}.png")
    plt.savefig(single_plot_path, dpi=300)
    if gradcam_plots_dir:
        plt.savefig(os.path.join(gradcam_plots_dir, "model_mean_roc_curves_single_graph.png"), dpi=300)
    plt.close()
    print(f"  -> Saved Single Graph ROC plot: {single_plot_path}")

    # =========================================================================
    # 2. ROC Bands Grid (Each Model with its ±1 SD Shaded Confidence Ribbon)
    # =========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
    axes_flat = axes.flat
    
    for idx, item in enumerate(roc_results):
        ax_sub = axes_flat[idx]
        m_name = item["model"]
        style = MODEL_STYLES.get(m_name, {"color": "black", "lw": 2.0, "ls": "-"})
        
        # Chance line
        ax_sub.plot([0, 1], [0, 1], linestyle="--", color="#95a5a6", lw=1.2)
        
        # Shaded Confidence Band (±1 SD)
        ax_sub.fill_between(
            item["fpr"],
            item["tpr_low"],
            item["tpr_high"],
            color=style["color"],
            alpha=0.25,
            label="±1 SD Bootstrap Band"
        )
        
        # Mean ROC line
        ax_sub.plot(
            item["fpr"],
            item["tpr_mean"],
            color=style["color"],
            lw=2.2,
            label=f"Mean ROC (AUC = {item['mean_auc']:.3f})"
        )
        
        ax_sub.set_title(f"{m_name}\nAUC = {item['mean_auc']:.3f} ± {item['std_auc']:.3f}", fontweight="bold", fontsize=11)
        ax_sub.legend(loc="lower right", fontsize=8.5)
        ax_sub.grid(True, linestyle="--", alpha=0.6)
        
    # Hide the 8th unused subplot (since 7 models)
    if len(roc_results) < len(axes_flat):
        axes_flat[-1].set_visible(False)
        
    for ax_sub in axes[1, :]:
        ax_sub.set_xlabel("False Positive Rate", fontweight="bold")
    for ax_sub in axes[:, 0]:
        ax_sub.set_ylabel("True Positive Rate", fontweight="bold")
        
    plt.suptitle(
        f"Individual Model ROC Curves with ±1 SD Bootstrap Confidence Bands\n{cohort_title}",
        fontsize=14, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    bands_plot_path = os.path.join(out_dir, f"roc_bands_grid_{dataset}_{side}.png")
    plt.savefig(bands_plot_path, dpi=300)
    if gradcam_plots_dir:
        plt.savefig(os.path.join(gradcam_plots_dir, "model_roc_bands_grid.png"), dpi=300)
    plt.close()
    print(f"  -> Saved ROC Bands Grid plot: {bands_plot_path}")

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_output_dir = os.path.join(repo_root, "Model_Results_Excel", "06_ROC_Curves")
    
    print("=" * 70)
    print("GENERATING MODEL MEAN ROC CURVES IN SINGLE GRAPH & CONFIDENCE BANDS")
    print("=" * 70)
    
    for dataset, side in COHORTS:
        plot_cohort_roc(dataset, side, repo_root, base_output_dir)
        
    print("\n" + "=" * 70)
    print(f"ALL ROC PLOTS COMPLETED SUCCESSFULLY!")
    print(f"Stored in: {base_output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
