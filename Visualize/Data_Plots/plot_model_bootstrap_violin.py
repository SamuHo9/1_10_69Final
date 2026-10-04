"""
================================================================================
Model Bootstrap Violin Plot Generator: Direct from Latest Model Predictions (.npz)
================================================================================
Computes 1,000 paired/individual bootstrap iterations dynamically from the
latest `test_predictions.npz` files for each model architecture:
- Ds005602 (Left, Right)
- Ds004469 (Left, Right)
- All_Augment_tain (Left, Right)
- Architectures: MLP, MobileNet, PointNet, ResNet+AE, ResNet, SqueezeNet, SVM
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

# Set scientific visualization aesthetic
sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "figure.titlesize": 14
})

MODEL_PALETTE = {
    "MLP": "#7f8c8d",          # Neutral Gray
    "MobileNet": "#3498db",    # Blue
    "PointNet": "#9b59b6",     # Purple
    "ResNet+AE": "#e67e22",    # Orange
    "ResNet": "#e74c3c",       # Crimson Red (Highlighted)
    "SqueezeNet": "#1abc9c",   # Teal
    "SVM": "#2ecc71"           # Emerald Green
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

def load_bootstrap_data_from_csv(csv_path, legend_path):
    """
    Loads bootstrap distributions directly from the user's latest Combined_Bootstrap_Results.csv
    in Model_Results_Excel, perfectly matching their official benchmark summary.
    """
    df_boot = pd.read_csv(csv_path)
    
    # Parse legend
    legend_mapping = {}
    with open(legend_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            parts = line.split(":")
            try:
                idx = int(parts[0].strip())
                name = parts[1].strip()
                legend_mapping[idx] = name
            except ValueError:
                continue
                
    return df_boot, legend_mapping

def process_cohort_from_csv(dataset, side, df_boot, legend_mapping, base_output_dir, gradcam_plot_dir=None):
    cohort_title = f"{dataset} [{side.capitalize()} Hippocampus]"
    print(f"\nProcessing Bootstrap Violin Plots from CSV for: {cohort_title}...")
    
    out_dir = os.path.join(base_output_dir, dataset, side)
    os.makedirs(out_dir, exist_ok=True)
    
    # Target models
    target_models = ["MLP", "MobileNet", "PointNet", "ResNet+AE", "ResNet", "SqueezeNet", "SVM"]
    
    # Find matching columns
    rows = []
    summary_stats = []
    
    for m_name in target_models:
        key_name = f"{dataset}_{side}_{m_name}"
        # Match index
        found_idx = None
        for idx, full_name in legend_mapping.items():
            if full_name == key_name:
                found_idx = idx
                break
                
        if found_idx is None:
            print(f"  [WARN] Model {key_name} not found in legend!")
            continue
            
        acc_col = f"Accuracy_{found_idx}"
        auc_col = f"AUC_{found_idx}"
        f1_col = f"F1_{found_idx}"
        sens_col = f"Sensitivity_{found_idx}"
        spec_col = f"Specificity_{found_idx}"
        
        accs = df_boot[acc_col].values
        aucs = df_boot[auc_col].values
        f1s = df_boot[f1_col].values
        senss = df_boot[sens_col].values
        specs = df_boot[spec_col].values
        
        summary_stats.append({
            "Model": m_name,
            "AUC_Mean": float(np.mean(aucs)),
            "AUC_SD": float(np.std(aucs)),
            "AUC_Median": float(np.median(aucs)),
            "Accuracy_Mean": float(np.mean(accs)),
            "Accuracy_SD": float(np.std(accs)),
            "F1_Mean": float(np.mean(f1s)),
            "F1_SD": float(np.std(f1s)),
            "Sensitivity_Mean": float(np.mean(senss)),
            "Specificity_Mean": float(np.mean(specs)),
        })
        
        for r_i in range(len(accs)):
            rows.append({
                "Round": r_i + 1,
                "Model": m_name,
                "Accuracy": accs[r_i],
                "AUC": aucs[r_i],
                "F1 Score": f1s[r_i],
                "Sensitivity": senss[r_i],
                "Specificity": specs[r_i]
            })
            
    if not rows:
        print(f"  [ERROR] No data found for {cohort_title}")
        return
        
    df_tidy = pd.DataFrame(rows)
    df_sum = pd.DataFrame(summary_stats)
    
    # Save CSVs: Both Summary Statistics and Complete 1,000-Round Raw Iterations
    stats_csv = os.path.join(out_dir, f"bootstrap_1000_summary_{dataset}_{side}.csv")
    df_sum.to_csv(stats_csv, index=False)
    
    raw_csv = os.path.join(out_dir, f"bootstrap_1000_raw_iterations_{dataset}_{side}.csv")
    df_tidy.to_csv(raw_csv, index=False)
    
    if gradcam_plot_dir:
        df_sum.to_csv(os.path.join(gradcam_plot_dir, f"bootstrap_1000_summary_{dataset}_{side}.csv"), index=False)
        df_tidy.to_csv(os.path.join(gradcam_plot_dir, f"bootstrap_1000_raw_iterations_{dataset}_{side}.csv"), index=False)
        
    # Best models
    available_models = [r["Model"] for r in summary_stats]
    best_auc_row = df_sum.sort_values("AUC_Mean", ascending=False).iloc[0]
    best_auc_model = best_auc_row["Model"]
    best_auc_val = best_auc_row["AUC_Mean"]
    
    best_acc_row = df_sum.sort_values("Accuracy_Mean", ascending=False).iloc[0]
    best_acc_model = best_acc_row["Model"]
    best_acc_val = best_acc_row["Accuracy_Mean"]

    # 1. Figure 1: AUC Violin
    fig, ax = plt.subplots(figsize=(11, 6))
    best_pos = available_models.index(best_auc_model)
    ax.axvspan(best_pos - 0.45, best_pos + 0.45, color="#fef9e7", alpha=0.8, zorder=0)
    ax.text(best_pos, 1.02, f"★ Top AUC: {best_auc_val:.3f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#d35400")

    sns.violinplot(
        data=df_tidy, x="Model", y="AUC", hue="Model", palette=MODEL_PALETTE,
        legend=False, inner="quartile", cut=0, linewidth=1.2, ax=ax
    )
    for pos, m_name in enumerate(available_models):
        m_mean = df_sum[df_sum["Model"] == m_name]["AUC_Mean"].values[0]
        m_sd = df_sum[df_sum["Model"] == m_name]["AUC_SD"].values[0]
        ax.plot(pos, m_mean, marker="o", markersize=6, color="white", markeredgecolor="black", markeredgewidth=1.2, zorder=5)
        ax.text(pos, m_mean + 0.02, f"{m_mean:.3f}\n±{m_sd:.3f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#2c3e50")
                
    ax.set_title(f"Bootstrap ROC-AUC Distribution (1,000 Iterations - Violin): {cohort_title}\n[Source: Combined_Bootstrap_Results.csv]",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Model Architecture", fontsize=12, fontweight="bold")
    ax.set_ylabel("ROC-AUC Score", fontsize=12, fontweight="bold")
    ax.set_ylim(min(0.5, df_tidy["AUC"].min() - 0.05), 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    fig1_path = os.path.join(out_dir, f"bootstrap_violin_auc_{dataset}_{side}.png")
    plt.savefig(fig1_path, dpi=300)
    if gradcam_plot_dir:
        plt.savefig(os.path.join(gradcam_plot_dir, "model_bootstrap_violin_auc.png"), dpi=300)
    plt.close()
    print(f"  -> Saved AUC Violin plot: {fig1_path}")

    # 1B. Figure 1B: AUC Box Plot (Publication Standard)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axvspan(best_pos - 0.45, best_pos + 0.45, color="#fef9e7", alpha=0.8, zorder=0)
    ax.text(best_pos, 1.02, f"★ Top AUC: {best_auc_val:.3f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#d35400")

    sns.boxplot(
        data=df_tidy, x="Model", y="AUC", hue="Model", palette=MODEL_PALETTE,
        showmeans=True, meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 7},
        boxprops={"alpha": 0.85}, ax=ax, legend=False
    )
    for pos, m_name in enumerate(available_models):
        m_mean = df_sum[df_sum["Model"] == m_name]["AUC_Mean"].values[0]
        m_sd = df_sum[df_sum["Model"] == m_name]["AUC_SD"].values[0]
        ax.text(pos, m_mean + 0.02, f"{m_mean:.3f}\n±{m_sd:.3f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#2c3e50")

    ax.set_title(f"Bootstrap ROC-AUC Distribution (1,000 Iterations - Box Plot): {cohort_title}\n[Source: Combined_Bootstrap_Results.csv]",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Model Architecture", fontsize=12, fontweight="bold")
    ax.set_ylabel("ROC-AUC Score", fontsize=12, fontweight="bold")
    ax.set_ylim(min(0.5, df_tidy["AUC"].min() - 0.05), 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig1b_path = os.path.join(out_dir, f"bootstrap_boxplot_auc_{dataset}_{side}.png")
    plt.savefig(fig1b_path, dpi=300)
    if gradcam_plot_dir:
        plt.savefig(os.path.join(gradcam_plot_dir, "model_bootstrap_boxplot_auc.png"), dpi=300)
    plt.close()
    print(f"  -> Saved AUC Box plot: {fig1b_path}")

    # 2. Figure 2: Accuracy Violin
    fig, ax = plt.subplots(figsize=(11, 6))
    best_acc_pos = available_models.index(best_acc_model)
    ax.axvspan(best_acc_pos - 0.45, best_acc_pos + 0.45, color="#eafaf1", alpha=0.8, zorder=0)
    ax.text(best_acc_pos, 1.02, f"★ Top Acc: {best_acc_val*100:.1f}%", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#27ae60")

    sns.violinplot(
        data=df_tidy, x="Model", y="Accuracy", hue="Model", palette=MODEL_PALETTE,
        legend=False, inner="quartile", cut=0, linewidth=1.2, ax=ax
    )
    for pos, m_name in enumerate(available_models):
        m_mean = df_sum[df_sum["Model"] == m_name]["Accuracy_Mean"].values[0]
        m_sd = df_sum[df_sum["Model"] == m_name]["Accuracy_SD"].values[0]
        ax.plot(pos, m_mean, marker="o", markersize=6, color="white", markeredgecolor="black", markeredgewidth=1.2, zorder=5)
        ax.text(pos, m_mean + 0.02, f"{m_mean*100:.1f}%\n±{m_sd*100:.1f}%", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#2c3e50")
                
    ax.set_title(f"Bootstrap Classification Accuracy (1,000 Iterations - Violin): {cohort_title}\n[Source: Combined_Bootstrap_Results.csv]",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Model Architecture", fontsize=12, fontweight="bold")
    ax.set_ylabel("Classification Accuracy", fontsize=12, fontweight="bold")
    ax.set_ylim(min(0.5, df_tidy["Accuracy"].min() - 0.05), 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    fig2_path = os.path.join(out_dir, f"bootstrap_violin_accuracy_{dataset}_{side}.png")
    plt.savefig(fig2_path, dpi=300)
    if gradcam_plot_dir:
        plt.savefig(os.path.join(gradcam_plot_dir, "model_bootstrap_violin_accuracy.png"), dpi=300)
    plt.close()
    print(f"  -> Saved Accuracy Violin plot: {fig2_path}")

    # 2B. Figure 2B: Accuracy Box Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axvspan(best_acc_pos - 0.45, best_acc_pos + 0.45, color="#eafaf1", alpha=0.8, zorder=0)
    ax.text(best_acc_pos, 1.02, f"★ Top Acc: {best_acc_val*100:.1f}%", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#27ae60")

    sns.boxplot(
        data=df_tidy, x="Model", y="Accuracy", hue="Model", palette=MODEL_PALETTE,
        showmeans=True, meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 7},
        boxprops={"alpha": 0.85}, ax=ax, legend=False
    )
    for pos, m_name in enumerate(available_models):
        m_mean = df_sum[df_sum["Model"] == m_name]["Accuracy_Mean"].values[0]
        m_sd = df_sum[df_sum["Model"] == m_name]["Accuracy_SD"].values[0]
        ax.text(pos, m_mean + 0.02, f"{m_mean*100:.1f}%\n±{m_sd*100:.1f}%", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#2c3e50")

    ax.set_title(f"Bootstrap Classification Accuracy (1,000 Iterations - Box Plot): {cohort_title}\n[Source: Combined_Bootstrap_Results.csv]",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Model Architecture", fontsize=12, fontweight="bold")
    ax.set_ylabel("Classification Accuracy", fontsize=12, fontweight="bold")
    ax.set_ylim(min(0.5, df_tidy["Accuracy"].min() - 0.05), 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig2b_path = os.path.join(out_dir, f"bootstrap_boxplot_accuracy_{dataset}_{side}.png")
    plt.savefig(fig2b_path, dpi=300)
    if gradcam_plot_dir:
        plt.savefig(os.path.join(gradcam_plot_dir, "model_bootstrap_boxplot_accuracy.png"), dpi=300)
    plt.close()
    print(f"  -> Saved Accuracy Box plot: {fig2b_path}")

    # 3. Figure 3: 4-Metrics Dashboard Violin
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    sns.violinplot(data=df_tidy, x="Model", y="AUC", hue="Model", palette=MODEL_PALETTE, legend=False,
                   inner="quartile", cut=0, ax=axes[0, 0])
    axes[0, 0].set_title("(A) ROC-AUC (1,000 Bootstraps)", fontweight="bold", fontsize=12)
    axes[0, 0].set_ylabel("AUC Score", fontweight="bold")
    axes[0, 0].set_ylim(min(0.5, df_tidy["AUC"].min() - 0.04), 1.02)
    
    sns.violinplot(data=df_tidy, x="Model", y="Accuracy", hue="Model", palette=MODEL_PALETTE, legend=False,
                   inner="quartile", cut=0, ax=axes[0, 1])
    axes[0, 1].set_title("(B) Classification Accuracy (1,000 Bootstraps)", fontweight="bold", fontsize=12)
    axes[0, 1].set_ylabel("Accuracy", fontweight="bold")
    axes[0, 1].set_ylim(min(0.5, df_tidy["Accuracy"].min() - 0.04), 1.02)
    
    sns.violinplot(data=df_tidy, x="Model", y="F1 Score", hue="Model", palette=MODEL_PALETTE, legend=False,
                   inner="quartile", cut=0, ax=axes[1, 0])
    axes[1, 0].set_title("(C) F1-Score (1,000 Bootstraps)", fontweight="bold", fontsize=12)
    axes[1, 0].set_ylabel("F1 Score", fontweight="bold")
    axes[1, 0].set_ylim(min(0.4, df_tidy["F1 Score"].min() - 0.04), 1.02)
    
    df_sens_spec = []
    for m in available_models:
        sub_m = df_tidy[df_tidy["Model"] == m]
        for v in sub_m["Sensitivity"]:
            df_sens_spec.append({"Model": m, "Score": v, "Metric": "Sensitivity (Recall)"})
        for v in sub_m["Specificity"]:
            df_sens_spec.append({"Model": m, "Score": v, "Metric": "Specificity"})
    df_ss = pd.DataFrame(df_sens_spec)
    
    sns.violinplot(data=df_ss, x="Model", y="Score", hue="Metric",
                   palette={"Sensitivity (Recall)": "#e74c3c", "Specificity": "#3498db"},
                   split=True, inner="quartile", cut=0, ax=axes[1, 1])
    axes[1, 1].set_title("(D) Sensitivity vs Specificity Balance (1,000 Bootstraps)", fontweight="bold", fontsize=12)
    axes[1, 1].set_ylabel("Score", fontweight="bold")
    axes[1, 1].set_ylim(min(0.3, df_ss["Score"].min() - 0.04), 1.02)
    axes[1, 1].legend(loc="lower left", framealpha=0.95)
    
    for ax_sub in axes.flat:
        ax_sub.grid(axis="y", linestyle="--", alpha=0.6)
        ax_sub.set_xlabel("")
        
    plt.suptitle(f"Multi-Metric 1,000-Round Bootstrap Evaluation (Violin): {cohort_title}\n[Based on Combined_Bootstrap_Results.csv]",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    fig3_path = os.path.join(out_dir, f"bootstrap_violin_4metrics_{dataset}_{side}.png")
    plt.savefig(fig3_path, dpi=300)
    if gradcam_plot_dir:
        plt.savefig(os.path.join(gradcam_plot_dir, "model_bootstrap_violin_4metrics.png"), dpi=300)
    plt.close()
    print(f"  -> Saved 4-Metrics Dashboard Violin: {fig3_path}")

    # 3B. Figure 3B: 4-Metrics Dashboard Box Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    sns.boxplot(data=df_tidy, x="Model", y="AUC", hue="Model", palette=MODEL_PALETTE, legend=False,
                showmeans=True, meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5},
                boxprops={"alpha": 0.85}, ax=axes[0, 0])
    axes[0, 0].set_title("(A) ROC-AUC (1,000 Bootstraps)", fontweight="bold", fontsize=12)
    axes[0, 0].set_ylabel("AUC Score", fontweight="bold")
    axes[0, 0].set_ylim(min(0.5, df_tidy["AUC"].min() - 0.04), 1.02)

    sns.boxplot(data=df_tidy, x="Model", y="Accuracy", hue="Model", palette=MODEL_PALETTE, legend=False,
                showmeans=True, meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5},
                boxprops={"alpha": 0.85}, ax=axes[0, 1])
    axes[0, 1].set_title("(B) Classification Accuracy (1,000 Bootstraps)", fontweight="bold", fontsize=12)
    axes[0, 1].set_ylabel("Accuracy", fontweight="bold")
    axes[0, 1].set_ylim(min(0.5, df_tidy["Accuracy"].min() - 0.04), 1.02)

    sns.boxplot(data=df_tidy, x="Model", y="F1 Score", hue="Model", palette=MODEL_PALETTE, legend=False,
                showmeans=True, meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5},
                boxprops={"alpha": 0.85}, ax=axes[1, 0])
    axes[1, 0].set_title("(C) F1-Score (1,000 Bootstraps)", fontweight="bold", fontsize=12)
    axes[1, 0].set_ylabel("F1 Score", fontweight="bold")
    axes[1, 0].set_ylim(min(0.4, df_tidy["F1 Score"].min() - 0.04), 1.02)

    sns.boxplot(data=df_ss, x="Model", y="Score", hue="Metric",
                palette={"Sensitivity (Recall)": "#e74c3c", "Specificity": "#3498db"},
                boxprops={"alpha": 0.85}, ax=axes[1, 1])
    axes[1, 1].set_title("(D) Sensitivity vs Specificity Balance (1,000 Bootstraps)", fontweight="bold", fontsize=12)
    axes[1, 1].set_ylabel("Score", fontweight="bold")
    axes[1, 1].set_ylim(min(0.3, df_ss["Score"].min() - 0.04), 1.02)
    axes[1, 1].legend(loc="lower left", framealpha=0.95)

    for ax_sub in axes.flat:
        ax_sub.grid(axis="y", linestyle="--", alpha=0.6)
        ax_sub.set_xlabel("")

    plt.suptitle(f"Multi-Metric 1,000-Round Bootstrap Evaluation (Box Plot): {cohort_title}\n[Based on Combined_Bootstrap_Results.csv]",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    fig3b_path = os.path.join(out_dir, f"bootstrap_boxplot_4metrics_{dataset}_{side}.png")
    plt.savefig(fig3b_path, dpi=300)
    if gradcam_plot_dir:
        plt.savefig(os.path.join(gradcam_plot_dir, "model_bootstrap_boxplot_4metrics.png"), dpi=300)
    plt.close()
    print(f"  -> Saved 4-Metrics Dashboard Box plot: {fig3b_path}")

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_output_dir = os.path.join(repo_root, "Model_Results_Excel", "05_Bootstrap_Violin_Plots")
    csv_path = os.path.join(repo_root, "Model_Results_Excel", "02_Bootstrap_and_Statistical_Tests", "Combined_Bootstrap_Results.csv")
    legend_path = os.path.join(repo_root, "Model_Results_Excel", "02_Bootstrap_and_Statistical_Tests", "Model_Legend.txt")
    
    print("=" * 70)
    print("GENERATING 1,000 BOOTSTRAP VIOLIN PLOTS DIRECTLY FROM COMBINED_BOOTSTRAP_RESULTS.CSV")
    print("=" * 70)
    
    if not os.path.exists(csv_path) or not os.path.exists(legend_path):
        print(f"[ERROR] Missing CSV or legend file at:\n  {csv_path}\n  {legend_path}")
        return
        
    df_boot, legend_mapping = load_bootstrap_data_from_csv(csv_path, legend_path)
    print(f"Loaded {len(df_boot)} bootstrap rounds and {len(legend_mapping)} models from legend.")
    
    for dataset, side in COHORTS:
        gradcam_dir = os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", dataset, side, "plots")
        if not os.path.exists(os.path.dirname(gradcam_dir)):
            gradcam_dir = None
        process_cohort_from_csv(dataset, side, df_boot, legend_mapping, base_output_dir, gradcam_dir)
        
    print("\n" + "=" * 70)
    print("ALL BOOTSTRAP VIOLIN PLOTS GENERATED SUCCESSFULLY FROM YOUR LATEST CSV!")
    print(f"Stored in: {base_output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
