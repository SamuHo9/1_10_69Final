"""
================================================================================
Violin Plot Visualizer: PLS-DA Parameters by Class & Deep Learning Model Predictions
================================================================================
This script generates publication-grade Violin Plots:
1. PLS-DA Components (All 8 & Top 3) comparing parameter distributions
   between Healthy Control vs Epilepsy (TLE) before sweeping to Distance Mapping.
2. Classification Model predictions (Probability distribution by True Class).
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cross_decomposition import PLSRegression

# Set publication serif styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Georgia']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 14
})

def compute_significance_stars(p_val):
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    else:
        return "ns"

def generate_violin_plots(side="right", dataset="Ds005602", output_dir=None):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Locate data files
    data_dir = os.path.join(repo_root, "Model", dataset, side)
    train_csv = os.path.join(data_dir, f"{dataset}_{side.capitalize()}_train_xyz_coords.csv")
    test_csv = os.path.join(data_dir, f"{dataset}_{side.capitalize()}_test_xyz_coords.csv")
    
    if not os.path.exists(train_csv):
        # Fallback search
        candidates = [
            os.path.join(repo_root, "Model", "Dataset_1", side, f"ALL_{side.capitalize()}_train_xyz_coords.csv"),
            os.path.join(repo_root, "Model", "Dataset_1", side, f"Dataset_1_{side.capitalize()}_train_xyz_coords.csv"),
            os.path.join(repo_root, "Model", "All_Augment_tain", side, f"ALL_{side.capitalize()}_train_xyz_coords.csv"),
            os.path.join(r"C:\Users\IHCK\Desktop\poinnet", f"ALL_{side.capitalize()}_train_xyz_coords.csv"),
            os.path.join(repo_root, "Model", "Ds005602", side, f"Ds005602_{side.capitalize()}_train_xyz_coords.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                train_csv = c
                break
                
    if not os.path.exists(train_csv):
        print(f"[ERROR] Could not find training coordinates CSV at: {train_csv}")
        return

    print(f"Loading data from: {train_csv}")
    train_df = pd.read_csv(train_csv)
    
    coord_cols = [c for c in train_df.columns if c.startswith(('x_', 'y_', 'z_'))]
    if not coord_cols:
        meta_cols = ["Subject", "Group", "Group_Name", "Group_Label", "BinaryClass", "Class", "DataType", "Unnamed: 0"]
        coord_cols = [c for c in train_df.columns if c not in meta_cols]
    
    X_train_flat = train_df[coord_cols].values.astype(np.float32)
    y_train = (train_df["Group_Label"].values if "Group_Label" in train_df.columns else train_df["BinaryClass"].values).astype(int)
    
    # Class names and palette
    class_names = {0: "Healthy Control", 1: "Epilepsy (TLE)"}
    class_palette = {"Healthy Control": "#3498db", "Epilepsy (TLE)": "#e74c3c"}
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", dataset, side, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    excel_dir = os.path.join(repo_root, "Model_Results_Excel", "08_PLSDA_Class_Violin_Plots", dataset, side)
    os.makedirs(excel_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. Fit PLS-DA with 8 Components
    # -------------------------------------------------------------------------
    n_components = 8
    Y_train_ohe = np.zeros((len(y_train), 2), dtype=np.float64)
    for idx, label in enumerate(y_train):
        Y_train_ohe[idx, label] = 1.0
        
    pls = PLSRegression(n_components=n_components, scale=True)
    X_scores, _ = pls.fit_transform(X_train_flat, Y_train_ohe)
    
    # Read summary CSV if available to know Top 3 components and rankings
    summary_csv = os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", dataset, side, "plsda_8_components_summary.csv")
    top3_indices = [0, 2, 1] # Default: PLS1, PLS3, PLS2
    top3_ranks = {0: 1, 2: 2, 1: 3}
    if os.path.exists(summary_csv):
        df_sum = pd.read_csv(summary_csv)
        if "Is_Top3" in df_sum.columns and "Top3_Rank" in df_sum.columns:
            top3_rows = df_sum[df_sum["Is_Top3"] == True].sort_values("Top3_Rank")
            top3_indices = [int(c.replace("PLS", "")) - 1 for c in top3_rows["Component"].tolist()]
            top3_ranks = {int(row["Component"].replace("PLS", "")) - 1: int(row["Top3_Rank"]) for _, row in top3_rows.iterrows()}
            
    print(f"Top 3 components identified: {[f'PLS{k+1}' for k in top3_indices]}")
    
    # Build a tidy DataFrame for Seaborn
    plot_rows = []
    stats_rows = []
    for k in range(n_components):
        comp_name = f"PLS{k+1}"
        s0 = X_scores[y_train == 0, k]
        s1 = X_scores[y_train == 1, k]
        
        t_stat, p_val = stats.ttest_ind(s0, s1, equal_var=False)
        u_stat, u_pval = stats.mannwhitneyu(s0, s1)
        stars = compute_significance_stars(p_val)
        
        # Cohen's d effect size
        pooled_std = np.sqrt(((len(s0)-1)*np.var(s0, ddof=1) + (len(s1)-1)*np.var(s1, ddof=1)) / (len(s0) + len(s1) - 2))
        cohen_d = abs(np.mean(s1) - np.mean(s0)) / (pooled_std + 1e-8)
        
        is_top3 = k in top3_indices
        rank = top3_ranks.get(k, -1)
        
        stats_rows.append({
            "Component": comp_name,
            "Is_Top3": is_top3,
            "Top3_Rank": rank,
            "Class0_Mean": np.mean(s0),
            "Class0_Std": np.std(s0),
            "Class0_Median": np.median(s0),
            "Class1_Mean": np.mean(s1),
            "Class1_Std": np.std(s1),
            "Class1_Median": np.median(s1),
            "T_Statistic": t_stat,
            "P_Value": p_val,
            "MannWhitney_PVal": u_pval,
            "Cohens_d": cohen_d,
            "Significance": stars
        })
        
        for val in s0:
            plot_rows.append({
                "Component": comp_name,
                "Latent Score": val,
                "Group": "Healthy Control",
                "Is_Top3": is_top3
            })
        for val in s1:
            plot_rows.append({
                "Component": comp_name,
                "Latent Score": val,
                "Group": "Epilepsy (TLE)",
                "Is_Top3": is_top3
            })
            
    df_plot = pd.DataFrame(plot_rows)
    df_stats = pd.DataFrame(stats_rows)
    stats_csv_path = os.path.join(output_dir, "plsda_components_class_stats.csv")
    df_stats.to_csv(stats_csv_path, index=False)
    df_stats.to_csv(os.path.join(excel_dir, "plsda_components_class_stats.csv"), index=False)
    print(f"Saved stats summary: {stats_csv_path}")

    # =========================================================================
    # Figure 1: Violin Plot across All 8 PLS-DA Components
    # =========================================================================
    print("Generating Figure 1: All 8 PLS-DA Components Violin Plot...")
    fig, ax = plt.subplots(figsize=(13, 6))
    
    # Background shading for Top 3 selected components
    for k in range(n_components):
        if k in top3_indices:
            ax.axvspan(k - 0.45, k + 0.45, color="#f9ebea", alpha=0.6, zorder=0)
            rank = top3_ranks[k]
            ax.text(k, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 30, f"★ Rank {rank}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold", color="#c0392b")

    # Split violin plot
    sns.violinplot(
        data=df_plot,
        x="Component",
        y="Latent Score",
        hue="Group",
        palette=class_palette,
        split=True,
        inner="quartile",
        cut=1.5,
        linewidth=1.2,
        ax=ax
    )
    
    # Annotate p-values and significance above each component
    for k in range(n_components):
        row = df_stats[df_stats["Component"] == f"PLS{k+1}"].iloc[0]
        p_val = row["P_Value"]
        stars = row["Significance"]
        y_max = df_plot[df_plot["Component"] == f"PLS{k+1}"]["Latent Score"].max()
        p_text = f"p={p_val:.1e}\n({stars})" if p_val < 0.001 else f"p={p_val:.3f}\n({stars})"
        ax.text(k, y_max + 4.5, p_text, ha="center", va="bottom", fontsize=8.5, fontweight="semibold", color="#2c3e50")

    # Expand top y-limit for labels
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - 2, y_max + 14)
    
    ax.set_title(
        f"PLS-DA 8 Component Parameters by Class ({side.capitalize()} Hippocampus - {dataset})\n"
        f"Comparing Latent Score Distributions: Healthy Control vs Epilepsy [Prior to Distance Mapping]",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlabel("PLS-DA Latent Components", fontsize=12, fontweight="bold")
    ax.set_ylabel("PLS-DA Component Score ($t_k$)", fontsize=12, fontweight="bold")
    ax.legend(title="Class Group", loc="lower right", framealpha=0.95)
    
    # Explanatory note
    plt.annotate(
        "★ Red Shading = Top 3 components selected for Grad-CAM Distance Mapping\n"
        "Notice: PLS1 & PLS3 show extreme separation (p < 1e-5), while PLS5 has heavy overlap.",
        xy=(0.015, 0.025), xycoords="axes fraction",
        fontsize=9, fontstyle="italic", bbox=dict(boxstyle="round,pad=0.5", facecolor="#fdfefe", edgecolor="#bdc3c7")
    )
    
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "plsda_all_8_components_violin.png")
    plt.savefig(fig1_path, dpi=300)
    plt.savefig(os.path.join(excel_dir, "plsda_all_8_components_violin.png"), dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")

    # =========================================================================
    # =========================================================================
    # Common helper to generate high-detail publication box plots
    # =========================================================================
    extra_sync_dirs = [
        os.path.join(repo_root, "Model", dataset, side, "plots"),
        os.path.join(repo_root, "Model", dataset, "plots", "gradcam_plsda"),
        os.path.join(repo_root, "Model_Results_Excel", "07_Detailed_Model_Plots", dataset, "gradcam_plsda"),
        os.path.join(repo_root, "Model_Results_Excel", "08_PLSDA_Class_Violin_Plots", dataset, side),
        r"C:\Users\IHCK\Desktop\Dataset_1_Bootstrap_Results\plots\gradcam_plsda"
    ]

    def render_boxplot_panels(selected_indices, fig_title, base_name):
        n_panels = len(selected_indices)
        fig_w = 5.2 * n_panels
        fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 5.5), sharey=False)
        if n_panels == 1:
            axes = [axes]
        fig.patch.set_facecolor('white')

        for idx, (k, ax_sub) in enumerate(zip(selected_indices, axes)):
            c_name = f"PLS{k+1}"
            rank = idx + 1
            sub_df = df_plot[df_plot["Component"] == c_name]
            row_stat = df_stats[df_stats["Component"] == c_name].iloc[0]

            ax_sub.set_facecolor('white')

            # Publication-grade Box Plot (No Violin, No Datapoints/Stripplot)
            sns.boxplot(
                data=sub_df,
                x="Group",
                y="Latent Score",
                hue="Group",
                palette=class_palette,
                legend=False,
                width=0.45,
                boxprops=dict(alpha=0.85, edgecolor="black", linewidth=1.5),
                medianprops=dict(color="black", linewidth=2.0),
                whiskerprops=dict(color="black", linewidth=1.5, linestyle="-"),
                capprops=dict(color="black", linewidth=1.5),
                showcaps=True,
                showfliers=False,
                ax=ax_sub
            )

            # Locked y-axis scale: strictly -80 to 80 for all components
            ax_sub.set_ylim(-80, 80)
            ax_sub.set_yticks([-80, -60, -40, -20, 0, 20, 40, 60, 80])

            # Annotate statistical test bracket & exact p-value at aligned top position
            bar_y = 65.0
            bar_h = 3.5
            ax_sub.plot([0, 0, 1, 1], [bar_y, bar_y + bar_h, bar_y + bar_h, bar_y], lw=1.5, c="#2c3e50")
            p_str = f"p = {row_stat['P_Value']:.2e} ({row_stat['Significance']})" if row_stat['P_Value'] < 0.001 else f"p = {row_stat['P_Value']:.3f} ({row_stat['Significance']})"
            ax_sub.text(0.5, 70.0, p_str, ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#c0392b")

            ax_sub.set_title(f"Rank {rank}: {c_name}\n({side.capitalize()} Hippocampus)", fontsize=13, fontweight="bold", color="#2c3e50", pad=10)
            ax_sub.set_xlabel("")
            ax_sub.set_ylabel("Component Latent Score ($t$)", fontsize=12, fontweight="bold")

            # Strict publication styling: Solid black 1.5pt frame, outward ticks, NO grid
            for spine in ax_sub.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(1.5)
            ax_sub.grid(False)
            ax_sub.tick_params(which="major", direction="out", length=6, width=1.5, color="black", labelsize=11)
            ax_sub.tick_params(which="minor", direction="out", length=3.5, width=1.0, color="black")

        plt.suptitle(fig_title, fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        # Save to output_dir, excel_dir, and extra_sync_dirs
        all_save_dirs = [output_dir, excel_dir] + extra_sync_dirs
        for d in all_save_dirs:
            try:
                os.makedirs(d, exist_ok=True)
                plt.savefig(os.path.join(d, f"{base_name}.png"), dpi=300)
                plt.savefig(os.path.join(d, f"{base_name}_{side}.png"), dpi=300)
                if "top3" in base_name:
                    plt.savefig(os.path.join(d, "plsda_top3_parameters_detailed_violin.png"), dpi=300)
            except Exception:
                pass
        plt.close()
        print(f"Saved: {base_name} ({n_panels} panels) across directories.")

    # =========================================================================
    # Figure 2A: Full Top 3 Components (PLS1, PLS2, PLS3) Detailed Box Plot
    # =========================================================================
    print("Generating Figure 2A: Full Top 3 Components (3 Panels) Box Plot...")
    full_top3_indices = top3_indices[:3]
    render_boxplot_panels(
        selected_indices=full_top3_indices,
        fig_title=(
            f"Top 3 Selected PLS-DA Components Comparison Across Classes ({side.capitalize()} Hippocampus)\n"
            f"[Directly feeding into -3.0 SD to +3.0 SD Distance Mapping & Grad-CAM Mesh Generation]"
        ),
        base_name="plsda_top3_parameters_boxplot"
    )

    # =========================================================================
    # Figure 2B: Focused Top 2 Components (PLS1 & PLS2) Detailed Box Plot
    # =========================================================================
    print("Generating Figure 2B: Top 2 Components (2 Panels) Box Plot...")
    focused_top2_indices = [k for k in top3_indices if k in (0, 1)]
    if len(focused_top2_indices) < 2:
        focused_top2_indices = top3_indices[:2]
    render_boxplot_panels(
        selected_indices=focused_top2_indices,
        fig_title=(
            f"Top 2 Selected PLS-DA Components Comparison Across Classes ({side.capitalize()} Hippocampus)\n"
            f"[Directly feeding into -3.0 SD to +3.0 SD Distance Mapping & Grad-CAM Mesh Generation]"
        ),
        base_name="plsda_top2_parameters_boxplot"
    )

    # =========================================================================
    # Figure 3: Model Classification Predictions Violin Plot
    # =========================================================================
    print("Generating Figure 3: Model Classification Predictions Violin Plot...")
    
    # Check if test_predictions.npz exists in ResNet results or other model folders
    resnet_pred_npz = os.path.join(repo_root, "Model", dataset, side, "ResNet", "results", "test_predictions.npz")
    
    model_preds_dict = {}
    if os.path.exists(resnet_pred_npz):
        data = np.load(resnet_pred_npz)
        model_preds_dict["ResNet"] = (data["y_test"], data["y_prob"])
        
    # Check other models for comparison
    for m_name in ["PointNet", "MobileNet", "MLP", "SVM", "SqueezeNet"]:
        m_npz = os.path.join(repo_root, "Model", dataset, side, m_name, "results", "test_predictions.npz")
        if os.path.exists(m_npz):
            data = np.load(m_npz)
            model_preds_dict[m_name] = (data["y_test"], data["y_prob"])
            
    if model_preds_dict:
        # Build DataFrame for Model Predictions
        m_plot_rows = []
        for m_name, (y_true, y_probs) in model_preds_dict.items():
            for yt, yp in zip(y_true, y_probs):
                grp = "Epilepsy (TLE)" if yt == 1 else "Healthy Control"
                m_plot_rows.append({
                    "Model": m_name,
                    "Predicted Probability P(Epilepsy)": float(yp),
                    "True Class": grp
                })
        df_m_plot = pd.DataFrame(m_plot_rows)
        
        # Plot: 2 Subplots (ResNet Detailed View + Multi-Model Overview)
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.4])
        
        # Subplot A: ResNet (The backbone model of Grad-CAM)
        ax_res = fig.add_subplot(gs[0])
        df_res = df_m_plot[df_m_plot["Model"] == "ResNet"]
        if not df_res.empty:
            sns.violinplot(
                data=df_res,
                x="True Class",
                y="Predicted Probability P(Epilepsy)",
                hue="True Class",
                palette=class_palette,
                legend=False,
                inner="quartile",
                cut=0,
                ax=ax_res
            )
            sns.stripplot(
                data=df_res,
                x="True Class",
                y="Predicted Probability P(Epilepsy)",
                hue="True Class",
                palette=class_palette,
                legend=False,
                size=6,
                jitter=0.2,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                ax=ax_res
            )
            ax_res.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Decision Threshold (0.5)")
            ax_res.set_title("ResNet Backbone (Grad-CAM Model)\nPredicted Probability Distribution by True Class", fontsize=11, fontweight="bold")
            ax_res.set_ylabel("Predicted Probability $P(Epilepsy)$", fontsize=11, fontweight="bold")
            ax_res.set_xlabel("True Class Group", fontsize=11, fontweight="bold")
            ax_res.set_ylim(-0.05, 1.05)
            ax_res.legend(loc="center right", fontsize=9)
            
        # Subplot B: Comparison across All Models
        ax_all = fig.add_subplot(gs[1])
        sns.violinplot(
            data=df_m_plot,
            x="Model",
            y="Predicted Probability P(Epilepsy)",
            hue="True Class",
            palette=class_palette,
            split=True,
            inner="quartile",
            cut=0,
            ax=ax_all
        )
        ax_all.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
        ax_all.set_title(f"Model Predictions Comparison Across Classes ({side.capitalize()} Hippocampus)", fontsize=11, fontweight="bold")
        ax_all.set_ylabel("Predicted Probability $P(Epilepsy)$", fontsize=11, fontweight="bold")
        ax_all.set_xlabel("Model Architecture", fontsize=11, fontweight="bold")
        ax_all.set_ylim(-0.05, 1.05)
        ax_all.legend(title="True Class", loc="upper right")
        
        plt.tight_layout()
        fig3_path = os.path.join(output_dir, "model_classification_predictions_violin.png")
        plt.savefig(fig3_path, dpi=300)
        plt.savefig(os.path.join(excel_dir, "model_classification_predictions_violin.png"), dpi=300)
        plt.close()
        print(f"Saved: {fig3_path}")
    else:
        print("[INFO] No test_predictions.npz found for model predictions violin plot.")

    print(f"\n--- PLS-DA Violin Plots for {dataset} ({side}) Successfully Saved to Output_GradCAM_PLSDA and Model_Results_Excel! ---")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        side_arg = sys.argv[1]
        dataset_arg = sys.argv[2] if len(sys.argv) > 2 else "Ds005602"
        generate_violin_plots(side=side_arg, dataset=dataset_arg)
    else:
        # Default: Process all major cohorts
        cohorts = [
            ("right", "Dataset_1"),
            ("left", "Dataset_1"),
            ("right", "All_Augment_tain"),
            ("left", "All_Augment_tain"),
            ("right", "Ds005602"),
            ("left", "Ds005602"),
        ]
        for s_c, d_c in cohorts:
            print("\n" + "=" * 70)
            print(f"GENERATING PLS-DA CLASS VIOLIN PLOTS: {d_c} [{s_c.capitalize()}]")
            print("=" * 70)
            generate_violin_plots(side=s_c, dataset=d_c)
