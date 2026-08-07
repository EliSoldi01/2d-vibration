import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from core.utils.r2_metrics import compute_r2_metrics as r2_metrics

# ----------------------------
# LOOCV leave-one-subject-out
# ----------------------------
def loocv_leave_one_subject(df, output_path=None):
    """
    LOOCV for linear regression model:
     - Features X: [pb_sum*duration, pt_sum*duration]
     - Target y: angle_deg
     - Weights: vividness/3.0
    """
    print("[LOOCV] Building feature matrix...")
    
    X, y, weights, groups = [], [], [], []
    for _, row in df.iterrows():
        b_part, t_part = row["pattern_pair"].split("_")
        pb_sum = sum(int(bit) for bit in b_part)
        pt_sum = sum(int(bit) for bit in t_part)
        X.append([pb_sum * row["duration"], pt_sum * row["duration"]])
        y.append(row["angle_deg"])
        weights.append(row["vividness"]/3.0)
        groups.append(row["subject"])

    X = np.array(X)
    y = np.array(y)
    weights = np.array(weights)
    groups = np.array(groups)

    logo = LeaveOneGroupOut()
    subjects_tested = []
    r2_zero_list, r2_mean_list = [], []
    r2_zero_unw_list, r2_mean_unw_list = [], []
    mae_list, rmse_list = [], []
    Kb_list, Kt_list = [], []
    y_pred_dict = {}

    print(f"[LOOCV] {len(np.unique(groups))} folds (one per subject)")

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_subject = np.unique(groups[test_idx])[0]
        subjects_tested.append(test_subject)

        model = LinearRegression(fit_intercept=False)
        model.fit(X[train_idx], y[train_idx], sample_weight=weights[train_idx])

        Kb_list.append(model.coef_[0])
        Kt_list.append(model.coef_[1])

        y_pred = model.predict(X[test_idx])
        y_true = y[test_idx]
        w_test = weights[test_idx]

        # Weighted and unweighted R² metrics
        r2_w = r2_metrics(y_true, y_pred, weights=w_test)
        r2_zero = r2_w["R2_zero"]
        r2_mean = r2_w["R2_mean"]

        r2_unw = r2_metrics(y_true, y_pred, weights=None)
        r2_zero_unw = r2_unw["R2_zero"]
        r2_mean_unw = r2_unw["R2_mean"]

        # Other metrics
        rmse = np.sqrt(np.average((y_true - y_pred)**2, weights=w_test))
        mae = np.average(np.abs(y_true - y_pred), weights=w_test)

        # Saving results
        r2_zero_list.append(r2_zero)
        r2_mean_list.append(r2_mean)
        r2_zero_unw_list.append(r2_zero_unw)
        r2_mean_unw_list.append(r2_mean_unw)
        mae_list.append(mae)
        rmse_list.append(rmse)
        y_pred_dict[test_subject] = y_pred

        print(f" Fold {fold+1}: {test_subject} | "
              f"R²_zero={r2_zero:.3f}, R²_mean={r2_mean:.3f}, "
              f"RMSE={rmse:.3f}, MAE={mae:.3f}")

    # ----------------------------
    # DataFrame for each subject
    # ----------------------------
    results_df = pd.DataFrame({
        "subject": subjects_tested,
        "Kb": [round(x,2) for x in Kb_list],
        "Kt": [round(x,2) for x in Kt_list],
        "R2_zero_weighted": [round(x,2) for x in r2_zero_list],
        "R2_mean_weighted": [round(x,2) for x in r2_mean_list],
        "R2_zero_unweighted": [round(x,2) for x in r2_zero_unw_list],
        "R2_mean_unweighted": [round(x,2) for x in r2_mean_unw_list],
        "RMSE_weighted": [round(x,2) for x in rmse_list],
        "MAE_weighted": [round(x,2) for x in mae_list]
    })

    # ----------------------------
    # Summary 
    # ----------------------------
    summary_df = pd.DataFrame({
        "Metric": ["Mean_R2_zero_weighted", "Std_R2_zero_weighted",
                   "Mean_R2_mean_weighted", "Std_R2_mean_weighted",
                   "Mean_R2_zero_unweighted", "Std_R2_zero_unweighted",
                   "Mean_R2_mean_unweighted", "Std_R2_mean_unweighted",
                   "Mean_RMSE_weighted", "Std_RMSE_weighted",
                   "Mean_MAE_weighted", "Std_MAE_weighted"],
        "Value": [
            round(np.mean(r2_zero_list),2), round(np.std(r2_zero_list),2),
            round(np.mean(r2_mean_list),2), round(np.std(r2_mean_list),2),
            round(np.mean(r2_zero_unw_list),2), round(np.std(r2_zero_unw_list),2),
            round(np.mean(r2_mean_unw_list),2), round(np.std(r2_mean_unw_list),2),
            round(np.mean(rmse_list),2), round(np.std(rmse_list),2),
            round(np.mean(mae_list),2), round(np.std(mae_list),2)
        ]
    })

    plot_loocv_mae_vs_r2_vividness(results_df, df, output_folder=os.path.join(os.path.dirname(output_path), "LOOCV_plots") if output_path else None)

    # ----------------------------
    # Excel saving
    # ----------------------------
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            results_df.to_excel(writer, sheet_name="LOOCV_per_subject", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            coeffs_df = pd.DataFrame({"subject": subjects_tested, "Kb": Kb_list, "Kt": Kt_list})
            coeffs_df.to_excel(writer, sheet_name="Coefficients", index=False)
        print(f"[LOOCV] ✅ Excel salvato: {output_path}")

    return results_df

def plot_loocv_mae_vs_r2_vividness(results_df, df, output_folder=None,
                                    r2_col="R2_zero_weighted", mae_col="MAE_weighted"):
    """
    Scatter plot MAE vs R² for each subject in LOOCV, color-coded by mean vividness.
    
    results_df: DataFrame from loocv_leave_one_subject
    df: original DataFrame to compute mean vividness per subject
    output_folder (optional): folder to save the plot
    r2_col: R² column to plot (default: "R2_zero_weighted")
    mae_col: MAE column to plot (default: "MAE_weighted")
    """
    # Calcolo vividness media per soggetto
    subjects = results_df['subject'].values
    vividness_mean = []
    for subj in subjects:
        subj_viv = df[df['subject'] == subj]['vividness'].values
        vividness_mean.append(np.mean(subj_viv))
    vividness_mean = np.array(vividness_mean)
    
    plt.figure(figsize=(10,6))
    sc = plt.scatter(results_df[mae_col], results_df[r2_col],
                     c=vividness_mean, cmap='Reds', s=150, alpha=0.8, vmin=1, vmax=3)
    
    plt.xlabel('MAE (°)', fontsize=16)
    plt.ylabel(f'R²', fontsize=16)
    plt.title('Leave-One-Subject-Out Cross-Validation', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(results_df[mae_col])*1.5)
    
    # Annotazioni soggetti
    for i, subj in enumerate(subjects):
        plt.annotate(subj, (results_df[mae_col].iloc[i], results_df[r2_col].iloc[i]),
                     xytext=(5,5), textcoords='offset points', fontsize=9)
    
    cbar = plt.colorbar(sc)
    cbar.set_label('Mean Vividness', rotation=270, labelpad=15, size=16)
    cbar.set_ticks([1, 1.5, 2, 2.5, 3])
    
    plt.tight_layout()
    
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        plt_path = os.path.join(output_folder, 'LOOCV_MAE_vs_R2_vividness.png')
        plt.savefig(plt_path, dpi=300, bbox_inches='tight')
        print(f"[LOOCV] Scatter plot salvato in {plt_path}")
    
    plt.show()