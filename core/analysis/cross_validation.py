import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from core.utils.r2_metrics import compute_r2_metrics as r2_metrics


# ============================================================
# Helper
# ============================================================

PURE_PATTERNS = ["001_000", "000_001"]


def is_combined_pattern(pattern):
    """
    Return True for combined/multi-muscle stimulation patterns.

    Pure patterns used to estimate Kb and Kt:
        001_000
        000_001
    """
    return pattern not in PURE_PATTERNS


def compute_weighted_mae_rmse(y_true, y_pred, weights):
    """
    Compute weighted MAE and RMSE.
    """
    mae = np.average(
        np.abs(y_true - y_pred),
        weights=weights
    )

    rmse = np.sqrt(
        np.average(
            (y_true - y_pred) ** 2,
            weights=weights
        )
    )

    return mae, rmse


# ============================================================
# LOOCV leave-one-subject-out
# ============================================================

def loocv_leave_one_subject(df, output_path=None):
    """
    LOOCV for linear regression model.

    Features:
        X = [pb_sum * duration, pt_sum * duration]

    Target:
        y = angle_deg

    Weights:
        vividness / 3.0

    The model is fitted separately in each fold, leaving one
    subject completely out.

    In addition to the original subject-level metrics, the
    function computes:

        1. LOOCV metrics on all patterns
        2. LOOCV metrics on combined patterns only
        3. Pooled LOOCV metrics across all out-of-sample
           predictions
        4. Pooled LOOCV metrics on combined patterns only

    Excel sheets:
        - LOOCV_per_subject
        - Summary
        - Coefficients
        - Pooled_LOOCV
        - LOOCV_predictions
    """

    print("[LOOCV] Building feature matrix...")

    X, y, weights, groups, patterns = [], [], [], [], []

    for _, row in df.iterrows():

        b_part, t_part = row["pattern_pair"].split("_")

        pb_sum = sum(int(bit) for bit in b_part)
        pt_sum = sum(int(bit) for bit in t_part)

        X.append([
            pb_sum * row["duration"],
            pt_sum * row["duration"]
        ])

        y.append(row["angle_deg"])
        weights.append(row["vividness"] / 3.0)
        groups.append(row["subject"])
        patterns.append(row["pattern_pair"])

    X = np.array(X)
    y = np.array(y)
    weights = np.array(weights)
    groups = np.array(groups)
    patterns = np.array(patterns)

    logo = LeaveOneGroupOut()

    subjects_tested = []

    # --------------------------------------------------------
    # Subject-level metrics: ALL patterns
    # --------------------------------------------------------

    r2_zero_list = []
    r2_mean_list = []

    r2_zero_unw_list = []
    r2_mean_unw_list = []

    mae_list = []
    rmse_list = []

    # --------------------------------------------------------
    # Subject-level metrics: COMBINED patterns only
    # --------------------------------------------------------

    r2_comb_list = []
    r2_comb_mean_list = []

    r2_comb_unw_list = []
    r2_comb_mean_unw_list = []

    mae_comb_list = []
    rmse_comb_list = []

    # --------------------------------------------------------
    # Coefficients
    # --------------------------------------------------------

    Kb_list = []
    Kt_list = []

    # --------------------------------------------------------
    # Pooled out-of-sample predictions
    # --------------------------------------------------------

    pooled_predictions = []

    print(
        f"[LOOCV] {len(np.unique(groups))} folds "
        f"(one per subject)"
    )

    # ========================================================
    # LOOCV
    # ========================================================

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups)
    ):

        test_subject = np.unique(groups[test_idx])[0]
        subjects_tested.append(test_subject)

        # ----------------------------------------------------
        # Fit model on training subjects
        # ----------------------------------------------------

        model = LinearRegression(fit_intercept=False)

        model.fit(
            X[train_idx],
            y[train_idx],
            sample_weight=weights[train_idx]
        )

        Kb = model.coef_[0]
        Kt = model.coef_[1]

        Kb_list.append(Kb)
        Kt_list.append(Kt)

        # ----------------------------------------------------
        # Predict held-out subject
        # ----------------------------------------------------

        y_pred = model.predict(X[test_idx])

        y_true = y[test_idx]
        w_test = weights[test_idx]
        pattern_test = patterns[test_idx]

        # ====================================================
        # ALL PATTERNS
        # ====================================================

        # Weighted
        r2_w = r2_metrics(
            y_true,
            y_pred,
            weights=w_test
        )

        r2_zero = r2_w["R2_zero"]
        r2_mean = r2_w["R2_mean"]

        # Unweighted
        r2_unw = r2_metrics(
            y_true,
            y_pred,
            weights=None
        )

        r2_zero_unw = r2_unw["R2_zero"]
        r2_mean_unw = r2_unw["R2_mean"]

        # MAE / RMSE
        mae, rmse = compute_weighted_mae_rmse(
            y_true,
            y_pred,
            w_test
        )

        # Save
        r2_zero_list.append(r2_zero)
        r2_mean_list.append(r2_mean)

        r2_zero_unw_list.append(r2_zero_unw)
        r2_mean_unw_list.append(r2_mean_unw)

        mae_list.append(mae)
        rmse_list.append(rmse)

        # ====================================================
        # COMBINED PATTERNS ONLY
        # ====================================================

        is_combined = np.array([
            is_combined_pattern(pattern)
            for pattern in pattern_test
        ])

        y_true_comb = y_true[is_combined]
        y_pred_comb = y_pred[is_combined]
        w_comb = w_test[is_combined]

        # Safety check
        if len(y_true_comb) > 0:

            # Weighted
            r2_comb = r2_metrics(
                y_true_comb,
                y_pred_comb,
                weights=w_comb
            )

            r2_comb_zero = r2_comb["R2_zero"]
            r2_comb_mean = r2_comb["R2_mean"]

            # Unweighted
            r2_comb_unw = r2_metrics(
                y_true_comb,
                y_pred_comb,
                weights=None
            )

            r2_comb_zero_unw = r2_comb_unw["R2_zero"]
            r2_comb_mean_unw = r2_comb_unw["R2_mean"]

            # MAE / RMSE
            mae_comb, rmse_comb = compute_weighted_mae_rmse(
                y_true_comb,
                y_pred_comb,
                w_comb
            )

        else:

            r2_comb_zero = np.nan
            r2_comb_mean = np.nan

            r2_comb_zero_unw = np.nan
            r2_comb_mean_unw = np.nan

            mae_comb = np.nan
            rmse_comb = np.nan

        # Save combined metrics
        r2_comb_list.append(r2_comb_zero)
        r2_comb_mean_list.append(r2_comb_mean)

        r2_comb_unw_list.append(r2_comb_zero_unw)
        r2_comb_mean_unw_list.append(r2_comb_mean_unw)

        mae_comb_list.append(mae_comb)
        rmse_comb_list.append(rmse_comb)

        # ====================================================
        # Save every out-of-sample prediction
        # ====================================================

        for local_idx, global_idx in enumerate(test_idx):

            pooled_predictions.append({
                "fold": fold + 1,
                "subject": groups[global_idx],
                "pattern": patterns[global_idx],
                "duration": df.iloc[global_idx]["duration"],
                "vividness": df.iloc[global_idx]["vividness"],
                "weight": weights[global_idx],
                "angle_true": y[global_idx],
                "angle_pred": y_pred[local_idx],
                "residual": y[global_idx] - y_pred[local_idx],
                "is_combined": is_combined_pattern(
                    patterns[global_idx]
                ),
                "Kb": Kb,
                "Kt": Kt
            })

        # ====================================================
        # Print fold results
        # ====================================================

        print(
            f" Fold {fold + 1}: {test_subject} | "
            f"ALL: "
            f"R²_zero={r2_zero:.3f}, "
            f"R²_mean={r2_mean:.3f}, "
            f"RMSE={rmse:.3f}, "
            f"MAE={mae:.3f} | "
            f"COMBINED: "
            f"R²_zero={r2_comb_zero:.3f}, "
            f"RMSE={rmse_comb:.3f}, "
            f"MAE={mae_comb:.3f}"
        )

    # ========================================================
    # DataFrame: predictions
    # ========================================================

    predictions_df = pd.DataFrame(pooled_predictions)

    # ========================================================
    # DataFrame: per-subject results
    # ========================================================

    results_df = pd.DataFrame({

        "subject": subjects_tested,

        # Coefficients
        "Kb": [round(x, 2) for x in Kb_list],
        "Kt": [round(x, 2) for x in Kt_list],

        # ----------------------------------------------------
        # ALL PATTERNS
        # ----------------------------------------------------

        "R2_zero_weighted": [
            round(x, 2) for x in r2_zero_list
        ],

        "R2_mean_weighted": [
            round(x, 2) for x in r2_mean_list
        ],

        "R2_zero_unweighted": [
            round(x, 2) for x in r2_zero_unw_list
        ],

        "R2_mean_unweighted": [
            round(x, 2) for x in r2_mean_unw_list
        ],

        "RMSE_weighted": [
            round(x, 2) for x in rmse_list
        ],

        "MAE_weighted": [
            round(x, 2) for x in mae_list
        ],

        # ----------------------------------------------------
        # COMBINED PATTERNS ONLY
        # ----------------------------------------------------

        "R2_zero_combined": [
            round(x, 2) if not np.isnan(x) else np.nan
            for x in r2_comb_list
        ],

        "R2_mean_combined": [
            round(x, 2) if not np.isnan(x) else np.nan
            for x in r2_comb_mean_list
        ],

        "R2_zero_combined_unweighted": [
            round(x, 2) if not np.isnan(x) else np.nan
            for x in r2_comb_unw_list
        ],

        "R2_mean_combined_unweighted": [
            round(x, 2) if not np.isnan(x) else np.nan
            for x in r2_comb_mean_unw_list
        ],

        "RMSE_combined": [
            round(x, 2) if not np.isnan(x) else np.nan
            for x in rmse_comb_list
        ],

        "MAE_combined": [
            round(x, 2) if not np.isnan(x) else np.nan
            for x in mae_comb_list
        ]
    })

    # ========================================================
    # Subject-level summary
    # ========================================================

    summary_df = pd.DataFrame({

        "Metric": [

            # ALL
            "Mean_R2_zero_weighted",
            "Std_R2_zero_weighted",

            "Mean_R2_mean_weighted",
            "Std_R2_mean_weighted",

            "Mean_R2_zero_unweighted",
            "Std_R2_zero_unweighted",

            "Mean_R2_mean_unweighted",
            "Std_R2_mean_unweighted",

            "Mean_RMSE_weighted",
            "Std_RMSE_weighted",

            "Mean_MAE_weighted",
            "Std_MAE_weighted",

            # COMBINED
            "Mean_R2_zero_combined",
            "Std_R2_zero_combined",

            "Mean_R2_mean_combined",
            "Std_R2_mean_combined",

            "Mean_R2_zero_combined_unweighted",
            "Std_R2_zero_combined_unweighted",

            "Mean_R2_mean_combined_unweighted",
            "Std_R2_mean_combined_unweighted",

            "Mean_RMSE_combined",
            "Std_RMSE_combined",

            "Mean_MAE_combined",
            "Std_MAE_combined"
        ],

        "Value": [

            # ALL
            round(np.nanmean(r2_zero_list), 2),
            round(np.nanstd(r2_zero_list), 2),

            round(np.nanmean(r2_mean_list), 2),
            round(np.nanstd(r2_mean_list), 2),

            round(np.nanmean(r2_zero_unw_list), 2),
            round(np.nanstd(r2_zero_unw_list), 2),

            round(np.nanmean(r2_mean_unw_list), 2),
            round(np.nanstd(r2_mean_unw_list), 2),

            round(np.nanmean(rmse_list), 2),
            round(np.nanstd(rmse_list), 2),

            round(np.nanmean(mae_list), 2),
            round(np.nanstd(mae_list), 2),

            # COMBINED
            round(np.nanmean(r2_comb_list), 2),
            round(np.nanstd(r2_comb_list), 2),

            round(np.nanmean(r2_comb_mean_list), 2),
            round(np.nanstd(r2_comb_mean_list), 2),

            round(np.nanmean(r2_comb_unw_list), 2),
            round(np.nanstd(r2_comb_unw_list), 2),

            round(np.nanmean(r2_comb_mean_unw_list), 2),
            round(np.nanstd(r2_comb_mean_unw_list), 2),

            round(np.nanmean(rmse_comb_list), 2),
            round(np.nanstd(rmse_comb_list), 2),

            round(np.nanmean(mae_comb_list), 2),
            round(np.nanstd(mae_comb_list), 2)
        ]
    })

    # ========================================================
    # POOLED LOOCV METRICS
    # ========================================================

    print("\n[LOOCV] Computing pooled out-of-sample metrics...")

    # --------------------------------------------------------
    # All patterns
    # --------------------------------------------------------

    y_true_all = predictions_df["angle_true"].values
    y_pred_all = predictions_df["angle_pred"].values
    weights_all = predictions_df["weight"].values

    pooled_all = r2_metrics(
        y_true_all,
        y_pred_all,
        weights=weights_all
    )

    pooled_all_unweighted = r2_metrics(
        y_true_all,
        y_pred_all,
        weights=None
    )

    pooled_mae_all, pooled_rmse_all = compute_weighted_mae_rmse(
        y_true_all,
        y_pred_all,
        weights_all
    )

    # --------------------------------------------------------
    # Combined patterns
    # --------------------------------------------------------

    combined_predictions_df = predictions_df[
        predictions_df["is_combined"]
    ].copy()

    y_true_combined = combined_predictions_df[
        "angle_true"
    ].values

    y_pred_combined = combined_predictions_df[
        "angle_pred"
    ].values

    weights_combined = combined_predictions_df[
        "weight"
    ].values

    pooled_combined = r2_metrics(
        y_true_combined,
        y_pred_combined,
        weights=weights_combined
    )

    pooled_combined_unweighted = r2_metrics(
        y_true_combined,
        y_pred_combined,
        weights=None
    )

    pooled_mae_combined, pooled_rmse_combined = (
        compute_weighted_mae_rmse(
            y_true_combined,
            y_pred_combined,
            weights_combined
        )
    )

    # ========================================================
    # Pooled DataFrame
    # ========================================================

    pooled_metrics_df = pd.DataFrame({

        "Dataset": [
            "All patterns",
            "Combined patterns only"
        ],

        "N_predictions": [
            len(predictions_df),
            len(combined_predictions_df)
        ],

        # Weighted R²
        "R2_zero_weighted": [
            pooled_all["R2_zero"],
            pooled_combined["R2_zero"]
        ],

        "R2_mean_weighted": [
            pooled_all["R2_mean"],
            pooled_combined["R2_mean"]
        ],

        # Unweighted R²
        "R2_zero_unweighted": [
            pooled_all_unweighted["R2_zero"],
            pooled_combined_unweighted["R2_zero"]
        ],

        "R2_mean_unweighted": [
            pooled_all_unweighted["R2_mean"],
            pooled_combined_unweighted["R2_mean"]
        ],

        # Error metrics
        "MAE_weighted": [
            pooled_mae_all,
            pooled_mae_combined
        ],

        "RMSE_weighted": [
            pooled_rmse_all,
            pooled_rmse_combined
        ]
    })

    # ========================================================
    # Print pooled results
    # ========================================================

    print("\n" + "=" * 70)
    print("POOLED LOOCV RESULTS")
    print("=" * 70)

    print("\nALL PATTERNS")
    print(
        f"  N = {len(predictions_df)}"
    )
    print(
        f"  R²_zero weighted = "
        f"{pooled_all['R2_zero']:.3f}"
    )
    print(
        f"  R²_mean weighted = "
        f"{pooled_all['R2_mean']:.3f}"
    )
    print(
        f"  MAE weighted = "
        f"{pooled_mae_all:.3f}°"
    )
    print(
        f"  RMSE weighted = "
        f"{pooled_rmse_all:.3f}°"
    )

    print("\nCOMBINED PATTERNS ONLY")
    print(
        f"  N = {len(combined_predictions_df)}"
    )
    print(
        f"  R²_zero weighted = "
        f"{pooled_combined['R2_zero']:.3f}"
    )
    print(
        f"  R²_mean weighted = "
        f"{pooled_combined['R2_mean']:.3f}"
    )
    print(
        f"  MAE weighted = "
        f"{pooled_mae_combined:.3f}°"
    )
    print(
        f"  RMSE weighted = "
        f"{pooled_rmse_combined:.3f}°"
    )

    print("=" * 70)

    # ========================================================
    # Plot
    # ========================================================

    plot_loocv_mae_vs_r2_vividness(
        results_df,
        df,
        output_folder=(
            os.path.join(
                os.path.dirname(output_path),
                "LOOCV_plots"
            )
            if output_path
            else None
        )
    )

    # ========================================================
    # Excel saving
    # ========================================================

    if output_path:

        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True
            )

        with pd.ExcelWriter(
            output_path,
            engine="openpyxl"
        ) as writer:

            # Original sheets
            results_df.to_excel(
                writer,
                sheet_name="LOOCV_per_subject",
                index=False
            )

            summary_df.to_excel(
                writer,
                sheet_name="Summary",
                index=False
            )

            coeffs_df = pd.DataFrame({
                "subject": subjects_tested,
                "Kb": Kb_list,
                "Kt": Kt_list
            })

            coeffs_df.to_excel(
                writer,
                sheet_name="Coefficients",
                index=False
            )

            # New pooled metrics sheet
            pooled_metrics_df.to_excel(
                writer,
                sheet_name="Pooled_LOOCV",
                index=False
            )

            # New predictions sheet
            predictions_df.to_excel(
                writer,
                sheet_name="LOOCV_predictions",
                index=False
            )

        print(
            f"[LOOCV] ✅ Excel salvato: "
            f"{output_path}"
        )

    return results_df


# ============================================================
# Plot
# ============================================================

def plot_loocv_mae_vs_r2_vividness(
    results_df,
    df,
    output_folder=None,
    r2_col="R2_zero_weighted",
    mae_col="MAE_weighted"
):
    """
    Scatter plot MAE vs R² for each subject in LOOCV,
    color-coded by mean vividness.

    results_df:
        DataFrame from loocv_leave_one_subject

    df:
        Original DataFrame to compute mean vividness
        per subject

    output_folder:
        Folder to save the plot

    r2_col:
        R² column to plot

    mae_col:
        MAE column to plot
    """

    # --------------------------------------------------------
    # Mean vividness per subject
    # --------------------------------------------------------

    subjects = results_df["subject"].values

    vividness_mean = []

    for subj in subjects:

        subj_viv = df[
            df["subject"] == subj
        ]["vividness"].values

        vividness_mean.append(
            np.mean(subj_viv)
        )

    vividness_mean = np.array(vividness_mean)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    sc = plt.scatter(
        results_df[mae_col],
        results_df[r2_col],
        c=vividness_mean,
        cmap="Reds",
        s=150,
        alpha=0.8,
        vmin=1,
        vmax=3
    )

    plt.xlabel(
        "MAE (°)",
        fontsize=16
    )

    plt.ylabel(
        "R²",
        fontsize=16
    )

    plt.title(
        "Leave-One-Subject-Out Cross-Validation",
        fontsize=18
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.xlim(
        0,
        max(results_df[mae_col]) * 1.5
    )

    # --------------------------------------------------------
    # Subject annotations
    # --------------------------------------------------------

    for i, subj in enumerate(subjects):

        plt.annotate(
            subj,
            (
                results_df[mae_col].iloc[i],
                results_df[r2_col].iloc[i]
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9
        )

    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    cbar = plt.colorbar(sc)

    cbar.set_label(
        "Mean Vividness",
        rotation=270,
        labelpad=15,
        size=16
    )

    cbar.set_ticks(
        [1, 1.5, 2, 2.5, 3]
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if output_folder:

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        plt_path = os.path.join(
            output_folder,
            "LOOCV_MAE_vs_R2_vividness.png"
        )

        plt.savefig(
            plt_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"[LOOCV] Scatter plot salvato in "
            f"{plt_path}"
        )

    plt.show()