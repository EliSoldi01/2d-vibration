import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression
from core.utils.r2_metrics import compute_r2_metrics

# ============================================================
# REGRESSION PLOT — SINGOLO SOGGETTO (comportamento originale)
# ============================================================
def plot_regression(df, x_col, y_col, title, xlabel, ylabel, save_path,
                    weights_col="vividness", protocol=None):
    """
    Regression plot for a single subject and pattern, using original trial data.
    - Linear regression forced through origin, weighted by vividness
    """
    if len(df) < 2:
        print(f"⚠️ Not enough points for {title}")
        return

    if protocol is not None and weights_col == "vividness":
        vividness_values = protocol["scales"]["vividness"]["values"]
    else:
        vividness_values = [0, 3]

    X = df[x_col].values.reshape(-1, 1)
    y = df[y_col].values
    w = df[weights_col].values / 3.0 if weights_col in df.columns else np.ones_like(y)

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y, sample_weight=w)
    slope = model.coef_[0]
    y_pred = model.predict(X)
    r2 = compute_r2_metrics(y, y_pred, weights=w)["R2_zero"]

    plt.figure(figsize=(9, 5))
    plt.scatter(X, y, s=80, alpha=0.8)
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    plt.plot(x_line, model.predict(x_line), color="red", linewidth=2,
             label=f"Fit y={slope:.2f}x\nR² (zero)={r2:.2f}")

    ax = plt.gca()
    if x_col.lower() in ["duration", "presentation_order"]:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    if y_col.lower() == "vividness":
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if x_col.lower() == "vividness":
        plt.xlim(vividness_values[0], vividness_values[-1])
    if y_col.lower() == "vividness":
        plt.ylim(vividness_values[0], vividness_values[-1])
    if y_col.lower().startswith("angle"):
        plt.ylim(-60, 60)

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# REGRESSION PLOT — GROUP AVERAGE (scatter + mean ± SD)
# ============================================================
def plot_regression_group(df, x_col, y_col, title, xlabel, ylabel, save_path,
                          weights_col="vividness", protocol=None):
    """
    Regression plot for group-level data (all subjects combined).
    Layers:
      1. Scatter of raw data (all trials, all subjects, semi-transparent)
      2. Mean ± SD for each level of x_col
      3. Regression line forced through origin (weighted by vividness)

    Legend includes R² with respect to 0 for both raw data and means.
    """
    if len(df) < 2:
        print(f"⚠️ Not enough points for {title}")
        return

    if protocol is not None and weights_col == "vividness":
        vividness_values = protocol["scales"]["vividness"]["values"]
    else:
        vividness_values = [0, 3]

    X_raw = df[x_col].values.reshape(-1, 1)
    y_raw = df[y_col].values
    w_raw = df[weights_col].values / 3.0 if weights_col in df.columns else np.ones(len(y_raw))

    # ── Regression on raw data ──────────────────────────
    model = LinearRegression(fit_intercept=False)
    model.fit(X_raw, y_raw, sample_weight=w_raw)
    slope = model.coef_[0]
    y_pred_raw = model.predict(X_raw)
    r2_raw = compute_r2_metrics(y_raw, y_pred_raw, weights=w_raw)["R2_zero"]

    # ── Medie e SD per ogni livello di x_col ────────────────
    df_agg = (
        df.groupby(x_col)[y_col]
        .agg(mean="mean", sd="std")
        .reset_index()
    )
    y_pred_means = model.predict(df_agg[x_col].values.reshape(-1, 1))
    r2_means = compute_r2_metrics(df_agg["mean"].values, y_pred_means)["R2_zero"]

    # ── Plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(X_raw, y_raw, s=40, alpha=0.25, color="steelblue",
               label="Individual trials", zorder=1)

    x_line = np.linspace(X_raw.min(), X_raw.max(), 200).reshape(-1, 1)
    ax.plot(x_line, model.predict(x_line), color="red", linewidth=2,
            label=f"Fit  y = {slope:.2f}x\n"
                  f"R²_zero (raw) = {r2_raw:.2f}\n"
                  f"R²_zero (means) = {r2_means:.2f}",
            zorder=3)

    ax.errorbar(df_agg[x_col], df_agg["mean"], yerr=df_agg["sd"],
                fmt="o", color="black", markerfacecolor="white",
                markeredgewidth=2, capsize=5, linewidth=2,
                markersize=8, label="Mean ± SD", zorder=4)

    # ── Assi e formattazione ─────────────────────────────────
    if x_col.lower() in ["duration", "presentation_order"]:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    if y_col.lower() == "vividness":
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if x_col.lower() == "vividness":
        ax.set_xlim(vividness_values[0], vividness_values[-1])
    if y_col.lower() == "vividness":
        ax.set_ylim(vividness_values[0], vividness_values[-1])
    if y_col.lower().startswith("angle"):
        ax.set_ylim(-60, 60)

    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ============================================================
# PLOT PER SOGGETTO E PATTERN
# ============================================================
def plot_subject_pattern(subj_df, subject, pattern, protocol,
                         output_folder="Results/regressions",
                         x_col="duration", x_label="Duration (s)"):
    """
    Plo for a single subject and pattern:
    - Per rep
    - Mean of reps
    x_col can be 'duration' or 'presentation_order'.
    """
    df_sub = subj_df[subj_df["pattern_pair"] == pattern]

    # --------------------- Plot per rep ---------------------
    for rep in sorted(df_sub["rep"].unique()):
        df_rep = df_sub[df_sub["rep"] == rep]

        rep_dir = os.path.join(output_folder, subject, "Single-reps", pattern)
        os.makedirs(rep_dir, exist_ok=True)

        plot_regression(df_rep, x_col=x_col, y_col="angle_deg",
                        title=f"{subject} – {pattern} – rep {rep} – Angle",
                        xlabel=x_label, ylabel="Angle (deg)",
                        save_path=os.path.join(rep_dir, f"rep{rep}_angle.png"),
                        protocol=protocol)

        plot_regression(df_rep, x_col=x_col, y_col="vividness",
                        title=f"{subject} – {pattern} – rep {rep} – Vividness",
                        xlabel=x_label, ylabel="Vividness",
                        save_path=os.path.join(rep_dir, f"rep{rep}_vividness.png"),
                        protocol=protocol)

        plot_regression(df_rep, x_col="vividness", y_col="angle_deg",
                        title=f"{subject} – {pattern} – rep {rep} – Angle vs Vividness",
                        xlabel="Vividness", ylabel="Angle (deg)",
                        save_path=os.path.join(rep_dir, f"rep{rep}_angle_vs_vividness.png"),
                        protocol=protocol)

    # --------------------- Mean ---------------------
    if x_col == "presentation_order":
        df_mean = df_sub.groupby("presentation_order", as_index=False).agg({
            "angle_deg": "mean",
            "vividness": "mean",
            "duration": "first"
        })
    else:
        df_mean = df_sub.groupby("duration", as_index=False).agg({
            "angle_deg": "mean",
            "vividness": "mean",
            "presentation_order": "first"
        })

    mean_dir = os.path.join(output_folder, subject, "Mean-per-subject", pattern)
    os.makedirs(mean_dir, exist_ok=True)

    plot_regression(df_mean, x_col=x_col, y_col="angle_deg",
                    title=f"{subject} – {pattern} – Mean reps – Angle",
                    xlabel=x_label, ylabel="Angle (deg)",
                    save_path=os.path.join(mean_dir, "mean_angle.png"),
                    protocol=protocol)

    plot_regression(df_mean, x_col=x_col, y_col="vividness",
                    title=f"{subject} – {pattern} – Mean reps – Vividness",
                    xlabel=x_label, ylabel="Vividness",
                    save_path=os.path.join(mean_dir, "mean_vividness.png"),
                    protocol=protocol)

    plot_regression(df_mean, x_col="vividness", y_col="angle_deg",
                    title=f"{subject} – {pattern} – Mean reps – Angle vs Vividness",
                    xlabel="Vividness", ylabel="Angle (deg)",
                    save_path=os.path.join(mean_dir, "mean_angle_vs_vividness.png"),
                    protocol=protocol)


# ============================================================
# PLOT MEDIA SU TUTTI I SOGGETTI
# ============================================================
def plot_group_average(df, patterns_list, protocol,
                       output_folder="Results/regressions",
                       x_col="duration", x_label="Duration (s)"):
    """
    Regression plot for group-level data (all subjects combined).
    Layers:
        1. Scatter of raw data (all trials, all subjects, semi-transparent)
        2. Mean ± SD for each level of x_col
        3. Regression line forced through origin (weighted by vividness)
    Legend includes R² with respect to 0 for both raw data and means.
    """
    os.makedirs(output_folder, exist_ok=True)

    for pattern in patterns_list:
        df_pat = df[df["pattern_pair"] == pattern].copy()
        save_folder = os.path.join(output_folder, pattern)
        os.makedirs(save_folder, exist_ok=True)

        # Angle vs Duration
        plot_regression_group(df_pat, x_col=x_col, y_col="angle_deg",
                              title=f"ALL SUBJECTS – {pattern} – Angle\n"
                                    f"(scatter = individual trials,  ● = mean ± SD)",
                              xlabel=x_label, ylabel="Angle (deg)",
                              save_path=os.path.join(save_folder, "angle_all_subjects.png"),
                              protocol=protocol)

        # Vividness vs Duration
        plot_regression_group(df_pat, x_col=x_col, y_col="vividness",
                              title=f"ALL SUBJECTS – {pattern} – Vividness\n"
                                    f"(scatter = individual trials,  ● = mean ± SD)",
                              xlabel=x_label, ylabel="Vividness",
                              save_path=os.path.join(save_folder, "vividness_all_subjects.png"),
                              protocol=protocol)

        # Angle vs Vividness
        plot_regression_group(df_pat, x_col="vividness", y_col="angle_deg",
                              title=f"ALL SUBJECTS – {pattern} – Angle vs Vividness\n"
                                    f"(scatter = individual trials,  ● = mean ± SD)",
                              xlabel="Vividness", ylabel="Angle (deg)",
                              save_path=os.path.join(save_folder, "angle_vs_vividness_all_subjects.png"),
                              protocol=protocol)