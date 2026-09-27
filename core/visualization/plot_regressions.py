# core/visualization/regression_plots.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from core.analysis.regressions import (
    aggregate_subject_means,
    aggregate_global_means,
    fit_linear_regression,
)


# ============================================================
# PLOT SETTINGS
# ============================================================

OUTPUT_FOLDER = Path("Results") / "regressions"


# ============================================================
# SUBJECT-LEVEL PLOT
# ============================================================

def plot_subject_angle_vs_duration(
    df_subject,
    subject,
    pattern,
    protocol,
    output_folder=cfg.REGRESSIONS_RESULTS_PATH / "subject_level",
):
    """
    Plot angle vs duration for one subject and one pattern.

    Input:
        df_subject:
            Data already aggregated as:
            subject × pattern × duration

    The regression is fitted on the three subject-level means
    (one mean per duration).

    Returns:
        regression result dictionary.
    """

    df = df_subject[
        (df_subject["subject"] == subject)
        & (df_subject["pattern_pair"] == pattern)
    ].copy()

    if df.empty:
        print(
            f"WARNING: no data for subject {subject}, "
            f"pattern {pattern}"
        )
        return None

    df = df.sort_values("duration")

    if len(df) < 2:
        print(
            f"WARNING: not enough data for subject {subject}, "
            f"pattern {pattern}"
        )
        return None

    # --------------------------------------------------------
    # Regression
    # --------------------------------------------------------

    result = fit_linear_regression(
        x=df["duration"],
        y=df["angle_deg"],
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    # Subject-level means
    ax.scatter(
        df["duration"],
        df["angle_deg"],
        s=70,
        zorder=3,
        label="Subject mean",
    )

    # --------------------------------------------------------
    # Regression line
    # --------------------------------------------------------

    x_line = np.linspace(
        df["duration"].min(),
        df["duration"].max(),
        200,
    )

    y_line = result["slope"] * x_line

    ax.plot(
        x_line,
        y_line,
        linewidth=2,
        label=(
            f"Fit: y = {result['slope']:.2f}x\n"
            f"R²₀ = {result['r_squared']:.2f}"
        ),
        zorder=2,
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.axhline(
        0,
        linewidth=1,
        linestyle="--",
        alpha=0.5,
    )

    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Angle (°)")

    ax.set_ylim(-20, +20)
    ax.set_xticks(protocol["blocks"]["durations"])

    ax.set_title(
        f"{subject} – {pattern}\n"
        "Angle vs Duration"
    )

    ax.grid(False)

    ax.legend(
        fontsize=9,
        loc="best",
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_folder = (
        Path(output_folder)
        / subject
        / pattern
    )

    save_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_path = (
        save_folder
        / "angle_vs_duration.png"
    )

    fig.tight_layout()
    fig.savefig(
        save_path,
        dpi=200,
    )
    plt.close(fig)

    return result


# ============================================================
# GROUP-LEVEL PLOT
# ============================================================

def plot_group_angle_vs_duration(
    df,
    pattern,
    protocol,
    output_folder=cfg.REGRESSIONS_RESULTS_PATH / "group_level",
):
    """
    Plot group-level angle vs duration for one pattern.

    Aggregation:
        repetitions
            ↓
        subject × pattern × duration means
            ↓
        group mean ± SD across subjects
            ↓
        regression on the three group means

    Returns:
        regression result dictionary.
    """

    # --------------------------------------------------------
    # Aggregate repetitions within subject
    # --------------------------------------------------------

    df_subject = aggregate_subject_means(df)

    # --------------------------------------------------------
    # Aggregate subjects into group means
    # --------------------------------------------------------

    df_group = aggregate_global_means(df_subject)

    df_group = df_group[
        df_group["pattern_pair"] == pattern
    ].copy()

    if df_group.empty:
        print(
            f"WARNING: no group-level data for pattern {pattern}"
        )
        return None

    df_group = df_group.sort_values("duration")

    if len(df_group) < 2:
        print(
            f"WARNING: not enough group-level data "
            f"for pattern {pattern}"
        )
        return None

    # --------------------------------------------------------
    # Calculate SD across subjects
    # --------------------------------------------------------

    df_subject_pattern = df_subject[
        df_subject["pattern_pair"] == pattern
    ].copy()

    df_sd = (
        df_subject_pattern
        .groupby("duration")["angle_deg"]
        .agg(
            mean="mean",
            sd="std",
        )
        .reset_index()
    )

    df_group = df_group.merge(
        df_sd[["duration", "sd"]],
        on="duration",
        how="left",
    )

    # --------------------------------------------------------
    # Regression on group means
    # --------------------------------------------------------

    result = fit_linear_regression(
        x=df_group["duration"],
        y=df_group["angle_deg"],
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    # Group mean ± SD
    ax.errorbar(
        df_group["duration"],
        df_group["angle_deg"],
        yerr=df_group["sd"],
        fmt="o",
        markersize=8,
        capsize=5,
        linewidth=2,
        label="Group mean ± SD",
        zorder=3,
    )

    # --------------------------------------------------------
    # Regression line
    # --------------------------------------------------------

    x_line = np.linspace(
        df_group["duration"].min(),
        df_group["duration"].max(),
        200,
    )

    y_line = result["slope"] * x_line

    ax.plot(
        x_line,
        y_line,
        linewidth=2,
        label=(
            f"Fit: y = {result['slope']:.2f}x\n"
            f"R²₀ = {result['r_squared']:.2f}"
        ),
        zorder=2,
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.axhline(
        0,
        linewidth=1,
        linestyle="--",
        alpha=0.5,
    )

    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Angle (°)")

    ax.set_xticks(protocol["blocks"]["durations"])

    ax.set_title(
        f"Group – {pattern}\n"
        "Angle vs Duration"
    )

    ax.grid(False)

    ax.legend(
        fontsize=9,
        loc="best",
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_folder = (
        Path(output_folder)
        / pattern
    )

    save_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_path = (
        save_folder
        / "angle_vs_duration.png"
    )

    fig.tight_layout()
    fig.savefig(
        save_path,
        dpi=200,
    )
    plt.close(fig)

    return result


# ============================================================
# RUN ALL REGRESSION PLOTS
# ============================================================

def run_angle_vs_duration_plots(
    df,
    patterns,
    protocol
):
    """
    Generate angle-vs-duration regression plots.

    Group-level plots are controlled by:
        cfg.INCLUDE_GROUP_LEVEL_REGRESSION_PLOTS

    Subject-level plots are controlled by:
        cfg.INCLUDE_SUBJECT_LEVEL_REGRESSION_PLOTS

    Returns:
        Dictionary containing regression results.
    """

    results = {
        "group_level": {},
        "subject_level": {},
    }

    # --------------------------------------------------------
    # Group-level
    # --------------------------------------------------------

    if cfg.INCLUDE_GROUP_LEVEL_REGRESSION_PLOTS:

        for pattern in patterns:

            result = plot_group_angle_vs_duration(
                df=df,
                pattern=pattern,
                protocol=protocol
            )

            results["group_level"][pattern] = result

    # --------------------------------------------------------
    # Subject-level
    # --------------------------------------------------------

    if cfg.INCLUDE_SUBJECT_LEVEL_REGRESSION_PLOTS:

        df_subject = aggregate_subject_means(df)

        subjects = sorted(
            df_subject["subject"].unique()
        )

        for subject in subjects:

            for pattern in patterns:

                result = plot_subject_angle_vs_duration(
                    df_subject=df_subject,
                    subject=subject,
                    pattern=pattern,
                    protocol=protocol
                )

                results["subject_level"].setdefault(
                    pattern,
                    {},
                )

                results["subject_level"][
                    pattern
                ][subject] = result

    return results