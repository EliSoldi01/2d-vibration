# core/visualization/plot_regressions.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config as cfg


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

ANALYSIS_COLUMNS = {
    "angle_vs_duration": {
        "x": "duration",
        "y": "angle_deg",
        "xlabel": "Duration (s)",
        "ylabel": "Angle (°)",
    },
    "vividness_vs_duration": {
        "x": "duration",
        "y": "vividness",
        "xlabel": "Duration (s)",
        "ylabel": "Vividness",
    },
    "angle_vs_vividness": {
        "x": "vividness",
        "y": "angle_deg",
        "xlabel": "Vividness",
        "ylabel": "Angle (°)",
    },
}


# ============================================================
# GENERIC HELPERS
# ============================================================

def _get_analysis_settings(analysis_name):
    """
    Return plotting settings for a regression analysis.

    Parameters
    ----------
    analysis_name : str
        Name of the regression analysis.

    Returns
    -------
    dict
        X/y columns and axis labels.
    """

    if analysis_name not in ANALYSIS_COLUMNS:
        raise ValueError(
            f"Unknown regression analysis: {analysis_name}"
        )

    return ANALYSIS_COLUMNS[analysis_name]

def _get_subject_result(subject_results, subject):
    """
    Return the regression result for a specific subject.

    Parameters
    ----------
    subject_results : list of dict
        Subject-level regression results. Each dictionary must contain
        the key "subject".

    subject : str
        Subject identifier.

    Returns
    -------
    dict or None
        Regression result of the subject, or None if the subject
        is not found.
    """


    for result in subject_results:
        if result["subject"] == subject:
            return result

    return None

def _get_pattern_to_process():
    """
    Return the list of patterns to process.

    If cfg.PATTERNS_TO_PROCESS is None, return all patterns
    found in the regression results.

    Returns
    -------
    list of str
        Patterns to process.
    """

    if cfg.PATTERNS_TO_PROCESS is not None:
        return cfg.PATTERNS_TO_PROCESS

    # If no specific patterns are specified, return all patterns
    # found in the regression results.
    all_patterns = set()

    for level_results in [
        "group_level",
        "subject_level",
        "trial_level",
    ]:
        if level_results in cfg.REGRESSION_RESULTS:
            for analysis_results in cfg.REGRESSION_RESULTS[level_results].values():
                all_patterns.update(analysis_results.keys())

    return sorted(all_patterns)


def _plot_regression_line(ax, result, x_min, x_max):
    """
    Plot a through-origin regression line from a fitted result.

    The line y = slope * x is drawn between x_min and x_max, with the
    slope and R²_zero in the legend label. Nothing is drawn if the
    result is None or the slope is not finite.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes where the line is drawn.

    result : dict or None
        Regression result containing the keys "slope" and "r_squared".

    x_min : float
        Left end of the line.

    x_max : float
        Right end of the line.

    Returns
    -------
    None
    """

    if result is None:
        return

    if not np.isfinite(result["slope"]):
        return

    x_line = np.linspace(x_min, x_max, 200)
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


def _format_regression_plot(ax, analysis_name, protocol, title):
    """
    Apply common formatting to a regression plot.

    Adds a dashed reference line at y = 0, the axis labels of the
    analysis, fixed y limits (-20, 20), the title and the legend. For
    analyses with duration on the x axis, the x ticks are set to the
    protocol durations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to format.

    analysis_name : str
        Name of the regression analysis, used to retrieve the
        axis labels.

    protocol : dict
        Experiment protocol. The durations are read from
        protocol["blocks"]["durations"].

    title : str
        Title of the plot.

    Returns
    -------
    None
    """

    settings = _get_analysis_settings(analysis_name)

    ax.axhline(
        0,
        linewidth=1,
        linestyle="--",
        alpha=0.5,
    )

    ax.set_xlabel(settings["xlabel"])
    ax.set_ylabel(settings["ylabel"])

    ax.set_ylim(-20, 20)

    if settings["x"] == "duration":
        ax.set_xticks(protocol["blocks"]["durations"])

    ax.set_title(title)
    ax.grid(False)
    ax.legend(fontsize=9, loc="best")


# ============================================================
# GROUP-LEVEL PLOT
# ============================================================

def plot_group_regression(
    plot_data,
    results,
    pattern,
    analysis_name,
    protocol,
    output_folder=cfg.REGRESSIONS_RESULTS_PATH / "group_level",
):
    """
    Plot one group-level regression.

    Parameters
    ----------
    plot_data : pandas.DataFrame
        Group-level data already prepared by the regression analysis.
        Must contain the group mean and, when available, SD columns.

    results : dict
        Regression result for the selected pattern and analysis.

    pattern : str
        Stimulation pattern.

    analysis_name : str
        Regression analysis to plot.

    protocol : dict
        Experiment protocol.

    output_folder : Path
        Output directory.

    Returns
    -------
    Path
        Path of the saved figure.
    """

    settings = _get_analysis_settings(analysis_name)

    df = plot_data[
        plot_data["pattern_pair"] == pattern
    ].copy()

    if df.empty:
        print(
            f"WARNING: no group-level data for "
            f"pattern {pattern}, analysis {analysis_name}"
        )
        return None

    df = df.sort_values(settings["x"])

    x = df[settings["x"]]
    y = df[settings["y"]]

    fig, ax = plt.subplots(figsize=(8, 6))

    # --------------------------------------------------------
    # Group mean ± SD
    # --------------------------------------------------------

    yerr = df["sd"] if "sd" in df.columns else None

    ax.errorbar(
        x,
        y,
        yerr=yerr,
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

    result = results.get(pattern)

    if result is not None:
        _plot_regression_line(
            ax=ax,
            result=result,
            x_min=x.min(),
            x_max=x.max(),
        )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    _format_regression_plot(
        ax=ax,
        analysis_name=analysis_name,
        protocol=protocol,
        title=f"Group – {pattern}\n{analysis_name.replace('_', ' ').title()}",
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_folder = (
        Path(output_folder)
        / analysis_name
        / pattern
    )

    save_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_path = save_folder / f"{analysis_name}.png"

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


# ============================================================
# SUBJECT-LEVEL PLOT
# ============================================================

def plot_subject_regression(
    plot_data,
    results,
    subject,
    pattern,
    analysis_name,
    protocol,
    output_folder=cfg.REGRESSIONS_RESULTS_PATH / "subject_level",
):
    """
    Plot one subject-level regression.

    Parameters
    ----------
    plot_data : pandas.DataFrame
        Subject-level means already prepared by the regression analysis.

    results : dict
        Subject-level regression results.

    subject : str
        Subject identifier.

    pattern : str
        Stimulation pattern.

    analysis_name : str
        Regression analysis to plot.

    protocol : dict
        Experiment protocol.

    output_folder : Path
        Output directory.

    Returns
    -------
    Path
        Path of the saved figure.
    """

    settings = _get_analysis_settings(analysis_name)

    df = plot_data[
        (plot_data["subject"] == subject)
        & (plot_data["pattern_pair"] == pattern)
    ].copy()

    if df.empty:
        print(
            f"WARNING: no data for subject {subject}, "
            f"pattern {pattern}, analysis {analysis_name}"
        )
        return None

    df = df.sort_values(settings["x"])

    x = df[settings["x"]]
    y = df[settings["y"]]

    fig, ax = plt.subplots(figsize=(8, 6))

    # --------------------------------------------------------
    # Subject means
    # --------------------------------------------------------

    ax.scatter(
        x,
        y,
        s=70,
        zorder=3,
        label="Subject mean",
    )

    # --------------------------------------------------------
    # Regression line
    # --------------------------------------------------------

    result = results.get(pattern)

    if result is not None:
        _plot_regression_line(
            ax=ax,
            result=result,
            x_min=x.min(),
            x_max=x.max(),
        )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    _format_regression_plot(
        ax=ax,
        analysis_name=analysis_name,
        protocol=protocol,
        title=(
            f"{subject} – {pattern}\n"
            f"{analysis_name.replace('_', ' ').title()}"
        ),
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_folder = (
        Path(output_folder)
        / analysis_name
        / subject
        / pattern
    )

    save_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_path = save_folder / f"{analysis_name}.png"

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


# ============================================================
# TRIAL-LEVEL PLOT
# ============================================================

def plot_trial_regression(
    plot_data,
    results,
    pattern,
    analysis_name,
    protocol,
    output_folder=cfg.REGRESSIONS_RESULTS_PATH / "trial_level",
):
    """
    Plot a trial-level regression.

    Unlike group- and subject-level plots, trial-level plots show
    the individual experimental trials together with the fitted
    through-origin regression line.

    This is particularly useful for visualizing the pure-pattern
    regressions used to estimate Kb and Kt.

    The figure shows the individual trials (grey), the mean across
    trials for each x value and the regression line. The y limits are
    set to the range of the data, plus a margin of 1.

    Parameters
    ----------
    plot_data : pandas.DataFrame
        Trial-level data. Must contain the column "pattern_pair" and
        the x and y columns of the analysis.

    results : dict
        Regression results of the analysis, mapping each pattern to its
        regression result.

    pattern : str
        Stimulation pattern.

    analysis_name : str
        Regression analysis to plot.

    protocol : dict
        Experiment protocol.

    output_folder : Path, optional
        Output directory. Defaults to
        cfg.REGRESSIONS_RESULTS_PATH / "trial_level".

    Returns
    -------
    Path or None
        Path of the saved figure, or None if no data are available for
        the pattern.
    """

    settings = _get_analysis_settings(analysis_name)

    df = plot_data[
        plot_data["pattern_pair"] == pattern
    ].copy()

    if df.empty:
        print(
            f"WARNING: no trial-level data for "
            f"pattern {pattern}, analysis {analysis_name}"
        )
        return None

    df = df.sort_values(settings["x"])

    x = df[settings["x"]]
    y = df[settings["y"]]

    max_y = max(y)
    min_y = min(y)

    fig, ax = plt.subplots(figsize=(8, 6))

    # --------------------------------------------------------
    # Individual trials
    # --------------------------------------------------------

    ax.scatter(
        x,
        y,
        s=35,
        alpha=0.6,
        label="Individual trials",
        color = "#b4b4b4",
        zorder=3,
    )

    # --------------------------------------------------------
    # Mean across trials
    # --------------------------------------------------------

    mean_data = (
        df.groupby(settings["x"], as_index=False)[settings["y"]]
        .mean()
    )

    ax.scatter(
        mean_data[settings["x"]],
        mean_data[settings["y"]],
        s=80,
        label="Trial mean",
        zorder=5,
    )

    # --------------------------------------------------------
    # Regression line
    # --------------------------------------------------------

    result = results.get(pattern)

    if result is not None:
        _plot_regression_line(
            ax=ax,
            result=result,
            x_min=x.min(),
            x_max=x.max(),
        )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    _format_regression_plot(
        ax=ax,
        analysis_name=analysis_name,
        protocol=protocol,
        title=(
            f"Trial level – {pattern}\n"
            f"{analysis_name.replace('_', ' ').title()}"
        ),
    )

    ax.set_ylim(min_y - 1, max_y + 1)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_folder = (
        Path(output_folder)
        / analysis_name
        / pattern
    )

    save_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_path = save_folder / f"{analysis_name}.png"

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path

# ============================================================
# RUN REGRESSION PLOTS
# ============================================================

def run_regression_plots(
    plot_data,
    results,
    protocol,
    group_level=True,
    subject_level=True,
    trial_level=True,
    analyses=None,
):
    """
    Generate the requested regression plots.

    Parameters
    ----------
    plot_data : dict
        Data prepared by ``run_regression_analysis``.

    results : dict
        Regression results prepared by ``run_regression_analysis``.

    protocol : dict
        Experiment protocol.

    group_level : bool, default=True
        Whether to generate group-level plots.

    subject_level : bool, default=True
        Whether to generate subject-level plots.

    trial_level : bool, default=True
        Whether to generate trial-level plots.

    analyses : list of str or None
        Analyses to plot. If None, all available analyses are plotted.

    Returns
    -------
    dict
        Paths of the generated plots.
    """

    if analyses is None:
        analyses = [
            "angle_vs_duration",
            "vividness_vs_duration",
            "angle_vs_vividness",
        ]

    plot_paths = {
        "group_level": {},
        "subject_level": {},
        "trial_level": {},
    }

    # ========================================================
    # GROUP LEVEL
    # ========================================================

    if group_level and plot_data.get("group_level"):

        group_data = plot_data["group_level"]["group_means"]
        group_results = results["group_level"]

        patterns = _get_pattern_to_process()

        for analysis_name in analyses:

            if analysis_name not in group_results:
                continue

            plot_paths["group_level"][analysis_name] = {}

            for pattern in patterns:

                path = plot_group_regression(
                    plot_data=group_data,
                    results=group_results[analysis_name],
                    pattern=pattern,
                    analysis_name=analysis_name,
                    protocol=protocol,
                )

                plot_paths["group_level"][
                    analysis_name
                ][pattern] = path

    # ========================================================
    # SUBJECT LEVEL
    # ========================================================

    if subject_level and plot_data.get("subject_level"):

        subject_data = plot_data["subject_level"]["subject_means"]
        subject_results = results["subject_level"]

        subjects = sorted(
            subject_data["subject"].unique()
        )

        patterns = _get_pattern_to_process()

        for analysis_name in analyses:

            if analysis_name not in subject_results:
                continue

            plot_paths["subject_level"][analysis_name] = {}

            for subject in subjects:

                for pattern in patterns:

                    subject_result = _get_subject_result(
                        subject_results[analysis_name][pattern],
                        subject,
                    )

                    path = plot_subject_regression(
                        plot_data=subject_data,
                        results={pattern: subject_result},
                        subject=subject,
                        pattern=pattern,
                        analysis_name=analysis_name,
                        protocol=protocol,
                    )

                    plot_paths["subject_level"][
                        analysis_name
                    ].setdefault(subject, {})[pattern] = path

    # ========================================================
    # TRIAL LEVEL
    # ========================================================

    if trial_level and plot_data.get("trial_level"):

        trial_data = plot_data["trial_level"]["trial_data"]
        trial_results = results["trial_level"]

        for analysis_name in analyses:

            if analysis_name not in trial_results:
                continue

            plot_paths["trial_level"][analysis_name] = {}


            patterns = _get_pattern_to_process()

            for pattern in patterns:

                path = plot_trial_regression(
                    plot_data=trial_data,
                    results=trial_results[analysis_name],
                    pattern=pattern,
                    analysis_name=analysis_name,
                    protocol=protocol,
                )

                plot_paths["trial_level"][
                    analysis_name
                ][pattern] = path

    return plot_paths