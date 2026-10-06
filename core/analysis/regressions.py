# core/analysis/regressions.py

import numpy as np
from sklearn.linear_model import LinearRegression

import config as cfg
from core.analysis.metrics import compute_r2_metrics


# ============================================================
# CORE REGRESSION
# ============================================================

def fit_linear_regression(x, y, weights=None):
    """
    Fit a linear regression forced through the origin:

        y = slope * x

    Parameters
    ----------
    x : array-like
        Independent variable.
    y : array-like
        Dependent variable.
    weights : array-like, optional
        Sample weights.

    Returns
    -------
    dict
        Regression parameters and performance metrics.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        valid &= np.isfinite(weights) & (weights > 0)

    x = x[valid]
    y = y[valid]

    if weights is not None:
        weights = weights[valid]

    if len(x) < 2:
        return {"slope": np.nan, "r_squared": np.nan, "mae": np.nan, "rmse": np.nan, "n": len(x)}

    X = x.reshape(-1, 1)

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y, sample_weight=weights)

    slope = float(model.coef_[0])
    y_pred = model.predict(X)

    metrics = compute_r2_metrics(y, y_pred, weights=weights)

    if weights is None:
        mae = float(np.mean(np.abs(y - y_pred)))
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    else:
        mae = float(np.average(np.abs(y - y_pred), weights=weights))
        rmse = float(np.sqrt(np.average((y - y_pred) ** 2, weights=weights)))

    return {
        "slope": slope,
        "r_squared": metrics["R2_zero"],
        "mae": mae,
        "rmse": rmse,
        "n": len(x),
    }


# ============================================================
# WEIGHTS
# ============================================================

def _get_weights(df, max_vividness=None):
    """
    Return regression weights according to the project configuration.

    If cfg.USE_VIVIDNESS_WEIGHTS is True, the weight is defined as:

        weight = vividness / max_vividness

    Otherwise, None is returned.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing the "vividness" column.

    max_vividness : float, optional
        Value used to normalize vividness scores. Required when
        cfg.USE_VIVIDNESS_WEIGHTS is True.

    Returns
    -------
    np.ndarray or None
        Array of weights, or None if vividness weighting is disabled.
    """

    if not cfg.USE_VIVIDNESS_WEIGHTS:
        return None

    return df["vividness"].to_numpy(dtype=float) / max_vividness


# ============================================================
# REGRESSION SUMMARY
# ============================================================

def regression_summary(df, x_col, y_col, group_col=None, max_vividness=None):
    """
    Compute through-origin linear regression.

    If group_col is None, one regression is computed on the complete
    dataframe. If group_col is provided, one regression is computed
    separately for each group.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    x_col : str
        Name of the independent-variable column.
    y_col : str
        Name of the dependent-variable column.
    group_col : str, optional
        Column used to divide the data into separate regressions.
    max_vividness : float, optional
        Value used to normalize vividness scores when computing weights.

    Returns
    -------
    dict or list of dict
        Regression result, or one result per group. In the grouped case,
        each result also contains the group value under the key group_col.
    """
    if group_col is None:
        weights = _get_weights(df, max_vividness=max_vividness)
        return fit_linear_regression(df[x_col], df[y_col], weights=weights)

    results = []

    for group, df_group in df.groupby(group_col):
        weights = _get_weights(df_group, max_vividness=max_vividness)
        result = fit_linear_regression(df_group[x_col], df_group[y_col], weights=weights)
        results.append({group_col: group, **result})

    return results


# ============================================================
# DATA AGGREGATION
# ============================================================

def aggregate_subject_means(df):
    """
    Average repetitions for each subject × pattern × duration.

    Parameters
    ----------
    df : pandas.DataFrame
        Trial-level data containing the columns "subject", "pattern_pair",
        "duration", "angle_deg" and "vividness".

    Returns
    -------
    pandas.DataFrame
        One row per subject × pattern × duration, with the mean
        "angle_deg" and "vividness".
    """
    return (
        df.groupby(["subject", "pattern_pair", "duration"], as_index=False)
        .agg(
            angle_deg=("angle_deg", "mean"),
            vividness=("vividness", "mean"),
        )
    )


def aggregate_global_means(df):
    """
    Compute group-level means from subject-level means.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing one row per subject × pattern × duration.

    Returns
    -------
    pandas.DataFrame
        One row per pattern × duration.
    """
    return (
        df.groupby(["pattern_pair", "duration"], as_index=False)
        .agg(
            angle_deg=("angle_deg", "mean"),
            vividness=("vividness", "mean"),
        )
    )


# ============================================================
# PRINTING
# ============================================================

def _print_regression_result(pattern, analysis_name, result):
    """
    Print a regression result in a consistent format.

    Parameters
    ----------
    pattern : str
        Pattern the regression refers to.
    analysis_name : str
        Description of the analysis, printed as a header.
    result : dict or None
        Regression result as returned by fit_linear_regression.
        If None, a "No result." message is printed.
    """
    if result is None:
        print(f"\nPattern: {pattern}\n  {analysis_name}\n  No result.")
        return

    print(
        f"\nPattern: {pattern}"
        f"\n  {analysis_name}"
        f"\n  n = {result['n']}"
        f"\n  slope = {result['slope']:.6f}"
        f"\n  R²_zero = {result['r_squared']:.6f}"
        f"\n  MAE = {result['mae']:.6f}"
        f"\n  RMSE = {result['rmse']:.6f}"
    )


# ============================================================
# GLOBAL / ALL SUBJECTS
# ============================================================

def run_global_regression(df, patterns, x_col, y_col, analysis_name, max_vividness = None):
    """
    Run one regression for each pattern at the global/group level.

    Repetitions are first averaged within each subject, then subjects
    are averaged within each pattern × duration combination.

    Parameters
    ----------
    df : pandas.DataFrame
        Trial-level input data.
    patterns : iterable
        Patterns to analyze.
    x_col : str
        Name of the independent-variable column.
    y_col : str
        Name of the dependent-variable column.
    analysis_name : str
        Description used when printing results.
    max_vividness : float, optional
        Value used to normalize vividness scores when computing weights.

    Returns
    -------
    dict
        Dictionary mapping each pattern to its regression result, or to
        None if no data are available for that pattern.
    """

    df_subject = aggregate_subject_means(df)
    df_global = aggregate_global_means(df_subject)
    results = {}

    for pattern in patterns:
        df_pattern = df_global[df_global["pattern_pair"] == pattern].copy()

        if df_pattern.empty:
            print(f"\nWARNING: no global data for pattern {pattern}")
            results[pattern] = None
            continue

        result = regression_summary(df_pattern, x_col=x_col, y_col=y_col, max_vividness=max_vividness)

        #_print_regression_result(pattern, analysis_name, result)
        results[pattern] = result

    return results


# ============================================================
# SUBJECT LEVEL
# ============================================================

def run_subject_regression(df, patterns, x_col, y_col, max_vividness=None):
    """
    Run one regression for each subject and pattern.

    Repetitions are first averaged within each subject × pattern ×
    duration combination.

    Parameters
    ----------
    df : pandas.DataFrame
        Trial-level input data.
    patterns : iterable
        Patterns to analyze.
    x_col : str
        Name of the independent-variable column.
    y_col : str
        Name of the dependent-variable column.
    max_vividness : float, optional
        Value used to normalize vividness scores when computing weights.

    Returns
    -------
    dict
        Dictionary mapping each pattern to a list of subject-level
        regression results (an empty list if no data are available
        for that pattern).
    """
    df_subject = aggregate_subject_means(df)
    results = {}

    for pattern in patterns:
        df_pattern = df_subject[df_subject["pattern_pair"] == pattern].copy()

        if df_pattern.empty:
            print(f"\nWARNING: no subject-level data for pattern {pattern}")
            results[pattern] = []
            continue

        results[pattern] = regression_summary(
            df_pattern,
            x_col=x_col,
            y_col=y_col,
            group_col="subject",
            max_vividness=max_vividness
        )

    return results


# ============================================================
# TRIAL LEVEL
# ============================================================

def run_trial_regression(df, patterns, x_col, y_col, analysis_name, max_vividness=None):
    """
    Run one regression for each pattern directly on individual trials.

    No averaging of repetitions is performed.

    Parameters
    ----------
    df : pandas.DataFrame
        Trial-level input data.
    patterns : iterable
        Patterns to analyze.
    x_col : str
        Name of the independent-variable column.
    y_col : str
        Name of the dependent-variable column.
    analysis_name : str
        Description used when printing results.
    max_vividness : float, optional
        Value used to normalize vividness scores when computing weights.

    Returns
    -------
    dict
        Dictionary mapping each pattern to its regression result, or to
        None if no data are available for that pattern.
    """
    results = {}

    for pattern in patterns:
        df_pattern = df[df["pattern_pair"] == pattern].copy()

        if df_pattern.empty:
            print(f"\nWARNING: no trial-level data for pattern {pattern}")
            results[pattern] = None
            continue

        result = regression_summary(df_pattern, x_col=x_col, y_col=y_col, max_vividness=max_vividness)

        #_print_regression_result(pattern, analysis_name, result)
        results[pattern] = result

    return results

# ============================================================
# REGRESSION ANALYSIS
# ============================================================

def run_regression_analysis(df, protocol, group_level=True, subject_level=True, trial_level=True, max_vividness=None):
    """
    Run the selected regression analyses and prepare data for plotting.

    Patterns are taken from cfg.PATTERNS_TO_PROCESS. If it is None, all
    patterns found in df["pattern_pair"] are used.

    Parameters
    ----------
    df : pandas.DataFrame
        Trial-level input data.
    protocol : dict
        Protocol information (currently unused).
    group_level : bool, optional
        Whether to run group-level regressions (angle vs duration,
        vividness vs duration, angle vs vividness).
    subject_level : bool, optional
        Whether to run subject-level regressions (same three analyses).
    trial_level : bool, optional
        Whether to run trial-level regressions (angle vs duration).
    max_vividness : float, optional
        Value used to normalize vividness scores when computing weights.

    Returns
    -------
    dict
        Dictionary containing:

        results
            Regression results, organized by level ("group_level",
            "subject_level", "trial_level") and then by analysis name.
        plot_data
            Data needed for plotting, organized by level:
            subject and group means for the group level, subject means
            for the subject level, and the trial data for the trial level.
            Levels that were not run are left empty.
    """

    patterns = cfg.PATTERNS_TO_PROCESS

    if patterns is None:
        patterns = sorted(df["pattern_pair"].dropna().unique())

    results = {
        "group_level": {},
        "subject_level": {},
        "trial_level": {},
    }

    plot_data = {
        "group_level": {},
        "subject_level": {},
        "trial_level": {},
    }

    # --------------------------------------------------------
    # Prepare subject means once if needed
    # --------------------------------------------------------

    df_subject = None

    if group_level or subject_level:
        df_subject = aggregate_subject_means(df)

    # --------------------------------------------------------
    # GROUP LEVEL
    # --------------------------------------------------------

    if group_level:
        print("     -> Running global regressions 1/3: Angle vs duration")
        results["group_level"]["angle_vs_duration"] = run_global_regression(
            df, patterns, "duration", "angle_deg",
            "GLOBAL: angle vs duration",
            max_vividness=max_vividness
        )

        print("     -> Running global regressions 2/3: Vividness vs duration")
        results["group_level"]["vividness_vs_duration"] = run_global_regression(
            df, patterns, "duration", "vividness",
            "GLOBAL: vividness vs duration",
            max_vividness=max_vividness
        )

        print("     -> Running global regressions 3/3: Angle vs vividness")
        results["group_level"]["angle_vs_vividness"] = run_global_regression(
            df, patterns, "vividness", "angle_deg",
            "GLOBAL: angle vs vividness",
            max_vividness=max_vividness
        )

        plot_data["group_level"] = {
            "subject_means": df_subject,
            "group_means": aggregate_global_means(df_subject),
        }

    # --------------------------------------------------------
    # SUBJECT LEVEL
    # --------------------------------------------------------

    if subject_level:
        print("     -> Running subject regressions 1/3: Angle vs duration")
        results["subject_level"]["angle_vs_duration"] = run_subject_regression(
            df, patterns, "duration", "angle_deg",
            max_vividness=max_vividness
        )

        print("     -> Running subject regressions 2/3: Vividness vs duration")
        results["subject_level"]["vividness_vs_duration"] = run_subject_regression(
            df, patterns, "duration", "vividness",
            max_vividness=max_vividness
        )

        print("     -> Running subject regressions 3/3: Angle vs vividness")
        results["subject_level"]["angle_vs_vividness"] = run_subject_regression(
            df, patterns, "vividness", "angle_deg",
            max_vividness=max_vividness
        )

        plot_data["subject_level"] = {
            "subject_means": df_subject,
        }

    # --------------------------------------------------------
    # TRIAL LEVEL
    # --------------------------------------------------------

    if trial_level:

        print("     -> Running trial level regressions: Angle vs duration")
        results["trial_level"]["angle_vs_duration"] = run_trial_regression(
            df, patterns, "duration", "angle_deg",
            "TRIAL LEVEL: angle vs duration",
            max_vividness=max_vividness
        )

        plot_data["trial_level"] = {
            "trial_data": df,
        }

    return {
        "results": results,
        "plot_data": plot_data,
    }