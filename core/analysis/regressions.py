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
        return {
            "slope": np.nan,
            "r_squared": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "n": len(x),
        }

    X = x.reshape(-1, 1)

    model = LinearRegression(fit_intercept=False)
    model.fit(
        X,
        y,
        sample_weight=weights,
    )

    slope = float(model.coef_[0])

    y_pred = model.predict(X)

    metrics = compute_r2_metrics(
        y,
        y_pred,
        weights=weights,
    )

    if weights is None:

        mae = float(
            np.mean(np.abs(y - y_pred))
        )

        rmse = float(
            np.sqrt(
                np.mean((y - y_pred) ** 2)
            )
        )

    else:

        mae = float(
            np.average(
                np.abs(y - y_pred),
                weights=weights,
            )
        )

        rmse = float(
            np.sqrt(
                np.average(
                    (y - y_pred) ** 2,
                    weights=weights,
                )
            )
        )

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

def _get_weights(df):
    """
    Return regression weights according to the project configuration.

    If cfg.USE_VIVIDNESS_WEIGHTS is True:
        weight = vividness / 3

    Otherwise:
        return None.
    """

    if not cfg.USE_VIVIDNESS_WEIGHTS:
        return None

    return (
        df["vividness"]
        .to_numpy(dtype=float)
        / 3.0
    )


# ============================================================
# REGRESSION SUMMARY
# ============================================================

def regression_summary(
    df,
    x_col,
    y_col,
    group_col=None,
):
    """
    Compute through-origin linear regression.

    If group_col is None:
        one regression is computed on the complete dataframe.

    If group_col is provided:
        one regression is computed for each group.

    Vividness weighting is controlled globally by:

        cfg.USE_VIVIDNESS_WEIGHTS
    """

    if group_col is None:

        weights = _get_weights(df)

        return fit_linear_regression(
            df[x_col],
            df[y_col],
            weights=weights,
        )

    results = []

    for group, df_group in df.groupby(group_col):

        weights = _get_weights(df_group)

        result = fit_linear_regression(
            df_group[x_col],
            df_group[y_col],
            weights=weights,
        )

        results.append({
            group_col: group,
            **result,
        })

    return results


# ============================================================
# DATA AGGREGATION
# ============================================================

def aggregate_subject_means(df):
    """
    Average repetitions for each:

        subject × pattern × duration

    This is the first aggregation step used by the
    global and subject-level analyses.

    Returns
    -------
    pandas.DataFrame
        One row per subject × pattern × duration.
    """

    return (
        df.groupby(
            [
                "subject",
                "pattern_pair",
                "duration",
            ],
            as_index=False,
        )
        .agg(
            angle_deg=("angle_deg", "mean"),
            vividness=("vividness", "mean"),
        )
    )


def aggregate_global_means(df):
    """
    Compute group-level means.

    The input dataframe is expected to contain one row per:

        subject × pattern × duration

    Therefore, this function computes the mean across subjects
    for each:

        pattern × duration

    Returns
    -------
    pandas.DataFrame
        One row per pattern × duration.
    """

    return (
        df.groupby(
            [
                "pattern_pair",
                "duration",
            ],
            as_index=False,
        )
        .agg(
            angle_deg=("angle_deg", "mean"),
            vividness=("vividness", "mean"),
        )
    )


# ============================================================
# PRINTING
# ============================================================

def _print_regression_result(
    pattern,
    analysis_name,
    result,
):
    """
    Print a regression result in a consistent format.
    """

    if result is None:
        print(
            f"\nPattern: {pattern}"
            f"\n  {analysis_name}"
            f"\n  No result."
        )
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

def run_global_angle_vs_duration(
    df,
    patterns,
):
    """
    Global / ALL SUBJECTS regression:

        angle_deg ~ duration

    Aggregation:

        repetitions
            ↓
        subject × pattern × duration means
            ↓
        pattern × duration group means
            ↓
        regression

    One regression is performed for each pattern.

    Returns
    -------
    dict
        {
            pattern: regression_result
        }
    """

    # --------------------------------------------------------
    # Step 1: average repetitions within each subject
    # --------------------------------------------------------

    df_subject = aggregate_subject_means(df)

    # --------------------------------------------------------
    # Step 2: average subjects within each pattern × duration
    # --------------------------------------------------------

    df_global = aggregate_global_means(df_subject)

    results = {}

    # --------------------------------------------------------
    # Step 3: regression for each pattern
    # --------------------------------------------------------

    for pattern in patterns:

        df_pattern = df_global[
            df_global["pattern_pair"] == pattern
        ].copy()

        if df_pattern.empty:
            print(
                f"\nWARNING: no global data for pattern {pattern}"
            )

            results[pattern] = None
            continue

        result = regression_summary(
            df=df_pattern,
            x_col="duration",
            y_col="angle_deg",
        )

        _print_regression_result(
            pattern=pattern,
            analysis_name=(
                "GLOBAL: angle vs duration"
            ),
            result=result,
        )

        results[pattern] = result

    return results


def run_global_vividness_vs_duration(
    df,
    patterns,
):
    """
    Global / ALL SUBJECTS regression:

        vividness ~ duration

    Aggregation:

        repetitions
            ↓
        subject × pattern × duration means
            ↓
        pattern × duration group means
            ↓
        regression

    One regression is performed for each pattern.
    """

    df_subject = aggregate_subject_means(df)

    df_global = aggregate_global_means(df_subject)

    results = {}

    for pattern in patterns:

        df_pattern = df_global[
            df_global["pattern_pair"] == pattern
        ].copy()

        if df_pattern.empty:
            print(
                f"\nWARNING: no global data for pattern {pattern}"
            )

            results[pattern] = None
            continue

        result = regression_summary(
            df=df_pattern,
            x_col="duration",
            y_col="vividness",
        )

        _print_regression_result(
            pattern=pattern,
            analysis_name=(
                "GLOBAL: vividness vs duration"
            ),
            result=result,
        )

        results[pattern] = result

    return results


def run_global_angle_vs_vividness(
    df,
    patterns,
):
    """
    Global / ALL SUBJECTS regression:

        angle_deg ~ vividness

    Aggregation:

        repetitions
            ↓
        subject × pattern × duration means
            ↓
        pattern × duration group means
            ↓
        regression

    One regression is performed for each pattern.
    """

    df_subject = aggregate_subject_means(df)

    df_global = aggregate_global_means(df_subject)

    results = {}

    for pattern in patterns:

        df_pattern = df_global[
            df_global["pattern_pair"] == pattern
        ].copy()

        if df_pattern.empty:
            print(
                f"\nWARNING: no global data for pattern {pattern}"
            )

            results[pattern] = None
            continue

        result = regression_summary(
            df=df_pattern,
            x_col="vividness",
            y_col="angle_deg",
        )

        _print_regression_result(
            pattern=pattern,
            analysis_name=(
                "GLOBAL: angle vs vividness"
            ),
            result=result,
        )

        results[pattern] = result

    return results


# ============================================================
# SUBJECT LEVEL
# ============================================================

def run_subject_angle_vs_duration(
    df,
    patterns,
):
    """
    Subject-level regression:

        angle_deg ~ duration

    Aggregation:

        repetitions
            ↓
        subject × pattern × duration means
            ↓
        regression separately for each subject and pattern.

    Returns
    -------
    dict
        {
            pattern: [
                {
                    "subject": ...,
                    "slope": ...,
                    ...
                },
                ...
            ]
        }
    """

    df_subject = aggregate_subject_means(df)

    results = {}

    for pattern in patterns:

        df_pattern = df_subject[
            df_subject["pattern_pair"] == pattern
        ].copy()

        if df_pattern.empty:
            print(
                f"\nWARNING: no subject-level data "
                f"for pattern {pattern}"
            )

            results[pattern] = []
            continue

        pattern_results = regression_summary(
            df=df_pattern,
            x_col="duration",
            y_col="angle_deg",
            group_col="subject",
        )

        results[pattern] = pattern_results

    return results


def run_subject_vividness_vs_duration(
    df,
    patterns,
):
    """
    Subject-level regression:

        vividness ~ duration

    Regression is performed separately for each
    subject and pattern.
    """

    df_subject = aggregate_subject_means(df)

    results = {}

    for pattern in patterns:

        df_pattern = df_subject[
            df_subject["pattern_pair"] == pattern
        ].copy()

        if df_pattern.empty:
            print(
                f"\nWARNING: no subject-level data "
                f"for pattern {pattern}"
            )

            results[pattern] = []
            continue

        pattern_results = regression_summary(
            df=df_pattern,
            x_col="duration",
            y_col="vividness",
            group_col="subject",
        )

        results[pattern] = pattern_results

    return results


def run_subject_angle_vs_vividness(
    df,
    patterns,
):
    """
    Subject-level regression:

        angle_deg ~ vividness

    Regression is performed separately for each
    subject and pattern.
    """

    df_subject = aggregate_subject_means(df)

    results = {}

    for pattern in patterns:

        df_pattern = df_subject[
            df_subject["pattern_pair"] == pattern
        ].copy()

        if df_pattern.empty:
            print(
                f"\nWARNING: no subject-level data "
                f"for pattern {pattern}"
            )

            results[pattern] = []
            continue

        pattern_results = regression_summary(
            df=df_pattern,
            x_col="vividness",
            y_col="angle_deg",
            group_col="subject",
        )

        results[pattern] = pattern_results

    return results


# ============================================================
# TRIAL LEVEL
# ============================================================

def run_trial_angle_vs_duration(
    df,
    patterns,
):
    """
    Trial-level regression:

        angle_deg ~ duration

    Regression is performed directly on the individual trials.

    No averaging of repetitions is performed.

    This analysis is intended to preserve the original
    trial-level regression used for the pure-pattern
    Kb / Kt estimation.
    """

    results = {}

    for pattern in patterns:

        df_pattern = df[
            df["pattern_pair"] == pattern
        ].copy()

        if df_pattern.empty:
            print(
                f"\nWARNING: no trial-level data "
                f"for pattern {pattern}"
            )

            results[pattern] = None
            continue

        result = regression_summary(
            df=df_pattern,
            x_col="duration",
            y_col="angle_deg",
        )

        _print_regression_result(
            pattern=pattern,
            analysis_name=(
                "TRIAL LEVEL: angle vs duration"
            ),
            result=result,
        )

        results[pattern] = result

    return results