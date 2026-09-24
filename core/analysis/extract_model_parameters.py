import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import config as cfg

from core.analysis.r2_metrics import compute_r2_metrics
from core.utils.patterns import pattern_sums

# ============================================================
# WEIGHTED REGRESSION
# ============================================================
def get_weights(vividness, n=None):
    """
    Return sample weights according to the configured analysis mode.

    If USE_VIVIDNESS_WEIGHTS is True, weights are vividness / 3.
    Otherwise, all samples receive equal weight.
    """

    if vividness is None:
        if n is None:
            raise ValueError(
                "n must be provided when vividness is None."
            )
        return np.ones(n, dtype=float)

    vividness = np.asarray(vividness)

    if cfg.USE_VIVIDNESS_WEIGHTS:
        return vividness / 3.0

    return np.ones(len(vividness), dtype=float)

def compute_regression_slope(x, y, vividness):
    """
    Fit a linear regression through the origin.

    ```
    Sample weights are determined by cfg.USE_VIVIDNESS_WEIGHTS.

    Parameters
    ----------
    x : array-like
        Predictor variable.

    y : array-like
        Observed angle.

    vividness : array-like
        Trial vividness scores.

    Returns
    -------
    slope : float
        Estimated regression coefficient.

    y_pred : np.ndarray
        Model predictions.

    r2_zero : float
        R² referenced to zero.
    """

    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)

    weights = get_weights(vividness, n=len(vividness))

    model = LinearRegression(
        fit_intercept=False
    )

    model.fit(
        x,
        y,
        sample_weight=weights
    )

    slope = model.coef_[0]

    y_pred = model.predict(x)

    r2_zero = compute_r2_metrics(
        y,
        y_pred,
        weights=weights
    )["R2_zero"]

    return slope, y_pred, r2_zero

# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_angle(pattern, duration, Kb, Kt):
    """
    Predict the perceived angle for a stimulation pattern.

    ```
    Model
    -----
    angle = Kb * pb_sum * duration
        + Kt * pt_sum * duration

    Parameters
    ----------
    pattern : str
        Stimulation pattern encoded as 'BBB_TTT'.

    duration : float
        Stimulation duration.

    Kb : float
        Biceps model parameter.

    Kt : float
        Triceps model parameter.

    Returns
    -------
    float
        Predicted angle.
    """

    pb, pt = pattern_sums(pattern)

    return (
        Kb * pb * duration
        + Kt * pt * duration
    )

def _predict_raw(raw_df, Kb, Kt):
    """
    Compute model predictions for raw trial-level data.

    ```
    Returns
    -------
    y_true : np.ndarray
    y_pred : np.ndarray
    weights : np.ndarray
    """

    if raw_df.empty:
        return (
            np.array([]),
            np.array([]),
            np.array([])
        )

    y_true = raw_df["angle_deg"].values

    y_pred = np.array([
        predict_angle(
            pattern=pattern,
            duration=duration,
            Kb=Kb,
            Kt=Kt
        )
        for pattern, duration
        in zip(
            raw_df["pattern_pair"],
            raw_df["duration"]
        )
    ])

    weights = get_weights(raw_df["vividness"].values)

    return y_true, y_pred, weights

# ============================================================
# SUBJECT PARAMETERS
# ============================================================

def extract_subject_parameters(subj_df):
    """
    Estimate Kb and Kt for one subject.

    ```
    Pure patterns are taken from cfg.PURE_PATTERNS.

    Returns
    -------
    params : dict
        Dictionary containing:

        Kb
        R2_Kb
        Kt
        R2_Kt
    """

    params = {}

    for pattern, label in cfg.PURE_PATTERNS.items():

        pat_df = subj_df[
            subj_df["pattern_pair"] == pattern
        ]

        if pat_df.empty:

            params[f"K{label}"] = np.nan
            params[f"R2_K{label}"] = np.nan

            continue

        slope, _, r2 = compute_regression_slope(
            x=pat_df["duration"].values,
            y=pat_df["angle_deg"].values,
            vividness=pat_df["vividness"].values
        )

        params[f"K{label}"] = slope
        params[f"R2_K{label}"] = r2

    return params

# ============================================================
# MODEL PERFORMANCE
# ============================================================

def compute_model_metrics(raw_df, Kb, Kt):
    """
    Evaluate an already-defined model on raw trial data.

    ```
    Metrics
    -------
    R²_zero
    MAE
    RMSE

    All metrics use the weighting scheme defined by cfg.USE_VIVIDNESS_WEIGHTS.

    Important
    ---------
    This function does not estimate Kb or Kt.
    It only evaluates the supplied parameters.
    """

    if (
        raw_df.empty
        or np.isnan(Kb)
        or np.isnan(Kt)
    ):

        return {
            "R2_zero": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan
        }

    y_true, y_pred, weights = _predict_raw(
        raw_df,
        Kb,
        Kt
    )

    r2_zero = compute_r2_metrics(
        y_true,
        y_pred,
        weights=weights
    )["R2_zero"]

    mae = float(
        np.average(
            np.abs(y_true - y_pred),
            weights=weights
        )
    )

    rmse = float(
        np.sqrt(
            np.average(
                (y_true - y_pred) ** 2,
                weights=weights
            )
        )
    )

    return {
        "R2_zero": r2_zero,
        "MAE": mae,
        "RMSE": rmse
    }

# ============================================================
# GLOBAL PARAMETERS
# ============================================================
def compute_global_parameters(df):
    """
    Estimate global Kb and Kt from pooled trial-level data.

    ```
    Kb is estimated from the pure pattern labeled "b"
    in cfg.PURE_PATTERNS.

    Kt is estimated from the pure pattern labeled "t"
    in cfg.PURE_PATTERNS.

    Parameters
    ----------
    df : pandas.DataFrame
        Trial-level processed data.

    Returns
    -------
    dict
        Dictionary containing:

        Kb
        Kt
        R2_Kb
        R2_Kt
    """

    biceps_pattern = next(
        pattern
        for pattern, label in cfg.PURE_PATTERNS.items()
        if label == "b"
    )

    triceps_pattern = next(
        pattern
        for pattern, label in cfg.PURE_PATTERNS.items()
        if label == "t"
    )

    biceps_df = df[
        df["pattern_pair"] == biceps_pattern
    ]

    triceps_df = df[
        df["pattern_pair"] == triceps_pattern
    ]

    if biceps_df.empty:
        Kb = np.nan
        R2_Kb = np.nan
    else:
        Kb, _, R2_Kb = compute_regression_slope(
            x=biceps_df["duration"].values,
            y=biceps_df["angle_deg"].values,
            vividness=biceps_df["vividness"].values
        )

    if triceps_df.empty:
        Kt = np.nan
        R2_Kt = np.nan
    else:
        Kt, _, R2_Kt = compute_regression_slope(
            x=triceps_df["duration"].values,
            y=triceps_df["angle_deg"].values,
            vividness=triceps_df["vividness"].values
        )

    return {
        "Kb": Kb,
        "Kt": Kt,
        "R2_Kb": R2_Kb,
        "R2_Kt": R2_Kt
    }


# ============================================================
# COMPLETE MODEL ANALYSIS
# ============================================================

def run_model_analysis(df, subjects, protocol):
    """
    Run the complete model analysis.

    This function performs the model-related calculations
    without building output tables or handling file I/O.

    Parameters
    ----------
    df : pandas.DataFrame
        Processed trial-level data for the selected subjects.

    subjects : list
        Subjects included in the analysis.

    protocol : dict
        Loaded experiment protocol.

    Returns
    -------
    dict
        Dictionary containing all model-analysis results:

        subject_parameters
            Subject-level Kb, Kt and model performance metrics.

        global_parameters
            Global Kb, Kt and fitting R² values.

        global_metrics_all
            Global model performance on all patterns.

        global_metrics_combined
            Global model performance on combined patterns only.

        df_combined
            Trial-level data containing combined patterns only.
    """

    # ========================================================
    # SUBJECT PARAMETERS
    # ========================================================

    subject_parameters = {}

    for subject in subjects:

        subj_df = df[
            df["subject"] == subject
        ].copy()

        params = extract_subject_parameters(
            subj_df
        )

        # ----------------------------------------------------
        # Subject-level model performance
        # ----------------------------------------------------

        metrics_all_subject = compute_model_metrics(
            raw_df=subj_df,
            Kb=params["Kb"],
            Kt=params["Kt"]
        )

        combined_df_subject = subj_df[
            ~subj_df["pattern_pair"].isin(
                cfg.PURE_PATTERNS.keys()
            )
        ].copy()

        metrics_combined_subject = compute_model_metrics(
            raw_df=combined_df_subject,
            Kb=params["Kb"],
            Kt=params["Kt"]
        )

        # ----------------------------------------------------
        # Store subject results
        # ----------------------------------------------------

        params["R2_full_all"] = (
            metrics_all_subject["R2_zero"]
        )

        params["MAE_full_all"] = (
            metrics_all_subject["MAE"]
        )

        params["RMSE_full_all"] = (
            metrics_all_subject["RMSE"]
        )

        params["R2_combined"] = (
            metrics_combined_subject["R2_zero"]
        )

        params["MAE_combined"] = (
            metrics_combined_subject["MAE"]
        )

        params["RMSE_combined"] = (
            metrics_combined_subject["RMSE"]
        )

        subject_parameters[subject] = params

    # ========================================================
    # GLOBAL PARAMETERS
    # ========================================================

    global_parameters = compute_global_parameters(
        df
    )

    Kb_global = global_parameters["Kb"]
    Kt_global = global_parameters["Kt"]

    # ========================================================
    # GLOBAL MODEL PERFORMANCE
    # ========================================================

    global_metrics_all = compute_model_metrics(
        raw_df=df,
        Kb=Kb_global,
        Kt=Kt_global
    )

    df_combined = df[
        ~df["pattern_pair"].isin(
            cfg.PURE_PATTERNS.keys()
        )
    ].copy()

    global_metrics_combined = compute_model_metrics(
        raw_df=df_combined,
        Kb=Kb_global,
        Kt=Kt_global
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "subject_parameters": subject_parameters,
        "global_parameters": global_parameters,
        "global_metrics_all": global_metrics_all,
        "global_metrics_combined": global_metrics_combined,
        "df_combined": df_combined
    }
