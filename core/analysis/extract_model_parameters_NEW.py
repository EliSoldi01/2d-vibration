import os
import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.linear_model import LinearRegression

from core.utils.r2_metrics import compute_r2_metrics as r2_metrics


# =============================================================================
# HELPERS
# =============================================================================

def _pattern_sums(pattern):
    """
    Return the number of biceps and triceps stimulation pulses contained
    in a stimulation pattern.

    Example
    -------
    pattern = "011_100"

    -> biceps = 0 + 1 + 1 = 2
    -> triceps = 1 + 0 + 0 = 1

    Returns
    -------
    pb_sum : int
        Number of biceps stimulation pulses.
    pt_sum : int
        Number of triceps stimulation pulses.
    """

    b, t = pattern.split("_")

    pb_sum = sum(int(x) for x in b)
    pt_sum = sum(int(x) for x in t)

    return pb_sum, pt_sum


# =============================================================================
# MODEL PREDICTION
# =============================================================================

def _predict_raw(raw_df, Kb, Kt):
    """
    Compute model predictions for every raw trial.

    Model
    -----
        Δθ = Kb * pb_sum * D + Kt * pt_sum * D

    where:

        Kb      = biceps-specific gain [deg/s]
        Kt      = triceps-specific gain [deg/s]
        pb_sum  = number of biceps stimulation pulses
        pt_sum  = number of triceps stimulation pulses
        D       = vibration duration [s]

    The function also returns vividness-based weights:

        weight = vividness / 3

    These weights are used for the primary (weighted) analysis.

    Parameters
    ----------
    raw_df : pandas.DataFrame
        Raw trial-level data.
    Kb : float
        Biceps coefficient.
    Kt : float
        Triceps coefficient.

    Returns
    -------
    y_true : ndarray
        Observed perceived angle.
    y_pred : ndarray
        Model-predicted angle.
    weights : ndarray
        Normalized vividness weights.
    """

    # Mapping from pattern to number of biceps/triceps stimulation pulses
    pb_map = {
        p: _pattern_sums(p)[0]
        for p in raw_df["pattern_pair"].unique()
    }

    pt_map = {
        p: _pattern_sums(p)[1]
        for p in raw_df["pattern_pair"].unique()
    }

    # Observed angle
    y_true = raw_df["angle_deg"].values

    # Model prediction
    y_pred = np.array([
        Kb * pb_map[p] * d +
        Kt * pt_map[p] * d
        for p, d in zip(
            raw_df["pattern_pair"],
            raw_df["duration"]
        )
    ])

    # Normalized vividness
    weights = raw_df["vividness"].values / 3.0

    return y_true, y_pred, weights


# =============================================================================
# LINEAR REGRESSION FOR Kb / Kt
# =============================================================================

def compute_slope(
    x,
    y,
    vividness=None,
    weighted=True
):
    """
    Estimate a linear regression slope forced through the origin.

    Model
    -----
        y = K * x

    This corresponds to the muscle-specific model:

        Δθ = K * D

    for pure biceps or pure triceps stimulation.

    Parameters
    ----------
    x : array-like
        Vibration durations.
    y : array-like
        Observed perceived angular displacement.
    vividness : array-like or None
        Vividness ratings.
    weighted : bool
        If True:
            weight = vividness / 3
        If False:
            all trials have equal weight.

    Returns
    -------
    slope : float
        Estimated coefficient K.
    y_pred : ndarray
        Regression predictions.
    r2 : float
        R²_zero.
    """

    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)

    # -------------------------------------------------------------------------
    # Define regression weights
    # -------------------------------------------------------------------------

    if weighted:

        if vividness is None:
            raise ValueError(
                "vividness must be provided when weighted=True."
            )

        weights = np.asarray(vividness) / 3.0

    else:

        # Every observation receives equal weight
        weights = np.ones_like(y, dtype=float)

    # -------------------------------------------------------------------------
    # Fit regression forced through the origin
    # -------------------------------------------------------------------------

    model = LinearRegression(
        fit_intercept=False
    )

    model.fit(
        x,
        y,
        sample_weight=weights
    )

    # Estimated slope
    slope = model.coef_[0]

    # Predictions
    y_pred = model.predict(x)

    # -------------------------------------------------------------------------
    # R²_zero
    #
    # We use R²_zero because the regression is explicitly constrained
    # to pass through the origin.
    # -------------------------------------------------------------------------

    r2 = r2_metrics(
        y,
        y_pred,
        weights=weights
    )["R2_zero"]

    return slope, y_pred, r2


# =============================================================================
# SUBJECT-SPECIFIC PARAMETER ESTIMATION
# =============================================================================

def extract_subject_parameters(subj_df):
    """
    Estimate biceps and triceps coefficients for one subject.

    For each muscle, two estimates are obtained:

    1. Weighted regression
       -------------------
       weights = vividness / 3

       This is the PRIMARY analysis used in the original model.

    2. Unweighted regression
       ---------------------
       all trials receive equal weight

       This is the SENSITIVITY ANALYSIS requested by the reviewer.

    Pure stimulation patterns
    -------------------------
        001_000 -> biceps only
        000_001 -> triceps only

    Returns
    -------
    params : dict
        Dictionary containing weighted and unweighted coefficients
        and their corresponding R²_zero values.
    """

    params = {}

    # -------------------------------------------------------------------------
    # Loop over the two pure stimulation conditions
    # -------------------------------------------------------------------------

    for pattern, label in [
        ("001_000", "b"),
        ("000_001", "t")
    ]:

        pat_df = subj_df[
            subj_df["pattern_pair"] == pattern
        ]

        # ---------------------------------------------------------------------
        # If the subject has no observations for this pattern
        # ---------------------------------------------------------------------

        if pat_df.empty:

            params[f"K{label}_weighted"] = np.nan
            params[f"R2_K{label}_weighted"] = np.nan

            params[f"K{label}_unweighted"] = np.nan
            params[f"R2_K{label}_unweighted"] = np.nan

            continue

        # =====================================================================
        # WEIGHTED REGRESSION
        # =====================================================================

        slope_weighted, _, r2_weighted = compute_slope(
            x=pat_df["duration"].values,
            y=pat_df["angle_deg"].values,
            vividness=pat_df["vividness"].values,
            weighted=True
        )

        params[f"K{label}_weighted"] = slope_weighted
        params[f"R2_K{label}_weighted"] = r2_weighted

        # =====================================================================
        # UNWEIGHTED REGRESSION
        # =====================================================================

        slope_unweighted, _, r2_unweighted = compute_slope(
            x=pat_df["duration"].values,
            y=pat_df["angle_deg"].values,
            vividness=pat_df["vividness"].values,
            weighted=False
        )

        params[f"K{label}_unweighted"] = slope_unweighted
        params[f"R2_K{label}_unweighted"] = r2_unweighted

    return params


# =============================================================================
# FULL MODEL METRICS
# =============================================================================

def _compute_model_metrics(
    raw_df,
    Kb,
    Kt,
    weighted=True
):
    """
    Compute performance metrics for the complete model.

    Model
    -----
        Δθ = Kb * pb_sum * D + Kt * pt_sum * D

    Metrics
    -------
    R²_zero
    MAE
    RMSE

    Parameters
    ----------
    raw_df : pandas.DataFrame
        Raw trial-level data.
    Kb : float
        Biceps coefficient.
    Kt : float
        Triceps coefficient.
    weighted : bool
        If True:
            vividness / 3 is used as weight.
        If False:
            all trials are equally weighted.

    Notes
    -----
    The weighted analysis is the PRIMARY analysis.

    The unweighted analysis is included as a sensitivity analysis to determine
    whether the conclusions depend substantially on the use of vividness
    ratings as regression weights.

    Returns
    -------
    dict
        R²_zero, MAE and RMSE.
    """

    if raw_df.empty:
        return {
            "R2_zero": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan
        }

    # -------------------------------------------------------------------------
    # Obtain predictions
    # -------------------------------------------------------------------------

    y_true, y_pred, vividness_weights = _predict_raw(
        raw_df,
        Kb,
        Kt
    )

    # -------------------------------------------------------------------------
    # Choose weights
    # -------------------------------------------------------------------------

    if weighted:

        weights = vividness_weights

    else:

        weights = np.ones_like(
            y_true,
            dtype=float
        )

    # -------------------------------------------------------------------------
    # R²_zero
    # -------------------------------------------------------------------------

    r2 = r2_metrics(
        y_true,
        y_pred,
        weights=weights
    )["R2_zero"]

    # -------------------------------------------------------------------------
    # MAE
    # -------------------------------------------------------------------------

    mae = float(
        np.average(
            np.abs(y_true - y_pred),
            weights=weights
        )
    )

    # -------------------------------------------------------------------------
    # RMSE
    # -------------------------------------------------------------------------

    rmse = float(
        np.sqrt(
            np.average(
                (y_true - y_pred) ** 2,
                weights=weights
            )
        )
    )

    return {
        "R2_zero": r2,
        "MAE": mae,
        "RMSE": rmse
    }


# =============================================================================
# VALIDATION TABLE
# =============================================================================

def build_validation_table(
    subj_df,
    Kb,
    Kt,
    patterns,
    durations
):
    """
    Build a validation table comparing:

        ideal_angle
        real_angle
        absolute error

    The real angle is computed as a vividness-weighted mean.

    This table is used downstream for the sigmoid/tanh analysis.

    IMPORTANT
    ---------
    The validation table remains based on the WEIGHTED model because this
    corresponds to the primary analysis of the paper.
    """

    rows = []

    for pattern in patterns:

        pb, pt = _pattern_sums(pattern)

        for D in durations:

            # -----------------------------------------------------------------
            # Model prediction
            # -----------------------------------------------------------------

            ideal = (
                Kb * pb * D +
                Kt * pt * D
            )

            # -----------------------------------------------------------------
            # Select trials
            # -----------------------------------------------------------------

            subset = subj_df[
                (subj_df["pattern_pair"] == pattern) &
                (subj_df["duration"] == D)
            ]

            # -----------------------------------------------------------------
            # Weighted observed mean
            # -----------------------------------------------------------------

            if not subset.empty:

                real = np.average(
                    subset["angle_deg"],
                    weights=subset["vividness"] / 3.0
                )

            else:

                real = np.nan

            # -----------------------------------------------------------------
            # Store row
            # -----------------------------------------------------------------

            rows.append({

                "pattern": pattern,

                "duration": D,

                "ideal_angle": ideal,

                "real_angle": real,

                "abs_error": (
                    np.abs(ideal - real)
                    if not np.isnan(real)
                    else np.nan
                )
            })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def create_subject_and_global_excel(
    df,
    protocol,
    subject_output_folder
):
    """
    Estimate subject-specific and global model parameters.

    PRIMARY ANALYSIS
    ----------------
    Kb and Kt are estimated using vividness-weighted regression:

        weight = vividness / 3

    SENSITIVITY ANALYSIS
    --------------------
    Kb and Kt are also estimated using an unweighted regression, in which
    every trial receives equal weight.

    The purpose is to assess whether the main results depend substantially
    on the vividness weighting procedure.

    Outputs
    -------
    1. Subject-specific validation files.
    2. group_validation.xlsx containing:
       - Parameters
       - Validation
       - Weighting_Sensitivity
       - Kb_vs_Kt_Test

    Returns
    -------
    all_params
    validation_dfs
    Kb_global_weighted
    Kt_global_weighted
    group_validation_df
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    os.makedirs(
        subject_output_folder,
        exist_ok=True
    )

    subjects = df["subject"].unique()

    durations = sorted(
        protocol["blocks"]["durations"]
    )

    patterns = sorted(
        df["pattern_pair"].unique()
    )

    # Dictionary containing all subject-specific parameters
    all_params = {}

    # Dictionary containing validation tables
    validation_dfs = {}


    # =========================================================================
    # SUBJECT-SPECIFIC ANALYSIS
    # =========================================================================

    for subj in subjects:

        print(
            f"\n{'=' * 70}\n"
            f"SUBJECT: {subj}\n"
            f"{'=' * 70}"
        )

        # ---------------------------------------------------------------------
        # Select subject
        # ---------------------------------------------------------------------

        subj_df = df[
            df["subject"] == subj
        ].copy()

        # ---------------------------------------------------------------------
        # Estimate weighted and unweighted Kb/Kt
        # ---------------------------------------------------------------------

        params = extract_subject_parameters(
            subj_df
        )

        # =====================================================================
        # FULL MODEL — WEIGHTED
        # =====================================================================

        weighted_metrics = _compute_model_metrics(
            subj_df,
            params["Kb_weighted"],
            params["Kt_weighted"],
            weighted=True
        )

        params["R2_full_weighted"] = weighted_metrics["R2_zero"]
        params["MAE_full_weighted"] = weighted_metrics["MAE"]
        params["RMSE_full_weighted"] = weighted_metrics["RMSE"]

        # =====================================================================
        # FULL MODEL — UNWEIGHTED
        # =====================================================================

        unweighted_metrics = _compute_model_metrics(
            subj_df,
            params["Kb_unweighted"],
            params["Kt_unweighted"],
            weighted=False
        )

        params["R2_full_unweighted"] = unweighted_metrics["R2_zero"]
        params["MAE_full_unweighted"] = unweighted_metrics["MAE"]
        params["RMSE_full_unweighted"] = unweighted_metrics["RMSE"]

        # ---------------------------------------------------------------------
        # Save parameters
        # ---------------------------------------------------------------------

        all_params[subj] = params

        # =====================================================================
        # VALIDATION TABLE
        #
        # IMPORTANT:
        # We use the WEIGHTED parameters because they belong to the primary
        # analysis.
        # =====================================================================

        validation_dfs[subj] = build_validation_table(
            subj_df,
            params["Kb_weighted"],
            params["Kt_weighted"],
            patterns,
            durations
        )

        # ---------------------------------------------------------------------
        # Save subject validation table
        # ---------------------------------------------------------------------

        subj_file = os.path.join(
            subject_output_folder,
            f"{subj}_validation.xlsx"
        )

        validation_dfs[subj].to_excel(
            subj_file,
            index=False
        )


    # =========================================================================
    # GLOBAL ANALYSIS
    # =========================================================================

    df_all = df.copy()

    # =========================================================================
    # GLOBAL WEIGHTED Kb
    # =========================================================================

    kb_data = df_all[
        df_all["pattern_pair"] == "001_000"
    ]

    Kb_global_weighted, _, R2_Kb_global_weighted = compute_slope(
        x=kb_data["duration"].values,
        y=kb_data["angle_deg"].values,
        vividness=kb_data["vividness"].values,
        weighted=True
    )

    # =========================================================================
    # GLOBAL WEIGHTED Kt
    # =========================================================================

    kt_data = df_all[
        df_all["pattern_pair"] == "000_001"
    ]

    Kt_global_weighted, _, R2_Kt_global_weighted = compute_slope(
        x=kt_data["duration"].values,
        y=kt_data["angle_deg"].values,
        vividness=kt_data["vividness"].values,
        weighted=True
    )

    # =========================================================================
    # GLOBAL UNWEIGHTED Kb
    # =========================================================================

    Kb_global_unweighted, _, R2_Kb_global_unweighted = compute_slope(
        x=kb_data["duration"].values,
        y=kb_data["angle_deg"].values,
        vividness=kb_data["vividness"].values,
        weighted=False
    )

    # =========================================================================
    # GLOBAL UNWEIGHTED Kt
    # =========================================================================

    Kt_global_unweighted, _, R2_Kt_global_unweighted = compute_slope(
        x=kt_data["duration"].values,
        y=kt_data["angle_deg"].values,
        vividness=kt_data["vividness"].values,
        weighted=False
    )


    # =========================================================================
    # GLOBAL FULL MODEL — WEIGHTED
    # =========================================================================

    global_metrics_weighted = _compute_model_metrics(
        df_all,
        Kb_global_weighted,
        Kt_global_weighted,
        weighted=True
    )


    # =========================================================================
    # GLOBAL FULL MODEL — UNWEIGHTED
    # =========================================================================

    global_metrics_unweighted = _compute_model_metrics(
        df_all,
        Kb_global_unweighted,
        Kt_global_unweighted,
        weighted=False
    )


    # =========================================================================
    # PRINT GLOBAL RESULTS
    # =========================================================================

    print("\n")
    print("=" * 70)
    print("GLOBAL PARAMETERS")
    print("=" * 70)

    print("\nPRIMARY ANALYSIS — VIVIDNESS WEIGHTED")

    print(
        f"Kb = {Kb_global_weighted:.3f} deg/s "
        f"(R²={R2_Kb_global_weighted:.3f})"
    )

    print(
        f"Kt = {Kt_global_weighted:.3f} deg/s "
        f"(R²={R2_Kt_global_weighted:.3f})"
    )

    print(
        f"Full model: "
        f"R²_zero={global_metrics_weighted['R2_zero']:.3f}, "
        f"MAE={global_metrics_weighted['MAE']:.2f}°, "
        f"RMSE={global_metrics_weighted['RMSE']:.2f}°"
    )


    print("\nSENSITIVITY ANALYSIS — UNWEIGHTED")

    print(
        f"Kb = {Kb_global_unweighted:.3f} deg/s "
        f"(R²={R2_Kb_global_unweighted:.3f})"
    )

    print(
        f"Kt = {Kt_global_unweighted:.3f} deg/s "
        f"(R²={R2_Kt_global_unweighted:.3f})"
    )

    print(
        f"Full model: "
        f"R²_zero={global_metrics_unweighted['R2_zero']:.3f}, "
        f"MAE={global_metrics_unweighted['MAE']:.2f}°, "
        f"RMSE={global_metrics_unweighted['RMSE']:.2f}°"
    )


    # =========================================================================
    # GLOBAL VALIDATION TABLE
    #
    # This remains based on the weighted model because it feeds the main
    # sigmoid/tanh analysis.
    # =========================================================================

    group_rows = []

    for pattern in patterns:

        pb, pt = _pattern_sums(pattern)

        for D in durations:

            # -----------------------------------------------------------------
            # Weighted model prediction
            # -----------------------------------------------------------------

            ideal_weighted = (
                Kb_global_weighted * pb * D +
                Kt_global_weighted * pt * D
            )

            # -----------------------------------------------------------------
            # Unweighted model prediction
            # -----------------------------------------------------------------

            ideal_unweighted = (
                Kb_global_unweighted * pb * D +
                Kt_global_unweighted * pt * D
            )

            # -----------------------------------------------------------------
            # Calculate subject-level weighted means
            # -----------------------------------------------------------------

            real_values = []
            vividness_values = []

            trial_value = np.nan

            for subj in subjects:

                subset = df[
                    (df["subject"] == subj) &
                    (df["pattern_pair"] == pattern) &
                    (df["duration"] == D)
                ]

                if not subset.empty:

                    # Save trial identifier if available
                    if "trial" in subset.columns:
                        trial_value = subset["trial"].iloc[0]

                    # Weighted mean of perceived angle
                    real_values.append(
                        np.average(
                            subset["angle_deg"],
                            weights=subset["vividness"] / 3.0
                        )
                    )

                    # Mean vividness
                    vividness_values.append(
                        np.average(
                            subset["vividness"]
                        )
                    )

            # -----------------------------------------------------------------
            # Save row
            # -----------------------------------------------------------------

            group_rows.append({

                "trial": trial_value,

                "pattern": pattern,

                "duration": D,

                "ideal_angle_weighted": ideal_weighted,

                "ideal_angle_unweighted": ideal_unweighted,

                "real_mean": (
                    np.nanmean(real_values)
                    if real_values
                    else np.nan
                ),

                "vividness_mean": (
                    np.nanmean(vividness_values)
                    if vividness_values
                    else np.nan
                )
            })


    group_validation_df = pd.DataFrame(
        group_rows
    )


    # =========================================================================
    # SENSITIVITY ANALYSIS TABLE
    #
    # This table directly addresses the reviewer comment.
    # =========================================================================

    # -------------------------------------------------------------------------
    # Global coefficient comparison
    # -------------------------------------------------------------------------

    kb_abs_difference = (
        Kb_global_weighted -
        Kb_global_unweighted
    )

    kt_abs_difference = (
        Kt_global_weighted -
        Kt_global_unweighted
    )

    kb_relative_difference = (
        100 *
        kb_abs_difference /
        abs(Kb_global_unweighted)
        if Kb_global_unweighted != 0
        else np.nan
    )

    kt_relative_difference = (
        100 *
        kt_abs_difference /
        abs(Kt_global_unweighted)
        if Kt_global_unweighted != 0
        else np.nan
    )


    # -------------------------------------------------------------------------
    # Create sensitivity table
    # -------------------------------------------------------------------------

    sensitivity_df = pd.DataFrame({

        "Metric": [

            "Kb",
            "Kt",

            "R² Kb fit",
            "R² Kt fit",

            "Full model R²_zero",
            "Full model MAE [°]",
            "Full model RMSE [°]"
        ],

        "Weighted": [

            Kb_global_weighted,
            Kt_global_weighted,

            R2_Kb_global_weighted,
            R2_Kt_global_weighted,

            global_metrics_weighted["R2_zero"],
            global_metrics_weighted["MAE"],
            global_metrics_weighted["RMSE"]
        ],

        "Unweighted": [

            Kb_global_unweighted,
            Kt_global_unweighted,

            R2_Kb_global_unweighted,
            R2_Kt_global_unweighted,

            global_metrics_unweighted["R2_zero"],
            global_metrics_unweighted["MAE"],
            global_metrics_unweighted["RMSE"]
        ]
    })


    # -------------------------------------------------------------------------
    # Add absolute and relative differences
    # -------------------------------------------------------------------------

    sensitivity_df["Absolute difference"] = (
        sensitivity_df["Weighted"] -
        sensitivity_df["Unweighted"]
    )

    sensitivity_df["Relative difference [%]"] = (
        100 *
        sensitivity_df["Absolute difference"] /
        sensitivity_df["Unweighted"].abs()
    )


    # =========================================================================
    # SUBJECT-LEVEL SENSITIVITY TABLE
    #
    # Useful if you want to report how stable the coefficients are across
    # participants, rather than only comparing the global coefficients.
    # =========================================================================

    subject_sensitivity_rows = []

    for subj, p in all_params.items():

        # Kb comparison
        kb_w = p["Kb_weighted"]
        kb_uw = p["Kb_unweighted"]

        # Kt comparison
        kt_w = p["Kt_weighted"]
        kt_uw = p["Kt_unweighted"]

        # Relative differences
        kb_rel = (
            100 * (kb_w - kb_uw) / abs(kb_uw)
            if not np.isnan(kb_uw) and kb_uw != 0
            else np.nan
        )

        kt_rel = (
            100 * (kt_w - kt_uw) / abs(kt_uw)
            if not np.isnan(kt_uw) and kt_uw != 0
            else np.nan
        )

        subject_sensitivity_rows.append({

            "Subject": subj,

            "Kb weighted": kb_w,
            "Kb unweighted": kb_uw,
            "Kb difference": kb_w - kb_uw,
            "Kb relative difference [%]": kb_rel,

            "Kt weighted": kt_w,
            "Kt unweighted": kt_uw,
            "Kt difference": kt_w - kt_uw,
            "Kt relative difference [%]": kt_rel,

            "R² full weighted":
                p["R2_full_weighted"],

            "R² full unweighted":
                p["R2_full_unweighted"],

            "MAE weighted [°]":
                p["MAE_full_weighted"],

            "MAE unweighted [°]":
                p["MAE_full_unweighted"],

            "RMSE weighted [°]":
                p["RMSE_full_weighted"],

            "RMSE unweighted [°]":
                p["RMSE_full_unweighted"]
        })


    subject_sensitivity_df = pd.DataFrame(
        subject_sensitivity_rows
    )


    # =========================================================================
    # WILCOXON TEST: Kb vs |Kt|
    #
    # This remains based on the PRIMARY weighted coefficients.
    # =========================================================================

    kb_samples = []
    kt_abs_samples = []

    for subj, p in all_params.items():

        kb = p["Kb_weighted"]
        kt = p["Kt_weighted"]

        if (
            not np.isnan(kb) and
            not np.isnan(kt)
        ):

            kb_samples.append(kb)
            kt_abs_samples.append(abs(kt))


    # -------------------------------------------------------------------------
    # Run Wilcoxon test
    # -------------------------------------------------------------------------

    stat, p_value = stats.wilcoxon(
        kb_samples,
        kt_abs_samples
    )


    # -------------------------------------------------------------------------
    # Prepare statistics table
    # -------------------------------------------------------------------------

    stats_results = pd.DataFrame({

        "Metric": [

            "Mean Kb",
            "Std Kb",

            "Mean |Kt|",
            "Std |Kt|",

            "Difference (Kb - |Kt|)",

            "Wilcoxon Stat",
            "p-value"
        ],

        "Value": [

            np.mean(kb_samples),
            np.std(kb_samples, ddof=1),

            np.mean(kt_abs_samples),
            np.std(kt_abs_samples, ddof=1),

            (
                np.mean(kb_samples) -
                np.mean(kt_abs_samples)
            ),

            stat,
            p_value
        ]
    })


    # -------------------------------------------------------------------------
    # Add interpretation
    # -------------------------------------------------------------------------

    stats_results["Significance"] = ""

    stats_results.loc[
        stats_results["Metric"] == "p-value",
        "Significance"
    ] = (
        "p < 0.05 (*)"
        if p_value < 0.05
        else "n.s."
    )


    # =========================================================================
    # SAVE EVERYTHING TO EXCEL
    # =========================================================================

    group_file = os.path.join(
        subject_output_folder,
        "group_validation.xlsx"
    )


    with pd.ExcelWriter(
        group_file,
        engine="openpyxl"
    ) as writer:

        # =====================================================================
        # SHEET 1 — PARAMETERS
        # =====================================================================

        params_rows = []

        # ---------------------------------------------------------------------
        # Subject-level parameters
        # ---------------------------------------------------------------------

        for subj, p in all_params.items():

            params_rows.append({

                "Subject": subj,

                # Biceps
                "Kb weighted": p["Kb_weighted"],
                "R² Kb weighted": p["R2_Kb_weighted"],

                "Kb unweighted": p["Kb_unweighted"],
                "R² Kb unweighted": p["R2_Kb_unweighted"],

                # Triceps
                "Kt weighted": p["Kt_weighted"],
                "R² Kt weighted": p["R2_Kt_weighted"],

                "Kt unweighted": p["Kt_unweighted"],
                "R² Kt unweighted": p["R2_Kt_unweighted"],

                # Full model
                "R² full weighted": p["R2_full_weighted"],
                "MAE full weighted [°]": p["MAE_full_weighted"],
                "RMSE full weighted [°]": p["RMSE_full_weighted"],

                "R² full unweighted": p["R2_full_unweighted"],
                "MAE full unweighted [°]": p["MAE_full_unweighted"],
                "RMSE full unweighted [°]": p["RMSE_full_unweighted"]
            })


        # ---------------------------------------------------------------------
        # GLOBAL row
        # ---------------------------------------------------------------------

        params_rows.append({

            "Subject": "GLOBAL",

            "Kb weighted": Kb_global_weighted,
            "R² Kb weighted": R2_Kb_global_weighted,

            "Kb unweighted": Kb_global_unweighted,
            "R² Kb unweighted": R2_Kb_global_unweighted,

            "Kt weighted": Kt_global_weighted,
            "R² Kt weighted": R2_Kt_global_weighted,

            "Kt unweighted": Kt_global_unweighted,
            "R² Kt unweighted": R2_Kt_global_unweighted,

            "R² full weighted":
                global_metrics_weighted["R2_zero"],

            "MAE full weighted [°]":
                global_metrics_weighted["MAE"],

            "RMSE full weighted [°]":
                global_metrics_weighted["RMSE"],

            "R² full unweighted":
                global_metrics_unweighted["R2_zero"],

            "MAE full unweighted [°]":
                global_metrics_unweighted["MAE"],

            "RMSE full unweighted [°]":
                global_metrics_unweighted["RMSE"]
        })


        params_df = pd.DataFrame(
            params_rows
        )


        params_df.round(3).to_excel(
            writer,
            sheet_name="Parameters",
            index=False
        )


        # =====================================================================
        # SHEET 2 — VALIDATION
        #
        # Primary weighted model.
        # =====================================================================

        group_validation_df.round(3).to_excel(
            writer,
            sheet_name="Validation",
            index=False
        )


        # =====================================================================
        # SHEET 3 — WEIGHTING SENSITIVITY
        #
        # Main table to report in response to reviewer.
        # =====================================================================

        sensitivity_df.round(4).to_excel(
            writer,
            sheet_name="Weighting_Sensitivity",
            index=False
        )


        # =====================================================================
        # SHEET 4 — SUBJECT-LEVEL SENSITIVITY
        # =====================================================================

        subject_sensitivity_df.round(4).to_excel(
            writer,
            sheet_name="Subject_Sensitivity",
            index=False
        )


        # =====================================================================
        # SHEET 5 — Kb VS |Kt| TEST
        # =====================================================================

        stats_results.round(4).to_excel(
            writer,
            sheet_name="Kb_vs_Kt_Test",
            index=False
        )


    # =========================================================================
    # ADDITIONAL PRINTS
    # =========================================================================

    print("\n")
    print("=" * 70)
    print("SUBJECT-LEVEL COEFFICIENT SUMMARY")
    print("=" * 70)

    kb_values = [
        p["Kb_weighted"]
        for p in all_params.values()
        if not np.isnan(p["Kb_weighted"])
    ]

    kt_values = [
        p["Kt_weighted"]
        for p in all_params.values()
        if not np.isnan(p["Kt_weighted"])
    ]

    kb_mean = np.mean(kb_values)
    kb_std = np.std(kb_values, ddof=1)

    kt_mean = np.mean(kt_values)
    kt_std = np.std(kt_values, ddof=1)

    print(
        f"Kb weighted = "
        f"{kb_mean:.3f} ± {kb_std:.3f}"
    )

    print(
        f"Kt weighted = "
        f"{kt_mean:.3f} ± {kt_std:.3f}"
    )


    # =========================================================================
    # SENSITIVITY SUMMARY
    # =========================================================================

    print("\n")
    print("=" * 70)
    print("WEIGHTING SENSITIVITY ANALYSIS")
    print("=" * 70)

    print(
        f"Kb weighted   = {Kb_global_weighted:.4f}"
    )

    print(
        f"Kb unweighted = {Kb_global_unweighted:.4f}"
    )

    print(
        f"Kb difference = {kb_abs_difference:.4f} "
        f"({kb_relative_difference:.2f}%)"
    )

    print()

    print(
        f"Kt weighted   = {Kt_global_weighted:.4f}"
    )

    print(
        f"Kt unweighted = {Kt_global_unweighted:.4f}"
    )

    print(
        f"Kt difference = {kt_abs_difference:.4f} "
        f"({kt_relative_difference:.2f}%)"
    )

    print("\nFull model:")

    print(
        f"Weighted   → "
        f"R²={global_metrics_weighted['R2_zero']:.4f}, "
        f"MAE={global_metrics_weighted['MAE']:.2f}°, "
        f"RMSE={global_metrics_weighted['RMSE']:.2f}°"
    )

    print(
        f"Unweighted → "
        f"R²={global_metrics_unweighted['R2_zero']:.4f}, "
        f"MAE={global_metrics_unweighted['MAE']:.2f}°, "
        f"RMSE={global_metrics_unweighted['RMSE']:.2f}°"
    )


    # =========================================================================
    # WILCOXON TEST PRINT
    # =========================================================================

    print("\n")
    print("=" * 70)
    print("STATISTICAL TEST — Kb VS |Kt|")
    print("=" * 70)

    print(
        f"Mean Kb = {np.mean(kb_samples):.4f}"
    )

    print(
        f"Mean |Kt| = {np.mean(kt_abs_samples):.4f}"
    )

    print(
        f"Wilcoxon statistic = {stat:.4f}"
    )

    print(
        f"p-value = {p_value:.4f}"
    )

    if p_value < 0.05:

        print(
            "→ Significant difference between "
            "biceps and triceps sensitivity."
        )

    else:

        print(
            "→ No significant difference between "
            "biceps and triceps sensitivity."
        )


    # =========================================================================
    # FINAL RETURN
    # =========================================================================

    return (
        all_params,
        validation_dfs,
        Kb_global_weighted,
        Kt_global_weighted,
        group_validation_df
    )