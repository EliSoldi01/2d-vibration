import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from core.utils.r2_metrics import compute_r2_metrics as r2_metrics


# ============================================================
# CONSTANTS
# ============================================================

PURE_PATTERNS = ["001_000", "000_001"]


# ============================================================
# HELPERS
# ============================================================

def _pattern_sums(pattern):
    """
    Return (pb_sum, pt_sum) for a pattern string like '011_100'.

    Example:
        '011_100' -> pb_sum = 2, pt_sum = 1
    """
    b, t = pattern.split("_")
    return sum(int(x) for x in b), sum(int(x) for x in t)


def _is_combined_pattern(pattern):
    """
    Return True if pattern is a combined stimulation pattern.

    Pure patterns:
        001_000 -> biceps only
        000_001 -> triceps only

    All other patterns are considered combined.
    """
    return pattern not in PURE_PATTERNS


def _predict_raw(raw_df, Kb, Kt):
    """
    For every trial in raw_df, compute the model prediction.

    Model:
        ΔA = Kb · pb_sum · D + Kt · pt_sum · D

    Weights:
        vividness / 3.0

    Returns:
        y_true, y_pred, weights
    """

    if raw_df.empty:
        return (
            np.array([]),
            np.array([]),
            np.array([])
        )

    y_true = raw_df["angle_deg"].values

    y_pred = np.array([
        Kb * _pattern_sums(pattern)[0] * duration
        + Kt * _pattern_sums(pattern)[1] * duration
        for pattern, duration
        in zip(raw_df["pattern_pair"], raw_df["duration"])
    ])

    weights = raw_df["vividness"].values / 3.0

    return y_true, y_pred, weights


# ============================================================
# WEIGHTED REGRESSION FOR Kb / Kt
# ============================================================

def compute_weighted_slope(x, y, vividness):
    """
    Linear regression forced through origin with weights based on vividness.

    Returns:
        slope
        y_pred
        R²_zero

    R²_zero is used because the model is forced through the origin.
    """

    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)

    weights = (
        np.asarray(vividness) / 3.0
        if vividness is not None
        else np.ones_like(y)
    )

    model = LinearRegression(fit_intercept=False)

    model.fit(
        x,
        y,
        sample_weight=weights
    )

    slope = model.coef_[0]

    y_pred = model.predict(x)

    r2 = r2_metrics(
        y,
        y_pred,
        weights=weights
    )["R2_zero"]

    return slope, y_pred, r2


# ============================================================
# SUBJECT PARAMETERS
# ============================================================

def extract_subject_parameters(subj_df):
    """
    Compute Kb, Kt and their R²_zero for one subject.

    IMPORTANT:
    Kb and Kt are estimated ONLY from the pure patterns:

        001_000 -> Kb
        000_001 -> Kt

    Combined patterns are NOT used for parameter estimation.
    """

    params = {}

    for pattern, label in [
        ("001_000", "b"),
        ("000_001", "t")
    ]:

        pat_df = subj_df[
            subj_df["pattern_pair"] == pattern
        ]

        if pat_df.empty:

            params[f"K{label}"] = np.nan
            params[f"R2_K{label}"] = np.nan

            continue

        slope, _, r2 = compute_weighted_slope(
            x=pat_df["duration"].values,
            y=pat_df["angle_deg"].values,
            vividness=pat_df["vividness"].values
        )

        params[f"K{label}"] = slope
        params[f"R2_K{label}"] = r2

    return params


# ============================================================
# MODEL METRICS — RAW DATA
# ============================================================

def _compute_model_metrics(raw_df, Kb, Kt):
    """
    Compute model performance on raw trial data.

    Metrics:
        R²_zero
        MAE
        RMSE

    All metrics are weighted by vividness.

    IMPORTANT:
    This function does NOT estimate Kb/Kt.
    It only evaluates an already-defined model.

    Therefore it can be safely used for:
        - ALL patterns
        - COMBINED patterns only
    """

    if raw_df.empty or np.isnan(Kb) or np.isnan(Kt):

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

    r2 = r2_metrics(
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
        "R2_zero": r2,
        "MAE": mae,
        "RMSE": rmse
    }


# ============================================================
# VALIDATION TABLE
# ============================================================

def build_validation_table(
    subj_df,
    Kb,
    Kt,
    patterns,
    durations
):
    """
    Build table comparing ideal vs weighted-mean real angles.

    IMPORTANT:
    This table is used downstream for the sigmoid fit.

    It is NOT used for model performance metrics.

    Performance metrics are calculated from raw trial data
    using _compute_model_metrics().
    """

    rows = []

    for pattern in patterns:

        pb, pt = _pattern_sums(pattern)

        for D in durations:

            ideal = (
                Kb * pb * D
                + Kt * pt * D
            )

            subset = subj_df[
                (subj_df["pattern_pair"] == pattern)
                &
                (subj_df["duration"] == D)
            ]

            if not subset.empty:

                real = np.average(
                    subset["angle_deg"],
                    weights=subset["vividness"] / 3
                )

            else:

                real = np.nan

            rows.append({

                "pattern": pattern,
                "duration": D,
                "ideal_angle": ideal,
                "real_angle": real,
                "abs_error":
                    np.abs(ideal - real)
                    if not np.isnan(real)
                    else np.nan
            })

    return pd.DataFrame(rows)


# ============================================================
# MAIN FUNCTION
# ============================================================

def create_subject_and_global_excel(
    df,
    protocol,
    subject_output_folder
):
    """
    Compute Kb/Kt per subject and globally.

    Kb/Kt are ALWAYS estimated exclusively from pure patterns:

        001_000
        000_001

    Model performance is then evaluated separately on:

        1. ALL patterns
        2. COMBINED patterns only

    Metrics:
        R²_zero
        MAE
        RMSE

    Individual subjects:
        Kb/Kt are estimated from the subject's pure patterns.

    GLOBAL:
        Kb/Kt are estimated from all subjects' pure patterns pooled.

    The Validation sheet is retained for the sigmoid fit.
    """

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


    # ========================================================
    # PER-SUBJECT
    # ========================================================

    all_params = {}
    validation_dfs = {}

    for subj in subjects:

        print(
            f"\n[SUBJECT] Processing {subj}"
        )

        subj_df = df[
            df["subject"] == subj
        ].copy()


        # ----------------------------------------------------
        # STEP 1
        # Estimate Kb / Kt from PURE patterns only
        # ----------------------------------------------------

        params = extract_subject_parameters(
            subj_df
        )


        # ----------------------------------------------------
        # STEP 2
        # Evaluate model on ALL patterns
        # ----------------------------------------------------

        metrics_all = _compute_model_metrics(
            subj_df,
            params["Kb"],
            params["Kt"]
        )

        params["R2_full_all"] = (
            metrics_all["R2_zero"]
        )

        params["MAE_full_all"] = (
            metrics_all["MAE"]
        )

        params["RMSE_full_all"] = (
            metrics_all["RMSE"]
        )


        # ----------------------------------------------------
        # STEP 3
        # Evaluate model on COMBINED patterns only
        # ----------------------------------------------------

        combined_df = subj_df[
            ~subj_df["pattern_pair"].isin(
                PURE_PATTERNS
            )
        ].copy()

        metrics_combined = _compute_model_metrics(
            combined_df,
            params["Kb"],
            params["Kt"]
        )

        params["R2_combined"] = (
            metrics_combined["R2_zero"]
        )

        params["MAE_combined"] = (
            metrics_combined["MAE"]
        )

        params["RMSE_combined"] = (
            metrics_combined["RMSE"]
        )


        # ----------------------------------------------------
        # Store parameters
        # ----------------------------------------------------

        all_params[subj] = params


        # ----------------------------------------------------
        # Validation table
        # ----------------------------------------------------

        validation_dfs[subj] = build_validation_table(
            subj_df,
            params["Kb"],
            params["Kt"],
            patterns,
            durations
        )


        # ----------------------------------------------------
        # Save subject validation file
        # ----------------------------------------------------

        subj_file = os.path.join(
            subject_output_folder,
            f"{subj}_validation.xlsx"
        )

        validation_dfs[subj].to_excel(
            subj_file,
            index=False
        )


        print(
            f"[INFO] {subj}: "
            f"Kb={params['Kb']:.3f}, "
            f"Kt={params['Kt']:.3f}"
        )

        print(
            f"[INFO] {subj} ALL — "
            f"R²={params['R2_full_all']:.3f}, "
            f"MAE={params['MAE_full_all']:.2f}°, "
            f"RMSE={params['RMSE_full_all']:.2f}°"
        )

        print(
            f"[INFO] {subj} COMBINED — "
            f"R²={params['R2_combined']:.3f}, "
            f"MAE={params['MAE_combined']:.2f}°, "
            f"RMSE={params['RMSE_combined']:.2f}°"
        )


    # ========================================================
    # GLOBAL Kb / Kt
    # ========================================================

    df_all = df.copy()


    # --------------------------------------------------------
    # GLOBAL Kb
    # --------------------------------------------------------

    biceps_df = df_all[
        df_all["pattern_pair"] == "001_000"
    ]

    Kb_global, _, R2_Kb_global = (
        compute_weighted_slope(
            x=biceps_df["duration"].values,
            y=biceps_df["angle_deg"].values,
            vividness=biceps_df["vividness"].values
        )
    )


    # --------------------------------------------------------
    # GLOBAL Kt
    # --------------------------------------------------------

    triceps_df = df_all[
        df_all["pattern_pair"] == "000_001"
    ]

    Kt_global, _, R2_Kt_global = (
        compute_weighted_slope(
            x=triceps_df["duration"].values,
            y=triceps_df["angle_deg"].values,
            vividness=triceps_df["vividness"].values
        )
    )


    # ========================================================
    # GLOBAL MODEL — ALL
    # ========================================================

    global_metrics_all = _compute_model_metrics(
        df_all,
        Kb_global,
        Kt_global
    )


    # ========================================================
    # GLOBAL MODEL — COMBINED ONLY
    # ========================================================

    df_combined = df_all[
        ~df_all["pattern_pair"].isin(
            PURE_PATTERNS
        )
    ].copy()

    global_metrics_combined = _compute_model_metrics(
        df_combined,
        Kb_global,
        Kt_global
    )


    # ========================================================
    # GLOBAL VALIDATION TABLE
    # ========================================================

    group_rows = []

    for pattern in patterns:

        pb, pt = _pattern_sums(pattern)

        for D in durations:

            ideal = (
                Kb_global * pb * D
                + Kt_global * pt * D
            )

            real_values = []
            vividness_values = []

            trial_values = []

            for subj in subjects:

                subset = df[
                    (df["subject"] == subj)
                    &
                    (df["pattern_pair"] == pattern)
                    &
                    (df["duration"] == D)
                ]

                if not subset.empty:

                    real_values.append(
                        np.average(
                            subset["angle_deg"],
                            weights=subset["vividness"] / 3
                        )
                    )

                    vividness_values.append(
                        np.average(
                            subset["vividness"]
                        )
                    )

                    trial_values.append(
                        subset["trial"].iloc[0]
                    )


            group_rows.append({

                "trial":
                    trial_values[0]
                    if trial_values
                    else np.nan,

                "pattern": pattern,

                "duration": D,

                "ideal_angle": ideal,

                "real_mean":
                    np.nanmean(real_values)
                    if real_values
                    else np.nan,

                "real_std":
                    np.std(real_values, ddof=1)
                    if len(real_values) > 1
                    else np.nan,

                "N":
                    len(real_values),

                "vividness_mean":
                    np.nanmean(vividness_values)
                    if vividness_values
                    else np.nan
            })


    group_validation_df = pd.DataFrame(
        group_rows
    )


    # ========================================================
    # PRINT GLOBAL RESULTS
    # ========================================================

    print(
        f"\n[INFO] Global Kb="
        f"{Kb_global:.3f} "
        f"(R²={R2_Kb_global:.3f})"
    )

    print(
        f"[INFO] Global Kt="
        f"{Kt_global:.3f} "
        f"(R²={R2_Kt_global:.3f})"
    )

    print(
        f"\n[INFO] Global full model — ALL"
    )

    print(
        f"       R²_zero="
        f"{global_metrics_all['R2_zero']:.3f}"
    )

    print(
        f"       MAE="
        f"{global_metrics_all['MAE']:.2f}°"
    )

    print(
        f"       RMSE="
        f"{global_metrics_all['RMSE']:.2f}°"
    )


    print(
        f"\n[INFO] Global full model — COMBINED ONLY"
    )

    print(
        f"       R²_zero="
        f"{global_metrics_combined['R2_zero']:.3f}"
    )

    print(
        f"       MAE="
        f"{global_metrics_combined['MAE']:.2f}°"
    )

    print(
        f"       RMSE="
        f"{global_metrics_combined['RMSE']:.2f}°"
    )


    # ========================================================
    # EXCEL
    # ========================================================

    group_file = os.path.join(
        subject_output_folder,
        "group_validation.xlsx"
    )


    with pd.ExcelWriter(
        group_file,
        engine="openpyxl"
    ) as writer:


        # ====================================================
        # SHEET 1 — PARAMETERS
        # ====================================================

        params_df = pd.DataFrame(
            all_params
        ).T


        params_df = params_df[
            [
                "Kb",
                "R2_Kb",
                "Kt",
                "R2_Kt",

                "R2_full_all",
                "MAE_full_all",
                "RMSE_full_all",

                "R2_combined",
                "MAE_combined",
                "RMSE_combined"
            ]
        ]


        params_df.columns = [

            "Kb",

            "R²_zero (Kb fit)",

            "Kt",

            "R²_zero (Kt fit)",

            "R²_zero (full model - ALL)",

            "MAE (full model - ALL) [°]",

            "RMSE (full model - ALL) [°]",

            "R²_zero (full model - COMBINED)",

            "MAE (full model - COMBINED) [°]",

            "RMSE (full model - COMBINED) [°]"
        ]


        # ====================================================
        # GLOBAL ROW
        # ====================================================

        params_df.loc["GLOBAL"] = {

            "Kb":
                Kb_global,

            "R²_zero (Kb fit)":
                R2_Kb_global,

            "Kt":
                Kt_global,

            "R²_zero (Kt fit)":
                R2_Kt_global,

            "R²_zero (full model - ALL)":
                global_metrics_all["R2_zero"],

            "MAE (full model - ALL) [°]":
                global_metrics_all["MAE"],

            "RMSE (full model - ALL) [°]":
                global_metrics_all["RMSE"],

            "R²_zero (full model - COMBINED)":
                global_metrics_combined["R2_zero"],

            "MAE (full model - COMBINED) [°]":
                global_metrics_combined["MAE"],

            "RMSE (full model - COMBINED) [°]":
                global_metrics_combined["RMSE"]
        }


        params_df.round(3).to_excel(
            writer,
            sheet_name="Parameters",
            index=True
        )


        # ====================================================
        # SUBJECT COEFFICIENT SUMMARY
        # ====================================================

        kb_values = [
            p["Kb"]
            for p in all_params.values()
            if not np.isnan(p["Kb"])
        ]

        kt_values = [
            p["Kt"]
            for p in all_params.values()
            if not np.isnan(p["Kt"])
        ]


        kb_mean = np.mean(kb_values)
        kb_std = np.std(
            kb_values,
            ddof=1
        )

        kt_mean = np.mean(kt_values)
        kt_std = np.std(
            kt_values,
            ddof=1
        )


        print(
            f"\nKb = "
            f"{kb_mean:.3f} ± "
            f"{kb_std:.3f}"
        )

        print(
            f"Kt = "
            f"{kt_mean:.3f} ± "
            f"{kt_std:.3f}"
        )


        # ====================================================
        # SHEET 2 — GLOBAL VALIDATION
        # ====================================================

        group_validation_df.round(3).to_excel(
            writer,
            sheet_name="Validation",
            index=False
        )


        # ====================================================
        # SHEET 3 — MODEL PERFORMANCE SUMMARY
        # ====================================================

        performance_df = pd.DataFrame({

            "Dataset": [
                "ALL patterns",
                "COMBINED patterns only"
            ],

            "N_trials": [
                len(df_all),
                len(df_combined)
            ],

            "R²_zero": [
                global_metrics_all["R2_zero"],
                global_metrics_combined["R2_zero"]
            ],

            "MAE [°]": [
                global_metrics_all["MAE"],
                global_metrics_combined["MAE"]
            ],

            "RMSE [°]": [
                global_metrics_all["RMSE"],
                global_metrics_combined["RMSE"]
            ]
        })


        performance_df.round(3).to_excel(
            writer,
            sheet_name="Model_Performance",
            index=False
        )


    # ========================================================
    # STATISTICAL TEST: Kb VS |Kt|
    # ========================================================

    import scipy.stats as stats


    kb_samples = []
    kt_abs_samples = []


    for subj, p in all_params.items():

        if (
            not np.isnan(p["Kb"])
            and
            not np.isnan(p["Kt"])
        ):

            kb_samples.append(
                p["Kb"]
            )

            kt_abs_samples.append(
                abs(p["Kt"])
            )


    stat, p_value = stats.wilcoxon(
        kb_samples,
        kt_abs_samples
    )


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

            np.std(
                kb_samples
            ),

            np.mean(
                kt_abs_samples
            ),

            np.std(
                kt_abs_samples
            ),

            (
                np.mean(kb_samples)
                -
                np.mean(kt_abs_samples)
            ),

            stat,

            p_value
        ]
    })


    stats_results["Significance"] = ""


    # Put significance only on p-value row
    stats_results.loc[
        stats_results["Metric"] == "p-value",
        "Significance"
    ] = (
        "p < 0.05 (*)"
        if p_value < 0.05
        else "n.s."
    )


    # ========================================================
    # SAVE STATISTICAL TEST
    # ========================================================

    with pd.ExcelWriter(
        group_file,
        engine="openpyxl",
        mode="a"
    ) as writer:

        stats_results.to_excel(
            writer,
            sheet_name="Kb_vs_Kt_Test",
            index=False
        )


    print(
        f"\n[STAT TEST] "
        f"Confronto Kb vs |Kt|: "
        f"p-value = {p_value:.4f}"
    )


    if p_value < 0.05:

        print(
            "-> Esiste una differenza significativa "
            "tra l'efficacia di Bicipite e Tricipite."
        )

    else:

        print(
            "-> Non sono state trovate differenze "
            "significative tra i due muscoli."
        )


    # ========================================================
    # RETURN
    # ========================================================

    return (
        all_params,
        validation_dfs,
        Kb_global,
        Kt_global,
        group_validation_df
    )