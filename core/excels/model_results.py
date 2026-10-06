import numpy as np
import pandas as pd

from core.analysis.extract_model_parameters import predict_angle, get_weights

# ============================================================
# PARAMETERS TABLE
# ============================================================

def build_parameters_table(
    subject_parameters,
    global_parameters,
    global_metrics_all,
    global_metrics_complex
):
    """
    Build the Parameters table for Excel output.

    Parameters
    ----------
    subject_parameters : dict
        Subject-level model parameters and performance metrics.

    global_parameters : dict
        Global Kb, Kt and corresponding R² values.

    global_metrics_all : dict
        Global model performance across all patterns.

    global_metrics_complex : dict
        Global model performance across complex patterns only.

    Returns
    -------
    pandas.DataFrame
        Parameters table.
    """

    rows = []

    # --------------------------------------------------------
    # SUBJECTS
    # --------------------------------------------------------

    for subject, params in subject_parameters.items():

        rows.append({
            "subject": subject,

            "Kb": params["Kb"],
            "R²_zero (Kb fit)": params["R2_Kb"],

            "Kt": params["Kt"],
            "R²_zero (Kt fit)": params["R2_Kt"],

            "R²_zero (full model - ALL)":
                params["R2_full_all"],

            "MAE (full model - ALL) [°]":
                params["MAE_full_all"],

            "RMSE (full model - ALL) [°]":
                params["RMSE_full_all"],

            "R²_zero (full model - COMPLEX)":
                params["R2_complex"],

            "MAE (full model - COMPLEX) [°]":
                params["MAE_complex"],

            "RMSE (full model - COMPLEX) [°]":
                params["RMSE_complex"]
        })

    # --------------------------------------------------------
    # GLOBAL
    # --------------------------------------------------------

    rows.append({
        "subject": "GLOBAL",

        "Kb": global_parameters["Kb"],
        "R²_zero (Kb fit)": global_parameters["R2_Kb"],

        "Kt": global_parameters["Kt"],
        "R²_zero (Kt fit)": global_parameters["R2_Kt"],

        "R²_zero (full model - ALL)":
            global_metrics_all["R2_zero"],

        "MAE (full model - ALL) [°]":
            global_metrics_all["MAE"],

        "RMSE (full model - ALL) [°]":
            global_metrics_all["RMSE"],

        "R²_zero (full model - COMPLEX)":
            global_metrics_complex["R2_zero"],

        "MAE (full model - COMPLEX) [°]":
            global_metrics_complex["MAE"],

        "RMSE (full model - COMPLEX) [°]":
            global_metrics_complex["RMSE"]
    })

    df = pd.DataFrame(rows).set_index("subject")

    return df
# ============================================================
# VALIDATION TABLE
# ============================================================

def build_validation_table(subj_df, Kb, Kt, patterns, durations, max_vividness=None):
    """
    Build the validation table used by downstream analyses.

    ```
    For each pattern × duration combination, calculate:

        ideal_angle
        real_angle
        abs_error

    The real angle uses the weighting scheme defined by cfg.USE_VIVIDNESS_WEIGHTS.

    Important
    ---------
    This table is separate from the raw trial-level model
    performance metrics.
    """

    rows = []

    for pattern in patterns:

        for duration in durations:

            ideal = predict_angle(
                pattern=pattern,
                duration=duration,
                Kb=Kb,
                Kt=Kt
            )

            subset = subj_df[
                (subj_df["pattern_pair"] == pattern)
                &
                (subj_df["duration"] == duration)
            ]

            if not subset.empty:

                real = np.average(
                    subset["angle_deg"],
                    weights=get_weights(subset["vividness"].values, max_vividness=max_vividness)
                )

            else:

                real = np.nan

            rows.append({
                "pattern": pattern,
                "duration": duration,
                "ideal_angle": ideal,
                "real_angle": real,
                "abs_error": (
                    np.abs(ideal - real)
                    if not np.isnan(real)
                    else np.nan
                )
            })

    return pd.DataFrame(rows)

# ============================================================
# GROUP VALIDATION TABLE
# ============================================================

def build_group_validation_table(
    df,
    global_parameters,
    patterns,
    durations,
    max_vividness=None
):
    """
    Build the group-level validation table.

    For each pattern × duration combination, compute:

        ideal_angle
        real_mean
        real_std
        N
        vividness_mean

    The real angle is first averaged within each subject using
    the weighting scheme defined by the analysis configuration.
    Group statistics are then computed across subjects.

    Parameters
    ----------
    df : pandas.DataFrame
        Trial-level processed data.

    global_parameters : dict
        Global Kb and Kt values.

    patterns : list
        Pattern identifiers to include.

    durations : list
        Stimulation durations to include.

    max_vividness : float, optional
        Maximum vividness value to use for normalization.

    Returns
    -------
    pandas.DataFrame
        Group validation table.
    """

    rows = []

    Kb = global_parameters["Kb"]
    Kt = global_parameters["Kt"]

    subjects = sorted(
        df["subject"].unique()
    )

    for pattern in patterns:

        for duration in durations:

            # ------------------------------------------------
            # IDEAL MODEL ANGLE
            # ------------------------------------------------

            ideal = predict_angle(
                pattern=pattern,
                duration=duration,
                Kb=Kb,
                Kt=Kt
            )

            real_values = []
            vividness_values = []

            # ------------------------------------------------
            # SUBJECT-LEVEL VALUES
            # ------------------------------------------------

            for subject in subjects:

                subset = df[
                    (df["subject"] == subject)
                    &
                    (df["pattern_pair"] == pattern)
                    &
                    (df["duration"] == duration)
                ]

                if subset.empty:
                    continue

                real = np.average(
                    subset["angle_deg"],
                    weights=get_weights(
                        subset["vividness"].values, max_vividness=max_vividness
                    )
                )

                vividness = np.mean(
                    subset["vividness"]
                )

                real_values.append(real)
                vividness_values.append(vividness)

            # ------------------------------------------------
            # GROUP STATISTICS
            # ------------------------------------------------

            rows.append({
                "pattern": pattern,
                "duration": duration,
                "ideal_angle": ideal,

                "real_mean": (
                    np.mean(real_values)
                    if real_values
                    else np.nan
                ),

                "real_std": (
                    np.std(
                        real_values,
                        ddof=1
                    )
                    if len(real_values) > 1
                    else np.nan
                ),

                "N": len(real_values),

                "vividness_mean": (
                    np.mean(vividness_values)
                    if vividness_values
                    else np.nan
                )
            })

    return pd.DataFrame(rows)


# ============================================================
# MODEL PERFORMANCE TABLE
# ============================================================

def build_model_performance_table(
    df_all,
    df_complex,
    metrics_all,
    metrics_complex
):
    """
    Build the Model Performance table for Excel output.

    Parameters
    ----------
    df_all : pandas.DataFrame
        Trial-level data including all stimulation patterns.

    df_complex : pandas.DataFrame
        Trial-level data including complex stimulation patterns only.

    metrics_all : dict
        Model performance metrics for all patterns.

    metrics_complex : dict
        Model performance metrics for complex patterns only.

    Returns
    -------
    pandas.DataFrame
        Model performance table.
    """

    rows = [
        {
            "Dataset": "ALL patterns",
            "N_trials": len(df_all),
            "R²_zero": metrics_all["R2_zero"],
            "MAE [°]": metrics_all["MAE"],
            "RMSE [°]": metrics_all["RMSE"]
        },
        {
            "Dataset": "COMPLEX patterns only",
            "N_trials": len(df_complex),
            "R²_zero": metrics_complex["R2_zero"],
            "MAE [°]": metrics_complex["MAE"],
            "RMSE [°]": metrics_complex["RMSE"]
        }
    ]

    return pd.DataFrame(rows)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    analysis_results,
    df,
    subjects,
    protocol,
    output_path,
    max_vividness=None
):
    """
    Build and save all model-analysis results.

    Parameters
    ----------
    analysis_results : dict
        Results returned by run_model_analysis().

    df : pandas.DataFrame
        Processed trial-level data for the selected subjects.

    subjects : list
        Subjects included in the analysis.

    protocol : dict
        Loaded experiment protocol.

    output_path : pathlib.Path
        Directory where the result files will be saved.

    Returns
    -------
    dict
        Dictionary containing all generated result tables.
    """

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # EXTRACT ANALYSIS RESULTS
    # ========================================================

    subject_parameters = (
        analysis_results["subject_parameters"]
    )

    global_parameters = (
        analysis_results["global_parameters"]
    )

    global_metrics_all = (
        analysis_results["global_metrics_all"]
    )

    global_metrics_complex = (
        analysis_results["global_metrics_complex"]
    )

    df_complex = (
        analysis_results["df_complex"]
    )

    # ========================================================
    # PATTERNS, DURATIONS 
    # ========================================================

    patterns = sorted(
        df["pattern_pair"].unique()
    )

    durations = sorted(
        protocol["blocks"]["durations"]
    )

    # ========================================================
    # PARAMETERS TABLE
    # ========================================================

    parameters_table = build_parameters_table(
        subject_parameters=subject_parameters,
        global_parameters=global_parameters,
        global_metrics_all=global_metrics_all,
        global_metrics_complex=global_metrics_complex
    )

    # ========================================================
    # SUBJECT VALIDATION TABLES
    # ========================================================

    subject_validation = {}

    for subject in subjects:

        subj_df = df[
            df["subject"] == subject
        ].copy()

        subject_validation[subject] = (
            build_validation_table(
                subj_df=subj_df,
                Kb=subject_parameters[subject]["Kb"],
                Kt=subject_parameters[subject]["Kt"],
                patterns=patterns,
                durations=durations,
                max_vividness=max_vividness
            )
        )

    # ========================================================
    # GROUP VALIDATION TABLE
    # ========================================================

    group_validation = build_group_validation_table(
        df=df,
        global_parameters=global_parameters,
        patterns=patterns,
        durations=durations,
        max_vividness=max_vividness
    )

    # ========================================================
    # MODEL PERFORMANCE TABLE
    # ========================================================

    model_performance = build_model_performance_table(
        df_all=df,
        df_complex=df_complex,
        metrics_all=global_metrics_all,
        metrics_complex=global_metrics_complex
    )

    # ========================================================
    # SAVE PARAMETERS + GROUP VALIDATION + PERFORMANCE
    # ========================================================

    group_validation_path = (
        output_path / "group_validation.xlsx"
    )

    with pd.ExcelWriter(
        group_validation_path,
        engine="openpyxl"
    ) as writer:

        parameters_table.to_excel(
            writer,
            sheet_name="Parameters"
        )

        group_validation.to_excel(
            writer,
            sheet_name="Validation",
            index=False
        )

        model_performance.to_excel(
            writer,
            sheet_name="Model_Performance",
            index=False
        )

    # ========================================================
    # SAVE SUBJECT VALIDATION TABLES
    # ========================================================

    for subject, validation_table in subject_validation.items():

        subject_path = (
            output_path
            / f"{subject}_validation.xlsx"
        )

        validation_table.to_excel(
            subject_path,
            index=False
        )

    # ========================================================
    # RETURN TABLES
    # ========================================================

    return {
        "parameters": parameters_table,
        "subject_validation": subject_validation,
        "group_validation": group_validation,
        "model_performance": model_performance
    }

