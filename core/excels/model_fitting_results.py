# core/excels/model_fitting_results.py

import numpy as np
import pandas as pd

import config as cfg
from core.analysis import saturation


# ============================================================
# PARAMETER NAMES
# ============================================================

PARAMETER_NAMES = {
    "linear": [
        "slope",
    ],
    "tanh": [
        "L",
        "k",
    ],
    "logistic4": [
        "L_min",
        "L_max",
        "k",
        "D0",
    ],
}


# ============================================================
# PARAMETERS TABLE
# ============================================================

def build_parameters_table(results):
    """
    Build the parameter-estimate table for all fitted models.

    Models whose fit failed are skipped.

    Columns
    -------
    Model
    Parameter
    Estimate
    SE
    CI_lower
    CI_upper

    Parameters
    ----------
    results : dict
        Results returned by model_fitting.fit_models().

    Returns
    -------
    pandas.DataFrame
        One row per model × parameter, with the estimate, its standard
        error and its 95% confidence interval.
    """

    rows = []

    for model_name, result in results.items():

        if not result.get("ok", False):
            continue

        params = result.get("params")
        params_se = result.get("params_se")
        params_ci = result.get("params_ci")

        if params is None:
            continue

        parameter_names = PARAMETER_NAMES.get(
            model_name,
            [
                f"parameter_{i + 1}"
                for i in range(len(params))
            ],
        )

        for i, estimate in enumerate(params):

            se = (
                params_se[i]
                if params_se is not None
                else np.nan
            )

            if params_ci is not None:

                ci_lower = params_ci[i, 0]
                ci_upper = params_ci[i, 1]

            else:

                ci_lower = np.nan
                ci_upper = np.nan

            rows.append({
                "Model": model_name,
                "Parameter": parameter_names[i],
                "Estimate": estimate,
                "SE": se,
                "CI_lower": ci_lower,
                "CI_upper": ci_upper,
            })

    return pd.DataFrame(rows)


# ============================================================
# PERFORMANCE TABLE
# ============================================================

def build_performance_table(results):
    """
    Build the model-performance table.

    Models whose fit failed are skipped. Delta_AIC is computed relative
    to the lowest finite AIC among the fitted models.

    Columns
    -------
    Model
    R2
    AIC
    Delta_AIC

    Parameters
    ----------
    results : dict
        Results returned by model_fitting.fit_models().

    Returns
    -------
    pandas.DataFrame
        One row per model, with R² (referenced to the mean), AIC and
        Delta_AIC.
    """


    rows = []

    valid_aics = [
        result.get("aic", np.nan)
        for result in results.values()
        if result.get("ok", False)
        and np.isfinite(result.get("aic", np.nan))
    ]

    min_aic = (
        min(valid_aics)
        if valid_aics
        else np.nan
    )

    for model_name, result in results.items():

        if not result.get("ok", False):
            continue

        aic = result.get(
            "aic",
            np.nan,
        )

        if np.isfinite(aic) and np.isfinite(min_aic):
            delta_aic = aic - min_aic
        else:
            delta_aic = np.nan

        rows.append({
            "Model": model_name,
            "R2": result.get(
                "r2_mean",
                np.nan,
            ),
            "AIC": aic,
            "Delta_AIC": delta_aic,
        })

    return pd.DataFrame(rows)


# ============================================================
# SATURATION TABLE
# ============================================================

def build_saturation_table(
    results,
    Kb=None,
    Kt=None,
    threshold=cfg.SATURATION_THRESHOLDS_SENSITIVITY,
):
    """
    Build the saturation-information table.

    Saturation is computed for the AIC-selected model. If Kb and Kt are
    not provided, the saturation points are still reported but the
    equivalent stimulation durations are set to NaN. If the selected
    model has no saturation (linear), a single row with NaN values
    is returned for each threshold.

    Columns
    -------
    Model
    Threshold
    Side
    X_ideal_deg
    Y_real_deg
    Duration_ideal_s
    Duration_real_s

    Notes
    -----
    X_ideal_deg corresponds to the ideal/input angle of the fitted model.
    Y_real_deg corresponds to the real/model-predicted angle.
    The two corresponding stimulation durations are calculated separately.

    Parameters
    ----------
    results : dict
        Results returned by model_fitting.fit_models().

    Kb : float, optional
        Biceps coefficient used to convert angles into
        equivalent stimulation durations.

    Kt : float, optional
        Triceps coefficient used to convert angles into
        equivalent stimulation durations.

    threshold : float or iterable of float, optional
        Saturation threshold(s), e.g. 0.95 or [0.85, 0.90, 0.95].
        Defaults to cfg.SATURATION_THRESHOLDS_SENSITIVITY.

    Returns
    -------
    pandas.DataFrame
        One row per threshold × side (Biceps, Triceps).
    """


    best_model_name = None

    try:
        best_model_name = (
            __import__(
                "core.analysis.model_fitting",
                fromlist=["get_best_model"],
            ).get_best_model(results)
        )
    except Exception:
        pass

    saturation_infos = saturation.get_saturation_infos(
        results=results,
        best_model_name=best_model_name,
        threshold=threshold,
    )

    rows = []

    # --------------------------------------------------------
    # No finite saturation
    # --------------------------------------------------------

    for info in saturation_infos:

        if not info["has_saturation"]:

            rows.append({
                "Model": info["model"],
                "Threshold": info["threshold"],
                "Side": None,
                "X_ideal_deg": np.nan,
                "Y_real_deg": np.nan,
                "Duration_ideal_s": np.nan,
                "Duration_real_s": np.nan,
            })

            continue

        # ----------------------------------------------------
        # Without Kb/Kt we can still report x/y saturation,
        # but not equivalent stimulation durations.
        # ----------------------------------------------------

        if Kb is None or Kt is None:

            if info["model"] == "tanh":

                rows.extend([
                    {
                        "Model": info["model"],
                        "Threshold": info["threshold"],
                        "Side": "Biceps",
                        "X_ideal_deg": info["x_sat"],
                        "Y_real_deg": info["y_sat"],
                        "Duration_ideal_s": np.nan,
                        "Duration_real_s": np.nan,
                    },
                    {
                        "Model": info["model"],
                        "Threshold": info["threshold"],
                        "Side": "Triceps",
                        "X_ideal_deg": -info["x_sat"],
                        "Y_real_deg": -info["y_sat"],
                        "Duration_ideal_s": np.nan,
                        "Duration_real_s": np.nan,
                    },
                ])

            elif info["model"] == "logistic4":

                rows.extend([
                    {
                        "Model": info["model"],
                        "Threshold": info["threshold"],
                        "Side": "Biceps",
                        "X_ideal_deg": info["x_sat_high"],
                        "Y_real_deg": info["y_sat_high"],
                        "Duration_ideal_s": np.nan,
                        "Duration_real_s": np.nan,
                    },
                    {
                        "Model": info["model"],
                        "Threshold": info["threshold"],
                        "Side": "Triceps",
                        "X_ideal_deg": info["x_sat_low"],
                        "Y_real_deg": info["y_sat_low"],
                        "Duration_ideal_s": np.nan,
                        "Duration_real_s": np.nan,
                    },
                ])

            continue

        # ----------------------------------------------------
        # Full duration calculation
        # ----------------------------------------------------

        durations = saturation.calculate_saturation_durations(
            results=results,
            Kb=Kb,
            Kt=Kt,
            threshold=info["threshold"],
        )
        if info["model"] == "tanh":

            rows.extend([
                {
                    "Model": info["model"],
                    "Threshold": info["threshold"],
                    "Side": "Biceps",
                    "X_ideal_deg": info["x_sat"],
                    "Y_real_deg": info["y_sat"],
                    "Duration_ideal_s":
                        durations["duration_biceps_ideal"],
                    "Duration_real_s":
                        durations["duration_biceps_real"],
                },
                {
                    "Model": info["model"],
                    "Threshold": info["threshold"],
                    "Side": "Triceps",
                    "X_ideal_deg": -info["x_sat"],
                    "Y_real_deg": -info["y_sat"],
                    "Duration_ideal_s":
                        durations["duration_triceps_ideal"],
                    "Duration_real_s":
                        durations["duration_triceps_real"],
                },
            ])

        elif info["model"] == "logistic4":

            rows.extend([
                {
                    "Model": info["model"],
                    "Threshold": info["threshold"],
                    "Side": "Biceps",
                    "X_ideal_deg": info["x_sat_high"],
                    "Y_real_deg": info["y_sat_high"],
                    "Duration_ideal_s":
                        durations["duration_biceps_ideal"],
                    "Duration_real_s":
                        durations["duration_biceps_real"],
                },
                {
                    "Model": info["model"],
                    "Threshold": info["threshold"],
                    "Side": "Triceps",
                    "X_ideal_deg": info["x_sat_low"],
                    "Y_real_deg": info["y_sat_low"],
                    "Duration_ideal_s":
                        durations["duration_triceps_ideal"],
                    "Duration_real_s":
                        durations["duration_triceps_real"],
                },
            ])

    return pd.DataFrame(rows)


# ============================================================
# SAVE EXCEL
# ============================================================

def save_model_fitting_results(
    results,
    output_path,
    Kb=None,
    Kt=None,
    threshold=cfg.SATURATION_THRESHOLDS_SENSITIVITY,
):
    """
    Save model-fitting results to an Excel workbook.

    The workbook contains three sheets:

        Parameters
            Model parameter estimates, SEs and 95% CIs.
        Performance
            R², AIC and Delta-AIC.
        Saturation
            Saturation points and equivalent stimulation
            durations for the configured threshold(s).

    Parameters
    ----------
    results : dict
        Results returned by model_fitting.fit_models().

    output_path : str or Path
        Path of the Excel file to create.

    Kb : float, optional
        Biceps coefficient used to convert angles into
        equivalent stimulation durations.

    Kt : float, optional
        Triceps coefficient used to convert angles into
        equivalent stimulation durations.

    threshold : float or iterable of float, optional
        Saturation threshold(s).
        Defaults to cfg.SATURATION_THRESHOLDS_SENSITIVITY.

    Returns
    -------
    str or Path
        The path of the saved file (same as output_path).
    """

    parameters_df = build_parameters_table(
        results
    )

    performance_df = build_performance_table(
        results
    )

    saturation_df = build_saturation_table(
        results=results,
        Kb=Kb,
        Kt=Kt,
        threshold=threshold,
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:

        parameters_df.to_excel(
            writer,
            sheet_name="Parameters",
            index=False,
        )

        performance_df.to_excel(
            writer,
            sheet_name="Performance",
            index=False,
        )

        saturation_df.to_excel(
            writer,
            sheet_name="Saturation",
            index=False,
        )

    return output_path