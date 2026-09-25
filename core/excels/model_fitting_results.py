import numpy as np
import pandas as pd

import config as cfg
from core.analysis.model_fitting import model_confidence_band


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
    Build a table containing parameter estimates, SEs,
    and 95% confidence intervals for all fitted models.

    Parameters
    ----------
    results : dict
        Output dictionary returned by fit_models().

    Returns
    -------
    pandas.DataFrame
        One row per fitted parameter.
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
            [f"parameter_{i + 1}" for i in range(len(params))]
        )

        for i, estimate in enumerate(params):

            if params_se is not None:
                se = params_se[i]
            else:
                se = np.nan

            if params_ci is not None:
                ci_lower = params_ci[i, 0]
                ci_upper = params_ci[i, 1]
            else:
                ci_lower = np.nan
                ci_upper = np.nan

            rows.append(
                {
                    "Model": model_name,
                    "Parameter": parameter_names[i],
                    "Estimate": estimate,
                    "SE": se,
                    "CI_lower": ci_lower,
                    "CI_upper": ci_upper,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# PERFORMANCE TABLE
# ============================================================

def build_performance_table(
    results,
    confidence=0.95,
):
    """
    Build a table containing model performance metrics.

    Parameters
    ----------
    results : dict
        Output dictionary returned by fit_models().
    confidence : float
        Confidence level used for the confidence band.

    Returns
    -------
    pandas.DataFrame
        One row per fitted model.
    """

    rows = []

    # --------------------------------------------------------
    # Minimum AIC
    # --------------------------------------------------------

    valid_aics = [
        result["aic"]
        for result in results.values()
        if result.get("ok", False)
        and np.isfinite(result.get("aic", np.nan))
    ]

    min_aic = (
        min(valid_aics)
        if valid_aics
        else np.nan
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    for model_name, result in results.items():

        if not result.get("ok", False):
            continue

        aic = result.get(
            "aic",
            np.nan
        )

        delta_aic = (
            aic - min_aic
            if np.isfinite(aic) and np.isfinite(min_aic)
            else np.nan
        )

        rows.append(
            {
                "Model": model_name,
                "R2": result.get(
                    "r2_mean",
                    np.nan
                ),
                "AIC": aic,
                "Delta_AIC": delta_aic,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SAVE EXCEL
# ============================================================

def save_model_fitting_results(
    results,
    output_path,
    confidence=0.95,
):
    """
    Save model fitting results to an Excel workbook.

    Sheets
    ------
    Parameters
        Parameter estimates, SEs, and 95% CIs.

    Performance
        R², AIC, ΔAIC, and confidence-band width summaries.

    Parameters
    ----------
    results : dict
        Output dictionary returned by fit_models().
    output_path : str or Path
        Output Excel file path.
    confidence : float
        Confidence level used for the confidence band.

    Returns
    -------
    str or Path
        Path to the saved Excel file.
    """

    parameters_df = build_parameters_table(
        results
    )

    performance_df = build_performance_table(
        results,
        confidence=confidence,
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl"
    ) as writer:

        parameters_df.to_excel(
            writer,
            sheet_name="Parameters",
            index=False
        )

        performance_df.to_excel(
            writer,
            sheet_name="Performance",
            index=False
        )

    return output_path
