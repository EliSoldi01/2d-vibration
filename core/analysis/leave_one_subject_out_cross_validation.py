# core/analysis/leave_one_subject_out_cross_validation.py

import numpy as np

import config as cfg
from core.analysis.extract_model_parameters import (
    compute_global_parameters,
    compute_model_metrics,
    get_model_predictions
)
from core.analysis.metrics import compute_r2_metrics


# ===================================================
# HELPER FUNCTIONS
# ===================================================

def _compute_loso_summary(subjects_results):
    """
    Compute mean and standard deviation of LOSO performance metrics across subjects.

    Parameters
    ----------
    subjects_results : list of dict
        Per-subject LOSO results. Each dictionary must contain the keys
        "MAE" and "RMSE".

    Returns
    -------
    dict
        Dictionary containing:

        included_metrics
            List of the summarized metrics ("MAE", "RMSE").
        Mean_MAE
        Std_MAE
        Mean_RMSE
        Std_RMSE

        All values are rounded to two decimals.
    """

    mae_values = [result["MAE"] for result in subjects_results]
    rmse_values = [result["RMSE"] for result in subjects_results]

    return {
        "included_metrics": ["MAE", "RMSE"],
        "Mean_MAE": np.round(np.mean(mae_values), 2),
        "Std_MAE": np.round(np.std(mae_values), 2),
        "Mean_RMSE": np.round(np.mean(rmse_values), 2),
        "Std_RMSE": np.round(np.std(rmse_values), 2),
    }


def _compute_parameter_summary(kb_values, kt_values):
    """
    Compute mean and standard deviation of LOSO model parameters.

    Parameters
    ----------
    kb_values : list of float
        Biceps parameter Kb estimated in each LOSO training fold.

    kt_values : list of float
        Triceps parameter Kt estimated in each LOSO training fold.

    Returns
    -------
    dict
        Dictionary containing:

        included_metrics
            List of the summarized parameters ("Kb", "Kt").
        Mean_Kb
        Std_Kb
        Mean_Kt
        Std_Kt

        All values are rounded to two decimals.
    """

    return {
        "included_metrics": ["Kb", "Kt"],
        "Mean_Kb": np.round(np.mean(kb_values), 2),
        "Std_Kb": np.round(np.std(kb_values), 2),
        "Mean_Kt": np.round(np.mean(kt_values), 2),
        "Std_Kt": np.round(np.std(kt_values), 2),
    }


def _compute_pooled_metrics(y_true, y_pred, weights):
    """
    Compute pooled LOSO performance metrics across all held-out trials.

    Parameters
    ----------
    y_true : array-like
        Observed angles of all held-out trials.

    y_pred : array-like
        Predicted angles of all held-out trials.

    weights : array-like
        Sample weights of all held-out trials.

    Returns
    -------
    dict
        Dictionary containing "R2_zero", "MAE" and "RMSE",
        rounded to two decimals.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights)

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
        "R2_zero": np.round(r2_zero, 2),
        "MAE": np.round(mae, 2),
        "RMSE": np.round(rmse, 2)
    }


# ============================================================
# LOOCV leave-one-subject-out
# ============================================================

def run_leave_one_subject_out_cross_validation(df, protocol):
    """
    Run leave-one-subject-out cross-validation for all and complex patterns.

    For each subject, the global parameters Kb and Kt are estimated on the
    remaining subjects and then used to predict the held-out subject's trials.
    Performance is evaluated on all patterns and on complex patterns only
    (i.e. excluding the pure patterns in cfg.PURE_PATTERNS).

    Parameters
    ----------
    df : pandas.DataFrame
        Processed trial-level data for all subjects. Must contain the
        columns "subject", "pattern_pair", "duration", "angle_deg" and
        "vividness".

    protocol : dict
        Experimental protocol. The vividness scale values are read from
        protocol["scales"]["vividness"]["values"] to compute max_vividness
        when cfg.USE_VIVIDNESS_WEIGHTS is True.

    Returns
    -------
    dict
        Dictionary containing:

        model_parameters
            summary
                Mean and standard deviation of Kb and Kt across folds.

        all_patterns
            subjects_results
                Per-subject Kb, Kt, R2_zero, MAE and RMSE.
            summary
                Mean and standard deviation of MAE and RMSE across subjects.
            pooled
                R2_zero, MAE and RMSE pooled across all held-out trials.

        complex_patterns
            Same structure as all_patterns, restricted to complex patterns.
    """

    results = {
        "model_parameters": {
            "summary": {}
        },
        "all_patterns": {
            "subjects_results": [],
            "summary": {},
            "pooled": {}
        },
        "complex_patterns": {
            "subjects_results": [],
            "summary": {},
            "pooled": {}
        },
    }

    all_y_true = []
    all_y_pred = []
    all_weights = []

    complex_y_true = []
    complex_y_pred = []
    complex_weights = []

    kb_values = []
    kt_values = []

    max_vividness = (
        None
        if not cfg.USE_VIVIDNESS_WEIGHTS
        else max(protocol["scales"]["vividness"]["values"])
    )

    subjects = df["subject"].dropna().unique()

    for subject in subjects:

        test_df = df[df["subject"] == subject]
        train_df = df[df["subject"] != subject]

        # =====================================
        # TRAINING
        # =====================================

        parameters = compute_global_parameters(
            train_df,
            max_vividness=max_vividness
        )

        Kb = parameters["Kb"]
        Kt = parameters["Kt"]

        kb_values.append(Kb)
        kt_values.append(Kt)

        # =====================================
        # TESTING - ALL PATTERNS
        # =====================================

        metrics = compute_model_metrics(
            test_df,
            Kb,
            Kt,
            max_vividness=max_vividness
        )

        all_result = {
            "subject": subject,
            "included_patterns": "all",
            "Kb": np.round(Kb, 2),
            "Kt": np.round(Kt, 2),
            "R2_zero": np.round(metrics["R2_zero"], 2),
            "MAE": np.round(metrics["MAE"], 2),
            "RMSE": np.round(metrics["RMSE"], 2),
        }

        y_true, y_pred, weights = get_model_predictions(
            test_df,
            Kb,
            Kt,
            max_vividness=max_vividness
        )

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_weights.extend(weights)

        # =====================================
        # TESTING - COMPLEX PATTERNS
        # =====================================

        complex_df = test_df[
            ~test_df["pattern_pair"].isin(cfg.PURE_PATTERNS.keys())
        ].copy()

        metrics_complex = compute_model_metrics(
            complex_df,
            Kb,
            Kt,
            max_vividness=max_vividness
        )

        complex_result = {
            "subject": subject,
            "included_patterns": "complex",
            "Kb": np.round(Kb, 2),
            "Kt": np.round(Kt, 2),
            "R2_zero": np.round(metrics_complex["R2_zero"], 2),
            "MAE": np.round(metrics_complex["MAE"], 2),
            "RMSE": np.round(metrics_complex["RMSE"], 2),
        }

        y_true_complex, y_pred_complex, weights_complex = get_model_predictions(
            complex_df,
            Kb,
            Kt,
            max_vividness=max_vividness
        )

        complex_y_true.extend(y_true_complex)
        complex_y_pred.extend(y_pred_complex)
        complex_weights.extend(weights_complex)

        # =====================================
        # STORE SUBJECT RESULTS
        # =====================================

        results["all_patterns"]["subjects_results"].append(all_result)
        results["complex_patterns"]["subjects_results"].append(complex_result)

    # ========================================================
    # SUMMARY
    # ========================================================

    results["model_parameters"]["summary"] = _compute_parameter_summary(
        kb_values,
        kt_values
    )

    results["all_patterns"]["summary"] = _compute_loso_summary(
        results["all_patterns"]["subjects_results"]
    )

    results["complex_patterns"]["summary"] = _compute_loso_summary(
        results["complex_patterns"]["subjects_results"]
    )

    # ========================================================
    # POOLED PERFORMANCE
    # ========================================================

    results["all_patterns"]["pooled"] = _compute_pooled_metrics(
        all_y_true,
        all_y_pred,
        all_weights
    )

    results["complex_patterns"]["pooled"] = _compute_pooled_metrics(
        complex_y_true,
        complex_y_pred,
        complex_weights
    )

    return results