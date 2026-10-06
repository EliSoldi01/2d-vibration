# core/excels/loso_results.py
import pandas as pd

def _create_per_subject_dataframe(results):
    """
    Create a DataFrame containing LOSO metrics for each subject.

    Parameters
    ----------
    results : dict
        LOSO cross-validation results, as returned by
        run_leave_one_subject_out_cross_validation.

    Returns
    -------
    pandas.DataFrame
        One row per subject, with Kb, Kt and R2_zero, MAE and RMSE
        for all patterns and for complex patterns only.
    """


    all_results = results["all_patterns"]["subjects_results"]
    complex_results = results["complex_patterns"]["subjects_results"]

    rows = []

    for all_result, complex_result in zip(all_results, complex_results):

        rows.append({
            "subject": all_result["subject"],
            "Kb": all_result["Kb"],
            "Kt": all_result["Kt"],
            "R2_zero_all_patterns": all_result["R2_zero"],
            "MAE_all_patterns": all_result["MAE"],
            "RMSE_all_patterns": all_result["RMSE"],
            "R2_zero_complex_patterns": complex_result["R2_zero"],
            "MAE_complex_patterns": complex_result["MAE"],
            "RMSE_complex_patterns": complex_result["RMSE"],
        })

    return pd.DataFrame(rows)

def _create_summary_dataframe(results):
    """
    Create a DataFrame containing descriptive statistics of LOSO metrics.

    Parameters
    ----------
    results : dict
        LOSO cross-validation results, as returned by
        run_leave_one_subject_out_cross_validation.

    Returns
    -------
    pandas.DataFrame
        One row per dataset × metric, with columns "dataset", "metric",
        "mean" and "sd". The datasets are "model_parameters" (Kb, Kt),
        "all_patterns" and "complex_patterns" (MAE, RMSE).
    """

    rows = []

    # Model parameters
    parameter_summary = results["model_parameters"]["summary"]

    for metric in parameter_summary["included_metrics"]:
        rows.append({
            "dataset": "model_parameters",
            "metric": metric,
            "mean": parameter_summary[f"Mean_{metric}"],
            "sd": parameter_summary[f"Std_{metric}"]
        })

    # Performance metrics
    for dataset_name in ["all_patterns", "complex_patterns"]:

        summary = results[dataset_name]["summary"]

        for metric in summary["included_metrics"]:
            rows.append({
                "dataset": dataset_name,
                "metric": metric,
                "mean": summary[f"Mean_{metric}"],
                "sd": summary[f"Std_{metric}"]
            })

    return pd.DataFrame(rows)

def _create_pooled_dataframe(results):
    """
    Create a DataFrame containing pooled LOSO performance metrics.

    Parameters
    ----------
    results : dict
        LOSO cross-validation results, as returned by
        run_leave_one_subject_out_cross_validation.

    Returns
    -------
    pandas.DataFrame
        One row per dataset ("all_patterns", "complex_patterns"), with
        the pooled R2_zero, MAE and RMSE.
    """
    rows = []
    
    for dataset_name in ["all_patterns", "complex_patterns"]:

        pooled = results[dataset_name]["pooled"]

        rows.append({
            "dataset": dataset_name,
            "R2_zero": pooled["R2_zero"],
            "MAE": pooled["MAE"],
            "RMSE": pooled["RMSE"]
        })

    return pd.DataFrame(rows)

def save_loso_results(results, output_path):
    """
    Save LOSO cross-validation results to an Excel workbook.

    The workbook contains three sheets:

        LOSO_per_subject
            Metrics for each held-out subject.
        Summary
            Mean and standard deviation of parameters and metrics.
        Pooled
            Performance pooled across all held-out trials.

    Parameters
    ----------
    results : dict
        LOSO cross-validation results, as returned by
        run_leave_one_subject_out_cross_validation.

    output_path : str or Path
        Path of the Excel file to create.

    Returns
    -------
    None
    """

    per_subject_df = _create_per_subject_dataframe(results)
    summary_df = _create_summary_dataframe(results)
    pooled_df = _create_pooled_dataframe(results)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

        per_subject_df.to_excel(writer, sheet_name="LOSO_per_subject", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        pooled_df.to_excel(writer, sheet_name="Pooled", index=False)



        

