# core/excels/regressions_results.py

from pathlib import Path

import pandas as pd


# ============================================================
# SHEET NAMES
# ============================================================

SHEET_NAMES = {
    "angle_vs_duration": "angle_duration",
    "vividness_vs_duration": "vividness_duration",
    "angle_vs_vividness": "angle_vividness",
}


# ============================================================
# BUILD DATAFRAMES
# ============================================================
def build_analysis_dataframes(results):
    """
    Convert regression results into one DataFrame per
    analysis level and analysis type.

    Levels or analyses with no results are skipped. At the subject level
    each row corresponds to a subject × pattern combination; at the
    other levels each row corresponds to a pattern.

    Parameters
    ----------
    results : dict
        Regression results, organized by level ("group_level",
        "subject_level", "trial_level") and then by analysis name,
        as returned in run_regression_analysis()["results"].

    Returns
    -------
    dict
        Dictionary mapping "<level>_<analysis_name>" (e.g.
        "group_level_angle_vs_duration") to a pandas.DataFrame with
        the columns "pattern", "n", "slope", "r_squared", "mae" and
        "rmse" (plus "subject" at the subject level).
    """

    dataframes = {}

    for level, level_results in results.items():

        for analysis_name, analysis_results in level_results.items():

            if analysis_results is None:
                continue

            rows = []

            for pattern, result in analysis_results.items():

                if result is None:
                    continue

                if level == "subject_level":

                    for subject_result in result:

                        if subject_result is None:
                            continue

                        rows.append({
                            "subject": subject_result["subject"],
                            "pattern": pattern,
                            "n": subject_result["n"],
                            "slope": subject_result["slope"],
                            "r_squared": subject_result["r_squared"],
                            "mae": subject_result["mae"],
                            "rmse": subject_result["rmse"],
                        })

                else:

                    rows.append({
                        "pattern": pattern,
                        "n": result["n"],
                        "slope": result["slope"],
                        "r_squared": result["r_squared"],
                        "mae": result["mae"],
                        "rmse": result["rmse"],
                    })

            if rows:
                dataframes[
                    f"{level}_{analysis_name}"
                ] = pd.DataFrame(rows)

    return dataframes


# ============================================================
# SAVE EXCEL
# ============================================================

def save_regression_results(results, output_path):
    """
    Save regression results to an Excel workbook.

    A separate worksheet is created for each combination
    of analysis level and regression analysis. Column widths are
    adjusted to the content, and a summary of the saved file is
    printed to the console.

    Parameters
    ----------
    results : dict
        Regression results, as returned in
        run_regression_analysis()["results"].

    output_path : str or Path
        Path of the Excel file to create. Missing parent directories
        are created.

    Returns
    -------
    None
        Nothing is saved if there are no results.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframes = build_analysis_dataframes(results)

    if not dataframes:
        print("No regression results to save.")
        return

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:

        for key, df in dataframes.items():

            level, analysis_name = key.split("_level_", 1)

            analysis_sheet = SHEET_NAMES.get(
                analysis_name,
                analysis_name,
            )

            sheet_name = f"{level}_{analysis_sheet}"

            df.to_excel(
                writer,
                sheet_name=sheet_name[:31],
                index=False,
            )

            worksheet = writer.sheets[
                sheet_name[:31]
            ]

            for column_cells in worksheet.columns:

                max_length = max(
                    len(str(cell.value))
                    for cell in column_cells
                    if cell.value is not None
                )

                worksheet.column_dimensions[
                    column_cells[0].column_letter
                ].width = max_length + 2

    print("\n" + "=" * 70)
    print("REGRESSION RESULTS SAVED")
    print("=" * 70)
    print(f"Path: {output_path}")
    print(f"Sheets: {list(dataframes.keys())}")