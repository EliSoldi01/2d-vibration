from pathlib import Path

import pandas as pd


# ============================================================
# SHEET NAMES
# ============================================================

SHEET_NAMES = {
    "angle_vs_duration": "angle_vs_duration",
    "vividness_vs_duration": "vividness_vs_duration",
    "angle_vs_vividness": "angle_vs_vividness",
    "angle_vs_presentation_order": "angle_vs_order",
}


# ============================================================
# BUILD DATAFRAMES
# ============================================================

def build_analysis_dataframes(results):
    """
    Convert regression results into one DataFrame per
    analysis type.

    Parameters
    ----------
    results : dict
        Dictionary returned by the regression analyses.

    Returns
    -------
    dict
        Dictionary mapping analysis names to DataFrames.
    """

    dataframes = {}

    for analysis_name, analysis_results in results.items():

        if analysis_results is None:
            continue

        rows = []

        for pattern, result in analysis_results.items():

            if result is None:
                continue

            rows.append({
                "pattern": pattern,
                "n": result["n"],
                "slope": result["slope"],
                "r_squared": result["r_squared"],
                "mae": result["mae"],
                "rmse": result["rmse"],
            })

        if rows:
            dataframes[analysis_name] = pd.DataFrame(rows)

    return dataframes


# ============================================================
# SAVE EXCEL
# ============================================================

def save_regression_results(
    results,
    output_path,
):
    """
    Save regression results to an Excel workbook.

    One worksheet is created for each regression analysis.

    Parameters
    ----------
    results : dict
        Dictionary returned by the regression analyses.

    output_path : str or Path
        Output Excel file path.
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

        for analysis_name, df in dataframes.items():

            sheet_name = SHEET_NAMES.get(
                analysis_name,
                analysis_name,
            )

            df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

            # ------------------------------------------------
            # Basic formatting
            # ------------------------------------------------

            worksheet = writer.sheets[sheet_name]

            for column_cells in worksheet.columns:

                max_length = 0

                for cell in column_cells:

                    if cell.value is not None:
                        max_length = max(
                            max_length,
                            len(str(cell.value)),
                        )

                worksheet.column_dimensions[
                    column_cells[0].column_letter
                ].width = max_length + 2

    print("\n" + "=" * 70)
    print("REGRESSION RESULTS SAVED")
    print("=" * 70)
    print(f"Path: {output_path}")
    print(f"Sheets: {list(dataframes.keys())}")