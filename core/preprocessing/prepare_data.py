import pandas as pd
from pathlib import Path

from core.preprocessing import geometry
from core.preprocessing import ordering

def merge_subject_data(df_main, df_subject):
    """
    Merge main data with subject data on subject.

    Parameters
    ----------
    df_main : pandas.DataFrame
        Trial-level experimental data. Must contain the column "subject".

    df_subject : pandas.DataFrame
        Subjects worksheet. Must contain the columns "subject",
        "forearm_cm", "forearm_angle_deg" and "block_order".

    Returns
    -------
    pandas.DataFrame
        df_main with the columns "forearm_cm", "forearm_angle_deg" and
        "block_order" added (left join on "subject").
    """
    return pd.merge(
        df_main,
        df_subject[
            [
                "subject",
                "forearm_cm",
                "forearm_angle_deg",
                "block_order"
            ]
        ],
        on="subject",
        how="left"
    )


def add_pattern_pair(df):
    """
    Clean the pattern, duration and coordinate columns and add the pattern pair.

    The biceps and triceps patterns are converted to strings and
    zero-padded to 3 characters. Duration, x and y are converted to
    numeric values. Rows with missing duration, x, y or patterns are
    dropped. The new column "pattern_pair" (format "BBB_TTT") is inserted
    right after "pattern_triceps".

    Parameters
    ----------
    df : pandas.DataFrame
        Trial-level experimental data. Must contain the columns
        "pattern_biceps", "pattern_triceps", "duration", "x" and "y".

    Returns
    -------
    pandas.DataFrame
        Cleaned copy of df with the new column "pattern_pair".
    """
    df = df.copy()

    df["pattern_biceps"] = (
        df["pattern_biceps"].astype(str).str.zfill(3)
    )
    df["pattern_triceps"] = (
        df["pattern_triceps"].astype(str).str.zfill(3)
    )

    df["duration"] = pd.to_numeric(
        df["duration"],
        errors="coerce"
    )
    df = df.dropna(subset=["duration"])

    df["pattern_pair"] = (
        df["pattern_biceps"] + "_" +
        df["pattern_triceps"]
    )

    df.insert(df.columns.get_loc("pattern_triceps")  + 1, "pattern_pair", df.pop("pattern_pair"))

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    df = df.dropna(
        subset=[
            "x",
            "y",
            "pattern_biceps",
            "pattern_triceps",
            "pattern_pair"
        ]
    )

    return df

def add_angle(
    df,
    protocol
):
    """
    Add the angular deviation of each trial.

    The angle is computed with geometry.compute_angle, rounded to two
    decimals and inserted right after the "vividness" column.

    Required columns:
        x
        y
        vividness
        forearm_cm
        forearm_angle_deg

    Parameters
    ----------
    df : pandas.DataFrame
        Trial-level data including the subject geometry columns.

    protocol : dict
        Protocol configuration. The start cell, cell size and tested arm
        are read from protocol["grid"]["start_cell"],
        protocol["grid"]["cell_size_cm"] and protocol["arm"].

    Returns
    -------
    pandas.DataFrame
        Copy of df with the new column "angle_deg" (positive = extension,
        negative = flexion).
    """

    df = df.copy()

    start_cell=protocol["grid"]["start_cell"]
    cell_size=protocol["grid"]["cell_size_cm"]
    arm=protocol["arm"]

    angle_deg = df.apply(
        lambda row: geometry.compute_angle(
            start_cell=start_cell,
            index_cell=(row["x"], row["y"]),
            forearm_cm=row["forearm_cm"],
            forearm_angle_deg=row["forearm_angle_deg"],
            cell_size=cell_size,
            arm=arm
        ),
        axis=1
    ).round(2)

    df.insert(
        df.columns.get_loc("vividness") + 1,
        "angle_deg",
        angle_deg
    )

    return df

def add_presentation_order(df, subject_orders):
    """
    Add the presentation order based on each subject's block order.

    The duration column is first converted to integer.

    Parameters
    ----------
    df : pandas.DataFrame
        Trial-level data. Must contain the columns "subject" and
        "duration".

    subject_orders : dict or pandas.DataFrame
        Block order of each subject, either as a dictionary
        {subject: block_order} or as a DataFrame with the columns
        "subject" and "block_order".

    Returns
    -------
    pandas.DataFrame
        Copy of df with the new column "presentation_order" (1-based).
    """
    df = df.copy()

    df["duration"] = df["duration"].astype(int)

    df = ordering.add_presentation_order(
        df,
        subject_orders,
        duration_col="duration"
    )

    return df

def add_expected_illusion(df, protocol):
    """
    Add the expected kinesthetic illusion for each pattern pair,
    only if the column does not already exist.

    The new column "expected_kinesthetic_illusion" is inserted right
    after "pattern_pair". The effect of each pattern is read from the
    protocol definitions, matching "text" with the pattern pair.

    Parameters
    ----------
    df : pandas.DataFrame
        Trial-level data. Must contain the column "pattern_pair".

    protocol : dict
        Protocol configuration. The pattern definitions are read from
        protocol["patterns"]["definitions"].

    Returns
    -------
    pandas.DataFrame
        Copy of df with the new column, or df unchanged if the column
        already exists.
    """
    column_name = "expected_kinesthetic_illusion"

    # Se la colonna esiste già, restituisce il dataframe così com'è
    if column_name in df.columns:
        return df

    df = df.copy()

    effect_map = {
        pattern["text"]: pattern["effect"]
        for pattern in protocol["patterns"]["definitions"]
    }

    expected_illusion = df["pattern_pair"].map(effect_map)

    df.insert(
        df.columns.get_loc("pattern_pair") + 1,
        column_name,
        expected_illusion
    )

    return df

def add_protocol_metadata(df, protocol):
    """
    Add protocol metadata to the processed experimental data.

    The metadata are read from the protocol configuration and added as
    columns at the beginning of the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Trial-level experimental data.

    protocol : dict
        Protocol configuration. The metadata are read from the keys
        "experiment_id", "protocol_id", "arm" and "initial_angle".

    Returns
    -------
    pandas.DataFrame
        Copy of df with the protocol metadata columns added first.
    """

    df = df.copy()

    metadata = {
        "experiment_id": protocol["experiment_id"],
        "protocol_id": protocol["protocol_id"],
        "arm": protocol["arm"],
        "initial_angle": protocol["initial_angle"],
    }

    for column, value in reversed(metadata.items()):
        df.insert(0, column, value)

    return df

def save_processed_data(df, output_path):
    """
    Save the processed experimental data to an Excel file.

    Missing parent directories are created.

    Parameters
    ----------
    df : pandas.DataFrame
        Processed trial-level data.

    output_path : str or Path
        Path of the Excel file to create.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)

def prepare_data(df_main, df_subject, protocol, output_path=None):
    """
    Complete data preparation pipeline.

    Steps:
        1. Clean the data and add "pattern_pair".
        2. Merge the subject data.
        3. Add protocol metadata.
        4. Add the expected kinesthetic illusion.
        5. Add the angular deviation ("angle_deg").
        6. Add the presentation order.
        7. Optionally save the result.

    Parameters
    ----------
    df_main : pandas.DataFrame
        Trial-level experimental data.

    df_subject : pandas.DataFrame
        Subjects worksheet.

    protocol : dict
        Protocol configuration.

    output_path : str or Path, optional
        Directory where "data_processed.xlsx" is saved. If None, the
        data are not saved.

    Returns
    -------
    pandas.DataFrame
        Prepared experimental data.
    """

    df = add_pattern_pair(df_main)

    df = merge_subject_data(
        df,
        df_subject
    )

    df = add_protocol_metadata(
        df,
        protocol
    )

    df = add_expected_illusion(
        df,
        protocol
    )

    df = add_angle(
        df,
        protocol
    )

    df = add_presentation_order(
        df,
        df_subject
    )

    if output_path is not None:
        output_file = Path(
            output_path,
            "data_processed.xlsx"
        )
        save_processed_data(
            df,
            output_file
        )

    return df

def convert_dataframe_coordinates_to_cm(
    df: pd.DataFrame,
    cell_size_cm: float,
) -> pd.DataFrame:
    """
    Convert the grid coordinates of a DataFrame to physical coordinates (cm).

    The columns "x" and "y" are converted using
    geometry.grid_coordinate_to_cm, so that each value corresponds to
    the center of the cell.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing the columns "x" and "y" as 1-based cell
        coordinates.

    cell_size_cm : float
        Side length of a grid cell, in cm.

    Returns
    -------
    pandas.DataFrame
        Copy of df with "x" and "y" expressed in cm.
    """

    df_cm = df.copy()

    df_cm["x"] = df_cm["x"].apply(
        geometry.grid_coordinate_to_cm,
        cell_size_cm=cell_size_cm,
    )
    df_cm["y"] = df_cm["y"].apply(
        geometry.grid_coordinate_to_cm,
        cell_size_cm=cell_size_cm,
    )

    return df_cm

def prepare_quadratic_interpolation_data(
    df,
    duration,
    protocol,
    metric="vividness"
):
    """
    Select data for one duration and calculate the mean
    position and mean metric for each pattern × repetition.

    The grid coordinates are first converted to cm.

    Parameters
    ----------
    df : pandas.DataFrame
        Trial-level data containing the columns "pattern_pair", "rep",
        "duration", "x", "y" and the selected metric.

    duration : float
        Stimulation duration to select.

    protocol : dict
        Protocol configuration. The cell size is read from
        protocol["grid"]["cell_size_cm"].

    metric : str, optional
        Name of the column to average. Defaults to "vividness".

    Returns
    -------
    pandas.DataFrame
        One row per pattern × repetition, with the mean "x" and "y"
        (in cm) and the mean of the metric.
    """

    df = convert_dataframe_coordinates_to_cm(
            df=df,
            cell_size_cm=protocol["grid"]["cell_size_cm"],
        )
    
    df_dur = df[
        df["duration"] == duration
    ]

    df_mean = (
        df_dur
        .groupby(
            [
                "pattern_pair",
                "rep"
            ]
        )
        .agg({
            "x": "mean",
            "y": "mean",
            metric: "mean"
        })
        .reset_index()
    )

    return df_mean