import pandas as pd
from pathlib import Path

from core.preprocessing import geometry
from core.preprocessing import ordering

def merge_subject_data(df_main, df_subject):
    """
    Merge main data with subject data on subject.
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
    Preprocess the experimental data by handling missing values
    and converting data types.
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

    if "pattern_pair" not in df.columns:
        df["pattern_pair"] = (
            df["pattern_biceps"] + "_" +
            df["pattern_triceps"]
        )
    else:
        df["pattern_pair"] = df["pattern_pair"].astype(str)

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    df = df.dropna(
        subset=[
            "x",
            "y",
            "pattern_biceps",
            "pattern_triceps"
        ]
    )

    return df

def add_angle(
    df,
    protocol
):
    """
    Add 'angle_deg' to the DataFrame.

    Required columns:
        x
        y
        forearm_cm
        forearm_angle_deg
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
    Add presentation order and presentation position
    based on each subject's block order.
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
    Add the expected kinesthetic illusion for each pattern pair.
    """
    df = df.copy()

    effect_map = {
        pattern["text"]: pattern["effect"]
        for pattern in protocol["patterns"]["definitions"]
    }

    expected_illusion = df["pattern_pair"].map(effect_map)

    df.insert(
        df.columns.get_loc("pattern_pair") + 1,
        "expected_kinesthetic_illusion",
        expected_illusion
    )

    return df

def save_processed_data(df, output_path):
    """
    Save the processed experimental data to an Excel file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)

def prepare_data(df_main, df_subject, protocol, output_path = None):
    """
    Complete data preparation pipeline.

    Returns:
        pd.DataFrame: Prepared experimental data.
    """

    df = merge_subject_data(df_main, df_subject)

    df = add_pattern_pair(df)

    df = add_expected_illusion(
        df,
        protocol
    )

    df = add_angle(df, protocol)

    df = add_presentation_order(
        df,
        df_subject
    )

    if output_path is not None:
        save_processed_data(df, output_path)

    return df