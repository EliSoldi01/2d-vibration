import numpy as np
import pandas as pd
from .create_std_excel import std_normalized

def aggregate(df, groupby_cols, value_cols):
    """
    Generic aggregation:
    returns mean, std and std_norm for each value column.
    """
    agg_dict = {}
    for v in value_cols:
        agg_dict[f"{v}_mean"] = (v, "mean")
        agg_dict[f"{v}_std"] = (v, "std")

    summary = (
        df
        .groupby(groupby_cols)
        .agg(**agg_dict)
        .reset_index()
    )

    for v in value_cols:
        summary[f"{v}_std_norm"] = summary.apply(
            lambda r: std_normalized(
                r[f"{v}_mean"], r[f"{v}_std"]
            ),
            axis=1
        )

    return summary


def aggregate_reps(df):
    """
    Aggregation per subject, pattern_pair, rep, duration
    """
    return aggregate(
        df,
        groupby_cols=["subject", "pattern_pair", "rep", "duration"],
        value_cols=["vividness", "angle_deg"]
    )

def aggregate_subject(df):
    """
    Aggregation per subject, pattern_pair, duration
    """
    return aggregate(
        df,
        groupby_cols=["subject", "pattern_pair", "duration"],
        value_cols=["vividness", "angle_deg"]
    )

def aggregate_all_subjects(df):
    """
    Aggregation across subjects per pattern_pair, duration
    """
    return aggregate(
        df,
        groupby_cols=["pattern_pair", "duration"],
        value_cols=["vividness", "angle_deg"]
    )
