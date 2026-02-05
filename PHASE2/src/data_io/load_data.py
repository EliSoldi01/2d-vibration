from genericpath import exists
import json
import pandas as pd

def load_data_file(path):
    """
    Load data from a JSON or Excel file.
    Args:
        path (str): Path to the data file.
    Returns:
        dict or pd.DataFrame: Loaded data.
    """
    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
        return data
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        data = pd.read_excel(path)
        return data
    else:
        raise ValueError("Unsupported file format. Use .json or .xlsx/.xls")
    
def load_protocol(path="protocol.json"):
    """
    Load protocol configuration from a JSON file.
    Args: 
        path (str): Path to the protocol JSON file.
    Returns:
        dict: Protocol configuration.
    """
    with open(path, "r") as f:
        protocol = json.load(f)
    return protocol

def load_main_data(path):
    """
    Load main experimental data from an xlsx file.
    Args:
        path (str): Path to the xlsx file.
    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_excel(path)

def merge_subject_data(df_main, df_subject):
    """
    Merge main data with subject data on subject and block.
    Args:
        df_main (pd.DataFrame): Main experimental data.
        df_subject (pd.DataFrame): Subject data.
    Returns:
        pd.DataFrame: Merged data.
    """
    return pd.merge(
        df_main,
        df_subject[["subject", "forearm_cm", "forearm_angle_deg"]],
        on="subject",
        how="left"
    )

def preprocess_data(df):
    """
    Preprocess the data by handling missing values and converting types.
    Args:
        df (pd.DataFrame): Data to preprocess.
    Returns:
        pd.DataFrame: Preprocessed data.
    """
    df["pattern_biceps"] = df["pattern_biceps"].astype(str).str.zfill(3)
    df["pattern_triceps"] = df["pattern_triceps"].astype(str).str.zfill(3)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df = df.dropna(subset=["duration"])
    if not "pattern_pair" in df.columns:
        df["pattern_pair"] = df["pattern_biceps"] + "_" + df["pattern_triceps"] 
    else:
        df["pattern_pair"] = df["pattern_pair"].astype(str)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y", "pattern_biceps", "pattern_triceps"])

    return df