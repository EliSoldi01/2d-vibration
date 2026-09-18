import json
import pandas as pd
from pathlib import Path

def load_data_file(path):
    """
    Load data from a JSON or Excel file.
    Args:
        path (str): Path to the data file.
    Returns:
        dict or pd.DataFrame: Loaded data.
    """
    path = Path(path)
    
    if path.suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        return data
    elif path.suffix == ".xlsx" or path.suffix == ".xls":
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

def load_data_file(path, sheet_name):
    """
    Load main experimental data from an xlsx file.
    Args:
        path (str): Path to the xlsx file.
    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_excel(
        path,
        sheet_name=sheet_name
    )

