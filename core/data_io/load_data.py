# core/data_io/load_data.py

import json
import pandas as pd
from pathlib import Path

def load_file(path):
    """
    Load data from a JSON or Excel file.

    Parameters
    ----------
    path : str or Path
        Path to the data file (.json, .xlsx or .xls).

    Returns
    -------
    dict or pandas.DataFrame
        Parsed JSON content for .json files, or the first sheet of the
        workbook for Excel files.

    Raises
    ------
    ValueError
        If the file extension is not .json, .xlsx or .xls.
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

    Parameters
    ----------
    path : str or Path, optional
        Path to the protocol JSON file. Defaults to "protocol.json".

    Returns
    -------
    dict
        Protocol configuration.
    """
    with open(path, "r") as f:
        protocol = json.load(f)
    return protocol

def load_data_file(path, sheet_name=None):
    """
    Load main experimental data from an Excel file.

    Parameters
    ----------
    path : str or Path
        Path to the Excel file.

    sheet_name : str or int, optional
        Name or index of the sheet to load. If None, the first sheet
        is loaded.

    Returns
    -------
    pandas.DataFrame
        Loaded data.
    """
    if sheet_name is not None:
        return pd.read_excel(path,sheet_name=sheet_name)

    return pd.read_excel(path)


