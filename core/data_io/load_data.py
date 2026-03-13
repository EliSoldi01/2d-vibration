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
        df_subject[["subject", "forearm_cm", "forearm_angle_deg", "block_order"]],
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

# ─────────────────────────────────────────────────────────────────
# PHASE 3 — TRAJECTORY DATA
# ─────────────────────────────────────────────────────────────────
 
def load_trajectory_data(path):
    """
    Load trajectory data from the wide-format Excel produced by the
    trajectory_collector HTML tool and convert it to a tidy long-format
    DataFrame.
 
    Wide format (two sheets: Forward, Return)
    ------------------------------------------
    Each row = one rep of one pattern for one subject.
 
    Forward sheet columns:
        subject, trial, rep, duration,
        pattern_biceps, pattern_triceps, pattern_pair,
        forward_step1_x, forward_step1_y,
        forward_step2_x, forward_step2_y, ...
 
    Return sheet columns:
        subject, trial, rep, duration,
        pattern_biceps, pattern_triceps, pattern_pair,
        return_step1_x, return_step1_y,   ← step1 = peak (duplicated from forward)
        return_step2_x, return_step2_y, ...
 
    Long format output
    ------------------
    Columns:
        subject, trial, rep, duration,
        pattern_biceps, pattern_triceps, pattern_pair,
        direction   : 'forward' | 'return'
        step        : 1-based step index
        x           : column in the grid (int)
        y           : row    in the grid (int)
 
    Notes
    -----
    - NaN cells in the wide format signal the end of a variable-length
      trajectory and are silently dropped.
    - The duplicated peak (return_step1) is kept in the output with
      step=1 and direction='return' so that the return arc starts from
      the correct position.
    - If the Return sheet is missing, only forward data is returned.
    """
    meta_cols = [
        "subject", "trial", "rep", "duration",
        "pattern_biceps", "pattern_triceps", "pattern_pair",
    ]
 
    xl = pd.ExcelFile(path)
    rows = []
 
    # ── Forward ───────────────────────────────────────────────────
    if "Forward" in xl.sheet_names:
        df_fwd = xl.parse("Forward")
        rows.extend(_parse_direction(df_fwd, meta_cols, "forward", "forward_step"))
    else:
        print(f"[load_trajectory_data] WARNING: 'Forward' sheet not found in {path}")
 
    # ── Return ────────────────────────────────────────────────────
    if "Return" in xl.sheet_names:
        df_ret = xl.parse("Return")
        rows.extend(_parse_direction(df_ret, meta_cols, "return", "return_step"))
 
    if not rows:
        print(f"[load_trajectory_data] WARNING: no data loaded from {path}")
        return pd.DataFrame(columns=meta_cols + ["direction", "step", "x", "y"])
 
    df_long = pd.DataFrame(rows)
 
    # Enforce types
    df_long["step"]     = df_long["step"].astype(int)
    df_long["x"]        = df_long["x"].astype(int)
    df_long["y"]        = df_long["y"].astype(int)
    df_long["trial"]    = df_long["trial"].astype(int)
    df_long["rep"]      = df_long["rep"].astype(int)
    df_long["duration"] = pd.to_numeric(df_long["duration"], errors="coerce")
 
    return df_long.reset_index(drop=True)
 
 
def _parse_direction(df_wide, meta_cols, direction, step_prefix):
    """
    Internal helper: unpack step columns from one sheet into long-format rows.
 
    Parameters
    ----------
    df_wide     : wide DataFrame (one sheet)
    meta_cols   : list of metadata column names to carry through
    direction   : 'forward' or 'return'
    step_prefix : e.g. 'forward_step' or 'return_step'
 
    Returns
    -------
    list of dicts
    """
    # Find all step indices present in this sheet
    step_indices = sorted({
        int(col.replace(step_prefix, "").replace("_x", "").replace("_y", ""))
        for col in df_wide.columns
        if col.startswith(step_prefix) and (col.endswith("_x") or col.endswith("_y"))
    })
 
    rows = []
    for _, wide_row in df_wide.iterrows():
        base = {col: wide_row[col] for col in meta_cols if col in wide_row.index}
 
        for step in step_indices:
            x_col = f"{step_prefix}{step}_x"
            y_col = f"{step_prefix}{step}_y"
 
            if x_col not in df_wide.columns or y_col not in df_wide.columns:
                break
 
            x_val = wide_row[x_col]
            y_val = wide_row[y_col]
 
            # NaN = trajectory ended before this step
            if pd.isna(x_val) or pd.isna(y_val):
                break
 
            rows.append({
                **base,
                "direction": direction,
                "step":      step,
                "x":         int(x_val),
                "y":         int(y_val),
            })
 
    return rows
 
"""
def load_all_subjects_trajectory(folder_path):
    Load and concatenate trajectory Excel files for all subjects in a folder.
 
    Expects one file per subject named: {subject}_trajectory.xlsx
 
    Parameters
    ----------
    folder_path : str — path to folder containing subject Excel files
 
    Returns
    -------
    DataFrame in long format (same structure as load_trajectory_data)

    dfs = []
    folder = Path(folder_path) if not isinstance(folder_path, Path) else folder_path
 
    files = list(folder.glob("*_trajectory.xlsx"))
    if not files:
        print(f"[load_all_subjects_trajectory] No *_trajectory.xlsx files found in {folder_path}")
        return pd.DataFrame()
 
    for fpath in sorted(files):
        print(f"[load_all_subjects_trajectory] Loading {fpath.name}...")
        try:
            df = load_trajectory_data(str(fpath))
            dfs.append(df)
        except Exception as e:
            print(f"  WARNING: could not load {fpath.name}: {e}")
 
    if not dfs:
        return pd.DataFrame()
 
    return pd.concat(dfs, ignore_index=True)
 """
 
# needed for load_all_subjects_trajectory
from pathlib import Path