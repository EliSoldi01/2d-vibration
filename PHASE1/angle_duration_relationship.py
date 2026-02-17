import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# Constants
# -----------------------------
NX = 21
NY = 9
CELL_SIZE = 4  # cm

OUTPUT_FOLDER = "Results_Phase1\\LinRegression"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# Coordinate conversions
# -----------------------------
def cell_to_cm(x, y):
    return np.array([
        (x - 1) * CELL_SIZE + CELL_SIZE / 2,
        (y - 1) * CELL_SIZE + CELL_SIZE / 2
    ])

# -----------------------------
# Geometry
# -----------------------------
def compute_elbow(start_cm, forearm_length, forearm_angle_deg):
    """
    Compute elbow position given forearm length and orientation.

    Args:
        start_cm: wrist position in cm (np.array [x, y])
        forearm_length: forearm length in cm
        forearm_angle_deg: angle between horizontal axis and forearm (degrees)

    Returns:
        elbow position in cm (np.array [x, y])
    """
    theta = np.deg2rad(forearm_angle_deg)

    dx = forearm_length * np.cos(theta)
    dy = forearm_length * np.sin(theta)

    return start_cm + np.array([dx, dy])

def signed_angle(v_ref, v_cur):
    """Compute signed angle in degrees between two vectors.
    Positive if v_cur is counter-clockwise from v_ref.
    Args:
        v_ref: reference vector
        v_cur: current vector
    Returns: angle in degrees
    """
    dot = np.dot(v_ref, v_cur)
    det = v_ref[0]*v_cur[1] - v_ref[1]*v_cur[0]
    return np.degrees(np.arctan2(det, dot))

def compute_angle(start_cell, index_cell, forearm_cm, forearm_angle_deg):
    """Compute absolute angle at elbow given start cell, index cell, forearm length and angle.
    Args:   
        start_cell: (x, y) of wrist in cell coordinates
        index_cell: (x, y) of index finger in cell coordinates
        forearm_cm: forearm length in cm
        forearm_angle_deg: angle of forearm in degrees
    Returns: absolute angle at elbow in degrees
    """
    start_cm = cell_to_cm(*start_cell)
    elbow_cm = compute_elbow(start_cm, forearm_cm, forearm_angle_deg)

    baseline_vec = start_cm - elbow_cm

    idx_cm = cell_to_cm(*index_cell)
    current_vec = idx_cm - elbow_cm

    delta_angle = signed_angle(baseline_vec, current_vec)
    absolute_angle = 90 + delta_angle

    return absolute_angle

# -----------------------------
# Data loading functions
# -----------------------------
def load_main_data(excel_path):
    """Load main data and create pattern column.
     Args:
        excel_path: path to the Excel file
    Returns: DataFrame with main data
    """
    df = pd.read_excel(excel_path)
    df["pattern"] = (
        df["pattern_biceps"].astype(int).astype(str).str.zfill(3) + "_" +
        df["pattern_triceps"].astype(int).astype(str).str.zfill(3)
    )
    return df

def extract_metadata(df, patterns=None):
    """Extract unique subjects, durations, and patterns from DataFrame.
    Args:
        df: DataFrame with main data
        patterns: list of patterns to filter, or None for all
    Returns: (subjects, durations, patterns)
    """
    if patterns is not None:
        df = df[df["pattern"].isin(patterns)]
    subjects = sorted(df["subject"].unique())
    durations = sorted(df["duration"].unique())
    patterns_list = sorted(df["pattern"].unique())
    return subjects, durations, patterns_list

def get_coordinates(df, subject, duration, pattern, patterns=None):
    """Get list of (x, y) coordinates for given subject, duration, and pattern.
    Args:
        df: DataFrame with main data
        subject: subject identifier
        duration: duration value
        pattern: pattern identifier
        patterns: list of patterns to filter, or None for all
    Returns: list of (x, y) tuples
    """
    if patterns is not None and pattern not in patterns:
        return []
    subset = df[
        (df["subject"] == subject) &
        (df["duration"] == duration) &
        (df["pattern"] == pattern)
    ]
    return list(zip(subset["x"], subset["y"]))

def load_forearm_data(excel_path):
    """Load forearm length data.
    Args:
        excel_path: path to the Excel file
    Returns: DataFrame with forearm lengths
    """
    df = pd.read_excel(excel_path)
    df["subject"] = df["subject"].astype(str)
    return df

def merge_forearm(main_df, forearm_df):
    """Merge forearm lengths into main DataFrame.
    Args:
        main_df: main DataFrame
        forearm_df: DataFrame with forearm lengths
    Returns: merged DataFrame
    """
    return pd.merge(main_df, forearm_df, on="subject", how="left")

def check_missing_forearm(df, var_name="forearm_cm"):
    """Check for missing forearm lengths and print warnings.
    Args:
        df: DataFrame with forearm lengths
    """
    missing = df[df[var_name].isna()]["subject"].unique()
    if len(missing) > 0:
        print(f"⚠️ Attention: Missing {var_name} for subjects:")
        print(missing)
    else:
        print(f"✅ All subjects have {var_name}.")

def linear_regression_fit_and_plot(df, var = "angle_deg",mean = False):
    X = df["duration"].values
    y = df[var].values

    # Linear regression
    coeffs = np.polyfit(X, y, deg=1)
    y_fit = np.polyval(coeffs, X)

    # R^2
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    print(
        f"Pattern {pattern}: "
        f"{var} = {coeffs[0]:.4f} * Duration + {coeffs[1]:.4f}, "
        f"R^2 = {r_squared:.4f}"
    )

    # -----------------------------
    # PLOT
    # -----------------------------

    plt.figure(figsize=(10, 5))

    plt.scatter(
        df["duration"],
        df[var],
        marker="o"
    )

    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = coeffs[0] * x_line + coeffs[1]

    plt.plot(
        x_line,
        y_line,
        color="red",
        linewidth=2,
        label=f"Fit: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}\n$R^2$ = {r_squared:.2f}"
    )

    plt.xlabel("Duration (s)")
    plt.ylabel(f"{var}")
    plt.title(f"Pattern {pattern} – {var} vs Duration")
    plt.ylim(50, 130)
    plt.legend()
    plt.grid(True)

    if mean:
        filename = os.path.join(OUTPUT_FOLDER, f"{var}_duration_pattern_{pattern}_{selected_subjects}_mean.png")
    else:
        filename = os.path.join(OUTPUT_FOLDER, f"{var}_duration_pattern_{pattern}_{selected_subjects}.png")
    plt.savefig(filename)
    plt.close()

# -----------------------------
# MAIN
# -----------------------------

# Load data
main_df = load_main_data(
    "C:\\Users\\Utente\\Desktop\\Elisa\\Research\\2D-vibration\\Subjects\\data_all_subjects - RIDOTTO.xlsx"
)

forearm_df = load_forearm_data(
    "C:\\Users\\Utente\\Desktop\\Elisa\\Research\\2D-vibration\\Subjects\\RIDOTTO_results\\Subjects_list.xlsx"
)

df = merge_forearm(
    main_df,
    forearm_df[["subject", "forearm_cm", "forearm_angle_deg"]]
)

# Check missing values
check_missing_forearm(df, var_name="forearm_cm")
check_missing_forearm(df, var_name="forearm_angle_deg")

# -----------------------------
# USER SELECTIONS
# -----------------------------

# Patterns to analyze
selected_patterns = ["100_000", "000_100"]

# Optional: select only some subjects (None = all)
selected_subjects = ["S07"]

# Set to True to compute regression on mean angles per duration
compute_mean_regression = True

# -----------------------------
# FILTER DATA
# -----------------------------

if selected_patterns is not None:
    df = df[df["pattern"].isin(selected_patterns)]

if selected_subjects is not None:
    df = df[df["subject"].isin(selected_subjects)]

subjects, durations, patterns = extract_metadata(
    df,
    patterns=selected_patterns
)

# -----------------------------
# GEOMETRY
# -----------------------------

start_cell = (11, 5)

df["angle_deg"] = df.apply(
    lambda row: compute_angle(
        start_cell,
        (row["x"], row["y"]),
        row["forearm_cm"],
        forearm_angle_deg=row["forearm_angle_deg"]
    ),
    axis=1
)

# -----------------------------
# CLEAN DATA FOR ANALYSIS
# -----------------------------

df_analysis = df.dropna(
    subset=["angle_deg", "duration", "vividness"]
).reset_index(drop=True)

# -----------------------------
# SAVE CLEAN DATA
# -----------------------------

df_final = df_analysis[
    ["subject", "duration", "rep", "pattern",
     "y", "x", "forearm_cm", "forearm_angle_deg", "angle_deg"]
]

df_final = df_final.sort_values(
    by=["pattern", "subject", "duration", "rep"]
).reset_index(drop=True)

# output folder
filename = os.path.join(OUTPUT_FOLDER, f"{'_'.join(selected_patterns)}_{len(selected_subjects)}subjects.xlsx")
df_final.to_excel(filename, index=False)
print(f"✅ Saved file: {filename}")

# -----------------------------
# LINEAR REGRESSION ANALYSIS
# -----------------------------

for pattern in selected_patterns:

    df_linear_regression = df_analysis[
        (df_analysis["pattern"] == pattern) & 
        (df_analysis["subject"].isin(selected_subjects))
    ]

    print(f"Pattern {pattern} – number of valid points: {len(df_linear_regression)}")

    if compute_mean_regression:
        df_linear_regression_mean = df_linear_regression.groupby(["duration"], as_index=False)["angle_deg"].mean()

    print(f"Pattern {pattern} – number of mean points: {len(df_linear_regression_mean)}")

    linear_regression_fit_and_plot(df_linear_regression)
    linear_regression_fit_and_plot(df_linear_regression_mean, True)