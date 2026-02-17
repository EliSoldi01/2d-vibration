import numpy as np

CELL_SIZE_DEFAULT = 4  # cm

def cell_to_cm(x, y, cell_size=CELL_SIZE_DEFAULT):
    """
    Converts grid cell coordinates (1-based) to cm coordinates.
    """
    return np.array([
        (x - 1) * cell_size + cell_size / 2,
        (y - 1) * cell_size + cell_size / 2
    ])


def compute_elbow(start_cm, forearm_length, forearm_angle_deg):
    """
    Computes elbow position in cm given forearm length and angle.
    """
    theta = np.deg2rad(forearm_angle_deg)
    dx = forearm_length * np.cos(theta)
    dy = forearm_length * np.sin(theta)
    return start_cm + np.array([dx, dy])


def signed_angle(v_ref, v_cur):
    """
    Signed angle (deg) between reference and current vectors.
    """
    dot = np.dot(v_ref, v_cur)
    det = v_ref[0] * v_cur[1] - v_ref[1] * v_cur[0]
    return np.degrees(np.arctan2(det, dot))


def compute_angle(start_cell, index_cell, forearm_cm, forearm_angle_deg):
    """
    Computes elbow-centered angular deviation (deg).
    Returns NaN if geometry info is missing.
    """
    if np.isnan(forearm_cm) or np.isnan(forearm_angle_deg):
        return np.nan

    start_cm = cell_to_cm(*start_cell)
    elbow_cm = compute_elbow(start_cm, forearm_cm, forearm_angle_deg)

    baseline_vec = start_cm - elbow_cm
    idx_cm = cell_to_cm(*index_cell)
    current_vec = idx_cm - elbow_cm

    return signed_angle(baseline_vec, current_vec)

def add_angle_column(df, start_cell):
    """
    Adds column 'angle_deg' to DataFrame.
    Requires columns: x, y, forearm_cm, forearm_angle_deg
    """
    df = df.copy()
    df["angle_deg"] = df.apply(
        lambda r: compute_angle(
            start_cell,
            (r["x"], r["y"]),
            r["forearm_cm"],
            r["forearm_angle_deg"]
        ),
        axis=1
    )
    return df
