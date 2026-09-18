import numpy as np


def cell_to_cm(x, y, cell_size):
    """
    Convert 1-based grid cell coordinates to physical coordinates (cm).

    Grid convention:
        x increases to the right
        y increases downward
    """
    return np.array([
        (x - 1) * cell_size + cell_size / 2,
        (y - 1) * cell_size + cell_size / 2
    ])


def compute_elbow(
    start_cm,
    forearm_length,
    forearm_angle_deg,
    arm="right"
):
    """
    Compute elbow position in cm from the wrist/start position.

    For the left arm, the forearm direction is mirrored
    around the vertical axis.
    """
    theta = np.deg2rad(forearm_angle_deg)

    dx = forearm_length * np.cos(theta)
    dy = forearm_length * np.sin(theta)

    if arm == "left":
        dx = -dx

    return start_cm + np.array([dx, dy])


def signed_angle(v_ref, v_cur):
    """
    Compute the signed angle (degrees) between two vectors.

    Positive = counterclockwise in Cartesian coordinates.
    Negative = clockwise in Cartesian coordinates.
    """
    dot = np.dot(v_ref, v_cur)
    det = v_ref[0] * v_cur[1] - v_ref[1] * v_cur[0]

    return np.degrees(np.arctan2(det, dot))


def compute_angle(
    start_cell,
    index_cell,
    forearm_cm,
    forearm_angle_deg,
    cell_size,
    arm="right"
):
    """
    Compute elbow-centered angular deviation (degrees)
    relative to the baseline forearm direction.

    Positive = extension
    Negative = flexion

    Returns NaN if subject geometry information is missing.
    """

    if np.isnan(forearm_cm) or np.isnan(forearm_angle_deg):
        return np.nan

    start_cm = cell_to_cm(
        *start_cell,
        cell_size=cell_size
    )

    index_cm = cell_to_cm(
        *index_cell,
        cell_size=cell_size
    )

    elbow_cm = compute_elbow(
        start_cm,
        forearm_cm,
        forearm_angle_deg,
        arm=arm
    )

    baseline_vec = start_cm - elbow_cm
    current_vec = index_cm - elbow_cm

    angle = signed_angle(
        baseline_vec,
        current_vec
    )

    # Keep the convention used for the left arm
    if arm == "left":
        angle = -angle

    return angle


