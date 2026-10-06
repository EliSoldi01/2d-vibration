import numpy as np


def cell_to_cm(x, y, cell_size):
    """
    Convert 1-based grid cell coordinates to physical coordinates (cm).

    The returned point is the center of the cell.

    Grid convention:
        x increases to the right
        y increases downward

    Parameters
    ----------
    x : float
        Horizontal cell coordinate (1-based).

    y : float
        Vertical cell coordinate (1-based).

    cell_size : float
        Side length of a grid cell, in cm.

    Returns
    -------
    np.ndarray
        Array [x_cm, y_cm] with the physical coordinates of the cell center.
    """
    return np.array([
        (x - 1) * cell_size + cell_size / 2,
        (y - 1) * cell_size + cell_size / 2
    ])

def grid_coordinate_to_cm(
    coordinate: float,
    cell_size_cm: float,
) -> float:
    """
    Convert a single 1-based grid coordinate to a physical coordinate (cm).

    The returned value corresponds to the center of the cell along
    that axis.

    Parameters
    ----------
    coordinate : float
        Cell coordinate along one axis (1-based).

    cell_size_cm : float
        Side length of a grid cell, in cm.

    Returns
    -------
    float
        Physical coordinate of the cell center, in cm.
    """
    return (coordinate - 1.0) * cell_size_cm + cell_size_cm / 2.0

def grid_point_to_cm(
    x_cell: float,
    y_cell: float,
    cell_size_cm: float,
) -> tuple[float, float]:
    """
    Convert 1-based grid cell coordinates to physical coordinates (cm).

    The returned point is the center of the cell.

    Parameters
    ----------
    x_cell : float
        Horizontal cell coordinate (1-based).

    y_cell : float
        Vertical cell coordinate (1-based).

    cell_size_cm : float
        Side length of a grid cell, in cm.

    Returns
    -------
    tuple of float
        Physical coordinates (x_cm, y_cm) of the cell center.
    """
    return (
        grid_coordinate_to_cm(x_cell, cell_size_cm),
        grid_coordinate_to_cm(y_cell, cell_size_cm),
    )

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

    Parameters
    ----------
    start_cm : np.ndarray
        Wrist/start position [x_cm, y_cm].

    forearm_length : float
        Forearm length, in cm.

    forearm_angle_deg : float
        Angle of the forearm direction, in degrees.

    arm : str, optional
        Arm being tested, "right" or "left". Defaults to "right".

    Returns
    -------
    np.ndarray
        Elbow position [x_cm, y_cm].
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

    Parameters
    ----------
    v_ref : array-like
        Reference 2D vector.

    v_cur : array-like
        Current 2D vector.

    Returns
    -------
    float
        Signed angle from v_ref to v_cur, in degrees, in the
        range (-180, 180].
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

    Parameters
    ----------
    start_cell : tuple of float
        Grid cell (x, y) of the start position (1-based).

    index_cell : tuple of float
        Grid cell (x, y) of the indicated position (1-based).

    forearm_cm : float
        Forearm length, in cm.

    forearm_angle_deg : float
        Angle of the baseline forearm direction, in degrees.

    cell_size : float
        Side length of a grid cell, in cm.

    arm : str, optional
        Arm being tested, "right" or "left". For the left arm the
        sign of the angle is inverted so that the convention
        (positive = extension) is the same for both arms.
        Defaults to "right".

    Returns
    -------
    float
        Angular deviation, in degrees, or NaN if forearm_cm or
        forearm_angle_deg is missing.
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


