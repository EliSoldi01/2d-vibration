import numpy as np

import config as cfg

import core.analysis.model_fitting as mf

# ============================================================
# SATURATION
# ============================================================

def get_saturation_info(
    results,
    best_model_name=None,
    threshold=cfg.SATURATION_THRESHOLD,
):
    """
    Calculate saturation information for the
    AIC-selected model.

    Definitions
    -----------

    Linear:
        No finite saturation point.

    Tanh:
        Saturation is defined as the point at which
        the response reaches `threshold` of the asymptote.

        y_sat = threshold * L

        x_sat = atanh(threshold) / k

    Logistic4:
        Saturation is defined relative to the range
        between L_min and L_max.

        y_sat = L_min +
                threshold * (L_max - L_min)

        x_sat = D0 +
                log(threshold / (1-threshold)) / k

    Returns
    -------
    dict
    """

    if not (0 < threshold < 1):
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    if best_model_name is None:
        best_model_name = mf.get_best_model(
            results
        )

    if best_model_name not in results:
        raise ValueError(
            f"Model '{best_model_name}' "
            "not found in results."
        )

    result = results[
        best_model_name
    ]

    if not result.get("ok", False):
        raise ValueError(
            f"Model '{best_model_name}' "
            "is not valid."
        )

    params = result["params"]

    # --------------------------------------------------------
    # LINEAR
    # --------------------------------------------------------

    if best_model_name == "linear":

        return {
            "model": "linear",
            "has_saturation": False,
            "threshold": threshold,
        }

    # --------------------------------------------------------
    # TANH
    # --------------------------------------------------------

    elif best_model_name == "tanh":

        L, k = params

        x_sat = (
            np.arctanh(threshold)
            / abs(k)
        )

        y_sat = (
            threshold * abs(L)
        )

        return {
            "model": "tanh",
            "has_saturation": True,
            "threshold": threshold,
            "x_sat": x_sat,
            "y_sat": y_sat,
            "L": L,
            "k": k,
        }

    # --------------------------------------------------------
    # LOGISTIC 4
    # --------------------------------------------------------

    elif best_model_name == "logistic4":

        L_min, L_max, k, D0 = params

        # 5% and 95% points
        low = 1.0 - threshold
        high = threshold

        x_low = (
            D0
            + np.log(
                low / (1.0 - low)
            ) / k
        )

        x_high = (
            D0
            + np.log(
                high / (1.0 - high)
            ) / k
        )

        y_low = (
            L_min
            + low * (
                L_max - L_min
            )
        )

        y_high = (
            L_min
            + high * (
                L_max - L_min
            )
        )

        return {
            "model": "logistic4",
            "has_saturation": True,
            "threshold": threshold,

            "x_sat": x_high,
            "y_sat": y_high,

            "x_sat_low": x_low,
            "x_sat_high": x_high,

            "y_sat_low": y_low,
            "y_sat_high": y_high,

            "L_min": L_min,
            "L_max": L_max,
            "k": k,
            "D0": D0,
        }

    else:

        raise ValueError(
            f"Unknown model: {best_model_name}"
        )


def calculate_saturation_point(
    results,
    threshold=cfg.SATURATION_THRESHOLD,
):
    """
    Backward-compatible wrapper.

    Returns the saturation x and y values
    for the AIC-selected model.

    For logistic4, returns the upper
    95% saturation point.
    """

    info = get_saturation_info(
        results,
        threshold=threshold,
    )

    if not info["has_saturation"]:
        return None, None

    return (
        info["x_sat"],
        info["y_sat"],
    )


# ============================================================
# EQUIVALENT STIMULATION DURATION
# ============================================================

def calculate_equivalent_stimulation_duration(
    saturation_angle,
    Kb,
    Kt,
    pattern,
):
    """
    Convert a saturation angle into the equivalent
    stimulation duration using:

        angle = Kb * pb * D
              + Kt * pt * D

    therefore:

        D = angle /
            (Kb * pb + Kt * pt)

    Parameters
    ----------
    saturation_angle : float
        Saturation angle in degrees.

    Kb : float
        Biceps coefficient.

    Kt : float
        Triceps coefficient.

    pattern : str
        Pattern code in the format 'BBB_TTT',
        where the first part represents biceps
        activation and the second part represents
        triceps activation.

        For mixed patterns, the digits are interpreted
        as activation levels.

    Returns
    -------
    float
        Equivalent stimulation duration.
    """

    if not isinstance(pattern, str):
        raise ValueError(
            "pattern must be a string."
        )

    parts = pattern.split("_")

    if len(parts) != 2:
        raise ValueError(
            f"Invalid pattern '{pattern}'. "
            "Expected format '001_000'."
        )

    parts = pattern.split("_")

    if len(parts) != 2:
        raise ValueError(
            f"Invalid pattern '{pattern}'. "
            "Expected format 'BBB_TTT'."
        )

    try:
        pb = sum(
            int(value)
            for value in parts[0]
        )

        pt = sum(
            int(value)
            for value in parts[1]
        )

    except ValueError:
        raise ValueError(
            f"Invalid pattern '{pattern}'."
        )

    denominator = (
        Kb * pb
        + Kt * pt
    )

    if denominator == 0:
        raise ValueError(
            f"Pattern '{pattern}' produces "
            "a zero denominator."
        )

    return (
        saturation_angle
        / denominator
    )


def calculate_saturation_durations(
    results,
    Kb,
    Kt,
    threshold=cfg.SATURATION_THRESHOLD,
):
    """
    Calculate equivalent stimulation durations
    corresponding to saturation.

    For Tanh:
        - biceps uses the positive saturation angle
        - triceps uses the negative saturation angle

    For Logistic4:
        - biceps uses the upper 95% point
        - triceps uses the lower 5% point

    Returns
    -------
    dict
    """

    saturation = get_saturation_info(
        results,
        threshold=threshold,
    )

    if not saturation["has_saturation"]:

        return {
            "model": saturation["model"],
            "has_saturation": False,
        }

    # --------------------------------------------------------
    # Saturation angles
    # --------------------------------------------------------

    if saturation["model"] == "tanh":

        # Symmetric saturation
        theta_biceps = saturation["y_sat"]
        theta_triceps = -saturation["y_sat"]

    elif saturation["model"] == "logistic4":

        # Upper side = extension = biceps
        theta_biceps = saturation[
            "y_sat_high"
        ]

        # Lower side = flexion = triceps
        theta_triceps = saturation[
            "y_sat_low"
        ]

    else:

        raise ValueError(
            f"Unsupported model: "
            f"{saturation['model']}"
        )

    # --------------------------------------------------------
    # Equivalent durations
    # --------------------------------------------------------

    biceps_pattern = next(
        pattern
        for pattern, label in cfg.PURE_PATTERNS.items()
        if label == "b"
    )

    triceps_pattern = next(
        pattern
        for pattern, label in cfg.PURE_PATTERNS.items()
        if label == "t"
    )
    
    duration_biceps = (
        calculate_equivalent_stimulation_duration(
            theta_biceps,
            Kb,
            Kt,
            biceps_pattern,
        )
    )

    duration_triceps = (
        calculate_equivalent_stimulation_duration(
            theta_triceps,
            Kb,
            Kt,
            triceps_pattern,
        )
    )

    output = {
        "model": saturation["model"],
        "has_saturation": True,

        "threshold": threshold,

        "saturation_angle_biceps":
            theta_biceps,

        "saturation_angle_triceps":
            theta_triceps,

        "duration_biceps":
            duration_biceps,

        "duration_triceps":
            duration_triceps,
    }

    # --------------------------------------------------------
    # Logistic-specific information
    # --------------------------------------------------------

    if saturation["model"] == "logistic4":

        output.update({

            "saturation_angle_low":
                saturation["y_sat_low"],

            "saturation_angle_high":
                saturation["y_sat_high"],

            "duration_low_000_001":
                duration_triceps,

            "duration_high_001_000":
                duration_biceps,
        })

    return output