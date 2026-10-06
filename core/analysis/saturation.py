# core/analysis/saturation.py

import numpy as np

import config as cfg
import core.analysis.model_fitting as mf


# ============================================================
# THRESHOLD UTILITIES
# ============================================================

def _normalize_thresholds(threshold):
    """
    Convert a scalar threshold or a sequence of thresholds
    into a list of floats.

    Examples
    --------
    0.95
        -> [0.95]

    [0.85, 0.90, 0.95]
        -> [0.85, 0.90, 0.95]

    Parameters
    ----------
    threshold : float or iterable of float
        Single saturation threshold or sequence of thresholds.

    Returns
    -------
    list of float
        List of thresholds.

    Raises
    ------
    ValueError
        If any threshold is not strictly between 0 and 1.
    """


    if np.isscalar(threshold):
        thresholds = [float(threshold)]
    else:
        thresholds = [float(value) for value in threshold]

    for value in thresholds:
        if not (0 < value < 1):
            raise ValueError(
                "All saturation thresholds must be between 0 and 1."
            )

    return thresholds


# ============================================================
# SATURATION
# ============================================================

def get_saturation_info(
    results,
    best_model_name=None,
    threshold=cfg.SATURATION_THRESHOLD,
):
    """
    Calculate saturation information for the AIC-selected model.

    `threshold` must be a single value between 0 and 1.

    Definitions
    -----------

    Linear:
        No finite saturation point.

    Tanh:
        Saturation is defined as the point at which
        the response reaches `threshold` of the asymptote.

        y_sat = threshold * |L|

        x_sat = atanh(threshold) / |k|

    Logistic4:
        Saturation is defined relative to the range
        between L_min and L_max. Both a lower point
        (at 1 - threshold) and an upper point (at threshold)
        are returned; the upper point is the main saturation point.

        y_sat = L_min +
                threshold * (L_max - L_min)

        x_sat = D0 +
                log(threshold / (1-threshold)) / k

    Parameters
    ----------
    results : dict
        Model fitting results, as returned by fit_models.

    best_model_name : str, optional
        Name of the model to use. If None, the model with the
        lowest AIC is selected.

    threshold : float, optional
        Saturation threshold, strictly between 0 and 1.
        Defaults to cfg.SATURATION_THRESHOLD.

    Returns
    -------
    dict
        Saturation information for one threshold. Always contains
        "model", "has_saturation" and "threshold"; for tanh and
        logistic4 it also contains the saturation coordinates
        ("x_sat", "y_sat", plus "x_sat_low", "x_sat_high",
        "y_sat_low", "y_sat_high" for logistic4) and the fitted
        model parameters.

    Raises
    ------
    ValueError
        If threshold is not a single value between 0 and 1, if the
        model is not found in results or is not valid, or if the
        model name is unknown.
    """

    if not np.isscalar(threshold):
        raise ValueError(
            "get_saturation_info() expects a single threshold. "
            "Use get_saturation_infos() for multiple thresholds."
        )

    threshold = float(threshold)

    if not (0 < threshold < 1):
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    if best_model_name is None:
        best_model_name = mf.get_best_model(results)

    if best_model_name not in results:
        raise ValueError(
            f"Model '{best_model_name}' not found in results."
        )

    result = results[best_model_name]

    if not result.get("ok", False):
        raise ValueError(
            f"Model '{best_model_name}' is not valid."
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

        # Lower and upper points corresponding to the
        # selected saturation threshold.
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
            + low * (L_max - L_min)
        )

        y_high = (
            L_min
            + high * (L_max - L_min)
        )

        return {
            "model": "logistic4",
            "has_saturation": True,
            "threshold": threshold,

            # Main saturation point = upper point
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


def get_saturation_infos(
    results,
    best_model_name=None,
    threshold=cfg.SATURATION_THRESHOLD,
):
    """
    Calculate saturation information for multiple thresholds.

    `threshold` can be either:

        0.95

    or:

        [0.85, 0.90, 0.95]

    Parameters
    ----------
    results : dict
        Model fitting results, as returned by fit_models.

    best_model_name : str, optional
        Name of the model to use. If None, the model with the
        lowest AIC is selected.

    threshold : float or iterable of float, optional
        Single saturation threshold or sequence of thresholds.
        Defaults to cfg.SATURATION_THRESHOLD.

    Returns
    -------
    list of dict
        One saturation dictionary for each threshold
        (see get_saturation_info).
    """

    thresholds = _normalize_thresholds(threshold)

    if best_model_name is None:
        best_model_name = mf.get_best_model(results)

    return [
        get_saturation_info(
            results=results,
            best_model_name=best_model_name,
            threshold=value,
        )
        for value in thresholds
    ]


# ============================================================
# BACKWARD-COMPATIBLE SATURATION POINT
# ============================================================

def calculate_saturation_point(
    results,
    threshold=cfg.SATURATION_THRESHOLD,
):
    """
    Backward-compatible wrapper.

    Returns the saturation x and y values
    for the AIC-selected model.

    For logistic4, returns the upper saturation point.

    This function expects a single threshold.

    Parameters
    ----------
    results : dict
        Model fitting results, as returned by fit_models.

    threshold : float, optional
        Saturation threshold, strictly between 0 and 1.
        Defaults to cfg.SATURATION_THRESHOLD.

    Returns
    -------
    x_sat : float or None
        Saturation point on the x axis, or None if the selected
        model has no saturation (linear).

    y_sat : float or None
        Saturation point on the y axis, or None if the selected
        model has no saturation (linear).
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
    Calculate equivalent stimulation durations corresponding
    to saturation.

    `threshold` can be either a single value or a sequence.

    For Tanh:
        - biceps uses the positive saturation angle
        - triceps uses the negative saturation angle

    For Logistic4:
        - biceps uses the upper saturation point
        - triceps uses the lower saturation point

    The returned duration information distinguishes between:

        ideal duration:
            duration corresponding to x_sat

        real duration:
            duration corresponding to y_sat

    Parameters
    ----------
    results : dict
        Model fitting results, as returned by fit_models.

    Kb : float
        Biceps coefficient.

    Kt : float
        Triceps coefficient.

    threshold : float or iterable of float, optional
        Single saturation threshold or sequence of thresholds.
        Defaults to cfg.SATURATION_THRESHOLD.

    Returns
    -------
    dict or list of dict
        A dictionary for a single threshold, or a list of
        dictionaries for multiple thresholds. If the selected
        model has no saturation (linear), the dictionary contains
        only "model", "has_saturation" and "threshold".
    """

    if not isinstance(pattern, str):
        raise ValueError(
            "pattern must be a string."
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

    return saturation_angle / denominator


# ============================================================
# SATURATION DURATIONS
# ============================================================

def calculate_saturation_durations(
    results,
    Kb,
    Kt,
    threshold=cfg.SATURATION_THRESHOLD,
):
    """
    Calculate equivalent stimulation durations corresponding
    to saturation.

    `threshold` can be either a single value or a sequence.

    For Tanh:
        - biceps uses the positive saturation angle
        - triceps uses the negative saturation angle

    For Logistic4:
        - biceps uses the upper saturation point
        - triceps uses the lower saturation point

    The returned duration information distinguishes between:

        ideal duration:
            duration corresponding to x_sat

        real duration:
            duration corresponding to y_sat

    Returns
    -------
    dict or list of dict
        A dictionary for a single threshold, or a list of
        dictionaries for multiple thresholds.
    """

    thresholds = _normalize_thresholds(threshold)

    outputs = []

    for current_threshold in thresholds:

        saturation = get_saturation_info(
            results,
            threshold=current_threshold,
        )

        if not saturation["has_saturation"]:

            outputs.append({
                "model": saturation["model"],
                "has_saturation": False,
                "threshold": current_threshold,
            })

            continue

        # ----------------------------------------------------
        # Saturation angles
        # ----------------------------------------------------

        if saturation["model"] == "tanh":

            # Symmetric saturation
            theta_biceps_ideal = saturation["x_sat"]
            theta_biceps_real = saturation["y_sat"]

            theta_triceps_ideal = -saturation["x_sat"]
            theta_triceps_real = -saturation["y_sat"]

        elif saturation["model"] == "logistic4":

            # Upper side = extension = biceps
            theta_biceps_ideal = saturation["x_sat_high"]
            theta_biceps_real = saturation["y_sat_high"]

            # Lower side = flexion = triceps
            theta_triceps_ideal = saturation["x_sat_low"]
            theta_triceps_real = saturation["y_sat_low"]

        else:

            raise ValueError(
                f"Unsupported model: {saturation['model']}"
            )

        # ----------------------------------------------------
        # Pure stimulation patterns
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Ideal durations
        # ----------------------------------------------------

        duration_biceps_ideal = (
            calculate_equivalent_stimulation_duration(
                theta_biceps_ideal,
                Kb,
                Kt,
                biceps_pattern,
            )
        )

        duration_triceps_ideal = (
            calculate_equivalent_stimulation_duration(
                theta_triceps_ideal,
                Kb,
                Kt,
                triceps_pattern,
            )
        )

        # ----------------------------------------------------
        # Real durations
        # ----------------------------------------------------

        duration_biceps_real = (
            calculate_equivalent_stimulation_duration(
                theta_biceps_real,
                Kb,
                Kt,
                biceps_pattern,
            )
        )

        duration_triceps_real = (
            calculate_equivalent_stimulation_duration(
                theta_triceps_real,
                Kb,
                Kt,
                triceps_pattern,
            )
        )

        output = {
            "model": saturation["model"],
            "has_saturation": True,
            "threshold": current_threshold,

            # ------------------------------------------------
            # Saturation coordinates
            # ------------------------------------------------

            "x_sat_biceps":
                theta_biceps_ideal,

            "y_sat_biceps":
                theta_biceps_real,

            "x_sat_triceps":
                theta_triceps_ideal,

            "y_sat_triceps":
                theta_triceps_real,

            # ------------------------------------------------
            # Equivalent durations
            # ------------------------------------------------

            "duration_biceps_ideal":
                duration_biceps_ideal,

            "duration_biceps_real":
                duration_biceps_real,

            "duration_triceps_ideal":
                duration_triceps_ideal,

            "duration_triceps_real":
                duration_triceps_real,

            # ------------------------------------------------
            # Backward-compatible names
            # ------------------------------------------------

            "saturation_angle_biceps":
                theta_biceps_real,

            "saturation_angle_triceps":
                theta_triceps_real,

            "duration_biceps":
                duration_biceps_real,

            "duration_triceps":
                duration_triceps_real,
        }

        # ----------------------------------------------------
        # Logistic-specific information
        # ----------------------------------------------------

        if saturation["model"] == "logistic4":

            output.update({

                "saturation_angle_low":
                    saturation["y_sat_low"],

                "saturation_angle_high":
                    saturation["y_sat_high"],

                "x_saturation_angle_low":
                    saturation["x_sat_low"],

                "x_saturation_angle_high":
                    saturation["x_sat_high"],

                "duration_low_ideal":
                    duration_triceps_ideal,

                "duration_low_real":
                    duration_triceps_real,

                "duration_high_ideal":
                    duration_biceps_ideal,

                "duration_high_real":
                    duration_biceps_real,
            })

        outputs.append(output)

    # Keep the old scalar API when a scalar threshold was used.
    if np.isscalar(threshold):
        return outputs[0]

    return outputs
