import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import t

# ============================================================
# SATURATION CONFIGURATION
# ============================================================

SATURATION_THRESHOLD = 0.95

BICEPS_PATTERN = "001_000"
TRICEPS_PATTERN = "000_001"

# ============================================================
# MODELS
# ============================================================

def linear_model(x, slope):
    """
    Linear model constrained to pass through the origin.

    y = slope * x
    """
    return slope * x


def tanh_sigmoid(x, L, k):
    """
    Hyperbolic tangent sigmoid.

    y = L * tanh(k * x)
    """
    return L * np.tanh(k * x)


def logistic_sigmoid(x, L_min, L_max, k, D0):
    """
    Four-parameter logistic model.

    y = L_min + (L_max - L_min) /
        (1 + exp(-k * (x - D0)))
    """
    return (
        L_min
        + (L_max - L_min)
        / (1 + np.exp(-k * (x - D0)))
    )


# ============================================================
# BASIC METRICS
# ============================================================

def _r2_zero(y_true, y_pred):
    """
    R² for a model evaluated relative to zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true ** 2)

    if ss_tot == 0:
        return np.nan

    return 1 - ss_res / ss_tot


def _r2_mean(y_true, y_pred):
    """
    Conventional R² relative to the mean.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if ss_tot == 0:
        return np.nan

    return 1 - ss_res / ss_tot


def _mae(y_true, y_pred):
    """
    Mean absolute error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.mean(
        np.abs(y_true - y_pred)
    )


def _rmse(y_true, y_pred):
    """
    Root mean squared error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.sqrt(
        np.mean((y_true - y_pred) ** 2)
    )


# ============================================================
# AIC
# ============================================================

def _calculate_aic(y_true, y_pred, n_parameters):
    """
    Calculate AIC for a Gaussian-error model.

    AIC = n * ln(RSS / n) + 2k
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residuals = y_true - y_pred

    rss = np.sum(residuals ** 2)
    n = len(y_true)

    if n == 0 or rss <= 0:
        return np.nan

    return (
        n * np.log(rss / n)
        + 2 * n_parameters
    )


# ============================================================
# SINGLE MODEL FIT
# ============================================================

def _fit_one(
    model_func,
    x,
    y,
    weights=None,
    p0=None,
    bounds=(-np.inf, np.inf),
):
    """
    Fit one model using scipy curve_fit.

    Returns:
        params
        params_std
        r2_zero
        r2_mean
        mae
        rmse
        aic
        y_pred
        ok
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if weights is not None:
        weights = np.asarray(
            weights,
            dtype=float
        )[valid]

        weights = np.where(
            weights <= 0,
            np.nan,
            weights
        )

        valid_weights = np.isfinite(weights)

        x = x[valid_weights]
        y = y[valid_weights]
        weights = weights[valid_weights]

        sigma = 1.0 / weights
    else:
        sigma = None

    try:

        popt, pcov = curve_fit(
            model_func,
            x,
            y,
            p0=p0,
            bounds=bounds,
            sigma=sigma,
            absolute_sigma=False,
            maxfev=100000,
        )

        y_pred = model_func(
            x,
            *popt
        )

        # Standard errors of parameters
        if (
            pcov is not None
            and np.all(np.isfinite(pcov))
        ):
            params_std = np.sqrt(
                np.diag(pcov)
            )
        else:
            params_std = np.full(
                len(popt),
                np.nan
            )

        r2_zero = _r2_zero(
            y,
            y_pred
        )

        r2_mean = _r2_mean(
            y,
            y_pred
        )

        mae = _mae(
            y,
            y_pred
        )

        rmse = _rmse(
            y,
            y_pred
        )

        aic = _calculate_aic(
            y,
            y_pred,
            len(popt)
        )

        return {
            "params": popt,
            "params_std": params_std,
            "r2_zero": r2_zero,
            "r2_mean": r2_mean,
            "mae": mae,
            "rmse": rmse,
            "aic": aic,
            "y_pred": y_pred,
            "x": x,
            "y": y,
            "ok": True,
        }

    except Exception as e:

        print(
            f"Model fitting failed: {e}"
        )

        return {
            "params": None,
            "params_std": None,
            "r2_zero": np.nan,
            "r2_mean": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "aic": np.nan,
            "y_pred": None,
            "x": x,
            "y": y,
            "ok": False,
            "error": str(e),
        }


# ============================================================
# FIT ALL MODELS
# ============================================================

def fit_models(
    x,
    y,
    weights=None,
):
    """
    Fit the three candidate models:

        1. Linear through origin
        2. Tanh
        3. Four-parameter logistic

    Returns:
        dict containing the three model results.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # --------------------------------------------------------
    # LINEAR
    # --------------------------------------------------------

    linear_result = _fit_one(
        linear_model,
        x,
        y,
        weights=weights,
        p0=[1.0],
        bounds=(
            [-np.inf],
            [np.inf]
        ),
    )

    # --------------------------------------------------------
    # TANH
    # --------------------------------------------------------

    # Initial estimate of L:
    # use maximum absolute observed response.
    L0 = np.nanmax(
        np.abs(y)
    )

    if not np.isfinite(L0) or L0 <= 0:
        L0 = 1.0

    # Rough initial estimate for k.
    k0 = 0.1

    tanh_result = _fit_one(
        tanh_sigmoid,
        x,
        y,
        weights=weights,
        p0=[L0, k0],
        bounds=(
            [0.0, 0.0],
            [np.inf, np.inf]
        ),
    )

    # --------------------------------------------------------
    # LOGISTIC 4 PARAMETERS
    # --------------------------------------------------------

    L_min0 = np.nanmin(y)
    L_max0 = np.nanmax(y)

    if not np.isfinite(L_min0):
        L_min0 = -1.0

    if not np.isfinite(L_max0):
        L_max0 = 1.0

    if L_max0 <= L_min0:
        L_max0 = L_min0 + 1.0

    D00 = np.nanmedian(x)

    if not np.isfinite(D00):
        D00 = 0.0

    logistic_result = _fit_one(
        logistic_sigmoid,
        x,
        y,
        weights=weights,
        p0=[
            L_min0,
            L_max0,
            k0,
            D00,
        ],
        bounds=(
            [
                -np.inf,
                -np.inf,
                0.0,
                -np.inf,
            ],
            [
                np.inf,
                np.inf,
                np.inf,
                np.inf,
            ],
        ),
    )

    results = {
        "linear": linear_result,
        "tanh": tanh_result,
        "logistic4": logistic_result,
    }

    _print_summary(results)

    return results


# ============================================================
# BEST MODEL
# ============================================================

def get_best_model(results):
    """
    Select the best valid model according to minimum AIC.

    Returns:
        model name
    """

    valid_models = {}

    for name, result in results.items():

        if not isinstance(result, dict):
            continue

        if not result.get("ok", False):
            continue

        aic = result.get(
            "aic",
            np.nan
        )

        if not np.isfinite(aic):
            continue

        valid_models[name] = aic

    if not valid_models:
        raise ValueError(
            "No valid model with finite AIC."
        )

    return min(
        valid_models,
        key=valid_models.get
    )


# ============================================================
# SATURATION
# ============================================================

def get_saturation_info(
    results,
    best_model_name=None,
    threshold=SATURATION_THRESHOLD,
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
        best_model_name = get_best_model(
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
    threshold=SATURATION_THRESHOLD,
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
        Pattern code, e.g.

            '001_000' -> biceps
            '000_001' -> triceps

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

    try:
        pb = int(parts[0])
        pt = int(parts[1])
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
    threshold=SATURATION_THRESHOLD,
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

    duration_biceps = (
        calculate_equivalent_stimulation_duration(
            theta_biceps,
            Kb,
            Kt,
            "001_000",
        )
    )

    duration_triceps = (
        calculate_equivalent_stimulation_duration(
            theta_triceps,
            Kb,
            Kt,
            "000_001",
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

        "duration_001_000":
            duration_biceps,

        "duration_000_001":
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


# ============================================================
# PRINT SUMMARY
# ============================================================

def _print_summary(results):
    """
    Print model-fitting summary.
    """

    print("\n" + "=" * 80)
    print("MODEL FIT SUMMARY")
    print("=" * 80)

    print(
        f"{'Model':<15}"
        f"{'R2_zero':>12}"
        f"{'R2_mean':>12}"
        f"{'MAE':>12}"
        f"{'RMSE':>12}"
        f"{'AIC':>12}"
    )

    print("-" * 80)

    for name, result in results.items():

        if result.get("ok", False):

            print(
                f"{name:<15}"
                f"{result['r2_zero']:>12.4f}"
                f"{result['r2_mean']:>12.4f}"
                f"{result['mae']:>12.4f}"
                f"{result['rmse']:>12.4f}"
                f"{result['aic']:>12.3f}"
            )

        else:

            print(
                f"{name:<15}"
                f"{'FAILED':>12}"
            )

    try:

        best = get_best_model(
            results
        )

        print("-" * 80)
        print(
            f"Best model according to AIC: "
            f"{best}"
        )

    except ValueError:
        pass

    print("=" * 80 + "\n")


# ============================================================
# GROUP SIGMOID FIT
# ============================================================

def fit_group_sigmoid(
    group_validation_df,
    plot=True,
    output_folder=None,
):
    """
    Fit Linear, Tanh and Logistic4 models to the
    group validation dataset.

    Expected columns:

        ideal_angle
        real_mean
        pattern
        duration

    Returns
    -------
    results : dict
    """

    df = group_validation_df.copy()

    required_columns = [
        "ideal_angle",
        "real_mean",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    df = df[
        np.isfinite(
            df["ideal_angle"]
        )
        & np.isfinite(
            df["real_mean"]
        )
    ].copy()

    x = df[
        "ideal_angle"
    ].to_numpy()

    y = df[
        "real_mean"
    ].to_numpy()

    # --------------------------------------------------------
    # Optional weighting
    # --------------------------------------------------------

    weights = None

    if "weight" in df.columns:

        weights = df[
            "weight"
        ].to_numpy()

    # --------------------------------------------------------
    # Fit models
    # --------------------------------------------------------

    results = fit_models(
        x,
        y,
        weights=weights,
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_model = get_best_model(
        results
    )

    print(
        f"\nSelected model: "
        f"{best_model}"
    )

    # --------------------------------------------------------
    # Saturation
    # --------------------------------------------------------

    saturation = get_saturation_info(
        results,
        best_model_name=best_model,
        threshold=SATURATION_THRESHOLD,
    )

    if saturation["has_saturation"]:

        print(
            "\nSaturation analysis:"
        )

        print(
            f"  Model: "
            f"{saturation['model']}"
        )

        if saturation["model"] == "tanh":

            print(
                f"  x_sat = "
                f"{saturation['x_sat']:.3f}"
            )

            print(
                f"  y_sat = "
                f"{saturation['y_sat']:.3f}°"
            )

        elif (
            saturation["model"]
            == "logistic4"
        ):

            print(
                f"  x_5 = "
                f"{saturation['x_sat_low']:.3f}"
            )

            print(
                f"  x_95 = "
                f"{saturation['x_sat_high']:.3f}"
            )

            print(
                f"  y_5 = "
                f"{saturation['y_sat_low']:.3f}°"
            )

            print(
                f"  y_95 = "
                f"{saturation['y_sat_high']:.3f}°"
            )

    else:

        print(
            "\nNo finite saturation point "
            "for the selected linear model."
        )

    # --------------------------------------------------------
    # Optional model comparison plot
    # --------------------------------------------------------

    if plot:

        if output_folder is not None:
            os.makedirs(
                output_folder,
                exist_ok=True
            )

        _plot_comparison(
            x,
            y,
            results,
            output_folder=output_folder,
        )

    return results


# ============================================================
# COMPARISON PLOT
# ============================================================

def _plot_comparison(
    x,
    y,
    results,
    output_folder=None,
):
    """
    Plot all fitted models.
    """

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.scatter(
        x,
        y,
        alpha=0.65,
        label="Data",
    )

    x_min = np.nanmin(x)
    x_max = np.nanmax(x)

    x_range = np.linspace(
        x_min,
        x_max,
        500,
    )

    styles = {
        "linear": "b--",
        "tanh": "r-",
        "logistic4": "g-.",
    }

    for name, result in results.items():

        if not result.get(
            "ok",
            False
        ):
            continue

        params = result[
            "params"
        ]

        if name == "linear":
            y_range = linear_model(
                x_range,
                *params,
            )

        elif name == "tanh":
            y_range = tanh_sigmoid(
                x_range,
                *params,
            )

        elif name == "logistic4":
            y_range = logistic_sigmoid(
                x_range,
                *params,
            )

        else:
            continue

        ax.plot(
            x_range,
            y_range,
            styles.get(
                name,
                "-"
            ),
            label=name,
        )

    ax.set_xlabel(
        "Ideal angle (deg)"
    )

    ax.set_ylabel(
        "Real angle (deg)"
    )

    ax.legend()
    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    if output_folder is not None:

        path = os.path.join(
            output_folder,
            "model_comparison.png",
        )

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)


# ============================================================
# GROUPED SCATTER
# ============================================================

def plot_grouped_scatter(
    df,
    selected_patterns=None,
    protocol_path=None,
    output_folder=None,
    results=None,
    plot_ci=True,
    saturation_plot=False,
    Kb=None,
    Kt=None,
):
    """
    Generate the grouped validation scatter plot.

    Two versions can be generated:

        grouped_scatter.png
            Original plot without confidence intervals.

        grouped_scatter_CI.png
            Same plot with 95% CI error bars for the individual means,
            if plot_ci=True.

    The best model is selected according to AIC.

    Saturation is model-dependent:

        Tanh:
            saturation is defined at SATURATION_THRESHOLD of the
            positive/negative asymptote.

        Logistic4:
            upper threshold -> biceps / extension
            lower threshold -> triceps / flexion

        Linear:
            no finite saturation point.

    Args:
        df:
            DataFrame with columns:
            'ideal_angle', 'real_mean', 'pattern', 'duration'

        selected_patterns:
            List of patterns to plot. Default: all.

        protocol_path:
            Path to protocol.json.

        output_folder:
            Folder where plots are saved.

        results:
            Dictionary returned by fit_models().

        plot_ci:
            If True, also save a version with 95% CI error bars.

        saturation_plot:
            If True, draw saturation thresholds.

        Kb, Kt:
            Global biceps/triceps coefficients used to convert
            saturation angles into equivalent stimulation durations.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import json

    # ============================================================
    # CHECK RESULTS
    # ============================================================

    if results is None:
        raise ValueError(
            "plot_grouped_scatter requires 'results' from fit_models()."
        )

    # ============================================================
    # COPY DATAFRAME
    # ============================================================

    df = df.copy()

    # ------------------------------------------------------------
    # Filter duration
    # ------------------------------------------------------------

    df = df[df["duration"] != 0].copy()

    # ------------------------------------------------------------
    # Filter patterns
    # ------------------------------------------------------------

    if selected_patterns is not None:
        df = df[df["pattern"].isin(selected_patterns)].copy()

    if df.empty:
        raise ValueError(
            "No data available for plotting after filtering."
        )

    # ============================================================
    # BEST MODEL
    # ============================================================

    best_name = get_best_model(results)
    best_model = results[best_name]

    best_aic = best_model["aic"]

    if best_model is None:
        raise ValueError("No valid model available.")

    # ============================================================
    # SMOOTH X RANGE
    # ============================================================

    x_smooth = np.linspace(
        df["ideal_angle"].min(),
        df["ideal_angle"].max(),
        300,
    )

    # ============================================================
    # TANH PARAMETERS
    # ============================================================

    tanh_params = None

    if results.get("tanh", {}).get("ok", False):
        tanh_params = results["tanh"]["params"]

    # ============================================================
    # FONT
    # ============================================================

    plt.rcParams["font.family"] = "Times New Roman"

    # ============================================================
    # COLORS / LEGEND ORDER FROM PROTOCOL
    # ============================================================

    colors = {}
    legend_order = {}
    protocol = {}

    if protocol_path is not None:

        with open(protocol_path, "r", encoding="utf-8") as f:
            protocol = json.load(f)

        definitions = (
            protocol
            .get("patterns", {})
            .get("definitions", [])
        )

        for definition in definitions:

            text = definition.get("text")
            color = definition.get("color")
            position = definition.get(
                "legend_position",
                999,
            )

            if text and color:

                colors[text] = color
                legend_order[text] = position

    # ============================================================
    # MARKER FOR DURATION
    # ============================================================

    marker_map = {}

    if protocol:

        duration_markers = (
            protocol
            .get("blocks", {})
            .get("duration_markers", {})
        )

        for duration, marker in duration_markers.items():

            try:
                marker_map[int(duration)] = marker

            except (ValueError, TypeError):
                pass

    # ------------------------------------------------------------
    # Fallback markers
    # ------------------------------------------------------------

    if not marker_map:

        marker_map = {
            3: "o",
            6: "^",
            9: "s",
            15: "D",
        }

    # ============================================================
    # FALLBACK COLORS
    # ============================================================

    missing_patterns = (
        set(df["pattern"].unique())
        - set(colors.keys())
    )

    if missing_patterns:

        cmap = plt.cm.tab20

        for i, pattern in enumerate(
            sorted(missing_patterns)
        ):

            colors[pattern] = cmap(
                i / max(1, len(missing_patterns))
            )

            legend_order[pattern] = 999

    # ============================================================
    # BEST MODEL CURVE
    # ============================================================

    if best_name == "linear":

        y_fit = linear_model(
            x_smooth,
            best_model["params"][0],
        )

        style = "b--"

    elif best_name == "tanh":

        y_fit = tanh_sigmoid(
            x_smooth,
            best_model["params"][0],
            best_model["params"][1],
        )

        style = "r-"

    elif best_name == "logistic4":

        p = best_model["params"]

        y_fit = logistic_sigmoid(
            x_smooth,
            p[0],
            p[1],
            p[2],
            p[3],
        )

        style = "g-."

    else:

        raise ValueError(
            f"Unknown model: {best_name}"
        )

    # ============================================================
    # SATURATION INFORMATION
    # ============================================================

    saturation = get_saturation_info(
        results,
        best_model_name=best_name,
        threshold=SATURATION_THRESHOLD,
    )

    # ============================================================
    # SATURATION DURATIONS
    # ============================================================

    durations = None

    if (
        Kb is not None
        and Kt is not None
        and saturation["has_saturation"]
    ):

        durations = calculate_saturation_durations(
            results,
            Kb,
            Kt,
            threshold=SATURATION_THRESHOLD,
        )

    # ============================================================
    # PLOT FUNCTION
    # ============================================================

    def create_plot(
        include_ci=False,
        output_name="grouped_scatter.png",
    ):

        fig, ax = plt.subplots(
            figsize=(12, 7)
        )

        # ========================================================
        # SCATTER
        # ========================================================

        for _, row in df.iterrows():

            x = row["ideal_angle"]
            y = row["real_mean"]
            pattern = row["pattern"]
            duration = row["duration"]

            color = colors.get(
                pattern,
                "black",
            )

            marker = marker_map.get(
                duration,
                "o",
            )

            ax.scatter(
                x,
                y,
                color=color,
                marker=marker,
                s=100,
                alpha=0.9,
                label=f"{pattern} {duration}",
            )

            # ====================================================
            # 95% CI OF THE MEAN
            # ====================================================

            if (
                include_ci
                and "real_std" in df.columns
                and "N" in df.columns
            ):

                if (
                    pd.notna(row["real_std"])
                    and pd.notna(row["N"])
                    and row["N"] > 0
                ):

                    se = (
                        row["real_std"]
                        / np.sqrt(row["N"])
                    )

                    ci = 1.96 * se

                    ax.errorbar(
                        x,
                        y,
                        yerr=ci,
                        fmt="none",
                        ecolor=color,
                        elinewidth=1.5,
                        capsize=4,
                        alpha=0.7,
                    )

        # ========================================================
        # BEST MODEL OVERLAY
        # ========================================================

        d_best = 0.0

        ax.plot(
            x_smooth,
            y_fit,
            style,
            linewidth=2,
            label=(
                f"{best_name} (best AIC)\n"
                f"AIC={best_model['aic']:.1f}\n"
                f"ΔAIC={d_best:.1f}\n"
                f"R²={best_model['r2_mean']:.3f}"
            ),
        )

        # ========================================================
        # LEGEND
        # ========================================================

        handles, labels = (
            ax.get_legend_handles_labels()
        )

        unique = dict(
            zip(labels, handles)
        )

        # --------------------------------------------------------
        # Legend sorting
        # --------------------------------------------------------

        def sort_key(label):

            if "(best AIC)" in label:
                return 9999

            pattern = label.split()[0]

            return legend_order.get(
                pattern,
                999,
            )

        sorted_items = sorted(
            unique.items(),
            key=lambda x: sort_key(x[0]),
        )

        if sorted_items:

            sorted_labels, sorted_handles = zip(
                *sorted_items
            )

            sorted_labels = list(
                sorted_labels
            )

            sorted_handles = list(
                sorted_handles
            )

        else:

            sorted_labels = []
            sorted_handles = []

        # ========================================================
        # SATURATION LINES
        # ========================================================

        if (
            saturation_plot
            and saturation["has_saturation"]
        ):

            # ----------------------------------------------------
            # TANH
            # ----------------------------------------------------

            if best_name == "tanh":

                x_sat = saturation["x_sat"]
                y_sat = saturation["y_sat"]

                # Vertical thresholds
                sat_line_pos = ax.axvline(
                    x_sat,
                    color="red",
                    linestyle=":",
                    alpha=0.7,
                )

                ax.axvline(
                    -x_sat,
                    color="red",
                    linestyle=":",
                    alpha=0.7,
                )

                # Horizontal thresholds
                ax.axhline(
                    y_sat,
                    color="red",
                    linestyle=":",
                    alpha=0.7,
                )

                ax.axhline(
                    -y_sat,
                    color="red",
                    linestyle=":",
                    alpha=0.7,
                )

                sorted_handles.append(
                    sat_line_pos
                )

                sorted_labels.append(
                    f"Saturation "
                    f"{SATURATION_THRESHOLD:.0%} "
                    f"(±{x_sat:.1f}°; "
                    f"±{y_sat:.1f}°)"
                )

            # ----------------------------------------------------
            # LOGISTIC4
            # ----------------------------------------------------

            elif best_name == "logistic4":

                x_low = saturation["x_sat_low"]
                x_high = saturation["x_sat_high"]

                y_low = saturation["y_sat_low"]
                y_high = saturation["y_sat_high"]

                # Lower saturation
                sat_line_low = ax.axvline(
                    x_low,
                    color="red",
                    linestyle=":",
                    alpha=0.7,
                )

                # Upper saturation
                ax.axvline(
                    x_high,
                    color="red",
                    linestyle=":",
                    alpha=0.7,
                )

                # Horizontal thresholds
                ax.axhline(
                    y_low,
                    color="red",
                    linestyle=":",
                    alpha=0.7,
                )

                ax.axhline(
                    y_high,
                    color="red",
                    linestyle=":",
                    alpha=0.7,
                )

                sorted_handles.append(
                    sat_line_low
                )

                sorted_labels.append(
                    "Saturation "
                    f"{1 - SATURATION_THRESHOLD:.0%}"
                    " / "
                    f"{SATURATION_THRESHOLD:.0%}"
                    f" ({y_low:.1f}° / "
                    f"{y_high:.1f}°)"
                )

            # ----------------------------------------------------
            # LINEAR
            # ----------------------------------------------------

            elif best_name == "linear":

                # No finite saturation
                pass

        # ========================================================
        # CENTRAL AXES
        # ========================================================

        ax.axhline(
            0,
            color="gray",
            linewidth=1,
            alpha=0.5,
        )

        ax.axvline(
            0,
            color="gray",
            linewidth=1,
            alpha=0.5,
        )

        # ========================================================
        # SAME X/Y SCALE
        # ========================================================

        x_min = df["ideal_angle"].min()
        x_max = df["ideal_angle"].max()

        # Include CI in limits when plotting CI version
        if (
            include_ci
            and "real_std" in df.columns
            and "N" in df.columns
        ):

            ci_values = (
                1.96
                * df["real_std"]
                / np.sqrt(df["N"])
            )

            y_min = (
                df["real_mean"]
                - ci_values
            ).min()

            y_max = (
                df["real_mean"]
                + ci_values
            ).max()

        else:

            y_min = (
                df["real_mean"]
                - df["real_std"]
            ).min()

            y_max = (
                df["real_mean"]
                + df["real_std"]
            ).max()

        x_max_abs = max(
            abs(x_min),
            abs(x_max),
        )

        y_max_abs = max(
            abs(y_min),
            abs(y_max),
        )

        ax.set_xlim(
            -x_max_abs - x_max_abs * 0.1,
            x_max_abs + x_max_abs * 0.1,
        )

        ax.set_ylim(
            -y_max_abs,
            y_max_abs,
        )

        # ========================================================
        # LABELS
        # ========================================================

        ax.set_xlabel(
            "Ideal mean Δθ (°)",
            fontsize=20,
        )

        ax.set_ylabel(
            "Real mean Δθ (°)",
            fontsize=20,
        )

        # ========================================================
        # LEGEND
        # ========================================================

        if sorted_handles:

            ax.legend(
                sorted_handles,
                sorted_labels,
                bbox_to_anchor=(1.02, 1),
                loc="best",
                fontsize=16,
                frameon=False,
                ncol=2,
            )

        # ========================================================
        # GRID
        # ========================================================

        ax.grid(
            True,
            alpha=0.2,
        )

        fig.tight_layout(
            rect=[0, 0, 1, 0.95]
        )

        # ========================================================
        # SAVE
        # ========================================================

        if output_folder:

            os.makedirs(
                output_folder,
                exist_ok=True,
            )

            out_path = os.path.join(
                output_folder,
                output_name,
            )

            fig.savefig(
                out_path,
                dpi=300,
                bbox_inches="tight",
            )

            print(
                f"[plot_grouped_scatter] "
                f"Saved: {out_path}"
            )

        plt.show()
        plt.close(fig)

    # ============================================================
    # CREATE ORIGINAL PLOT
    # ============================================================

    create_plot(
        include_ci=False,
        output_name="grouped_scatter.png",
    )

    # ============================================================
    # CREATE CI VERSION
    # ============================================================

    if plot_ci:

        create_plot(
            include_ci=True,
            output_name="grouped_scatter_CI.png",
        )

    # ============================================================
    # PRINT SATURATION INFORMATION
    # ============================================================

    print("\n" + "=" * 80)
    print("SATURATION ANALYSIS")
    print("=" * 80)

    print(
        f"Best model: {best_name}"
    )

    print(
        f"Saturation threshold: "
        f"{SATURATION_THRESHOLD:.0%}"
    )

    # ------------------------------------------------------------
    # Linear
    # ------------------------------------------------------------

    if not saturation["has_saturation"]:

        print(
            "No finite saturation point "
            "for the linear model."
        )

    # ------------------------------------------------------------
    # Tanh
    # ------------------------------------------------------------

    elif best_name == "tanh":

        print("\nTanh saturation:")

        print(
            f"  Positive angle: "
            f"{saturation['y_sat']:.3f}°"
        )

        print(
            f"  Negative angle: "
            f"{-saturation['y_sat']:.3f}°"
        )

        print(
            f"  Positive x: "
            f"{saturation['x_sat']:.3f}"
        )

        print(
            f"  Negative x: "
            f"{-saturation['x_sat']:.3f}"
        )

    # ------------------------------------------------------------
    # Logistic4
    # ------------------------------------------------------------

    elif best_name == "logistic4":

        print("\nLogistic saturation:")

        print(
            f"  Lower response "
            f"({1 - SATURATION_THRESHOLD:.0%}): "
            f"{saturation['y_sat_low']:.3f}°"
        )

        print(
            f"  Upper response "
            f"({SATURATION_THRESHOLD:.0%}): "
            f"{saturation['y_sat_high']:.3f}°"
        )

        print(
            f"  x at lower threshold: "
            f"{saturation['x_sat_low']:.3f}"
        )

        print(
            f"  x at upper threshold: "
            f"{saturation['x_sat_high']:.3f}"
        )

    # ============================================================
    # EQUIVALENT STIMULATION DURATIONS
    # ============================================================

    if durations is not None:

        print(
            "\nEquivalent stimulation durations:"
        )

        print(
            f"  {BICEPS_PATTERN} "
            f"(biceps / extension): "
            f"{durations['duration_001_000']:.3f} s"
        )

        print(
            f"  {TRICEPS_PATTERN} "
            f"(triceps / flexion): "
            f"{durations['duration_000_001']:.3f} s"
        )

    print("=" * 80 + "\n")

    return saturation


# ============================================================
# OPTIONAL PLOT WITH LABELS
# ============================================================

def _plot_comparison_with_labels(
    x,
    y,
    results,
    output_folder=None,
):
    """
    Plot all models with model names and AIC values.
    """

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.scatter(
        x,
        y,
        alpha=0.65,
        label="Data",
    )

    x_range = np.linspace(
        np.nanmin(x),
        np.nanmax(x),
        500,
    )

    styles = {
        "linear": "b--",
        "tanh": "r-",
        "logistic4": "g-.",
    }

    for name, result in results.items():

        if not result.get(
            "ok",
            False
        ):
            continue

        params = result[
            "params"
        ]

        if name == "linear":

            y_range = linear_model(
                x_range,
                *params,
            )

        elif name == "tanh":

            y_range = tanh_sigmoid(
                x_range,
                *params,
            )

        elif name == "logistic4":

            y_range = logistic_sigmoid(
                x_range,
                *params,
            )

        else:
            continue

        aic = result[
            "aic"
        ]

        ax.plot(
            x_range,
            y_range,
            styles.get(
                name,
                "-"
            ),
            linewidth=2,
            label=(
                f"{name} "
                f"(AIC={aic:.2f})"
            ),
        )

    best_model = get_best_model(
        results
    )

    ax.set_title(
        f"Model comparison "
        f"(best: {best_model})"
    )

    ax.set_xlabel(
        "Ideal angle (deg)"
    )

    ax.set_ylabel(
        "Real angle (deg)"
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    if output_folder is not None:

        path = os.path.join(
            output_folder,
            "model_comparison_labels.png",
        )

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)