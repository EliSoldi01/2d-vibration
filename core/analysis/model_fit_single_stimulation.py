import os
import json
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .model_fit import (
    fit_models,
    calculate_saturation_point,
    calculate_saturation_durations,
    linear_model,
    tanh_sigmoid,
    logistic_sigmoid,
)

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

SELECTED_PATTERNS = [
    "001_000",
    "011_000",
    "111_000",
    "000_001",
    "000_011",
    "000_111",
    "001_001",       # control
]

SATURATION_THRESHOLD = 0.95

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
# FILTER
# ============================================================

def filter_single_stimulation_patterns(df):
    """
    Keep only the patterns selected for the
    single-stimulation analysis.

    Duration == 0 rows are excluded.
    """

    required_columns = [
        "pattern",
        "duration",
        "ideal_angle",
        "real_mean",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df_filtered = df.copy()

    # Selected patterns only
    df_filtered = df_filtered[
        df_filtered["pattern"].isin(
            SELECTED_PATTERNS
        )
    ].copy()

    # Remove zero-duration/control baseline
    df_filtered = df_filtered[
        df_filtered["duration"] != 0
    ].copy()

    if df_filtered.empty:
        raise ValueError(
            "No data remain after filtering "
            "selected stimulation patterns."
        )

    return df_filtered


# ============================================================
# MODEL FIT
# ============================================================

def fit_single_stimulation_model(df):
    """
    Filter the dataframe and fit the three models used
    in model_fit.py:

        - Linear
        - Tanh
        - Logistic4

    Returns
    -------
    df_filtered
        Filtered dataframe used for the analysis.

    results
        Dictionary containing the fitted models.

    best_name
        Name of the model with minimum AIC.

    best_model
        Dictionary containing the best model results.
    """

    df_filtered = filter_single_stimulation_patterns(df)

    # Remove invalid dependent-variable values
    mask = (
        np.isfinite(
            df_filtered["ideal_angle"].to_numpy(
                dtype=float
            )
        )
        &
        np.isfinite(
            df_filtered["real_mean"].to_numpy(
                dtype=float
            )
        )
    )

    df_filtered = df_filtered.loc[
        mask
    ].copy()

    if len(df_filtered) < 5:
        raise ValueError(
            "Too few valid points for model fitting."
        )

    x = df_filtered[
        "ideal_angle"
    ].to_numpy(dtype=float)

    y = df_filtered[
        "real_mean"
    ].to_numpy(dtype=float)

    # Same logic as the original fit:
    # if no specific weights are available, all points
    # receive equal weight.
    weights = np.ones(
        len(df_filtered),
        dtype=float
    )

    results = fit_models(
        x,
        y,
        weights=weights,
    )

    best_name = get_best_model(results)
    best_model = results[best_name]

    if best_name is None:
        raise RuntimeError(
            "No valid model could be fitted."
        )

    return (
        df_filtered,
        results,
        best_name,
        best_model,
    )


# ============================================================
# SATURATION
# ============================================================

def calculate_new_saturation(
    results,
    Kb=None,
    Kt=None,
):
    """
    Calculate saturation using the same logic as the
    original model_fit.py.

    IMPORTANT:
    Saturation is defined from the Tanh model, independently
    of which model has the minimum AIC.

    For the Tanh:
        x_sat = ± x_sat
        y_sat = ± y_sat

    This therefore represents both:
        - extension
        - flexion
    """

    tanh_result = results.get(
        "tanh",
        {}
    )

    if not tanh_result.get(
        "ok",
        False
    ):

        logger.warning(
            "[single_stimulation] "
            "Tanh fit unavailable; "
            "saturation cannot be calculated."
        )

        return {
            "x_sat": None,
            "y_sat": None,
            "durations": None,
        }

    # --------------------------------------------------------
    # Same calculation used by original model_fit.py
    # --------------------------------------------------------

    saturation = calculate_saturation_point(
        results,
        threshold=SATURATION_THRESHOLD,
    )

    if saturation is None:

        return {
            "x_sat": None,
            "y_sat": None,
            "durations": None,
        }

    x_sat, y_sat = saturation

    # --------------------------------------------------------
    # Equivalent stimulation durations
    # --------------------------------------------------------

    durations = None

    if (
        Kb is not None
        and Kt is not None
    ):

        try:

            durations = (
                calculate_saturation_durations(
                    results,
                    Kb,
                    Kt,
                    threshold=SATURATION_THRESHOLD,
                )
            )

        except Exception as exc:

            logger.warning(
                "[single_stimulation] "
                "Could not calculate equivalent "
                f"stimulation durations: {exc}"
            )

    return {
        "x_sat": x_sat,
        "y_sat": y_sat,
        "durations": durations,
    }


# ============================================================
# PROTOCOL INFORMATION
# ============================================================

def _load_protocol_info(
    protocol_path
):
    """
    Load pattern colors, legend positions and duration
    markers from protocol.json.

    This follows the same structure used in the original
    plot_grouped_scatter().
    """

    colors = {}
    legend_order = {}
    marker_map = {}

    if protocol_path is None:
        return (
            colors,
            legend_order,
            marker_map,
        )

    with open(
        protocol_path,
        "r"
    ) as f:

        protocol = json.load(f)

    # --------------------------------------------------------
    # Pattern colors and legend order
    # --------------------------------------------------------

    definitions = (
        protocol
        .get("patterns", {})
        .get("definitions", [])
    )

    for d in definitions:

        text = d.get("text")
        color = d.get("color")
        pos = d.get(
            "legend_position",
            999
        )

        if text and color:

            colors[text] = color
            legend_order[text] = pos

    # --------------------------------------------------------
    # Duration markers
    # --------------------------------------------------------

    duration_markers = (
        protocol
        .get("blocks", {})
        .get("duration_markers", {})
    )

    for duration, marker in (
        duration_markers.items()
    ):

        try:
            marker_map[int(duration)] = marker
        except (ValueError, TypeError):
            continue

    # Same fallback as original
    if not marker_map:

        marker_map = {
            3: "o",
            6: "^",
            9: "s",
            15: "D",
        }

    return (
        colors,
        legend_order,
        marker_map,
    )


# ============================================================
# PLOT
# ============================================================

def plot_single_stimulation_model(
    df,
    results,
    best_name,
    saturation,
    protocol_path=None,
    output_folder=None,
    plot_ci=True,
):
    """
    Plot the single-stimulation model using the same
    color/marker/legend logic as plot_grouped_scatter().

    Color:
        stimulation pattern

    Marker:
        stimulation duration

    Model:
        best model according to AIC

    Saturation:
        Tanh 95% threshold, shown on both sides
        (extension and flexion).
    """

    plt.rcParams[
        "font.family"
    ] = "Times New Roman"

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    # ========================================================
    # PROTOCOL
    # ========================================================

    (
        colors,
        legend_order,
        marker_map,
    ) = _load_protocol_info(
        protocol_path
    )

    # ========================================================
    # FALLBACK COLORS
    # ========================================================

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
                i / max(
                    1,
                    len(missing_patterns)
                )
            )

            legend_order[pattern] = 999

    # ========================================================
    # X RANGE
    # ========================================================

    x_smooth = np.linspace(
        df["ideal_angle"].min(),
        df["ideal_angle"].max(),
        300,
    )

    # ========================================================
    # SCATTER + CI
    # ========================================================

    for _, row in df.iterrows():

        x = row["ideal_angle"]
        y = row["real_mean"]

        pattern = row["pattern"]
        duration = row["duration"]

        ax.scatter(
            x,
            y,
            color=colors.get(
                pattern,
                "black"
            ),
            marker=marker_map.get(
                int(duration),
                "o"
            ),
            s=100,
            alpha=0.9,
            label=(
                f"{pattern} "
                f"{duration:g}"
            ),
        )

        # ----------------------------------------------------
        # 95% CI
        # ----------------------------------------------------

        if (
            plot_ci
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
                    ecolor=colors.get(
                        pattern,
                        "black"
                    ),
                    elinewidth=1.5,
                    capsize=4,
                    alpha=0.7,
                )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_model = results[
        best_name
    ]

    valid_aics = {
        name: result["aic"]
        for name, result
        in results.items()
        if result.get("ok", False)
        and not np.isnan(
            result["aic"]
        )
    }

    best_aic = min(
        valid_aics.values()
    )

    d_best = (
        best_model["aic"]
        - best_aic
    )

    if best_name == "linear":

        y_fit = linear_model(
            x_smooth,
            best_model["params"][0]
        )

        style = "b--"

    elif best_name == "tanh":

        y_fit = tanh_sigmoid(
            x_smooth,
            best_model["params"][0],
            best_model["params"][1]
        )

        style = "r-"

    elif best_name == "logistic4":

        p = best_model["params"]

        y_fit = logistic_sigmoid(
            x_smooth,
            p[0],
            p[1],
            p[2],
            p[3]
        )

        style = "g-."

    else:

        raise ValueError(
            f"Unknown model: {best_name}"
        )

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
    # LEGEND ORDER
    # ========================================================

    handles, labels = (
        ax.get_legend_handles_labels()
    )

    unique = dict(
        zip(
            labels,
            handles
        )
    )

    def sort_key(label):

        if "(best AIC)" in label:
            return 9999

        pattern = label.split()[0]

        return legend_order.get(
            pattern,
            999
        )

    sorted_items = sorted(
        unique.items(),
        key=lambda item:
            sort_key(item[0])
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
    # SATURATION
    # ========================================================

    if (
        saturation is not None
        and saturation.get("x_sat") is not None
        and saturation.get("y_sat") is not None
    ):

        x_sat = saturation[
            "x_sat"
        ]

        y_sat = saturation[
            "y_sat"
        ]

        # ----------------------------------------------------
        # Vertical thresholds
        # ----------------------------------------------------

        sat_line = ax.axvline(
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

        # ----------------------------------------------------
        # Horizontal thresholds
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Legend
        # ----------------------------------------------------

        sorted_handles.append(
            sat_line
        )

        sorted_labels.append(
            "Saturation 95% "
            f"(±{x_sat:.1f}°; "
            f"±{y_sat:.1f}°)"
        )

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
    # SAME SCALE LOGIC AS ORIGINAL
    # ========================================================

    x_min = df[
        "ideal_angle"
    ].min()

    x_max = df[
        "ideal_angle"
    ].max()

    if (
        "real_std" in df.columns
        and df["real_std"].notna().any()
    ):

        y_min = (
            df["real_mean"]
            - df["real_std"]
        ).min()

        y_max = (
            df["real_mean"]
            + df["real_std"]
        ).max()

    else:

        y_min = df[
            "real_mean"
        ].min()

        y_max = df[
            "real_mean"
        ].max()

    x_max_abs = max(
        abs(x_min),
        abs(x_max)
    )

    y_max_abs = max(
        abs(y_min),
        abs(y_max)
    )

    ax.set_xlim(
        -x_max_abs
        - x_max_abs * 0.1,
        x_max_abs
        + x_max_abs * 0.1,
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

    ax.legend(
        sorted_handles,
        sorted_labels,
        bbox_to_anchor=(1.02, 1),
        loc="best",
        fontsize=16,
        frameon=False,
        ncol=2,
    )

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
            exist_ok=True
        )

        out_path = os.path.join(
            output_folder,
            "single_stimulation_model_fit.png"
        )

        fig.savefig(
            out_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            "[single_stimulation] "
            f"Saved: {out_path}"
        )

    plt.show()
    plt.close(fig)

    return fig, ax