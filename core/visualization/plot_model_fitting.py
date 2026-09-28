import core.analysis.model_fitting as mf
import core.analysis.saturation as sat
import config as cfg
import math

def plot_comparison(
    x,
    y,
    results,
    output_folder=cfg.MODEL_FITTING_PATH,
):
    """
    Plot all fitted models together with the experimental data.

    The legend reports R², AIC, and ΔAIC for each model.
    Experimental data points are shown in black.
    """

    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    # ========================================================
    # EXPERIMENTAL DATA
    # ========================================================

    ax.scatter(
        x,
        y,
        color="black",
        alpha=0.65,
        label="Data",
    )

    # ========================================================
    # X RANGE
    # ========================================================

    x_min = np.nanmin(x)
    x_max = np.nanmax(x)

    x_range = np.linspace(
        x_min,
        x_max,
        500,
    )

    # ========================================================
    # AIC / DELTA AIC
    # ========================================================

    valid_aics = [
        result["aic"]
        for result in results.values()
        if result.get("ok", False)
        and np.isfinite(result.get("aic", np.nan))
    ]

    min_aic = (
        min(valid_aics)
        if valid_aics
        else np.nan
    )

    # ========================================================
    # MODEL STYLES
    # ========================================================

    styles = {
        "linear": "b--",
        "tanh": "r-",
        "logistic4": "g-.",
    }

    # ========================================================
    # FITTED MODELS
    # ========================================================

    for name, result in results.items():

        if not result.get(
            "ok",
            False
        ):
            continue

        params = result["params"]

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        if name == "linear":

            y_range = mf.linear_model(
                x_range,
                *params,
            )

        elif name == "tanh":

            y_range = mf.tanh_sigmoid(
                x_range,
                *params,
            )

        elif name == "logistic4":

            y_range = mf.logistic_sigmoid(
                x_range,
                *params,
            )

        else:
            continue

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        r2 = result.get(
            "r2_zero",
            np.nan
        )

        aic = result.get(
            "aic",
            np.nan
        )

        if np.isfinite(aic) and np.isfinite(min_aic):
            delta_aic = aic - min_aic
        else:
            delta_aic = np.nan

        # ----------------------------------------------------
        # Legend label
        # ----------------------------------------------------

        label = (
            f"{name} "
            f"($R^2$={r2:.3f}, "
            f"AIC={aic:.2f}, "
            f"$\\Delta$AIC={delta_aic:.2f})"
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        ax.plot(
            x_range,
            y_range,
            styles.get(
                name,
                "-"
            ),
            label=label,
        )

    # ========================================================
    # AXES
    # ========================================================

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

    # ========================================================
    # SAVE
    # ========================================================

    if output_folder is not None:

        output_folder = Path(
            output_folder
        )

        output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = (
            output_folder
            / "model_comparison.png"
        )

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"Model comparison saved to: {path}"
        )

    plt.close(fig)

def plot_best_model(
    df,
    protocol_path=None,
    output_folder=cfg.MODEL_FITTING_PATH,
    results=None,
    selected_patterns=None,
    plot_confidence_band=False,
    plot_saturation_points=False,
    confidence=0.95,
    output_name="best_model.png",
    forced_best_model=None
):
    """
    Plot experimental group means (± SD) together with
    the AIC-selected best-fitting model.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing:
            ideal_angle
            real_mean
            real_std
            pattern
            duration

    protocol_path : str, optional
        Path to protocol.json. Used to retrieve pattern
        colors, legend order, and duration markers.

    output_folder : str, optional
        Folder where the figure is saved.

    results : dict
        Results returned by fit_models().

    selected_patterns : list, optional
        Patterns to include in the plot.

    plot_confidence_band : bool, optional
        If True, plot the confidence band around
        the best-fitting model. Default is False.

    confidence : float, optional
        Confidence level of the model confidence band.
        Default is 0.95.

    output_name : str
        Name of the output figure.
    """

    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    # ========================================================
    # CHECK RESULTS
    # ========================================================

    if results is None:
        raise ValueError(
            "plot_best_model requires "
            "'results' from fit_models()."
        )

    # ========================================================
    # COPY / FILTER DATA
    # ========================================================

    df = df.copy()

    # Duration 0 is not part of the validation plot.
    df = df[
        df["duration"] != 0
    ].copy()

    if selected_patterns is not None:
        df = df[
            df["pattern"].isin(
                selected_patterns
            )
        ].copy()

    if df.empty:
        raise ValueError(
            "No data available for plotting "
            "after filtering."
        )

    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(14, 10)
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    if forced_best_model is not None:
        best_name = forced_best_model
    else:
        best_name = mf.get_best_model(
            results
        )

    best_model = results[
        best_name
    ]

    # ========================================================
    # SATURATION POINT
    # ========================================================

    saturation = None

    if plot_saturation_points:

        saturation = sat.get_saturation_info(
            results,
            best_model_name=best_name,
            threshold=cfg.SATURATION_THRESHOLD,
        )

    # ========================================================
    # SMOOTH X RANGE
    # ========================================================

    x_smooth = np.linspace(
        df["ideal_angle"].min(),
        df["ideal_angle"].max(),
        500,
    )

    # ========================================================
    # BEST MODEL CURVE
    # ========================================================

    if best_name == "linear":

        y_fit = mf.linear_model(
            x_smooth,
            *best_model["params"],
        )

        style = "k-"

    elif best_name == "tanh":

        y_fit = mf.tanh_sigmoid(
            x_smooth,
            *best_model["params"],
        )

        style = "k-"

    elif best_name == "logistic4":

        y_fit = mf.logistic_sigmoid(
            x_smooth,
            *best_model["params"],
        )

        style = "k-"

    else:

        raise ValueError(
            f"Unknown model: {best_name}"
        )

    # ========================================================
    # SATURATION POINT
    # ========================================================

    if (
        plot_saturation_points
        and saturation is not None
        and saturation["has_saturation"]
    ):

        ax.scatter(
            saturation["x_sat"],
            saturation["y_sat"],
            marker="x",
            s=120,
            color="red",
            zorder=7,
            label=(
                f"Saturation onset "
                f"({int(saturation['threshold'] * 100)}%)"
            ),
        )

        ax.scatter(
            -saturation["x_sat"],
            -saturation["y_sat"],
            marker="x",
            s=120,
            color="red",
            zorder=7
        )

    # ========================================================
    # CONFIDENCE BAND
    # ========================================================

    if plot_confidence_band:

        n_obs = len(
            best_model["x"]
        )

        n_params = len(
            best_model["params"]
        )

        df_resid = (
            n_obs
            - n_params
        )

        (
            _,
            lower_band,
            upper_band,
        ) = mf.model_confidence_band(
            model_name=best_name,
            x=x_smooth,
            params=best_model["params"],
            pcov=best_model["pcov"],
            confidence=confidence,
            df_resid=df_resid,
        )

    # ========================================================
    # FONT
    # ========================================================

    plt.rcParams[
        "font.family"
    ] = "Times New Roman"

    # ========================================================
    # COLORS / LEGEND ORDER / MARKERS
    # ========================================================

    colors = {}
    legend_order = {}
    marker_map = {}

    protocol = {}

    if protocol_path is not None:

        with open(
            protocol_path,
            "r",
            encoding="utf-8",
        ) as f:

            protocol = json.load(f)

        # ----------------------------------------------------
        # Pattern colors and legend order
        # ----------------------------------------------------

        definitions = (
            protocol
            .get("patterns", {})
            .get("definitions", [])
        )

        for definition in definitions:

            text = definition.get(
                "text"
            )

            color = definition.get(
                "color"
            )

            position = definition.get(
                "legend_position",
                999,
            )

            if text and color:

                colors[text] = color

                legend_order[
                    text
                ] = position

        # ----------------------------------------------------
        # Duration markers
        # ----------------------------------------------------

        duration_markers = (
            protocol
            .get("blocks", {})
            .get(
                "duration_markers",
                {}
            )
        )

        for duration, marker in (
            duration_markers.items()
        ):

            try:

                marker_map[
                    int(duration)
                ] = marker

            except (
                ValueError,
                TypeError,
            ):

                pass

    # ========================================================
    # FALLBACK MARKERS
    # ========================================================

    if not marker_map:

        marker_map = {
            3: "o",
            6: "^",
            9: "s",
            15: "D",
        }

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
                    len(missing_patterns),
                )
            )

            legend_order[
                pattern
            ] = 999

    # ========================================================
    # CONFIDENCE BAND
    # ========================================================

    if plot_confidence_band:

        ax.fill_between(
            x_smooth,
            lower_band,
            upper_band,
            color="#3d3d3d44",
            alpha=0.20,
            label=(
                f"{int(confidence * 100)}% CI"
            ),
            zorder=2,
        )

    # ========================================================
    # EXPERIMENTAL DATA: MEAN ± SD
    # ========================================================

    for _, row in df.iterrows():

        x = row[
            "ideal_angle"
        ]

        y = row[
            "real_mean"
        ]

        pattern = row[
            "pattern"
        ]

        duration = row[
            "duration"
        ]

        color = colors.get(
            pattern,
            "black",
        )

        marker = marker_map.get(
            duration,
            "o",
        )

        # ----------------------------------------------------
        # Mean
        # ----------------------------------------------------

        ax.scatter(
            x,
            y,
            color=color,
            marker=marker,
            s=100,
            alpha=0.9,
            label=(
                f"{pattern} "
                f"{duration}"
            ),
            zorder=4,
        )

        # ----------------------------------------------------
        # ± SD
        # ----------------------------------------------------

        sd = row[
            "real_std"
        ]

        if np.isfinite(sd):

            ax.errorbar(
                x,
                y,
                yerr=sd,
                fmt="none",
                ecolor=color,
                elinewidth=1.5,
                capsize=4,
                alpha=0.7,
                zorder=5,
            )

    # ========================================================
    # BEST MODEL
    # ========================================================

    ax.plot(
        x_smooth,
        y_fit,
        style,
        linewidth=2.5,
        label=(
            f"{best_name} "
            f"R²={best_model['r2_mean']:.3f}"
        ),
        zorder=6,
    )

    # ========================================================
    # LEGEND
    # ========================================================

    handles, labels = (
        ax.get_legend_handles_labels()
    )

    unique = dict(
        zip(
            labels,
            handles,
        )
    )

    def sort_key(label):

        if label.startswith(
            best_name
        ):
            return 9998
        
        if label.startswith("Saturation"):
            return 9999

        if label.endswith("CI"):

            return 10000

        pattern = label.split()[0]

        return legend_order.get(
            pattern,
            999,
        )

    sorted_items = sorted(
        unique.items(),
        key=lambda item:
            sort_key(item[0]),
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
    # CENTRAL AXES
    # ========================================================

    ax.axhline(
        0,
        color="gray",
        linewidth=1,
        alpha=0.5,
        zorder=0,
    )

    ax.axvline(
        0,
        color="gray",
        linewidth=1,
        alpha=0.5,
        zorder=0,
    )

    # ========================================================
    # AXIS LIMITS
    # ========================================================

    all_values = []

    # Experimental x values
    all_values.extend(
        df["ideal_angle"]
        .dropna()
        .to_numpy()
    )

    # Experimental means
    all_values.extend(
        df["real_mean"]
        .dropna()
        .to_numpy()
    )

    # Experimental SD intervals
    valid_sd = (
        df["real_mean"].notna()
        & df["real_std"].notna()
    )

    if valid_sd.any():

        means = (
            df.loc[
                valid_sd,
                "real_mean"
            ].to_numpy()
        )

        stds = (
            df.loc[
                valid_sd,
                "real_std"
            ].to_numpy()
        )

        all_values.extend(
            (means - stds).tolist()
        )

        all_values.extend(
            (means + stds).tolist()
        )

    all_values = np.asarray(
        all_values,
        dtype=float,
    )

    all_values = all_values[
        np.isfinite(all_values)
    ]

    if len(all_values) == 0:

        axis_max = 1.0

    else:

        axis_max = np.max(
            np.abs(all_values)
        )

    padding = (
        0.10 * axis_max
    )

    axis_limit = (
        axis_max + padding
    )

    # Same limits for X and Y
    ax.set_xlim(
        -axis_limit,
        axis_limit,
    )

    ax.set_ylim(
        -axis_limit,
        axis_limit,
    )

    # Equal aspect ratio
    ax.set_aspect(
        "equal",
        adjustable="box",
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
        num_items = len(sorted_labels)
        max_rows = 18  # Numero massimo di righe desiderato
        n_cols = math.ceil(num_items / max_rows)

        ax.legend(
            sorted_handles,
            sorted_labels,
            bbox_to_anchor=(1.02, 0.5),
            loc="center left",
            fontsize=16,
            frameon=False,
            ncol=n_cols,
            columnspacing=1.2,
            handletextpad=0.5,
            borderaxespad=0,
        )

    # ========================================================
    # GRID
    # ========================================================

    ax.grid(
        True,
        alpha=0.2,
        zorder=0,
    )

    # ========================================================
    # LAYOUT
    # ========================================================

    fig.subplots_adjust(
        left=0.10,
        right=0.72,
        bottom=0.12,
        top=0.95,
    )

    # ========================================================
    # SAVE
    # ========================================================

    if output_folder is not None:

        output_folder = Path(
            output_folder
        )

        output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        out_path = (
            output_folder
            / output_name
        )

        fig.savefig(
            out_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            "[plot_best_model] "
            f"Saved: {out_path}"
        )

    plt.show()
    plt.close(fig)

