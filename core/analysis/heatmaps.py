import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


# ============================================================
# Configuration helpers
# ============================================================

def get_pattern_colors(protocol, patterns=None):
    """
    Return pattern colors and ordered pattern list.

    Colors and legend positions are defined in the protocol.

    Parameters
    ----------
    protocol : dict
        Loaded protocol dictionary.

    patterns : iterable, optional
        Subset of pattern pairs to include.
        If None, all protocol patterns are used.

    Returns
    -------
    pattern_colors : dict
        Mapping pattern_pair -> color.

    ordered_patterns : list
        Pattern pairs ordered according to legend_position.
    """
    definitions = protocol["patterns"]["definitions"]

    color_mapping = {
        pattern["text"]: pattern.get("color", "#000000")
        for pattern in definitions
    }

    legend_mapping = {
        pattern["text"]: pattern.get("legend_position", 999)
        for pattern in definitions
    }

    if patterns is not None:
        patterns = set(patterns)

        color_mapping = {
            pattern: color
            for pattern, color in color_mapping.items()
            if pattern in patterns
        }

        legend_mapping = {
            pattern: position
            for pattern, position in legend_mapping.items()
            if pattern in patterns
        }

    ordered_patterns = sorted(
        color_mapping.keys(),
        key=lambda pattern: legend_mapping[pattern]
    )

    return color_mapping, ordered_patterns


def get_metric_scale(protocol, metric):
    """
    Return the maximum value used to scale the metric.

    Currently the scale is taken from the protocol.

    Parameters
    ----------
    protocol : dict
        Loaded protocol dictionary.

    metric : str
        Name of the metric to plot.

    Returns
    -------
    float
        Maximum metric value.
    """
    metric_values = protocol["scales"][metric]["values"]

    return max(metric_values)


def get_start_position(protocol):
    """
    Return the starting grid cell from the protocol.
    """
    return tuple(protocol["grid"]["start_cell"])


def get_durations(protocol, df, durations=None):
    """
    Determine which durations should be plotted.

    Parameters
    ----------
    protocol : dict
        Loaded protocol dictionary.

    df : pandas.DataFrame
        Experimental data.

    durations : iterable, optional
        Explicit subset of durations to plot.

        If None, all durations specified by the protocol
        and present in the dataframe are used.

    Returns
    -------
    list
        Sorted list of durations.
    """
    protocol_durations = protocol["blocks"]["durations"]
    available_durations = set(df["duration"].dropna().unique())

    if durations is None:
        selected_durations = [
            duration
            for duration in protocol_durations
            if duration in available_durations
        ]
    else:
        selected_durations = [
            duration
            for duration in durations
            if duration in protocol_durations
            and duration in available_durations
        ]

    return sorted(selected_durations)


# ============================================================
# Drawing helpers
# ============================================================

def draw_circles(
    ax,
    df,
    metric,
    pattern_colors,
    scale_max,
    start_pos
):
    """
    Draw one circle for each row of the dataframe.
    """
    for _, row in df.iterrows():

        x = row["x"]
        y = row["y"]
        pattern = row["pattern_pair"]

        if pattern not in pattern_colors:
            continue

        radius = 0.5 * (row[metric] / scale_max)

        ax.add_patch(
            plt.Circle(
                (x, y),
                radius,
                color=pattern_colors[pattern],
                fill=False,
                linewidth=2,
                alpha=0.7
            )
        )

    ax.scatter(
        *start_pos,
        c="red",
        s=120,
        marker="x",
        linewidths=1
    )


def setup_axis(
    ax,
    protocol
):
    """
    Configure the grid and axes according to the protocol.
    """
    xlim = (1, protocol["grid"]["x"])
    ylim = (1, protocol["grid"]["y"])

    ax.set_xlim(
        xlim[0] - 0.5,
        xlim[1] + 0.5
    )

    ax.set_ylim(
        ylim[0] - 0.5,
        ylim[1] + 0.5
    )

    ax.set_xticks(
        np.arange(
            xlim[0],
            xlim[1] + 1
        )
    )

    ax.set_yticks(
        np.arange(
            ylim[0],
            ylim[1] + 1
        )
    )

    ax.invert_yaxis()
    ax.set_aspect("equal")

    for x in range(
        xlim[0],
        xlim[1] + 1
    ):
        ax.vlines(
            x - 0.5,
            ylim[0] - 0.5,
            ylim[1] + 0.5,
            color="lightgrey",
            linewidth=0.8
        )

    for y in range(
        ylim[0],
        ylim[1] + 1
    ):
        ax.hlines(
            y - 0.5,
            xlim[0] - 0.5,
            xlim[1] + 0.5,
            color="lightgrey",
            linewidth=0.8
        )


def draw_pattern_legend(
    fig,
    pattern_colors,
    ordered_patterns,
    fontsize=13
):
    """
    Draw a global pattern legend.

    The layout adapts to the number of patterns.
    """
    patches = {
        pattern: mpatches.Patch(
            color=pattern_colors[pattern],
            label=pattern
        )
        for pattern in ordered_patterns
    }

    n_patterns = len(ordered_patterns)

    if n_patterns == 0:
        return

    if n_patterns == 11:
        rows = [
            ordered_patterns[:4],
            ordered_patterns[4:7],
            ordered_patterns[7:11]
        ]

        y_positions = [
            0.93,
            0.89,
            0.85
        ]

    else:
        half = int(np.ceil(n_patterns / 2))

        rows = [
            ordered_patterns[:half],
            ordered_patterns[half:]
        ]

        y_positions = [
            0.93,
            0.88
        ]

    for row, y_position in zip(
        rows,
        y_positions
    ):
        if not row:
            continue

        fig.legend(
            handles=[
                patches[pattern]
                for pattern in row
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, y_position),
            ncol=len(row),
            frameon=False,
            fontsize=fontsize
        )


# ============================================================
# Basic heatmap
# ============================================================

def plot_heatmap(
    df,
    filename,
    title,
    protocol,
    metric="vividness"
):
    """
    Plot a single heatmap.

    Parameters
    ----------
    df : pandas.DataFrame
        Data for the heatmap.

    filename : str
        Output path.

    title : str
        Figure title.

    protocol : dict
        Loaded protocol dictionary.

    metric : str
        Metric represented by circle size.
    """
    pattern_list = df["pattern_pair"].dropna().unique()

    pattern_colors, ordered_patterns = get_pattern_colors(
        protocol,
        pattern_list
    )

    scale_max = get_metric_scale(
        protocol,
        metric
    )

    start_pos = get_start_position(
        protocol
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.set_title(
        title,
        pad=60,
        fontsize=18,
        fontweight="bold"
    )

    draw_circles(
        ax=ax,
        df=df,
        metric=metric,
        pattern_colors=pattern_colors,
        scale_max=scale_max,
        start_pos=start_pos
    )

    setup_axis(
        ax,
        protocol
    )

    draw_pattern_legend(
        fig,
        pattern_colors,
        ordered_patterns
    )

    ax.set_xlabel(
        "Mediolateral axis"
    )

    ax.set_ylabel(
        "Anteroposterior axis"
    )

    fig.subplots_adjust(
        top=0.75
    )

    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Horizontal multi-duration heatmap
# ============================================================

def plot_heatmaps_by_duration(
    df,
    protocol,
    filename,
    durations=None,
    metric="vividness"
):
    """
    Plot multiple duration heatmaps horizontally.

    Parameters
    ----------
    df : pandas.DataFrame
        Experimental data.

    protocol : dict
        Loaded protocol dictionary.

    filename : str
        Output path.

    durations : iterable, optional
        Durations to plot.

        If None, all durations defined by the protocol
        and present in the dataframe are plotted.

    metric : str
        Metric represented by circle size.
    """
    durations = get_durations(
        protocol,
        df,
        durations
    )

    if not durations:
        print(
            "[WARNING] No valid durations available for heatmap."
        )
        return

    pattern_list = df["pattern_pair"].dropna().unique()

    pattern_colors, ordered_patterns = get_pattern_colors(
        protocol,
        pattern_list
    )

    scale_max = get_metric_scale(
        protocol,
        metric
    )

    start_pos = get_start_position(
        protocol
    )

    n_durations = len(durations)

    fig, axes = plt.subplots(
        1,
        n_durations,
        figsize=(5 * n_durations, 5),
        sharey=True
    )

    if n_durations == 1:
        axes = [axes]

    for ax, duration in zip(
        axes,
        durations
    ):
        df_duration = df[
            df["duration"] == duration
        ]

        df_mean = (
            df_duration
            .groupby(
                ["pattern_pair", "rep"]
            )
            .agg({
                "x": "mean",
                "y": "mean",
                metric: "mean"
            })
            .reset_index()
        )

        ax.set_title(
            f"Duration: {duration}s",
            fontsize=18,
            fontweight="bold",
            pad=10
        )

        setup_axis(
            ax,
            protocol
        )

        draw_circles(
            ax=ax,
            df=df_mean,
            metric=metric,
            pattern_colors=pattern_colors,
            scale_max=scale_max,
            start_pos=start_pos
        )

        ax.set_xlabel(
            "Mediolateral axis",
            fontsize=16
        )

    axes[0].set_ylabel(
        "Anteroposterior axis",
        fontsize=16
    )

    draw_pattern_legend(
        fig,
        pattern_colors,
        ordered_patterns,
        fontsize=13
    )

    fig.subplots_adjust(
        top=0.78
    )

    plt.savefig(
        filename,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[SUCCESS] Heatmap figure saved to: {filename}"
    )


# ============================================================
# Repetition plots
# ============================================================

def plot_reps(
    df,
    subject_id,
    filename,
    protocol,
    metric="vividness"
):
    """
    Plot separate heatmaps for each repetition.
    """
    reps = sorted(
        df["rep"].dropna().unique()
    )

    if not reps:
        print(
            f"[WARNING] No repetitions found for {subject_id}."
        )
        return

    scale_max = get_metric_scale(
        protocol,
        metric
    )

    start_pos = get_start_position(
        protocol
    )

    pattern_list = df["pattern_pair"].dropna().unique()

    pattern_colors, ordered_patterns = get_pattern_colors(
        protocol,
        pattern_list
    )

    n_reps = len(reps)

    fig, axes = plt.subplots(
        1,
        n_reps,
        figsize=(6 * n_reps, 6),
        sharey=True
    )

    if n_reps == 1:
        axes = [axes]

    for ax, rep in zip(
        axes,
        reps
    ):
        df_rep = df[
            df["rep"] == rep
        ]

        draw_circles(
            ax=ax,
            df=df_rep,
            metric=metric,
            pattern_colors=pattern_colors,
            scale_max=scale_max,
            start_pos=start_pos
        )

        setup_axis(
            ax,
            protocol
        )

        ax.set_title(
            f"Rep {rep}"
        )

        ax.set_xlabel(
            "X position"
        )

    axes[0].set_ylabel(
        "Y position"
    )

    draw_pattern_legend(
        fig,
        pattern_colors,
        ordered_patterns
    )

    fig.suptitle(
        f"Subject {subject_id} – Repetitions",
        fontsize=16,
        y=0.99
    )

    fig.subplots_adjust(
        top=0.82
    )

    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Image combining
# ============================================================

def combine_images(
    subject_folder,
    subject_id,
    durations,
    mode="reps",
    layout="horizontal",
    output_name=None
):
    """
    Combine previously generated duration images.
    """
    images = []

    filename_template = (
        "{sid}_{mode}_{dur}s.png"
    )

    for duration in durations:
        image_path = os.path.join(
            subject_folder,
            f"duration_{duration}s",
            filename_template.format(
                sid=subject_id,
                mode=mode,
                dur=duration
            )
        )

        if not os.path.exists(image_path):
            print(
                f"[WARNING] Missing file: {image_path}"
            )
            continue

        images.append(
            Image.open(image_path)
        )

    if not images:
        print(
            f"[WARNING] No images found for subject "
            f"{subject_id}"
        )
        return

    widths, heights = zip(
        *(image.size for image in images)
    )

    if layout == "horizontal":

        combined = Image.new(
            "RGB",
            (
                sum(widths),
                max(heights)
            ),
            "white"
        )

        x_offset = 0

        for image in images:
            combined.paste(
                image,
                (x_offset, 0)
            )

            x_offset += image.size[0]

    elif layout == "vertical":

        combined = Image.new(
            "RGB",
            (
                max(widths),
                sum(heights)
            ),
            "white"
        )

        y_offset = 0

        for image in images:
            combined.paste(
                image,
                (0, y_offset)
            )

            y_offset += image.size[1]

    else:
        raise ValueError(
            "layout must be 'horizontal' or 'vertical'."
        )

    if output_name is None:
        output_name = (
            f"{subject_id}_{mode}_ALL_durations.png"
        )

    combined.save(
        os.path.join(
            subject_folder,
            output_name
        )
    )


# ============================================================
# Subject heatmaps
# ============================================================

def save_subject_heatmaps(
    df,
    subject,
    protocol,
    output_folder="Results_",
    metric="vividness",
    recalc_subject=True,
    durations=None
):
    """
    Save heatmaps for a single subject.
    """
    subject_folder = os.path.join(
        output_folder,
        subject
    )

    if (
        os.path.exists(subject_folder)
        and not recalc_subject
    ):
        print(
            f"[INFO] Skipping existing subject: {subject}"
        )
        return

    df_subject = df[
        df["subject"] == subject
    ]

    durations = get_durations(
        protocol,
        df_subject,
        durations
    )

    if not durations:
        print(
            f"[WARNING] No valid durations for {subject}."
        )
        return

    for duration in durations:

        df_duration = df_subject[
            df_subject["duration"] == duration
        ]

        duration_folder = os.path.join(
            subject_folder,
            f"duration_{duration}s"
        )

        os.makedirs(
            duration_folder,
            exist_ok=True
        )

        plot_heatmap(
            df_duration,
            os.path.join(
                duration_folder,
                f"{subject}_global_{duration}s.png"
            ),
            f"{subject} – ({duration}s)",
            protocol,
            metric=metric
        )

        plot_reps(
            df_duration,
            subject_id=subject,
            filename=os.path.join(
                duration_folder,
                f"{subject}_reps_{duration}s.png"
            ),
            protocol=protocol,
            metric=metric
        )

    combine_images(
        subject_folder,
        subject,
        durations,
        mode="reps",
        layout="vertical",
        output_name=(
            f"reps_{subject}_durations.png"
        )
    )

    combine_images(
        subject_folder,
        subject,
        durations,
        mode="global",
        layout="horizontal",
        output_name=(
            f"global_{subject}_durations.png"
        )
    )


# ============================================================
# Group heatmaps
# ============================================================

def save_all_subjects_heatmaps(
    df,
    protocol,
    output_folder="Results_",
    metric="vividness",
    durations=None
):
    """
    Save average heatmaps across all subjects.
    """
    all_folder = os.path.join(
        output_folder,
        "ALL_SUBJECTS"
    )

    os.makedirs(
        all_folder,
        exist_ok=True
    )

    durations = get_durations(
        protocol,
        df,
        durations
    )

    if not durations:
        print(
            "[WARNING] No valid durations for group heatmaps."
        )
        return

    # --------------------------------------------------------
    # Individual duration figures
    # --------------------------------------------------------

    for duration in durations:

        df_duration = df[
            df["duration"] == duration
        ]

        df_mean = (
            df_duration
            .groupby(
                ["pattern_pair", "rep"]
            )
            .agg({
                "x": "mean",
                "y": "mean",
                metric: "mean"
            })
            .reset_index()
        )

        duration_folder = os.path.join(
            all_folder,
            f"duration_{duration}s"
        )

        os.makedirs(
            duration_folder,
            exist_ok=True
        )

        plot_heatmap(
            df_mean,
            os.path.join(
                duration_folder,
                f"ALL_global_{duration}s.png"
            ),
            f"ALL Subjects – ({duration}s)",
            protocol,
            metric=metric
        )

        plot_reps(
            df_mean,
            subject_id=(
                f"ALL Subjects – Repetitions "
                f"({duration}s)"
            ),
            filename=os.path.join(
                duration_folder,
                f"ALL_reps_{duration}s.png"
            ),
            protocol=protocol,
            metric=metric
        )

    # --------------------------------------------------------
    # Horizontal multi-duration figure
    # --------------------------------------------------------

    experiment_id = protocol["experiment_id"]
    protocol_id = protocol["protocol_id"]

    filename = os.path.join(
        all_folder,
        f"{experiment_id}_{protocol_id}.png"
    )

    plot_heatmaps_by_duration(
        df=df,
        protocol=protocol,
        filename=filename,
        durations=durations,
        metric=metric
    )

