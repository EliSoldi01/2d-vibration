import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# -----------------------------
# Helper functions
# -----------------------------

def _get_columns(protocol):
    """Return column mapping from protocol or default."""
    default_cols = {
        "x": "x",
        "y": "y",
        "pattern_pair": "pattern_pair",
        "rep": "rep"
    }
    return protocol.get("columns", default_cols)

def _get_pattern_colors(protocol, pattern_list=None):
    """Return dict of pattern_text -> color and ordered pattern list."""
    patterns = protocol["patterns"]["definitions"]
    color_mapping = {p['text']: p.get('color', "#000000") for p in patterns}
    legend_mapping = {p['text']: p.get('legend_position', 999) for p in patterns}

    if pattern_list is not None:
        color_mapping = {p: color_mapping[p] for p in pattern_list if p in color_mapping}
        legend_mapping = {p: legend_mapping[p] for p in pattern_list if p in legend_mapping}

    ordered_patterns = sorted(color_mapping.keys(), key=lambda p: legend_mapping[p])
    return color_mapping, ordered_patterns

def _draw_circles(ax, df, cols, metric_col, pattern_colors, scale_max, start_pos):
    """Draw circles for each row in df."""
    for _, row in df.iterrows():
        x, y = row[cols["x"]], row[cols["y"]]
        pair = row[cols["pattern_pair"]]
        if pair not in pattern_colors:
            continue
        radius = 0.5 * (row[metric_col] / scale_max)
        ax.add_patch(plt.Circle((x, y), radius, color=pattern_colors[pair],
                                fill=False, linewidth=2, alpha=0.7))
    ax.scatter(*start_pos, c="red", s=120, marker="x", linewidths=1)

def _setup_ax(ax, xlim, ylim):
    ax.set_xlim(xlim[0] - 0.5, xlim[1] + 0.5)
    ax.set_ylim(ylim[0] - 0.5, ylim[1] + 0.5)
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 1))
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 1))
    ax.invert_yaxis()
    ax.set_aspect("equal")
    for xi in range(xlim[0], xlim[1] + 1):
        ax.vlines(xi - 0.5, ylim[0] - 0.5, ylim[1] + 0.5, color="lightgrey", lw=0.8)
    for yj in range(ylim[0], ylim[1] + 1):
        ax.hlines(yj - 0.5, xlim[0] - 0.5, xlim[1] + 0.5, color="lightgrey", lw=0.8)

# -----------------------------
# Plotting functions
# -----------------------------

def plot_heatmap(df, filename, title, protocol, metric="vividness", start_pos=(11,5), xlim=None, ylim=None):
    """Plot global heatmap."""
    cols = _get_columns(protocol)
    metric_scale = protocol['scales'][metric]['values']
    scale_max = max(metric_scale)
    pattern_list = sorted(df[cols["pattern_pair"]].unique())
    pattern_colors, ordered_patterns = _get_pattern_colors(protocol, pattern_list)

    if xlim is None:
        xlim = (1, protocol["grid"]["x"])
    if ylim is None:
        ylim = (1, protocol["grid"]["y"])

    fig, ax = plt.subplots(figsize=(12,5))
    ax.set_title(title, pad=60)
    _draw_circles(ax, df, cols, metric, pattern_colors, scale_max, start_pos)
    _setup_ax(ax, xlim, ylim)

    # legend
    patches = [mpatches.Patch(color=pattern_colors[p], label=p) for p in ordered_patterns]
    fig.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5,0.92),
               ncol=4, title=f"Pattern ({cols['pattern_pair']})")

    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    fig.subplots_adjust(top=0.75)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

def plot_reps(df, subject_id, filename, protocol, metric="vividness", start_pos=(11,5)):
    """Plot subplots for each repetition."""
    cols = _get_columns(protocol)
    reps = sorted(df[cols["rep"]].unique())
    n_reps = len(reps)
    metric_scale = protocol['scales'][metric]['values']
    scale_max = max(metric_scale)

    pattern_list = sorted(df[cols["pattern_pair"]].unique())
    pattern_colors, ordered_patterns = _get_pattern_colors(protocol, pattern_list)

    fig, axes = plt.subplots(1, n_reps, figsize=(6*n_reps,6), sharey=True)
    if n_reps == 1: axes = [axes]

    for ax, rep in zip(axes, reps):
        df_rep = df[df[cols["rep"]]==rep]
        _draw_circles(ax, df_rep, cols, metric, pattern_colors, scale_max, start_pos)
        _setup_ax(ax, (1, protocol["grid"]["x"]), (1, protocol["grid"]["y"]))
        ax.set_title(f"Rep {rep}")
        ax.set_xlabel("X position")
    axes[0].set_ylabel("Y position")

    patches = [mpatches.Patch(color=pattern_colors[p], label=p) for p in ordered_patterns]
    fig.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5,0.92),
               ncol=4, title=f"Pattern ({cols['pattern_pair']})")
    fig.suptitle(f"Subject {subject_id} – Repetitions", fontsize=16, y=0.99)
    fig.subplots_adjust(top=1.15)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

# -----------------------------
# Image combining
# -----------------------------

def combine_images_vertical(subject_folder, subject_id, durations, mode="reps", output_name=None):
    images = []
    filename_template = "{sid}_{mode}_{dur}s.png".format(sid=subject_id, mode=mode, dur="{dur}")
    for dur in durations:
        img_path = os.path.join(subject_folder, f"duration_{dur}s", filename_template.format(dur=dur))
        if not os.path.exists(img_path):
            print(f"[WARNING] Missing file: {img_path}")
            continue
        images.append(Image.open(img_path))

    if not images:
        print(f"[WARNING] No images found for subject {subject_id}")
        return

    widths, heights = zip(*(img.size for img in images))
    combined = Image.new("RGB", (max(widths), sum(heights)), "white")
    y_offset = 0
    for img in images:
        combined.paste(img, (0, y_offset))
        y_offset += img.size[1]

    if output_name is None:
        output_name = f"{subject_id}_{mode}_ALL_durations_vertical.png"
    combined.save(os.path.join(subject_folder, output_name))

# -----------------------------
# Save functions
# -----------------------------

def save_subject_heatmaps(df, subject, protocol, output_folder="Results_", metric="vividness",
                          recalc_subject=True, start_pos=(11,5)):
    """Save heatmaps for a single subject."""
    subj_folder = os.path.join(output_folder+protocol["name"], "Heatmaps", subject)
    if os.path.exists(subj_folder) and not recalc_subject:
        print(f"[INFO] Skipping existing subject: {subject}")
        return

    df_subj = df[df["subject"]==subject]
    durations = sorted(df_subj["duration"].unique())

    for dur in durations:
        df_dur = df_subj[df_subj["duration"]==dur]
        dur_folder = os.path.join(subj_folder, f"duration_{dur}s")
        os.makedirs(dur_folder, exist_ok=True)

        # global heatmap
        plot_heatmap(df_dur,
                     os.path.join(dur_folder, f"{subject}_global_{dur}s.png"),
                     f"{subject} – Global Heatmap ({dur}s)",
                     protocol, metric=metric, start_pos=start_pos)

        # reps
        plot_reps(df_dur,
                  subject_id=subject,
                  filename=os.path.join(dur_folder, f"{subject}_reps_{dur}s.png"),
                  protocol=protocol,
                  metric=metric,
                  start_pos=start_pos)

    combine_images_vertical(subj_folder, subject, durations, mode="reps",
                            output_name=f"reps_{subject}_durations_vertical.png")
    combine_images_vertical(subj_folder, subject, durations, mode="global",
                            output_name=f"global_{subject}_durations_vertical.png")

def save_all_subjects_heatmaps(df, protocol, output_folder="Results_", metric="vividness", start_pos=(11,5)):
    """Save average heatmaps across all subjects."""
    all_folder = os.path.join(output_folder+protocol["name"], "Heatmaps", "ALL_SUBJECTS")
    os.makedirs(all_folder, exist_ok=True)
    cols = _get_columns(protocol)
    durations = sorted(df["duration"].unique())

    for dur in durations:
        df_dur = df[df["duration"]==dur]
        df_mean = df_dur.groupby([cols["pattern_pair"], cols["rep"]]).agg({
            cols["x"]: "mean",
            cols["y"]: "mean",
            metric: "mean"
        }).reset_index()

        dur_folder = os.path.join(all_folder, f"duration_{dur}s")
        os.makedirs(dur_folder, exist_ok=True)

        plot_heatmap(df_mean,
                     os.path.join(dur_folder, f"ALL_global_{dur}s.png"),
                     f"ALL Subjects – Global Heatmap ({dur}s)",
                     protocol, metric=metric, start_pos=start_pos)

        plot_reps(df_mean,
                  subject_id=f"ALL Subjects – Repetitions ({dur}s)",
                  filename=os.path.join(dur_folder, f"ALL_reps_{dur}s.png"),
                  protocol=protocol,
                  metric=metric,
                  start_pos=start_pos)

    combine_images_vertical(all_folder, "ALL", durations, mode="reps",
                            output_name="ALL_reps_ALL_durations_vertical.png")
    combine_images_vertical(all_folder, "ALL", durations, mode="global",
                            output_name="ALL_global_ALL_durations_vertical.png")
