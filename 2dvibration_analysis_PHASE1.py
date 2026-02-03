import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from PIL import Image

# -----------------------------
# Constants
# -----------------------------
NX = 21
NY = 9
START_POS = (11, 5)
OUTPUT_FOLDER = "Results_Phase1\\Heatmaps"

DURATION_LABEL = "duration"
REP_LABEL = "rep"
X_LABEL = "x"
Y_LABEL = "y"
PATTERN_BICEPS_LABEL = "pattern_biceps"
PATTERN_TRICEPS_LABEL = "pattern_triceps"
VIVIDNESS_LABEL = "vividness"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

PATTERN_PAIR_ORDER = [
    "100_000",
    "101_000",
    "011_100",
    "100_100",
    "010_010",
    "000_100",
    "000_101",
    "100_011"
]

CUSTOM_COLORS = [
    "#ffb716", # Ext 100
    "#ff8711", # Ext 101
    "#b36800", # Ext 011
    "#c9c9c9", # Noise 100
    "#c9c9c9", # Noise 010
    "#85e0fc", # Flex 100
    "#057dcd", # Flex 101
    "#194a7a" # Flex 011
]

def get_pattern_colors(pattern_pairs):
    """
    Map each pattern_pair to a fixed, highly distinguishable color.
    """

    # Order pattern pairs according to PATTERN_PAIR_ORDER
    ordered_pairs = [
        p for p in PATTERN_PAIR_ORDER if p in pattern_pairs
    ]

    if len(ordered_pairs) > len(CUSTOM_COLORS):
        raise ValueError(
            f"Too many pattern pairs ({len(pattern_pairs)}). "
            f"Max supported: {len(CUSTOM_COLORS)}"
        )

    # Create mapping
    return {
        p: CUSTOM_COLORS[i]
        for i, p in enumerate(ordered_pairs)
    }


# -----------------------------
# Plot: global heatmap
# 
# Create a global heatmap for a given subject and duration  
# -----------------------------
def plot_circles_heatmap(
    df_plot,
    filename,
    title,
    pattern_pairs,
    start_pos=START_POS,
    xlim=None
): 
    """
    Plot a global heatmap with circles representing vividness at each position.
    Args:
            df_plot: DataFrame containing the data to plot
            filename: Path to save the figure
            title: Title of the plot
            pattern_pairs: List of unique pattern pairs
            start_pos: Starting position tuple (x, y)
            xlim: Tuple defining x-axis limits; if None, use full range
    Returns: None
    """
    pattern_colors = get_pattern_colors(pattern_pairs)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(title, pad=60)

    for _, row in df_plot.iterrows():
        x, y = row[X_LABEL], row[Y_LABEL]
        pair = row["pattern_pair"]

        if pair not in pattern_colors:
            continue

        radius = 0.5 * (row[VIVIDNESS_LABEL] / 10)
        ax.add_patch(
            plt.Circle(
                (x, y),
                radius,
                color=pattern_colors[pair],
                fill=False,
                linewidth=2,
                alpha=0.7
            )
        )

    if xlim:
        ax.set_xlim(xlim[0] - 0.5, xlim[1] + 0.5)
        xticks = np.arange(xlim[0], xlim[1] + 1)
    else:
        ax.set_xlim(0.5, NX + 0.5)
        xticks = np.arange(1, NX + 1)

    ax.set_ylim(0.5, NY + 0.5)
    ax.set_xticks(xticks)
    ax.set_yticks(np.arange(1, NY + 1))
    ax.invert_yaxis()
    ax.set_aspect("equal")

    for xi in range(xticks[0], xticks[-1] + 1):
        ax.vlines(xi - 0.5, 0.5, NY + 0.5, color="lightgrey", lw=0.8)
    for yj in range(1, NY + 1):
        ax.hlines(yj - 0.5, xticks[0] - 0.5, xticks[-1] + 0.5, color="lightgrey", lw=0.8)

    ax.scatter(*start_pos, c="red", s=120, marker="x", linewidths=1)

    legend_pairs = [p for p in PATTERN_PAIR_ORDER if p in pattern_colors]
    patches = [mpatches.Patch(color=pattern_colors[p], label=p) for p in legend_pairs]
    fig.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=4,
        title="Pattern (Biceps_Triceps)"
    )

    fig.subplots_adjust(top=0.75)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

# -----------------------------
# Plot: reps in subplots
#
# Create subplots for each repetition for a given subject and duration
# -----------------------------
def plot_reps_subplots(
        df_subject,
        subject_id, 
        pattern_pairs, 
        start_pos=START_POS, 
        xlim=(1,21), 
        save_path=None
    ):
    """
    Plot subplots for each repetition showing vividness circles.
    Args: 
            df_subject: DataFrame for a specific subject and duration
            subject_id: Identifier for the subject
            pattern_pairs: List of unique pattern pairs
            start_pos: Starting position tuple (x, y)
            xlim: Tuple defining x-axis limits
            save_path: Path to save the figure; if None, show the plot
    Returns: None
    """
    pattern_colors = get_pattern_colors(pattern_pairs)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for i, rep in enumerate([1, 2, 3]):
        ax = axes[i]
        df_rep = df_subject[df_subject[REP_LABEL] == rep]
        
        for _, row in df_rep.iterrows():
            x, y = row[X_LABEL], row[Y_LABEL]
            pair = row["pattern_pair"]
            if pair not in pattern_colors:
                continue
            radius = 0.5 * (row[VIVIDNESS_LABEL] / 10)
            circle = plt.Circle((x, y), radius, color=pattern_colors[pair], fill=False, linewidth=2, alpha=0.7)
            ax.add_patch(circle)

        ax.scatter(start_pos[0], start_pos[1], c="red", s=120, marker="x", linewidths=1)
        
        ax.set_xlim(xlim[0]-0.5, xlim[1]+0.5)
        ax.set_ylim(0.5, NY+0.5)
        ax.set_xticks(np.arange(xlim[0], xlim[1]+1))
        ax.set_yticks(np.arange(1, NY+1))
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_title(f"Rep {rep}")
        
        for xi in range(xlim[0], xlim[1]+1):
            ax.vlines(xi-0.5, 0.5, NY+0.5, color="lightgrey", lw=0.8)
        for yj in range(1, NY+1):
            ax.hlines(yj-0.5, xlim[0]-0.5, xlim[1]+0.5, color="lightgrey", lw=0.8)
        if i == 0:
            ax.set_ylabel("Y position")
        ax.set_xlabel("X position")

    # Legend
    legend_pairs = [p for p in PATTERN_PAIR_ORDER if p in pattern_colors]
    patches = [mpatches.Patch(color=pattern_colors[p], label=p) for p in legend_pairs]
    fig.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=4, title="Pattern (Biceps_Triceps)")

    fig.suptitle(f"Subject {subject_id} – Repetitions", fontsize=16, y=0.99)
    fig.subplots_adjust(top=1.15)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# -----------------------------
# Combine images vertically
# 
# Combine images for a subject across all durations vertically
# -----------------------------
def combine_images_vertical(
    subject_folder,
    subject_id,
    durations,
    mode="reps", # "reps" o "global"
    output_name=None
):
    """
    Combine images vertically for a subject across all durations.
    Args: 
            subject_folder: Folder containing subject images
            subject_id: Identifier for the subject
            durations: List of durations to include
            mode: "reps" or "global" to specify image type
            output_name: Name for the output combined image; if None, use default 
    Returns: None
    """

    images = []

    # Determine filename template based on mode
    if mode == "reps":
        filename_template = "{sid}_reps_{dur}s.png"
    elif mode == "global":
        filename_template = "{sid}_global_{dur}s.png"
    else:
        raise ValueError("mode must be 'reps' or 'global'")

    # Load images for each duration
    for dur in durations:
        img_path = os.path.join(
            subject_folder,
            f"duration_{dur}s",
            filename_template.format(sid=subject_id, dur=dur)
        )

        if not os.path.exists(img_path):
            print(f"[WARNING] Missing file: {img_path}")
            continue

        images.append(Image.open(img_path))

    if len(images) == 0:
        print(f"[WARNING] No images found for subject {subject_id}")
        return

    # Size calculation for vertical stacking
    widths, heights = zip(*(img.size for img in images))
    max_width = max(widths)
    total_height = sum(heights)

    combined = Image.new("RGB", (max_width, total_height), "white")

    # Paste images vertically
    y_offset = 0
    for img in images:
        combined.paste(img, (0, y_offset))
        y_offset += img.size[1]

    if output_name is None:
        output_name = f"{subject_id}_reps_ALL_durations_vertical.png"

    out_path = os.path.join(subject_folder, output_name)
    combined.save(out_path)

    print(f"[OK] Saved vertical combined image: {out_path}")

# -----------------------------
# Save per subject
#
# Create and save heatmaps for a given subject and all durations
# -----------------------------
def save_subject_heatmaps(
        df, 
        subject, 
        recalc_subject=True
    ):
    """""
    Save heatmaps for a given subject and all durations.
    Args:
            df: DataFrame containing all data
            subject: Subject identifier
            recalc_subject: If False, skip if subject folder exists
    Returns: None
    """
    subj_folder = os.path.join(OUTPUT_FOLDER, subject)
    if os.path.exists(subj_folder) and not recalc_subject:
        print(f"[INFO] Subject folder exists, skipping: {subject}")
        return

    df_subj = df[df["subject"] == subject]
    durations = sorted(df_subj[DURATION_LABEL].unique())

    for dur in durations:
        df_dur = df_subj[df_subj[DURATION_LABEL]==dur]
        dur_folder = os.path.join(subj_folder, f"duration_{dur}s")
        os.makedirs(dur_folder, exist_ok=True)
        pattern_pairs = sorted(df_dur["pattern_pair"].unique())

        # Global heatmap
        plot_circles_heatmap(df_dur, os.path.join(dur_folder, f"{subject}_global_{dur}s.png"),
                             f"{subject} – Global Heatmap ({dur}s)", pattern_pairs)

        # Reps
        plot_reps_subplots(df_dur, subject, pattern_pairs,
                           save_path=os.path.join(dur_folder, f"{subject}_reps_{dur}s.png"))

    # Combine vertical 4×3 reps figure
    combine_images_vertical(subj_folder, subject, durations, mode="reps", output_name=f"reps_{subject}_durations_vertical.png")
    combine_images_vertical(subj_folder, subject, durations, mode="global", output_name=f"global_{subject}_durations_vertical.png")

# -----------------------------
# Save ALL SUBJECTS (mean)
# 
# Create average heatmaps across all subjects for each duration
# -----------------------------
def save_all_subjects_heatmaps(df):
    """Save average heatmaps across all subjects for each duration.
        Args:
            df: DataFrame containing all data   
        Returns: None
    """
    all_folder = os.path.join(OUTPUT_FOLDER, "ALL_SUBJECTS")
    os.makedirs(all_folder, exist_ok=True)

    for dur in sorted(df[DURATION_LABEL].unique()):
        df_dur = df[df[DURATION_LABEL] == dur]

        df_mean = (
            df_dur
            .groupby([DURATION_LABEL, "pattern_pair", REP_LABEL])
            .agg({
                X_LABEL: "mean",
                Y_LABEL: "mean",
                VIVIDNESS_LABEL: "mean"
            })
            .reset_index()
        )


        dur_folder = os.path.join(all_folder, f"duration_{dur}s")
        os.makedirs(dur_folder, exist_ok=True)

        pattern_pairs = sorted(df_mean["pattern_pair"].unique())

        plot_circles_heatmap(
            df_mean,
            os.path.join(dur_folder, f"ALL_global_{dur}s.png"),
            f"ALL Subjects – Global Heatmap ({dur}s)",
            pattern_pairs
        )

        plot_reps_subplots(
            df_mean,
            subject_id=f"ALL Subjects – Repetitions ({dur}s)",
            pattern_pairs=pattern_pairs,
            save_path=os.path.join(dur_folder, f"ALL_reps_{dur}s.png")
        )


    combine_images_vertical(all_folder, "ALL", sorted(df[DURATION_LABEL].unique()), mode="reps", output_name="ALL_reps_ALL_durations_vertical.png")
    combine_images_vertical(all_folder, "ALL", sorted(df[DURATION_LABEL].unique()), mode="global", output_name="ALL_global_ALL_durations_vertical.png")

# -----------------------------
# Main
# -----------------------------
df = pd.read_excel("C:\\Users\\Utente\\Desktop\\Elisa\\Research\\2D-vibration\\Subjects\\data_all_subjects - RIDOTTO.xlsx") # Load data

# Data preprocessing: cleaning and filtering
df[PATTERN_BICEPS_LABEL] = df[PATTERN_BICEPS_LABEL].astype(str).str.zfill(3)
df[PATTERN_TRICEPS_LABEL] = df[PATTERN_TRICEPS_LABEL].astype(str).str.zfill(3)
df[DURATION_LABEL] = pd.to_numeric(df[DURATION_LABEL], errors="coerce")
df = df.dropna(subset=[DURATION_LABEL])
df["pattern_pair"] = df[PATTERN_BICEPS_LABEL] + "_" + df[PATTERN_TRICEPS_LABEL] 
df[X_LABEL] = pd.to_numeric(df[X_LABEL], errors="coerce")
df[Y_LABEL] = pd.to_numeric(df[Y_LABEL], errors="coerce")
df = df.dropna(subset=[X_LABEL, Y_LABEL])
df = df[df[X_LABEL].between(1, NX) & df[Y_LABEL].between(1, NY)]

# --- CONFIGURATION ---
subjects_to_process = ["S07"]  # None = all subjects; else list e.g., ["subj1", "subj2"]
recalc_subject = True       # False = skip existing subject folder
update_group_average = True # True = recalculate ALL SUBJECTS average heatmaps
# ---------------------

subjects_list = df["subject"].unique() if subjects_to_process is None else subjects_to_process

for subj in subjects_list:
    print(f"[INFO] Processing subject: {subj}")
    save_subject_heatmaps(df, subj, recalc_subject=recalc_subject)

if update_group_average:
    print("[INFO] Processing ALL SUBJECTS average heatmaps")
    save_all_subjects_heatmaps(df)
    
