import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ==========================
# CONFIG
# ==========================
DATA_PATH = "C:\\Users\\Utente\\Desktop\\Elisa\\Research\\2D-vibration\\Subjects\\data_all_subjects - RIDOTTO.xlsx"
ORDER_PATH = "C:\\Users\\Utente\\Desktop\\Elisa\\Research\\2D-vibration\\Subjects\\RIDOTTO_results\\Subjects_list.xlsx"
OUT_DIR = "Results_Phase1\\LinRegressionComplete_v2"
START_CELL = (11, 5)
NX = 21
NY = 9
CELL_SIZE = 4  # cm

os.makedirs(OUT_DIR, exist_ok=True)

# ==========================
# LOAD DATA
# ==========================
df = pd.read_excel(DATA_PATH)
order_df = pd.read_excel(ORDER_PATH)

# ==========================
# FUNCTIONS
# ==========================
def normalize_std(mean, std):
    return std / mean if mean != 0 else np.nan


def reorder_by_presentation(df_subj, block_order):
    """
    Riordina le durate secondo l'ordine di presentazione
    """
    order = [int(d) for d in str(block_order)]
    df_subj["presentation_order"] = df_subj["duration"].apply(
        lambda x: order.index(x) + 1
    )
    return df_subj


def plot_and_regression(x, y, yerr, title, ylabel, save_path, xticks_order=None):
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        x, y, yerr=yerr,
        fmt='o', capsize=5
    )

    slope, intercept, r, p, stderr = linregress(x, y)
    y_fit = intercept + slope * np.array(x)

    plt.plot(x, y_fit,
        color="red", linewidth=2,
        label=f"y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r**2:.2f}")
    plt.xlabel("Duration (s) in presentation order")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.title(title)

    if xticks_order is not None:
        plt.xticks(ticks=range(1, len(xticks_order)+1), labels=xticks_order)

    if ylabel == "Vividness":
        plt.ylim(0, 10)
    elif ylabel == "Angle (deg)":
        plt.ylim(50, 130)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return slope, r**2, p

def load_main_data(path):
    df = pd.read_excel(path)

    df["pattern"] = (
        df["pattern_biceps"].astype(int).astype(str).str.zfill(3)
        + "_" +
        df["pattern_triceps"].astype(int).astype(str).str.zfill(3)
    )

    df["subject"] = df["subject"].astype(str)
    return df

def load_forearm_data(path):
    df = pd.read_excel(path)
    df["subject"] = df["subject"].astype(str)
    return df

def merge_forearm(main_df, forearm_df):
    return pd.merge(
        main_df,
        forearm_df[["subject", "forearm_cm", "forearm_angle_deg"]],
        on="subject",
        how="left"
    )
    
def cell_to_cm(x, y):
    return np.array([
        (x - 1) * CELL_SIZE + CELL_SIZE / 2,
        (y - 1) * CELL_SIZE + CELL_SIZE / 2
    ])

def compute_elbow(start_cm, forearm_length, forearm_angle_deg):
    theta = np.deg2rad(forearm_angle_deg)
    dx = forearm_length * np.cos(theta)
    dy = forearm_length * np.sin(theta)
    return start_cm + np.array([dx, dy])

def signed_angle(v_ref, v_cur):
    dot = np.dot(v_ref, v_cur)
    det = v_ref[0]*v_cur[1] - v_ref[1]*v_cur[0]
    return np.degrees(np.arctan2(det, dot))

def compute_angle(start_cell, index_cell, forearm_cm, forearm_angle_deg):
    if np.isnan(forearm_cm) or np.isnan(forearm_angle_deg):
        return np.nan

    start_cm = cell_to_cm(*start_cell)
    elbow_cm = compute_elbow(start_cm, forearm_cm, forearm_angle_deg)

    baseline_vec = start_cm - elbow_cm
    idx_cm = cell_to_cm(*index_cell)
    current_vec = idx_cm - elbow_cm

    return 90 + signed_angle(baseline_vec, current_vec)

# ==========================
# ALL SUBJECTS PLOT per rep
# ==========================
def plot_all_subjects_per_rep(summary_df, reps, xticks_order, ylabel, save_path):
    plt.figure(figsize=(8,5))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    
    for i, rep in enumerate(reps):
        rep_df = summary_df[summary_df["rep"] == rep]
        if rep_df.empty:
            continue
        plt.errorbar(
            rep_df["presentation_pos"],
            rep_df[f"{ylabel}_mean"],
            yerr=rep_df[f"{ylabel}_std_norm"],
            fmt='o',
            capsize=4,
            markersize=6,
            color=colors[i % len(colors)],
            label=f"Rep {rep}",
            linestyle=None
        )
    
    plt.xlabel("Ordine di presentazione")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} – ALL SUBJECTS per Rep")
    plt.grid(True)
    plt.legend()
    
    # Imposta ticks secondo l'ordine di presentazione dei blocchi
    plt.xticks(ticks=range(1, len(xticks_order)+1), labels=xticks_order)
    if ylabel == "vividness":
        plt.ylim(0, 10)
    elif ylabel == "angle":
        plt.ylim(50, 130)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def reorder_by_presentation_2(df_subj, block_order):
    """
    Riordina le durate secondo l'ordine di presentazione e assegna
    un indice universale 1,2,3,... alla posizione nella sequenza
    """
    # block_order è una stringa tipo "3241"
    order = [int(d) for d in str(block_order)]
    
    # mapping durata -> posizione nella sequenza
    duration_to_pos = {dur: i+1 for i, dur in enumerate(order)}
    
    df_subj["presentation_pos"] = df_subj["duration"].map(duration_to_pos)
    return df_subj


# ==========================
# MAIN LOOP
# ==========================
results = []

main_df = load_main_data(
    "C:\\Users\\Utente\\Desktop\\Elisa\\Research\\2D-vibration\\Subjects\\data_all_subjects - RIDOTTO.xlsx"
)

forearm_df = load_forearm_data(
    "C:\\Users\\Utente\\Desktop\\Elisa\\Research\\2D-vibration\\Subjects\\RIDOTTO_results\\Subjects_list.xlsx"
)

df = merge_forearm(main_df, forearm_df)

print("1) Data loaded")

# -------------------------
# USER SELECTION
# -------------------------
do_regression_and_plots = False
do_std_normalized_excel = False

selected_patterns = ["100_000", "000_100"] # None for all patterns, or list of patterns (e.g., ["100_000", "000_100"])
selected_subjects = None # None for all subjects, or list of subject IDs (e.g., ["S01", "S02"])

if selected_patterns is None:
    selected_patterns = sorted(df["pattern"].dropna().unique())

if selected_subjects is None:
    selected_subjects = sorted(df["subject"].dropna().unique())

df = df[
    df["pattern"].isin(selected_patterns) &
    df["subject"].isin(selected_subjects)
]

df["angle_deg"] = df.apply(
    lambda r: compute_angle(
        START_CELL,
        (r["x"], r["y"]),
        r["forearm_cm"],
        r["forearm_angle_deg"]
    ),
    axis=1
)

for subject in df["subject"].unique():

    subj_df = df[df["subject"] == subject]
    block_order = order_df.loc[
        order_df["subject"] == subject, "block_order"
    ].values[0]

    subj_df = reorder_by_presentation(subj_df, block_order)

    for pattern in subj_df["pattern"].unique():

        pat_df = subj_df[subj_df["pattern"] == pattern]

        summary = (
            pat_df
            .groupby(["presentation_order", "duration"])
            .agg(
                vividness_mean=("vividness", "mean"),
                vividness_std=("vividness", "std"),
                angle_mean=("angle_deg", "mean"),
                angle_std=("angle_deg", "std")
            )
            .reset_index()
            .sort_values("presentation_order")
        )

        summary["vividness_std_norm"] = summary.apply(
            lambda r: normalize_std(r["vividness_mean"], r["vividness_std"]), axis=1
        )
        summary["angle_std_norm"] = summary.apply(
            lambda r: normalize_std(r["angle_mean"], r["angle_std"]), axis=1
        )

        # ===== OUTPUT DIR =====
        out_path = os.path.join(
            OUT_DIR, f"{subject}", f"{pattern}"
        )
        os.makedirs(out_path, exist_ok=True)

        # ===== PLOT =====
        slope, r2, p = plot_and_regression(
            x=summary["presentation_order"],
            y=summary["angle_mean"],
            yerr=summary["angle_std_norm"],
            title=f"Subject {subject} – Pattern {pattern}",
            ylabel="Angle (deg)",
            save_path=os.path.join(out_path, "angle_vs_presentation_order.png"),
            xticks_order=[int(d) for d in str(block_order)]
        )

        slope, r2, p = plot_and_regression(
            x=summary["presentation_order"],
            y=summary["vividness_mean"],
            yerr=summary["vividness_std_norm"],
            title=f"Subject {subject} – Pattern {pattern}",
            ylabel="Vividness",
            save_path=os.path.join(out_path, "vividness_vs_presentation_order.png"),
            xticks_order=[int(d) for d in str(block_order)]
        )

        results.append({
            "subject": subject,
            "pattern": pattern,
            "slope": slope,
            "R2": r2,
            "p_value": p
        })


## Loop su pattern
for pattern in selected_patterns:
    all_summary = []

    for subject in selected_subjects:
        subj_df = df[df["subject"] == subject]
        
        block_order = int(order_df.loc[order_df["subject"]==subject, "block_order"].values[0])
        subj_df = reorder_by_presentation_2(subj_df, block_order)

        print(f"\nSubject {subject}, block_order originale: {block_order}")
        print(subj_df[["duration", "presentation_pos"]].drop_duplicates().sort_values("presentation_pos"))

        pat_df = subj_df[subj_df["pattern"]==pattern]
        
        for rep in sorted(pat_df["rep"].unique()):
            rep_df = pat_df[pat_df["rep"]==rep]
            
            # Group per presentation_pos del singolo soggetto
            summary = rep_df.groupby("presentation_pos", as_index=False).agg(
                vividness_mean=("vividness", "mean"),
                vividness_std=("vividness", "std"),
                angle_mean=("angle_deg", "mean"),
                angle_std=("angle_deg", "std")
            )
            summary["vividness_std_norm"] = summary.apply(
                lambda r: normalize_std(r["vividness_mean"], r["vividness_std"]), axis=1
            )
            summary["angle_std_norm"] = summary.apply(
                lambda r: normalize_std(r["angle_mean"], r["angle_std"]), axis=1
            )
            summary["rep"] = rep
            all_summary.append(summary)

    all_summary_df = pd.concat(all_summary, ignore_index=True)
    
    # Raggruppa per rep e posizione universale nella sequenza
    all_summary_grouped = all_summary_df.groupby(["rep","presentation_pos"], as_index=False).agg(
        vividness_mean=("vividness_mean", "mean"),
        vividness_std=("vividness_std", "std"),
        angle_mean=("angle_mean", "mean"),
        angle_std=("angle_std", "std")
    )
    all_summary_grouped["vividness_std_norm"] = all_summary_grouped.apply(
        lambda r: normalize_std(r["vividness_mean"], r["vividness_std"]), axis=1
    )
    all_summary_grouped["angle_std_norm"] = all_summary_grouped.apply(
        lambda r: normalize_std(r["angle_mean"], r["angle_std"]), axis=1
    )


    xticks_order = [1,2,3,4]  # per i tick
    out_path = os.path.join(OUT_DIR, f"Pattern_{pattern}")
    os.makedirs(out_path, exist_ok=True)

    # Plot Vividness
    plot_all_subjects_per_rep(
        all_summary_grouped, 
        reps=sorted(df["rep"].unique()), 
        xticks_order=xticks_order,
        ylabel="vividness",
        save_path=os.path.join(out_path, "vividness_all_subjects_per_rep.png")
    )

    # Plot Angle
    plot_all_subjects_per_rep(
        all_summary_grouped, 
        reps=sorted(df["rep"].unique()), 
        xticks_order=xticks_order,
        ylabel="angle",
        save_path=os.path.join(out_path, "angle_all_subjects_per_rep.png")
    )
