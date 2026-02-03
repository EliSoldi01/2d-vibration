import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# CONSTANTS
# ============================================================
NX = 21
NY = 9
CELL_SIZE = 4  # cm

OUTPUT_FOLDER = "Results_Phase1\\LinRegressionComplete"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

START_CELL = (11, 5)

# ============================================================
# UTILS
# ============================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# ============================================================
# GEOMETRY
# ============================================================
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

# ============================================================
# DATA LOADING
# ============================================================
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

# ============================================================
# PLOTTING
# ============================================================
def plot_regression(
    df,
    x_col,
    y_col,
    title,
    ylabel,
    xlabel,
    save_path
):
    if len(df) < 2:
        print(f"⚠️ Not enough points for {title}")
        return

    X = df[x_col].values
    y = df[y_col].values

    coeffs = np.polyfit(X, y, 1)
    y_fit = np.polyval(coeffs, X)

    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    plt.figure(figsize=(9, 5))
    plt.scatter(X, y, alpha=0.8)

    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = coeffs[0] * x_line + coeffs[1]

    plt.plot(
        x_line, y_line,
        color="red", linewidth=2,
        label=f"y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}\n$R^2$ = {r2:.2f}"
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xlabel == "Vividness":
        plt.xlim(0, 10)

    if ylabel == "Angle (deg)":
        plt.ylim(50, 130)
    elif ylabel == "Vividness":
        plt.ylim(0, 10)

    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()

# ============================================================
# MAIN
# ============================================================

# -------------------------
# LOAD DATA
# -------------------------
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

print("2) User selection applied. \nSelected patterns:", selected_patterns, "\nSelected subjects:", selected_subjects)

# -------------------------
# GEOMETRY
# -------------------------
df["angle_deg"] = df.apply(
    lambda r: compute_angle(
        START_CELL,
        (r["x"], r["y"]),
        r["forearm_cm"],
        r["forearm_angle_deg"]
    ),
    axis=1
)

print("3) Geometry computed")

# -------------------------
# CLEAN
# -------------------------
df = df.dropna(subset=["duration"]).reset_index(drop=True)

# ============================================================
# ANALYSIS & PLOTS
# ============================================================

print("4) Starting regression analysis and plotting...")
for pattern in selected_patterns:

    df_pat = df[df["pattern"] == pattern]

    # ==========================
    # PER SUBJECT
    # ==========================
    for subject in selected_subjects:

        print(f"Processing pattern {pattern}, subject {subject}...")
        df_sub = df_pat[df_pat["subject"] == subject]

        base_dir = os.path.join(OUTPUT_FOLDER, subject)
        single_dir = os.path.join(base_dir, "Single-reps")
        mean_dir = os.path.join(base_dir, "Mean-per-subject")

        ensure_dir(single_dir)
        ensure_dir(mean_dir)

        # ---------- SINGLE REPS ----------
        for rep in sorted(df_sub["rep"].unique()):

            df_rep = df_sub[df_sub["rep"] == rep]

            # Angle vs Duration
            plot_regression(
                df_rep.dropna(subset=["angle_deg"]),
                "duration",
                "angle_deg",
                f"{subject} – {pattern} – rep {rep} – Angle",
                "Angle (deg)",
                "Duration (s)",
                os.path.join(single_dir, f"{pattern}_rep{rep}_angle.png")
            )
            print(f"   Plotted angle vs duration for rep {rep}")

            # Vividness vs Duration
            plot_regression(
                df_rep.dropna(subset=["vividness"]),
                "duration",
                "vividness",
                f"{subject} – {pattern} – rep {rep} – Vividness",
                "Vividness",
                "Duration (s)",
                os.path.join(single_dir, f"{pattern}_rep{rep}_vividness.png")
            )
            print(f"   Plotted vividness vs duration for rep {rep}")

            # Angle vs Vividness
            plot_regression(
                df_rep.dropna(subset=["vividness"]),
                "vividness",
                "angle_deg",
                f"{subject} – {pattern} – rep {rep} – Angle vs Vividness",
                "Angle (deg)",
                "Vividness",
                os.path.join(single_dir, f"{pattern}_rep{rep}_angle_vs_vividness.png")
            )
            print(f"   Plotted angle vs vividness for rep {rep}")


        # ---------- MEAN PER SUBJECT ----------
        df_mean_sub = (
            df_sub
            .groupby("duration", as_index=False)
            .agg({
                "angle_deg": "mean",
                "vividness": "mean"
            })
        )

        # Angle vs Duration
        plot_regression(
            df_mean_sub.dropna(subset=["angle_deg"]),
            "duration",
            "angle_deg",
            f"{subject} – {pattern} – Mean reps – Angle",
            "Angle (deg)",
            "Duration (s)",
            os.path.join(mean_dir, f"{pattern}_mean_angle.png")
        )
        print(f"   Plotted mean angle vs duration for subject {subject}")

        # Vividness vs Duration
        plot_regression(
            df_mean_sub.dropna(subset=["vividness"]),
            "duration",
            "vividness",
            f"{subject} – {pattern} – Mean reps – Vividness",
            "Vividness",
            "Duration (s)",
            os.path.join(mean_dir, f"{pattern}_mean_vividness.png")
        )
        print(f"   Plotted mean vividness vs duration for subject {subject}")

        # Angle vs Vividness
        plot_regression(
            df_mean_sub.dropna(subset=["vividness"]),
            "vividness",
            "angle_deg",
            f"{subject} – {pattern} – Mean reps – Angle vs Vividness",
            "Angle (deg)",
            "Vividness",
            os.path.join(mean_dir, f"{pattern}_mean_angle_vs_vividness.png")
        )
        print(f"   Plotted mean angle vs vividness for subject {subject}")

    # ==========================
    # ALL SUBJECTS
    # ==========================
    all_dir = os.path.join(OUTPUT_FOLDER, "ALL_SUBJECTS")
    single_all = os.path.join(all_dir, "Single-reps")
    mean_all = os.path.join(all_dir, "Mean")

    ensure_dir(single_all)
    ensure_dir(mean_all)

    # ---------- MEAN PER REP ----------
    for rep in sorted(df_pat["rep"].unique()):

        df_rep_all = (
            df_pat[df_pat["rep"] == rep]
            .groupby("duration", as_index=False)
            .agg({
                "angle_deg": "mean",
                "vividness": "mean"
            })
        )

        # Angle vs Duration
        plot_regression(
            df_rep_all.dropna(subset=["angle_deg"]),
            "duration",
            "angle_deg",
            f"ALL – {pattern} – rep {rep} – Angle",
            "Angle (deg)",
            "Duration (s)",
            os.path.join(single_all, f"{pattern}_rep{rep}_angle.png")
        )
        print(f"Plotted ALL subjects angle vs duration for rep {rep}")

        # Vividness vs Duration
        plot_regression(
            df_rep_all.dropna(subset=["vividness"]),
            "duration",
            "vividness",
            f"ALL – {pattern} – rep {rep} – Vividness",
            "Vividness",
            "Duration (s)",
            os.path.join(single_all, f"{pattern}_rep{rep}_vividness.png")
        )
        print(f"Plotted ALL subjects vividness vs duration for rep {rep}")

        # Vividness vs Angle
        plot_regression(
            df_rep_all.dropna(subset=["vividness"]),
            "vividness",
            "angle_deg",
            f"ALL – {pattern} – rep {rep} – Angle vs Vividness",
            "Angle (deg)",
            "Vividness",
            os.path.join(single_all, f"{pattern}_rep{rep}_angle_vs_vividness.png")
        )
        print(f"Plotted ALL subjects angle vs vividness for rep {rep}")

    # ---------- MEAN TOTALE ----------
    df_mean_all = (
        df_pat
        .groupby("duration", as_index=False)
        .agg({
            "angle_deg": "mean",
            "vividness": "mean"
        })
    )

    # Angle vs Duration
    plot_regression(
        df_mean_all.dropna(subset=["angle_deg"]),
        "duration",
        "angle_deg",
        f"ALL – {pattern} – Mean – Angle",
        "Angle (deg)",
        "Duration (s)",
        os.path.join(mean_all, f"{pattern}_mean_angle.png")
    )
    print(f"Plotted ALL subjects mean angle vs duration")

    # Vividness vs Duration
    plot_regression(
        df_mean_all.dropna(subset=["vividness"]),
        "duration",
        "vividness",
        f"ALL – {pattern} – Mean – Vividness",
        "Vividness",
        "Duration (s)",
        os.path.join(mean_all, f"{pattern}_mean_vividness.png")
    )
    print(f"Plotted ALL subjects mean vividness vs duration")

    # Vividness vs Angle
    plot_regression(
        df_mean_all.dropna(subset=["vividness"]),
        "vividness",
        "angle_deg",
        f"ALL – {pattern} – Mean – Angle vs Vividness",
        "Angle (deg)",
        "Vividness",
        os.path.join(mean_all, f"{pattern}_mean_angle_vs_vividness.png")
    )
    print(f"Plotted ALL subjects mean angle vs vividness")

print("-> Regression analysis and plotting completed.")
