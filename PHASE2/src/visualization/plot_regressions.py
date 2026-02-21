import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression

# ============================================================
# LINEAR REGRESSION PLOT
# ============================================================
def plot_regression(df, x_col, y_col, title, ylabel, xlabel, save_path, protocol=None):
    if len(df) < 2:
        print(f"⚠️ Not enough points for {title}")
        return

    if protocol is not None:
        vividness_values = protocol["scales"]["vividness"]["values"]
    else:
        vividness_values = [0,10]

# ... dentro la funzione plot_regression ...

    X = df[x_col].values.reshape(-1, 1) # Scikit-learn vuole X 2D
    y = df[y_col].values

    # Inizializza il modello SENZA intercetta
    model = LinearRegression(fit_intercept=False)
    
    # Se vuoi usare i pesi (Vividness), passali qui:
    w = df['vividness'].values
    model.fit(X, y, sample_weight=w/3)  # Normalizza i pesi tra 0 e 1

    slope = model.coef_[0]
    y_fit = model.predict(X)


    # R^2 calcolato da scikit-learn
    # Nota: se fit_intercept=False, scikit-learn calcola l'R^2 coerente
    r2 = model.score(X, y) 

    plt.figure(figsize=(9, 5))
    plt.scatter(X, y, alpha=0.8)

    # Per il plot, y_line usa il predict del modello
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)

    plt.plot(
        x_line, y_line,
        color="red", linewidth=2,
        label=f"y = {slope:.2f}x\n$R^2$ = {r2:.2f}"
    )

    ax = plt.gca()

    if x_col == "presentation_order":
        # 1. Prendiamo i valori unici di ordine e durata presenti nel df
        # Li ordiniamo per presentation_order per farli coincidere con l'asse X
        mapping = df[["presentation_order", "duration"]].drop_duplicates().sort_values("presentation_order")
        
        # Impostiamo i ticks sulle posizioni dell'ordine (interi)
        ax.set_xticks(mapping["presentation_order"].values)
        # Sostituiamo le etichette con i valori della durata
        ax.set_xticklabels(mapping["duration"].values)
    else:
        # Forza i ticks a essere interi per duration o vividness
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Forza i ticks interi anche per l'asse Y se è vividness
    if y_col == "vividness":
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xlabel.lower() == "vividness":
        plt.xlim(vividness_values[0], vividness_values[-1])

    if ylabel.lower().startswith("angle"):
        plt.ylim(-60, +60)

    elif ylabel.lower() == "vividness":
        plt.ylim(vividness_values[0], vividness_values[-1])

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

"""
def plot_regression(df, x_col, y_col, title, ylabel, xlabel, save_path, protocol=None):
    if len(df) < 2:
        print(f"⚠️ Not enough points for {title}")
        return

    if protocol is not None:
        vividness_values = protocol["scales"]["vividness"]["values"]
    else:
        vividness_values = [0,10]

    X = df[x_col].values
    y = df[y_col].values

    # Regression forced through origin
    denom = np.sum(X**2)

    if denom == 0:
        print(f"⚠️ Degenerate regression for {title}")
        return

    slope = np.sum(X * y) / denom
    y_fit = slope * X

    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum(y ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    plt.figure(figsize=(9, 5))
    plt.scatter(X, y, alpha=0.8)

    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = slope * x_line

    plt.plot(
        x_line, y_line,
        color="red", linewidth=2,
        label=f"y = {slope:.2f}x\n$R^2$ = {r2:.2f}"
    )


    ax = plt.gca()

    if x_col == "presentation_order":
        # 1. Prendiamo i valori unici di ordine e durata presenti nel df
        # Li ordiniamo per presentation_order per farli coincidere con l'asse X
        mapping = df[["presentation_order", "duration"]].drop_duplicates().sort_values("presentation_order")
        
        # Impostiamo i ticks sulle posizioni dell'ordine (interi)
        ax.set_xticks(mapping["presentation_order"].values)
        # Sostituiamo le etichette con i valori della durata
        ax.set_xticklabels(mapping["duration"].values)
    else:
        # Forza i ticks a essere interi per duration o vividness
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Forza i ticks interi anche per l'asse Y se è vividness
    if y_col == "vividness":
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xlabel.lower() == "vividness":
        plt.xlim(vividness_values[0], vividness_values[-1])

    if ylabel.lower().startswith("angle"):
        plt.ylim(-60, +60)

    elif ylabel.lower() == "vividness":
        plt.ylim(vividness_values[0], vividness_values[-1])

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
"""

# ============================================================
# PLOT PER SOGGETTO E PATTERN
# ============================================================
def plot_subject_pattern(subj_df, subject, pattern, protocol, per_reps_plot = True, mean_plot = True, output_folder="Results/regressions", x_col="duration", x_label="Duration (s)"):
    """
    Plot regressions per subject and pattern:
        - Single reps
        - Mean over reps
    """
    print(f"[plot_subject_pattern] Plotting regressions for subject {subject}, pattern {pattern}...")
    df_sub = subj_df[subj_df["pattern_pair"] == pattern]

    # --------- SINGLE REPS ----------
    if per_reps_plot:
        for rep in sorted(df_sub["rep"].unique()):
            df_rep = df_sub[df_sub["rep"] == rep]

            rep_dir = os.path.join(output_folder, subject, "Single-reps")
            os.makedirs(rep_dir, exist_ok=True)

            # Angle vs Duration
            plot_regression(
                df_rep.dropna(subset=["angle_deg"]),
                x_col=x_col,
                y_col="angle_deg",
                title=f"{subject} – {pattern} – rep {rep} – Angle",
                ylabel="Angle (deg)",
                xlabel=x_label,
                save_path=os.path.join(rep_dir, pattern, f"rep{rep}_angle.png")
            )

            # Vividness vs Duration
            plot_regression(
                df_rep.dropna(subset=["vividness"]),
                x_col=x_col,
                y_col="vividness",
                title=f"{subject} – {pattern} – rep {rep} – Vividness",
                ylabel="Vividness",
                xlabel=x_label,
                save_path=os.path.join(rep_dir, pattern, f"rep{rep}_vividness.png"),
                protocol=protocol
            )

            # Angle vs Vividness
            plot_regression(
                df_rep.dropna(subset=["vividness"]),
                x_col=x_col,
                y_col="angle_deg",
                title=f"{subject} – {pattern} – rep {rep} – Angle vs Vividness",
                ylabel="Angle (deg)",
                xlabel=x_label,
                save_path=os.path.join(rep_dir, pattern, f"rep{rep}_angle_vs_vividness.png"),
                protocol=protocol
            )
    
    if mean_plot:
        # --------- MEAN PER SUBJECT ----------
        df_mean = df_sub.groupby("duration", as_index=False).agg({
            "angle_deg": "mean",
            "vividness": "mean",
            "presentation_order": "first",
            "duration": "first"
        })

        mean_dir = os.path.join(output_folder, subject, "Mean-per-subject")
        os.makedirs(mean_dir, exist_ok=True)

        plot_regression(
            df_mean.dropna(subset=["angle_deg"]),
            x_col=x_col,
            y_col="angle_deg",
            title=f"{subject} – {pattern} – Mean reps – Angle",
            ylabel="Angle (deg)",
            xlabel=x_label,
            save_path=os.path.join(mean_dir, pattern, f"mean_angle.png")
        )

        plot_regression(
            df_mean.dropna(subset=["vividness"]),
            x_col=x_col,
            y_col="vividness",
            title=f"{subject} – {pattern} – Mean reps – Vividness",
            ylabel="Vividness",
            xlabel=x_label,
            save_path=os.path.join(mean_dir, pattern, f"mean_vividness.png"),
            protocol=protocol
        )

        plot_regression(
            df_mean.dropna(subset=["vividness"]),
            x_col="vividness",
            y_col="angle_deg",
            title=f"{subject} – {pattern} – Mean reps – Angle vs Vividness",
            ylabel="Angle (deg)",
            xlabel="Vividness",
            save_path=os.path.join(mean_dir, pattern, f"mean_angle_vs_vividness.png"),
            protocol=protocol
        )


# ============================================================
# PLOT MEDIA SU TUTTI I SOGGETTI
# ============================================================
def plot_group_average(df, patterns_list, protocol, per_reps_plot = True, mean_plot = True, output_folder="Results/regressions", x_col="duration", x_label="Duration (s)"):
    """
    Plot regressions mediati su tutti i soggetti:
        - Per rep
        - Media totale
    """
    print("[plot_group_average] Plotting group average regressions...")
    all_dir = os.path.join(output_folder, "ALL_SUBJECTS")
    single_all = os.path.join(all_dir, "Single-reps")
    mean_all = os.path.join(all_dir, "Mean")
    os.makedirs(all_dir, exist_ok=True)
    os.makedirs(single_all, exist_ok=True)
    os.makedirs(mean_all, exist_ok=True)

    for pattern in patterns_list:
        df_pat = df[df["pattern_pair"] == pattern]

        # Per rep
        if per_reps_plot:
            rep_dir = os.path.join(single_all, pattern)
            os.makedirs(rep_dir, exist_ok=True)
            for rep in sorted(df_pat["rep"].unique()):
                df_rep_all = df_pat[df_pat["rep"] == rep].groupby("duration", as_index=False).agg({
                    "angle_deg": "mean",
                    "vividness": "mean",
                    "presentation_order": "first",
                    "duration": "first"
                })

                rep_dir = os.path.join(single_all)
                os.makedirs(rep_dir, exist_ok=True)

                # Angle vs Duration
                plot_regression(
                    df_rep_all.dropna(subset=["angle_deg"]),
                    x_col=x_col,
                    y_col="angle_deg",
                    title=f"ALL – {pattern} – rep {rep} – Angle",
                    ylabel="Angle (deg)",
                    xlabel=x_label,
                    save_path=os.path.join(rep_dir, pattern, f"rep{rep}_angle.png")
                )

                # Vividness vs Duration
                plot_regression(
                    df_rep_all.dropna(subset=["vividness"]),
                    x_col=x_col,
                    y_col="vividness",
                    title=f"ALL – {pattern} – rep {rep} – Vividness",
                    ylabel="Vividness",
                    xlabel=x_label,
                    save_path=os.path.join(rep_dir, pattern, f"rep{rep}_vividness.png"),
                    protocol=protocol
                )

                # Angle vs Vividness
                plot_regression(
                    df_rep_all.dropna(subset=["vividness"]),
                    x_col="vividness",
                    y_col="angle_deg",
                    title=f"ALL – {pattern} – rep {rep} – Angle vs Vividness",
                    ylabel="Angle (deg)",
                    xlabel="Vividness",
                    save_path=os.path.join(rep_dir, pattern, f"rep{rep}_angle_vs_vividness.png"),
                    protocol=protocol
                )

        if mean_plot:
            mean_dir_pattern = os.path.join(mean_all, pattern)
            os.makedirs(mean_dir_pattern, exist_ok=True)
            # Media totale
            df_mean_all = df_pat.groupby(x_col, as_index=False).agg({
                "angle_deg": "mean",
                "vividness": "mean",
                "presentation_order": "first",
                "duration": "first"
            })

            # Angle vs Duration
            plot_regression(
                df_mean_all.dropna(subset=["angle_deg"]),
                x_col=x_col,
                y_col="angle_deg",
                title=f"ALL – {pattern} – Mean – Angle",
                ylabel="Angle (deg)",
                xlabel=x_label,
                save_path=os.path.join(mean_all, pattern, f"mean_angle.png")
            )

            # Vividness vs Duration
            plot_regression(
                df_mean_all.dropna(subset=["vividness"]),
                x_col=x_col,
                y_col="vividness",
                title=f"ALL – {pattern} – Mean – Vividness",
                ylabel="Vividness",
                xlabel=x_label,
                save_path=os.path.join(mean_all, pattern, f"mean_vividness.png"),
                protocol=protocol
            )

            # Angle vs Vividness
            plot_regression(
                df_mean_all.dropna(subset=["vividness"]),
                x_col="vividness",
                y_col="angle_deg",
                title=f"ALL – {pattern} – Mean – Angle vs Vividness",
                ylabel="Angle (deg)",
                xlabel="Vividness",
                save_path=os.path.join(mean_all, pattern, f"mean_angle_vs_vividness.png"),
                protocol=protocol
            )

# ============================================================
# PLOT MEDIA SU TUTTI I SOGGETTI
# ============================================================
def plot_group_average_2(df, patterns_list, protocol, per_reps_plot = True, mean_plot = True, output_folder="Results/regressions", x_col="duration", x_label="Duration (s)"):
    """
    Plot regressions mediati su tutti i soggetti:
        - Per rep
        - Media totale
    """
    all_dir = os.path.join(output_folder, "ALL_SUBJECTS")
    single_all = os.path.join(all_dir, "Single-reps")
    mean_all = os.path.join(all_dir, "Mean")
    os.makedirs(single_all, exist_ok=True)
    os.makedirs(mean_all, exist_ok=True)

    for pattern in patterns_list:
        df_pat = df[df["pattern_pair"] == pattern]

        # Per rep
        if per_reps_plot:
            for rep in sorted(df_pat["rep"].unique()):
                df_rep_all = df_pat[df_pat["rep"] == rep].groupby(["duration", "rep"], as_index=False).agg({
                    "angle_deg": "mean",
                    "vividness": "mean",
                    "presentation_order": "first",
                    "duration": "first"
                })

                rep_dir = os.path.join(single_all, f"rep_{rep}")
                os.makedirs(rep_dir, exist_ok=True)

                # Angle vs Duration
                plot_regression(
                    df_rep_all.dropna(subset=["angle_deg"]),
                    x_col=x_col,
                    y_col="angle_deg",
                    title=f"ALL – {pattern} – rep {rep} – Angle",
                    ylabel="Angle (deg)",
                    xlabel=x_label,
                    save_path=os.path.join(rep_dir, pattern, f"rep{rep}_angle.png")
                )

                # Vividness vs Duration
                plot_regression(
                    df_rep_all.dropna(subset=["vividness"]),
                    x_col=x_col,
                    y_col="vividness",
                    title=f"ALL – {pattern} – rep {rep} – Vividness",
                    ylabel="Vividness",
                    xlabel=x_label,
                    save_path=os.path.join(rep_dir, pattern, f"rep{rep}_vividness.png"),
                    protocol=protocol
                )

                # Angle vs Vividness
                plot_regression(
                    df_rep_all.dropna(subset=["vividness"]),
                    x_col="vividness",
                    y_col="angle_deg",
                    title=f"ALL – {pattern} – rep {rep} – Angle vs Vividness",
                    ylabel="Angle (deg)",
                    xlabel="Vividness",
                    save_path=os.path.join(rep_dir, pattern, f"rep{rep}_angle_vs_vividness.png"),
                    protocol=protocol
                )

        if mean_plot:

            # 🔥 MEDIA SUI SOGGETTI, MANTENENDO LE REP
            df_mean_all = (
                df_pat
                .groupby(["duration", "rep"], as_index=False)
                .agg({
                    "angle_deg": "mean",
                    "vividness": "mean"
                })
            )

            mean_dir = os.path.join(mean_all, pattern)
            os.makedirs(mean_dir, exist_ok=True)

            # -------- ANGLE vs DURATION (9 punti, 1 regressione) --------
            plot_regression(
                df_mean_all.dropna(subset=["angle_deg"]),
                x_col="duration",
                y_col="angle_deg",
                title=f"ALL – {pattern} – Mean subjects (rep separated) – Angle",
                ylabel="Angle (deg)",
                xlabel="Duration (s)",
                save_path=os.path.join(mean_dir, f"mean_angle_9points.png"),
                protocol=protocol
            )

            # -------- VIVIDNESS vs DURATION --------
            plot_regression(
                df_mean_all.dropna(subset=["vividness"]),
                x_col="duration",
                y_col="vividness",
                title=f"ALL – {pattern} – Mean subjects (rep separated) – Vividness",
                ylabel="Vividness",
                xlabel="Duration (s)",
                save_path=os.path.join(mean_dir, f"mean_vividness_9points.png"),
                protocol=protocol
            )

            # -------- ANGLE vs VIVIDNESS --------
            plot_regression(
                df_mean_all.dropna(subset=["vividness"]),
                x_col="vividness",
                y_col="angle_deg",
                title=f"ALL – {pattern} – Mean subjects (rep separated) – Angle vs Vividness",
                ylabel="Angle (deg)",
                xlabel="Vividness",
                save_path=os.path.join(mean_dir, f"mean_angle_vs_vividness_9points.png"),
                protocol=protocol
            )

