import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_pattern_trajectories(df, filename,
                              start_pos=(7,10),
                              grid_x=14,
                              grid_y=18):

    # --- individua colonne step ---
    x_cols = [c for c in df.columns if "_x" in c]
    y_cols = [c for c in df.columns if "_y" in c]

    x_cols = sorted(x_cols, key=lambda x: int(x.split("step")[1].split("_")[0]))
    y_cols = sorted(y_cols, key=lambda x: int(x.split("step")[1].split("_")[0]))

    patterns = sorted(df["pattern_pair"].unique())

    fig, axes = plt.subplots(3, 4, figsize=(16,12))
    axes = axes.flatten()

    for i, pattern in enumerate(patterns):

        ax = axes[i]
        df_p = df[df["pattern_pair"] == pattern]

        # --- setup griglia ---
        ax.set_xlim(0.5, grid_x + 0.5)
        ax.set_ylim(0.5, grid_y + 0.5)
        ax.set_xticks(np.arange(1, grid_x+1))
        ax.set_yticks(np.arange(1, grid_y+1))
        ax.set_aspect("equal")
        ax.invert_yaxis()

        # linee griglia
        for x in range(1, grid_x+1):
            ax.axvline(x-0.5, color="lightgrey", lw=0.8)

        for y in range(1, grid_y+1):
            ax.axhline(y-0.5, color="lightgrey", lw=0.8)

        # --- traiettorie ---
        for _, row in df_p.iterrows():

            xs = pd.to_numeric(row[x_cols], errors="coerce").values
            ys = pd.to_numeric(row[y_cols], errors="coerce").values

            mask = ~np.isnan(xs) & ~np.isnan(ys)
            xs = xs[mask]
            ys = ys[mask]

            if len(xs) > 0:
                ax.plot(xs, ys, marker="o", linewidth=1.5, alpha=0.7)

        # start position
        ax.scatter(*start_pos, c="red", s=80, marker="x", linewidths=2)

        ax.set_title(pattern)

    # rimuove subplot vuoti
    for j in range(len(patterns), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

    print("Saved:", filename)


df = pd.read_excel("C:\\Users\\Utente\\Desktop\\Elisa\\Research\\2D-vibration\\Phase4-Trajectory\\Results\\Subjects\\S01\\Exp-completo-01.xlsx")

plot_pattern_trajectories(
    df,
    "trajectories.png"
)