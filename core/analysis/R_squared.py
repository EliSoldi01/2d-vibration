import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt

def plot_r2_global_vs_individual(
    results_df,
    save_path=None,
    show_subject_labels=True
):
    """
    Plot R² obtained with the global model vs subject-specific model.

    X-axis: R² using global parameters
    Y-axis: R² using subject-specific parameters

    The dashed diagonal represents y = x.
    Points above the diagonal indicate better performance
    of subject-specific calibration.
    """

    x = results_df["R2_global"]
    y = results_df["R2_individual"]

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        x,
        y,
        s=60,
        alpha=0.8
    )

    # Determine limits including the diagonal
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())

    margin = 0.1 * (max_val - min_val)

    lower = min_val - margin
    upper = max_val + margin

    # Identity line
    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.5,
        label="y = x"
    )

    # Subject labels
    if show_subject_labels:
        for _, row in results_df.iterrows():
            ax.annotate(
                row["Subject"],
                (
                    row["R2_global"],
                    row["R2_individual"]
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8
            )

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)

    ax.set_xlabel(r"$R^2$ — Global model")
    ax.set_ylabel(r"$R^2$ — Subject-specific model")

    ax.set_title(
        "Global vs subject-specific model performance"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

def plot_mae_global_vs_individual(
    results_df,
    save_path=None,
    show_subject_labels=True
):
    """
    Plot MAE obtained with the global model vs subject-specific model.

    Lower MAE indicates better prediction.
    Points below the y=x line indicate better performance
    of the subject-specific model.
    """

    x = results_df["MAE_global_deg"]
    y = results_df["MAE_individual_deg"]

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        x,
        y,
        s=60,
        alpha=0.8
    )

    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())

    margin = 0.1 * (max_val - min_val)

    lower = max(0, min_val - margin)
    upper = max_val + margin

    # Identity line
    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.5,
        label="y = x"
    )

    if show_subject_labels:
        for _, row in results_df.iterrows():
            ax.annotate(
                row["Subject"],
                (
                    row["MAE_global_deg"],
                    row["MAE_individual_deg"]
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8
            )

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)

    ax.set_xlabel("MAE — Global model [°]")
    ax.set_ylabel("MAE — Subject-specific model [°]")

    ax.set_title(
        "Global vs subject-specific prediction error"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

def plot_r2_distribution(
    results_df,
    save_path=None
):
    """
    Compare the distribution of R² values obtained with
    global and subject-specific models.
    """

    data = [
        results_df["R2_global"].values,
        results_df["R2_individual"].values
    ]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.boxplot(
        data,
        labels=[
            "Global model",
            "Subject-specific model"
        ],
        widths=0.5,
        showfliers=False
    )

    # Add individual observations
    rng = np.random.default_rng(42)

    for i, values in enumerate(data, start=1):

        jitter = rng.normal(
            loc=i,
            scale=0.04,
            size=len(values)
        )

        ax.scatter(
            jitter,
            values,
            s=35,
            alpha=0.7
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_ylabel(r"$R^2$")
    ax.set_title("Distribution of model performance across participants")

    ax.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

def plot_delta_r2(
    results_df,
    save_path=None
):
    """
    Plot the change in R² obtained by using subject-specific
    parameters instead of global parameters.

    Delta R² = R²_individual - R²_global
    """

    df = results_df.sort_values("Delta_R2")

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(
        df["Subject"],
        df["Delta_R2"]
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel("Subject")
    ax.set_ylabel(r"$\Delta R^2$ (Individual − Global)")

    ax.set_title(
        "Change in model performance with subject-specific calibration"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("results/phase1/model/subject_models")
GLOBAL_FILE = DATA_DIR / "group_validation.xlsx"

# Soggetti presenti nel dataset
subjects = [
    "S01", "S02", "S04", "S05", "S06", "S07", "S08",
    "S09", "S10", "S11", "S12", "S13", "S14", "S15",
    "S16", "S17", "S18", "S19", "S20", "S21", "S22"
]


# ============================================================
# LOAD GLOBAL VALIDATION
# ============================================================

global_df = pd.read_excel(GLOBAL_FILE, sheet_name="Validation")

# Controllo colonne
required_columns = ["pattern", "duration", "ideal_angle", "real_angle", ""]

# ============================================================
# CALCULATE METRICS
# ============================================================

results = []

for subject in subjects:

    subject_file = DATA_DIR / f"{subject}_validation.xlsx"

    if not subject_file.exists():
        print(f"⚠️ File non trovato: {subject_file}")
        continue

    # --------------------------------------------------------
    # Individual model
    # --------------------------------------------------------

    individual_df = pd.read_excel(subject_file)

    # --------------------------------------------------------
    # Match individual and global data
    # --------------------------------------------------------

    # Prendiamo la predizione globale e la risposta reale
    # dal group_validation.
    #
    # Usiamo pattern + duration per fare il matching.

    merged = individual_df[
        ["pattern", "duration", "real_angle", "ideal_angle"]
    ].merge(
        global_df[
            ["pattern", "duration", "ideal_angle"]
        ].rename(columns={
            "ideal_angle": "ideal_angle_global"
        }),
        on=["pattern", "duration"],
        how="inner"
    )

    # Rinominare la predizione individuale
    merged = merged.rename(columns={
        "ideal_angle": "ideal_angle_individual"
    })

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    merged = merged.dropna(
        subset=[
            "real_angle",
            "ideal_angle_individual",
            "ideal_angle_global"
        ]
    )

    # --------------------------------------------------------
    # Metrics: individual model
    # --------------------------------------------------------

    r2_individual = r2_score(
        merged["real_angle"],
        merged["ideal_angle_individual"]
    )

    mae_individual = mean_absolute_error(
        merged["real_angle"],
        merged["ideal_angle_individual"]
    )

    rmse_individual = np.sqrt(
        mean_squared_error(
            merged["real_angle"],
            merged["ideal_angle_individual"]
        )
    )

    # --------------------------------------------------------
    # Metrics: global model
    # --------------------------------------------------------

    r2_global = r2_score(
        merged["real_angle"],
        merged["ideal_angle_global"]
    )

    mae_global = mean_absolute_error(
        merged["real_angle"],
        merged["ideal_angle_global"]
    )

    rmse_global = np.sqrt(
        mean_squared_error(
            merged["real_angle"],
            merged["ideal_angle_global"]
        )
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "Subject": subject,

        "R2_global": r2_global,
        "R2_individual": r2_individual,

        "MAE_global_deg": mae_global,
        "MAE_individual_deg": mae_individual,

        "RMSE_global_deg": rmse_global,
        "RMSE_individual_deg": rmse_individual,

        "N": len(merged)
    })


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

# Difference individual - global
results_df["Delta_R2"] = (
    results_df["R2_individual"]
    - results_df["R2_global"]
)

results_df["Delta_MAE_deg"] = (
    results_df["MAE_global_deg"]
    - results_df["MAE_individual_deg"]
)

results_df["Delta_RMSE_deg"] = (
    results_df["RMSE_global_deg"]
    - results_df["RMSE_individual_deg"]
)

plot_r2_global_vs_individual(
    results_df,
    save_path="R2_global_vs_individual.png"
)

plot_mae_global_vs_individual(
    results_df,
    save_path="MAE_global_vs_individual.png"
)

plot_r2_distribution(
    results_df,
    save_path="R2_distribution.png"
)

plot_delta_r2(
    results_df,
    save_path="Delta_R2.png"
)

# ============================================================
# PRINT
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.3f}".format)

print("\n==============================================")
print("GLOBAL vs INDIVIDUAL MODEL")
print("==============================================\n")

print(results_df)


# ============================================================
# SAVE
# ============================================================

output_file = DATA_DIR / "global_vs_individual_results.xlsx"

results_df.to_excel(
    output_file,
    index=False
)

print(f"\n✅ Risultati salvati in: {output_file}")