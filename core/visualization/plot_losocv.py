import matplotlib.pyplot as plt


def plot_loso_performance(results, output_path=None):
    """Plot individual and pooled LOSO performance for all and complex patterns."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # ========================================================
    # ALL PATTERNS
    # ========================================================

    ax = axes[0]

    all_results = results["all_patterns"]["subjects_results"]
    pooled_point = results["all_patterns"]["pooled"]

    for result in all_results:
        ax.scatter(
            result["MAE"],
            result["R2_zero"],
            color="#7cb4fd"
        )

        ax.annotate(
            result["subject"],
            (result["MAE"], result["R2_zero"]),
            xytext=(5, 5),
            textcoords="offset points"
        )

    ax.scatter(
        pooled_point["MAE"],
        pooled_point["R2_zero"],
        marker="D",
        color="red",
        s=80,
        label="Pooled"
    )

    all_r2 = [result["R2_zero"] for result in all_results]
    all_r2.append(pooled_point["R2_zero"])

    ax.set_ylim(
        min(all_r2) - 0.5,
        max(all_r2) + 0.5
    )

    ax.axhline(y=0, color="blue", linestyle="--", alpha=0.4)
    ax.set_title("All patterns")
    ax.set_xlabel("MAE (°)")
    ax.set_ylabel(r"$R^2$")
    ax.margins(x=0.15)
    ax.legend()

    # ========================================================
    # COMPLEX PATTERNS
    # ========================================================

    ax = axes[1]

    complex_results = results["complex_patterns"]["subjects_results"]
    complex_pooled_point = results["complex_patterns"]["pooled"]

    for result in complex_results:
        ax.scatter(
            result["MAE"],
            result["R2_zero"],
            color="#7cb4fd"
        )

        ax.annotate(
            result["subject"],
            (result["MAE"], result["R2_zero"]),
            xytext=(5, 5),
            textcoords="offset points"
        )

    ax.scatter(
        complex_pooled_point["MAE"],
        complex_pooled_point["R2_zero"],
        marker="D",
        color="red",
        s=80,
        label="Pooled"
    )

    complex_r2 = [
        result["R2_zero"]
        for result in complex_results
    ]
    complex_r2.append(complex_pooled_point["R2_zero"])

    ax.set_ylim(
        min(complex_r2) - 0.5,
        max(complex_r2) + 0.5
    )

    ax.axhline(y=0, color="blue", linestyle="--", alpha=0.4)
    ax.set_title("Complex patterns")
    ax.set_xlabel("MAE (°)")
    ax.set_ylabel(r"$R^2$")
    ax.margins(x=0.15)
    ax.legend()

    # ========================================================
    # FIGURE
    # ========================================================

    fig.suptitle("Leave-One-Subject-Out Cross-Validation")

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()