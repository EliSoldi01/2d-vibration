import matplotlib.pyplot as plt


def plot_loso_performance(results, output_path=None):
    """Plot individual and pooled LOSO performance for all and complex patterns."""

    plt.rcParams["font.family"] = "Times New Roman"
    
    def plot_panel(ax, pattern_type):
        """Plot LOSO performance for a single pattern type."""

        pattern_results = results[pattern_type]["subjects_results"]
        pooled_point = results[pattern_type]["pooled"]

        for result in pattern_results:
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

        r2_values = [
            result["R2_zero"]
            for result in pattern_results
        ]
        r2_values.append(pooled_point["R2_zero"])

        ax.set_ylim(
            min(r2_values) - 0.5,
            max(r2_values) + 0.5
        )

        ax.axhline(
            y=0,
            color="blue",
            linestyle="--",
            alpha=0.4
        )

        ax.set_title(
            "All patterns"
            if pattern_type == "all_patterns"
            else "Complex patterns"
        )
        ax.set_xlabel("MAE (°)")
        ax.set_ylabel(r"$R^2$")
        ax.margins(x=0.15)
        ax.legend()

    # ========================================================
    # COMBINED FIGURE
    # ========================================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    plot_panel(axes[0], "all_patterns")
    plot_panel(axes[1], "complex_patterns")

    fig.suptitle("Leave-One-Subject-Out Cross-Validation")

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    # plt.show()

    # ========================================================
    # INDIVIDUAL FIGURES
    # ========================================================

    if output_path is not None:

        output_path = output_path.parent

        for pattern_type in ["all_patterns", "complex_patterns"]:

            fig, ax = plt.subplots(figsize=(7, 7))

            plot_panel(ax, pattern_type)

            fig.suptitle(
                "Leave-One-Subject-Out Cross-Validation"
            )

            plt.tight_layout()

            filename = (
                "loso_performance_all_patterns.png"
                if pattern_type == "all_patterns"
                else "loso_performance_complex_patterns.png"
            )

            fig.savefig(
                output_path / filename,
                dpi=300,
                bbox_inches="tight"
            )

            # plt.show()