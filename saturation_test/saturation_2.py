import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "saturation_test\\data.xlsx"
SHEET_NAME = "Complete"

OUTPUT_DIR = "saturation_test\\saturation_2_results"

SUBJECT_COL = "subject"
DURATION_COL = "duration"
PATTERN_BICEPS_COL = "pattern_biceps"
PATTERN_TRICEPS_COL = "pattern_triceps"
PATTERN_COL = "pattern_pair"
DIRECTION_COL = "direction"
ANGLE_COL = "angle_deg"


# Directions to analyze with saturation models
SATURATION_DIRECTIONS = [
    "extension",
    "flexion",
]


# ============================================================
# SATURATION CONFIGURATION
# ============================================================

# Fraction of the fitted asymptote
ASYMPTOTE_THRESHOLDS = {
    "Tanh 90%": 0.90,
    "Tanh 95%": 0.95,
}

# Saturation according to derivative:
# saturation when slope <= this fraction
# of the initial slope.
SLOPE_THRESHOLD = 0.10

# Empirical plateau:
# the increment must remain below this fraction
# of the observed response range for TWO consecutive intervals.
EMPIRICAL_PLATEAU_THRESHOLD = 0.25

# Number of consecutive small increments required
EMPIRICAL_PLATEAU_CONSECUTIVE = 2

FIGURE_DPI = 300


# ============================================================
# MODEL FUNCTIONS
# ============================================================

def linear_origin(x, a):
    """
    Linear model constrained through the origin.

        y = a*x
    """
    return a * x


def tanh_model(x, L, k):
    """
    Saturating tanh model.

        y = L * tanh(k*x)

    L = asymptotic response
    k = growth rate
    """
    return L * np.tanh(k * x)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME
    )

    required_columns = [
        SUBJECT_COL,
        DURATION_COL,
        PATTERN_BICEPS_COL,
        PATTERN_TRICEPS_COL,
        PATTERN_COL,
        DIRECTION_COL,
        ANGLE_COL,
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {col}"
                for col in missing
            )
        )

    df = df.copy()

    df[DURATION_COL] = pd.to_numeric(
        df[DURATION_COL],
        errors="coerce"
    )

    df[ANGLE_COL] = pd.to_numeric(
        df[ANGLE_COL],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            SUBJECT_COL,
            DURATION_COL,
            PATTERN_COL,
            DIRECTION_COL,
            ANGLE_COL,
        ]
    )

    # Standardize direction strings
    df[DIRECTION_COL] = (
        df[DIRECTION_COL]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    print(
        f"\nNumber of observations: {len(df)}"
    )

    print(
        f"Subjects: "
        f"{df[SUBJECT_COL].nunique()}"
    )

    print("\nDirections found:")
    print(
        df[DIRECTION_COL]
        .value_counts()
        .to_string()
    )

    print("\nPatterns found:")
    print(
        df[PATTERN_COL]
        .value_counts()
        .to_string()
    )

    print("\nNominal durations found:")
    print(
        sorted(
            df[DURATION_COL]
            .unique()
            .tolist()
        )
    )

    return df


# ============================================================
# COUNT ACTIVE VIBRATORS
# ============================================================

def count_active_vibrators(pattern):

    """
    Count the number of active vibrators represented by
    the binary pattern.

    Examples:

        001 -> 1
        011 -> 2
        111 -> 3
        100 -> 1
        000 -> 0
    """

    pattern = str(pattern).strip()

    return pattern.count("1")


# ============================================================
# PREPARE EFFECTIVE STIMULATION DURATION
# ============================================================

def prepare_effective_duration(df):

    """
    Calculate the effective stimulation duration for each trial.

    The effective duration is defined as:

        nominal duration × total number of active vibrators

    where:

        total active vibrators =
            active biceps vibrators
            +
            active triceps vibrators

    Examples:

        001_000 @ 3 s
            -> 1 vibrator
            -> 3 s effective duration

        011_100 @ 3 s
            -> 2 + 1 = 3 vibrators
            -> 9 s effective duration

        100_011 @ 3 s
            -> 1 + 2 = 3 vibrators
            -> 9 s effective duration

        100_001 @ 3 s
            -> 1 + 1 = 2 vibrators
            -> 6 s effective duration

    The input Excel file is never modified.
    """

    df = df.copy()

    df["n_biceps_vibrators"] = (
        df[PATTERN_BICEPS_COL]
        .apply(count_active_vibrators)
    )

    df["n_triceps_vibrators"] = (
        df[PATTERN_TRICEPS_COL]
        .apply(count_active_vibrators)
    )

    df["n_total_vibrators"] = (
        df["n_biceps_vibrators"]
        + df["n_triceps_vibrators"]
    )

    df["effective_stimulation_duration"] = (
        df[DURATION_COL]
        * df["n_total_vibrators"]
    )

    # --------------------------------------------------------
    # Response magnitude
    #
    # Extension = positive magnitude
    # Flexion = positive magnitude
    #
    # Control / symmetric are kept separately.
    # --------------------------------------------------------

    df["response_magnitude"] = np.abs(
        df[ANGLE_COL]
    )

    df[
        "effective_stimulation_duration"
    ] = pd.to_numeric(
        df[
            "effective_stimulation_duration"
        ],
        errors="coerce"
    )

    print("\n")
    print("=" * 70)
    print("EFFECTIVE STIMULATION DURATIONS")
    print("=" * 70)

    summary = (
        df[
            [
                DIRECTION_COL,
                DURATION_COL,
                "n_total_vibrators",
                "effective_stimulation_duration",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                DIRECTION_COL,
                "effective_stimulation_duration",
            ]
        )
    )

    print(
        summary.to_string(
            index=False
        )
    )

    return df


# ============================================================
# AGGREGATE BY DIRECTION AND EFFECTIVE DURATION
# ============================================================

def aggregate_by_effective_duration(
    df,
    direction
):

    data = df[
        df[DIRECTION_COL] == direction
    ].copy()

    grouped = (
        data
        .groupby(
            "effective_stimulation_duration"
        )["response_magnitude"]
        .agg(
            mean_angle="mean",
            sd_angle="std",
            n="count"
        )
        .reset_index()
        .sort_values(
            "effective_stimulation_duration"
        )
    )

    grouped["se"] = (
        grouped["sd_angle"]
        / np.sqrt(
            grouped["n"]
        )
    )

    return grouped


# ============================================================
# FIT LINEAR MODEL
# ============================================================

def fit_linear_origin(
    x,
    y
):

    try:

        popt, _ = curve_fit(
            linear_origin,
            x,
            y
        )

        y_pred = linear_origin(
            x,
            *popt
        )

        residuals = (
            y - y_pred
        )

        rss = np.sum(
            residuals ** 2
        )

        n = len(y)
        n_params = 1

        if rss > 0:

            aic = (
                n
                * np.log(
                    rss / n
                )
                + 2 * n_params
            )

        else:

            aic = -np.inf

        ss_tot = np.sum(
            (
                y
                - np.mean(y)
            ) ** 2
        )

        if ss_tot > 0:

            r2 = (
                1
                - rss / ss_tot
            )

        else:

            r2 = np.nan

        return {
            "ok": True,
            "model": "linear",
            "params": popt,
            "rss": rss,
            "aic": aic,
            "r2": r2,
        }

    except Exception as e:

        return {
            "ok": False,
            "model": "linear",
            "error": str(e),
        }


# ============================================================
# FIT TANH
# ============================================================

def fit_tanh(
    x,
    y
):

    """
    Fit:

        y = L*tanh(k*x)

    Both L and k are constrained to positive values because
    response magnitude is used.
    """

    try:

        max_y = np.max(
            np.abs(y)
        )

        if max_y <= 0:
            return {
                "ok": False,
                "model": "tanh",
                "error": "No positive response."
            }

        L0 = max_y * 1.5

        x_scale = np.max(x)

        if x_scale <= 0:
            x_scale = 1.0

        k0 = 1.0 / x_scale

        popt, _ = curve_fit(
            tanh_model,
            x,
            y,
            p0=[
                L0,
                k0
            ],
            bounds=(
                [
                    1e-8,
                    1e-8
                ],
                [
                    np.inf,
                    np.inf
                ]
            ),
            maxfev=100000
        )

        y_pred = tanh_model(
            x,
            *popt
        )

        residuals = (
            y - y_pred
        )

        rss = np.sum(
            residuals ** 2
        )

        n = len(y)
        n_params = 2

        if rss > 0:

            aic = (
                n
                * np.log(
                    rss / n
                )
                + 2 * n_params
            )

        else:

            aic = -np.inf

        ss_tot = np.sum(
            (
                y
                - np.mean(y)
            ) ** 2
        )

        if ss_tot > 0:

            r2 = (
                1
                - rss / ss_tot
            )

        else:

            r2 = np.nan

        mae = np.mean(
            np.abs(residuals)
        )

        rmse = np.sqrt(
            np.mean(
                residuals ** 2
            )
        )

        return {
            "ok": True,
            "model": "tanh",
            "params": popt,
            "L": popt[0],
            "k": popt[1],
            "rss": rss,
            "aic": aic,
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
        }

    except Exception as e:

        return {
            "ok": False,
            "model": "tanh",
            "error": str(e),
        }


# ============================================================
# MODEL COMPARISON
# ============================================================

def compare_models(
    x,
    y
):

    linear = fit_linear_origin(
        x,
        y
    )

    tanh = fit_tanh(
        x,
        y
    )

    rows = []

    for result in [
        linear,
        tanh,
    ]:

        if result["ok"]:

            rows.append({
                "model":
                    result["model"],
                "AIC":
                    result["aic"],
                "R2":
                    result["r2"],
                "RSS":
                    result["rss"],
                "MAE":
                    result.get(
                        "mae",
                        np.nan
                    ),
                "RMSE":
                    result.get(
                        "rmse",
                        np.nan
                    ),
            })

        else:

            rows.append({
                "model":
                    result["model"],
                "AIC":
                    np.nan,
                "R2":
                    np.nan,
                "RSS":
                    np.nan,
                "MAE":
                    np.nan,
                "RMSE":
                    np.nan,
            })

    return (
        pd.DataFrame(rows),
        {
            "linear": linear,
            "tanh": tanh,
        }
    )


# ============================================================
# TANH ASYMPTOTIC SATURATION
# ============================================================

def calculate_tanh_saturation(
    tanh_result,
    threshold
):

    if not tanh_result["ok"]:
        return {
            "ok": False
        }

    L = tanh_result["L"]
    k = tanh_result["k"]

    x_sat = (
        np.arctanh(
            threshold
        )
        / k
    )

    y_sat = (
        threshold
        * L
    )

    return {
        "ok": True,
        "threshold": threshold,
        "effective_duration_sat": x_sat,
        "angle_sat": y_sat,
    }


# ============================================================
# SLOPE THRESHOLD
# ============================================================

def calculate_slope_threshold_saturation(
    tanh_result,
    slope_fraction
):

    if not tanh_result["ok"]:
        return {
            "ok": False
        }

    L = tanh_result["L"]
    k = tanh_result["k"]

    x_sat = (
        np.arccosh(
            1
            / np.sqrt(
                slope_fraction
            )
        )
        / k
    )

    y_sat = tanh_model(
        x_sat,
        L,
        k
    )

    return {
        "ok": True,
        "slope_fraction":
            slope_fraction,
        "effective_duration_sat":
            x_sat,
        "angle_sat":
            y_sat,
    }


# ============================================================
# EMPIRICAL PLATEAU
# ============================================================

def calculate_empirical_plateau(
    grouped,
    threshold,
    consecutive_required=2
):

    """
    Empirical sustained-plateau criterion.

    A plateau candidate is identified when the absolute
    increment remains below:

        threshold * observed response range

    for at least `consecutive_required` consecutive intervals.

    The candidate is accepted only if no subsequent increment
    exceeds the same threshold.
    """

    if len(grouped) < (
        consecutive_required + 1
    ):

        return {
            "ok": False,
            "reason":
                "Not enough duration levels."
        }

    x = grouped[
        "effective_stimulation_duration"
    ].to_numpy()

    y = grouped[
        "mean_angle"
    ].to_numpy()

    # --------------------------------------------------------
    # Exclude baseline
    # --------------------------------------------------------

    valid = x > 0

    x = x[valid]
    y = y[valid]

    if len(x) < (
        consecutive_required + 1
    ):

        return {
            "ok": False,
            "reason":
                "Not enough non-zero duration levels."
        }

    # --------------------------------------------------------
    # Observed response range
    # --------------------------------------------------------

    response_range = (
        np.max(y)
        - np.min(y)
    )

    if response_range <= 0:

        return {
            "ok": False,
            "reason":
                "Zero response range."
        }

    threshold_value = (
        threshold
        * response_range
    )

    increments = np.abs(
        np.diff(y)
    )

    n_intervals = len(increments)

    for start_index in range(
        n_intervals
        - consecutive_required
        + 1
    ):

        candidate_increments = (
            increments[
                start_index:
                start_index
                + consecutive_required
            ]
        )

        candidate_is_small = np.all(
            candidate_increments
            <= threshold_value
        )

        if not candidate_is_small:
            continue

        subsequent_increments = (
            increments[
                start_index
                + consecutive_required:
            ]
        )

        later_increase = np.any(
            subsequent_increments
            > threshold_value
        )

        if later_increase:
            continue

        plateau_start_index = start_index

        plateau_end_index = (
            start_index
            + consecutive_required
        )

        return {
            "ok": True,

            "effective_duration_sat":
                x[plateau_start_index],

            "angle_sat":
                y[plateau_start_index],

            "next_duration":
                x[plateau_end_index],

            "next_angle":
                y[plateau_end_index],

            "threshold_fraction":
                threshold,

            "threshold_value":
                threshold_value,

            "response_range":
                response_range,

            "n_consecutive":
                consecutive_required,

            "plateau_start_index":
                plateau_start_index,

            "plateau_end_index":
                plateau_end_index,
        }

    return {
        "ok": False,
        "reason":
            "No sustained empirical plateau detected.",
        "threshold_fraction":
            threshold,
        "threshold_value":
            threshold_value,
        "response_range":
            response_range,
    }


# ============================================================
# DETERMINE WHETHER SATURATION IS OBSERVED
# ============================================================

def classify_saturation_range(
    x_max_observed,
    x_sat
):

    if not np.isfinite(x_sat):

        return "undefined"

    if x_sat <= x_max_observed:

        return "within observed range"

    return "extrapolated beyond observed range"


# ============================================================
# PLOT INDIVIDUAL PATTERNS
# ============================================================

def plot_individual_patterns(
    df,
    direction
):

    data_direction = df[
        df[DIRECTION_COL] == direction
    ].copy()

    patterns = sorted(
        data_direction[
            PATTERN_COL
        ].unique()
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for pattern in patterns:

        data = data_direction[
            data_direction[PATTERN_COL]
            == pattern
        ].copy()

        grouped = (
            data
            .groupby(
                "effective_stimulation_duration"
            )["response_magnitude"]
            .agg(
                mean="mean",
                sd="std",
                n="count"
            )
            .reset_index()
            .sort_values(
                "effective_stimulation_duration"
            )
        )

        ax.errorbar(
            grouped[
                "effective_stimulation_duration"
            ],
            grouped["mean"],
            yerr=grouped["sd"],
            marker="o",
            capsize=4,
            label=pattern
        )

    ax.axhline(
        0,
        linewidth=0.8
    )

    ax.set_xlabel(
        "Effective stimulation duration (s)"
    )

    ax.set_ylabel(
        "Illusion magnitude (deg)"
    )

    ax.set_title(
        f"{direction.capitalize()}: individual patterns"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        f"{direction}_individual_patterns.png"
    )

    fig.savefig(
        path,
        dpi=FIGURE_DPI
    )

    plt.close(fig)

    return path


# ============================================================
# PLOT COMBINED SATURATION CURVE
# ============================================================

def plot_combined_curve(
    df,
    direction,
    tanh_result,
    saturation_results
):

    grouped = aggregate_by_effective_duration(
        df,
        direction
    )

    x_all = grouped[
        "effective_stimulation_duration"
    ].to_numpy()

    y_all = grouped[
        "mean_angle"
    ].to_numpy()

    x_max = np.max(
        x_all
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.errorbar(
        x_all,
        y_all,
        yerr=grouped[
            "sd_angle"
        ],
        fmt="o",
        capsize=4,
        label="Observed data"
    )

    # --------------------------------------------------------
    # Tanh
    # --------------------------------------------------------

    if tanh_result["ok"]:

        x_curve = np.linspace(
            0,
            max(
                x_max * 1.5,
                1
            ),
            500
        )

        y_curve = tanh_model(
            x_curve,
            tanh_result["L"],
            tanh_result["k"]
        )

        ax.plot(
            x_curve,
            y_curve,
            label="Tanh fit"
        )

    # --------------------------------------------------------
    # Saturation markers
    # --------------------------------------------------------

    for label, result in (
        saturation_results.items()
    ):

        if (
            result is None
            or not result.get(
                "ok",
                False
            )
        ):
            continue

        xs = result[
            "effective_duration_sat"
        ]

        ys = result[
            "angle_sat"
        ]

        ax.scatter(
            xs,
            ys,
            s=70,
            zorder=5,
            label=label
        )

        ax.axvline(
            xs,
            linestyle="--",
            alpha=0.5
        )

    ax.axhline(
        0,
        linewidth=0.8
    )

    ax.set_xlabel(
        "Effective stimulation duration (s)"
    )

    ax.set_ylabel(
        "Illusion magnitude (deg)"
    )

    ax.set_title(
        f"{direction.capitalize()}: saturation analysis"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        f"{direction}_saturation_curve.png"
    )

    fig.savefig(
        path,
        dpi=FIGURE_DPI
    )

    plt.close(fig)

    return path


# ============================================================
# ANALYZE ONE DIRECTION
# ============================================================

def analyze_direction(
    df,
    direction
):

    print("\n")
    print("=" * 70)
    print(
        f"{direction.upper()} SATURATION ANALYSIS"
    )
    print("=" * 70)

    grouped = aggregate_by_effective_duration(
        df,
        direction
    )

    print("\nDuration-level data:")
    print(
        grouped.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Remove baseline for model fitting
    # --------------------------------------------------------

    fit_data = grouped[
        grouped[
            "effective_stimulation_duration"
        ] > 0
    ].copy()

    if len(fit_data) < 2:

        print(
            "\nNot enough non-zero duration levels "
            "for model fitting."
        )

        return None

    x = fit_data[
        "effective_stimulation_duration"
    ].to_numpy()

    y = fit_data[
        "mean_angle"
    ].to_numpy()

    x_max_observed = np.max(x)

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    comparison, models = compare_models(
        x,
        y
    )

    print("\nModel comparison:")
    print(
        comparison.to_string(
            index=False
        )
    )

    tanh_result = models[
        "tanh"
    ]

    # --------------------------------------------------------
    # Tanh parameters
    # --------------------------------------------------------

    if tanh_result["ok"]:

        print("\nTanh parameters:")

        print(
            f"  L = "
            f"{tanh_result['L']:.6f}°"
        )

        print(
            f"  k = "
            f"{tanh_result['k']:.6f} s⁻¹"
        )

    else:

        print(
            "\nTanh fit failed:"
        )

        print(
            tanh_result.get(
                "error",
                "Unknown error"
            )
        )

    # --------------------------------------------------------
    # Saturation results
    # --------------------------------------------------------

    saturation_results = {}

    saturation_rows = []

    print(
        "\nSaturation estimates:"
    )

    # --------------------------------------------------------
    # Tanh 90 / 95
    # --------------------------------------------------------

    if tanh_result["ok"]:

        for label, threshold in (
            ASYMPTOTE_THRESHOLDS.items()
        ):

            result = (
                calculate_tanh_saturation(
                    tanh_result,
                    threshold
                )
            )

            saturation_results[
                label
            ] = result

            if result["ok"]:

                duration_sat = result[
                    "effective_duration_sat"
                ]

                angle_sat = result[
                    "angle_sat"
                ]

                status = (
                    classify_saturation_range(
                        x_max_observed,
                        duration_sat
                    )
                )

                print(
                    f"\n  {label}:"
                )

                print(
                    f"    Effective duration = "
                    f"{duration_sat:.3f} s"
                )

                print(
                    f"    Angle = "
                    f"{angle_sat:.3f}°"
                )

                print(
                    f"    Status = "
                    f"{status}"
                )

                saturation_rows.append({
                    "direction":
                        direction,

                    "method":
                        label,

                    "effective_duration_sat_s":
                        duration_sat,

                    "angle_sat_deg":
                        angle_sat,

                    "observed_max_duration_s":
                        x_max_observed,

                    "status":
                        status,
                })

    # --------------------------------------------------------
    # Slope threshold
    # --------------------------------------------------------

    if tanh_result["ok"]:

        slope_result = (
            calculate_slope_threshold_saturation(
                tanh_result,
                SLOPE_THRESHOLD
            )
        )

        label = (
            f"Slope <= "
            f"{SLOPE_THRESHOLD:.0%}"
        )

        saturation_results[
            label
        ] = slope_result

        if slope_result["ok"]:

            duration_sat = (
                slope_result[
                    "effective_duration_sat"
                ]
            )

            angle_sat = (
                slope_result[
                    "angle_sat"
                ]
            )

            status = (
                classify_saturation_range(
                    x_max_observed,
                    duration_sat
                )
            )

            print(
                f"\n  {label}:"
            )

            print(
                f"    Effective duration = "
                f"{duration_sat:.3f} s"
            )

            print(
                f"    Angle = "
                f"{angle_sat:.3f}°"
            )

            print(
                f"    Status = "
                f"{status}"
            )

            saturation_rows.append({
                "direction":
                    direction,

                "method":
                    label,

                "effective_duration_sat_s":
                    duration_sat,

                "angle_sat_deg":
                    angle_sat,

                "observed_max_duration_s":
                    x_max_observed,

                "status":
                    status,
            })

    # --------------------------------------------------------
    # Empirical plateau
    # --------------------------------------------------------

    empirical_result = (
        calculate_empirical_plateau(
            grouped,
            EMPIRICAL_PLATEAU_THRESHOLD,
            EMPIRICAL_PLATEAU_CONSECUTIVE
        )
    )

    saturation_results[
        "Empirical plateau"
    ] = empirical_result

    if empirical_result["ok"]:

        duration_sat = (
            empirical_result[
                "effective_duration_sat"
            ]
        )

        angle_sat = (
            empirical_result[
                "angle_sat"
            ]
        )

        next_duration = (
            empirical_result[
                "next_duration"
            ]
        )

        next_angle = (
            empirical_result[
                "next_angle"
            ]
        )

        threshold_value = (
            empirical_result[
                "threshold_value"
            ]
        )

        print(
            "\n  Empirical plateau:"
        )

        print(
            f"    Effective duration = "
            f"{duration_sat:.3f} s"
        )

        print(
            f"    Angle = "
            f"{angle_sat:.3f}°"
        )

        print(
            f"    Plateau remains stable through = "
            f"{next_duration:.3f} s"
        )

        print(
            f"    Angle at end of plateau window = "
            f"{next_angle:.3f}°"
        )

        print(
            f"    Threshold = "
            f"{EMPIRICAL_PLATEAU_THRESHOLD:.1%} "
            f"of observed range"
        )

        print(
            f"    Absolute increment threshold = "
            f"{threshold_value:.3f}°"
        )

        print(
            f"    Consecutive intervals = "
            f"{empirical_result['n_consecutive']}"
        )

        saturation_rows.append({
            "direction":
                direction,

            "method":
                "Empirical plateau",

            "effective_duration_sat_s":
                duration_sat,

            "angle_sat_deg":
                angle_sat,

            "plateau_end_duration_s":
                next_duration,

            "plateau_end_angle_deg":
                next_angle,

            "threshold_fraction":
                empirical_result[
                    "threshold_fraction"
                ],

            "threshold_value_deg":
                threshold_value,

            "observed_response_range_deg":
                empirical_result[
                    "response_range"
                ],

            "n_consecutive":
                empirical_result[
                    "n_consecutive"
                ],

            "observed_max_duration_s":
                x_max_observed,

            "status":
                "within observed range",
        })

    else:

        print(
            "\n  Empirical plateau:"
            " not detected."
        )

        print(
            f"    Reason: "
            f"{empirical_result.get('reason', '')}"
        )

        if (
            "threshold_value"
            in empirical_result
        ):

            print(
                f"    Threshold = "
                f"{EMPIRICAL_PLATEAU_THRESHOLD:.1%} "
                f"of observed range"
            )

            print(
                f"    Absolute increment threshold = "
                f"{empirical_result['threshold_value']:.3f}°"
            )

    saturation_table = pd.DataFrame(
        saturation_rows
    )

    return {
        "grouped":
            grouped,

        "fit_data":
            fit_data,

        "comparison":
            comparison,

        "models":
            models,

        "tanh":
            tanh_result,

        "saturation":
            saturation_table,

        "saturation_results":
            saturation_results,
    }


# ============================================================
# ANALYZE OTHER DIRECTIONS DESCRIPTIVELY
# ============================================================

def analyze_other_directions(
    df
):

    other_directions = [
        direction
        for direction in df[DIRECTION_COL].unique()
        if direction not in SATURATION_DIRECTIONS
    ]

    results = {}

    for direction in other_directions:

        print("\n")
        print("=" * 70)
        print(
            f"{direction.upper()} DESCRIPTIVE ANALYSIS"
        )
        print("=" * 70)

        grouped = aggregate_by_effective_duration(
            df,
            direction
        )

        print(
            grouped.to_string(
                index=False
            )
        )

        results[direction] = grouped

    return results


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    direction_results,
    other_direction_results
):

    output_file = os.path.join(
        OUTPUT_DIR,
        "saturation_results.xlsx"
    )

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl"
    ) as writer:

        # ----------------------------------------------------
        # Saturation directions
        # ----------------------------------------------------

        for direction, results in (
            direction_results.items()
        ):

            if results is None:
                continue

            direction_name = (
                direction.capitalize()
            )

            results[
                "grouped"
            ].to_excel(
                writer,
                sheet_name=f"{direction_name}_duration",
                index=False
            )

            results[
                "comparison"
            ].to_excel(
                writer,
                sheet_name=f"{direction_name}_models",
                index=False
            )

            results[
                "saturation"
            ].to_excel(
                writer,
                sheet_name=f"{direction_name}_saturation",
                index=False
            )

        # ----------------------------------------------------
        # Other directions
        # ----------------------------------------------------

        for direction, grouped in (
            other_direction_results.items()
        ):

            sheet_name = (
                direction.capitalize()
            )

            # Excel sheet names max 31 characters
            sheet_name = sheet_name[:31]

            grouped.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )

    print(
        f"\nResults saved to:\n"
        f"{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Effective duration
    # --------------------------------------------------------

    df = prepare_effective_duration(
        df
    )

    # --------------------------------------------------------
    # Analyze extension / flexion
    # --------------------------------------------------------

    direction_results = {}

    for direction in SATURATION_DIRECTIONS:

        if direction not in df[
            DIRECTION_COL
        ].unique():

            print(
                f"\nDirection '{direction}' "
                f"not found in dataset."
            )

            direction_results[
                direction
            ] = None

            continue

        direction_results[
            direction
        ] = analyze_direction(
            df,
            direction
        )

    # --------------------------------------------------------
    # Analyze control / symmetric descriptively
    # --------------------------------------------------------

    other_direction_results = (
        analyze_other_directions(
            df
        )
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    print(
        "\nGenerating plots..."
    )

    for direction, results in (
        direction_results.items()
    ):

        if results is None:
            continue

        plot_individual_patterns(
            df,
            direction
        )

        plot_combined_curve(
            df,
            direction,
            results["tanh"],
            results[
                "saturation_results"
            ]
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        direction_results,
        other_direction_results
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)

    for direction, results in (
        direction_results.items()
    ):

        if results is None:
            continue

        print(
            f"\n{direction.capitalize()}:"
        )

        tanh = results["tanh"]

        if not tanh["ok"]:
            continue

        print(
            f"  Tanh L = "
            f"{tanh['L']:.3f}°"
        )

        print(
            f"  Tanh k = "
            f"{tanh['k']:.5f} s⁻¹"
        )

        for threshold in [
            0.90,
            0.95,
        ]:

            result = (
                calculate_tanh_saturation(
                    tanh,
                    threshold
                )
            )

            duration_sat = result[
                "effective_duration_sat"
            ]

            status = (
                classify_saturation_range(
                    results["fit_data"][
                        "effective_stimulation_duration"
                    ].max(),
                    duration_sat
                )
            )

            print(
                f"  {threshold:.0%} saturation = "
                f"{duration_sat:.3f} s "
                f"({status})"
            )

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()