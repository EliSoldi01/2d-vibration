import numpy as np

from scipy.optimize import curve_fit
from scipy.stats import t
from core.analysis.metrics import (
    compute_r2_metrics,
    compute_mae,
    compute_rmse,
    compute_aic,
)


# ============================================================
# MODELS
# ============================================================

def linear_model(x, slope):
    """
    Linear model constrained to pass through the origin.

    y = slope * x
    """
    return slope * x


def tanh_sigmoid(x, L, k):
    """
    Hyperbolic tangent sigmoid.

    y = L * tanh(k * x)
    """
    return L * np.tanh(k * x)


def logistic_sigmoid(x, L_min, L_max, k, D0):
    """
    Four-parameter logistic model.

    y = L_min + (L_max - L_min) /
        (1 + exp(-k * (x - D0)))
    """
    return (
        L_min
        + (L_max - L_min)
        / (1 + np.exp(-k * (x - D0)))
    )

# ============================================================
# SINGLE MODEL FIT
# ============================================================

def fit_one(
    model_func,
    x,
    y,
    weights=None,
    p0=None,
    bounds=(-np.inf, np.inf),
):
    """
    Fit one model using scipy curve_fit.

    Parameters
    ----------
    model_func : callable
        Model function to fit.
    x : array-like
        Independent variable.
    y : array-like
        Observed dependent variable.
    weights : array-like, optional
        Weights used for the fit.
    p0 : array-like, optional
        Initial parameter estimates.
    bounds : tuple, optional
        Bounds for fitted parameters.

    Returns
    -------
    dict
        Fitting results, including parameters, covariance,
        confidence intervals, metrics, predictions, and
        fitting status.
    """

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if weights is not None:

        weights = np.asarray(
            weights,
            dtype=float
        )[valid]

        weights = np.where(
            weights <= 0,
            np.nan,
            weights
        )

        valid_weights = np.isfinite(
            weights
        )

        x = x[valid_weights]
        y = y[valid_weights]
        weights = weights[valid_weights]

        sigma = 1.0 / weights

    else:

        sigma = None

    try:

        popt, pcov = curve_fit(
            model_func,
            x,
            y,
            p0=p0,
            bounds=bounds,
            sigma=sigma,
            absolute_sigma=False,
            maxfev=100000,
        )

        y_pred = model_func(
            x,
            *popt
        )

        # ----------------------------------------------------
        # Standard errors and 95% CI of parameters
        # ----------------------------------------------------

        if (
            pcov is not None
            and np.all(np.isfinite(pcov))
        ):

            params_se = np.sqrt(
                np.diag(pcov)
            )

            # Residual degrees of freedom
            df_resid = len(y) - len(popt)

            if df_resid > 0:

                t_critical = t.ppf(
                    0.975,
                    df_resid
                )

                params_ci_lower = (
                    popt
                    - t_critical * params_se
                )

                params_ci_upper = (
                    popt
                    + t_critical * params_se
                )

                params_ci = np.column_stack(
                    (
                        params_ci_lower,
                        params_ci_upper
                    )
                )

            else:

                params_ci = np.full(
                    (len(popt), 2),
                    np.nan
                )

        else:

            params_se = np.full(
                len(popt),
                np.nan
            )

            params_ci = np.full(
                (len(popt), 2),
                np.nan
            )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        r2_metrics = compute_r2_metrics(
            y,
            y_pred
        )

        mae = compute_mae(
            y,
            y_pred
        )

        rmse = compute_rmse(
            y,
            y_pred
        )

        residuals = y - y_pred

        rss = np.sum(
            residuals ** 2
        )

        aic = compute_aic(
            n=len(y),
            rss=rss,
            k=len(popt)
        )

        return {
            "params": popt,
            "params_se": params_se,
            "params_ci": params_ci,
            "pcov": pcov,
            "r2_zero": r2_metrics["R2_zero"],
            "r2_mean": r2_metrics["R2_mean"],
            "mae": mae,
            "rmse": rmse,
            "aic": aic,
            "y_pred": y_pred,
            "x": x,
            "y": y,
            "ok": True,
        }

    except Exception as e:

        print(
            f"Model fitting failed: {e}"
        )

        return {
            "params": None,
            "params_se": None,
            "params_ci": None,
            "pcov": None,
            "r2_zero": np.nan,
            "r2_mean": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "aic": np.nan,
            "y_pred": None,
            "x": x,
            "y": y,
            "ok": False,
            "error": str(e),
        }

def fit_models(
    x,
    y,
    weights=None,
):
    """
    Fit all available models to the data.

    Models:
        - linear
        - tanh
        - logistic4

    Parameters
    ----------
    x : array-like
        Independent variable.
    y : array-like
        Dependent variable.
    weights : array-like or None
        Optional fitting weights.

    Returns
    -------
    dict
        Dictionary containing the fitting results for each model.
    """

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    # ========================================================
    # LINEAR
    # ========================================================

    linear_result = fit_one(
        linear_model,
        x,
        y,
        weights=weights,
        p0=[1.0],
        bounds=(
            [-np.inf],
            [np.inf]
        ),
    )

    # ========================================================
    # TANH SIGMOID
    # ========================================================

    L0 = np.nanmax(
        np.abs(y)
    )

    if not np.isfinite(L0) or L0 <= 0:
        L0 = 1.0

    k0 = 0.1

    tanh_result = fit_one(
        tanh_sigmoid,
        x,
        y,
        weights=weights,
        p0=[
            L0,
            k0,
        ],
        bounds=(
            [0.0, 0.0],
            [np.inf, np.inf]
        ),
    )

    # ========================================================
    # LOGISTIC 4-PARAMETER
    # ========================================================

    L_min0 = np.nanmin(y)
    L_max0 = np.nanmax(y)

    if not np.isfinite(L_min0):
        L_min0 = -1.0

    if not np.isfinite(L_max0):
        L_max0 = 1.0

    if L_max0 <= L_min0:
        L_max0 = L_min0 + 1.0

    D00 = np.nanmedian(x)

    if not np.isfinite(D00):
        D00 = 0.0

    logistic_result = fit_one(
        logistic_sigmoid,
        x,
        y,
        weights=weights,
        p0=[
            L_min0,
            L_max0,
            k0,
            D00,
        ],
        bounds=(
            [
                -np.inf,
                -np.inf,
                0.0,
                -np.inf,
            ],
            [
                np.inf,
                np.inf,
                np.inf,
                np.inf,
            ],
        ),
    )

    # ========================================================
    # RESULTS
    # ========================================================

    return {
        "linear": linear_result,
        "tanh": tanh_result,
        "logistic4": logistic_result,
    }

# ============================================================
# BEST MODEL
# ============================================================

def get_best_model(results):
    """
    Select the best valid model according to minimum AIC.

    Parameters
    ----------
    results : dict
        Dictionary containing model fitting results.

    Returns
    -------
    str
        Name of the model with the lowest finite AIC.

    Raises
    ------
    ValueError
        If no valid model with finite AIC is available.
    """

    valid_models = {}

    for name, result in results.items():

        if not isinstance(
            result,
            dict
        ):
            continue

        if not result.get(
            "ok",
            False
        ):
            continue

        aic = result.get(
            "aic",
            np.nan
        )

        if not np.isfinite(aic):
            continue

        valid_models[name] = aic

    if not valid_models:
        raise ValueError(
            "No valid model with finite AIC."
        )

    return min(
        valid_models,
        key=valid_models.get
    )


def model_confidence_band(
    model_name,
    x,
    params,
    pcov,
    confidence=0.95,
    df_resid=None,
):
    """
    Calculate a pointwise confidence band for a fitted model
    using the delta method.

    Parameters
    ----------
    model_name : str
        'linear', 'tanh', or 'logistic4'.

    x : array-like
        X values where the confidence band is evaluated.

    params : array-like
        Fitted model parameters.

    pcov : ndarray
        Parameter covariance matrix returned by curve_fit.

    confidence : float
        Confidence level, e.g. 0.95.

    df_resid : int or None
        Residual degrees of freedom. If provided, a t critical
        value is used. Otherwise, a normal approximation is used.

    Returns
    -------
    y_fit : ndarray
        Fitted model values.

    lower : ndarray
        Lower confidence limit.

    upper : ndarray
        Upper confidence limit.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    params = np.asarray(
        params,
        dtype=float,
    )

    # ========================================================
    # CHECK COVARIANCE MATRIX
    # ========================================================

    if pcov is None:

        raise ValueError(
            "Parameter covariance matrix is required "
            "to calculate the confidence band."
        )

    pcov = np.asarray(
        pcov,
        dtype=float,
    )

    if not np.all(
        np.isfinite(pcov)
    ):

        raise ValueError(
            "Parameter covariance matrix contains "
            "non-finite values."
        )

    n_params = len(params)

    if pcov.shape != (
        n_params,
        n_params,
    ):

        raise ValueError(
            "Parameter covariance matrix has "
            "an incompatible shape: "
            f"{pcov.shape}. Expected "
            f"({n_params}, {n_params})."
        )

    # ========================================================
    # FITTED CURVE + JACOBIAN
    # ========================================================

    if model_name == "linear":

        y_fit = linear_model(
            x,
            *params,
        )

        # y = slope * x
        #
        # dy / d(slope) = x

        J = x[:, None]

    elif model_name == "tanh":

        L, k = params

        y_fit = tanh_sigmoid(
            x,
            L,
            k,
        )

        # y = L * tanh(k*x)
        #
        # dy/dL = tanh(k*x)
        # dy/dk = L*x*sech²(k*x)

        tanh_value = np.tanh(
            k * x
        )

        sech2 = (
            1.0
            - tanh_value ** 2
        )

        J = np.column_stack([
            tanh_value,
            L * x * sech2,
        ])

    elif model_name == "logistic4":

        L_min, L_max, k, D0 = params

        z = k * (
            x - D0
        )

        # ----------------------------------------------------
        # Numerically stable logistic function
        # ----------------------------------------------------

        q = np.empty_like(z)

        positive = z >= 0
        negative = ~positive

        q[positive] = (
            1.0
            / (
                1.0
                + np.exp(
                    -z[positive]
                )
            )
        )

        exp_z = np.exp(
            z[negative]
        )

        q[negative] = (
            exp_z
            / (
                1.0
                + exp_z
            )
        )

        y_fit = logistic_sigmoid(
            x,
            L_min,
            L_max,
            k,
            D0,
        )

        # ----------------------------------------------------
        # Analytical Jacobian
        #
        # y = L_min + (L_max-L_min)*q
        #
        # dy/dL_min = 1-q
        # dy/dL_max = q
        # dy/dk =
        #   (L_max-L_min)*(x-D0)*q*(1-q)
        # dy/dD0 =
        #   -(L_max-L_min)*k*q*(1-q)
        # ----------------------------------------------------

        dq = (
            q
            * (1.0 - q)
        )

        J = np.column_stack([
            1.0 - q,
            q,
            (
                (L_max - L_min)
                * (x - D0)
                * dq
            ),
            (
                -(L_max - L_min)
                * k
                * dq
            ),
        ])

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    # ========================================================
    # DELTA-METHOD VARIANCE
    # ========================================================

    variance = np.einsum(
        "ij,jk,ik->i",
        J,
        pcov,
        J,
    )

    # Protect against tiny negative values caused by
    # floating-point precision.
    variance = np.maximum(
        variance,
        0.0,
    )

    se = np.sqrt(
        variance
    )

    # ========================================================
    # CRITICAL VALUE
    # ========================================================

    alpha = (
        1.0
        - confidence
    )

    if (
        df_resid is not None
        and df_resid > 0
    ):

        critical_value = t.ppf(
            1.0 - alpha / 2.0,
            df_resid,
        )

    else:

        # Normal approximation
        critical_value = 1.96

    # ========================================================
    # CONFIDENCE LIMITS
    # ========================================================

    margin = (
        critical_value
        * se
    )

    lower = (
        y_fit
        - margin
    )

    upper = (
        y_fit
        + margin
    )

    return (
        y_fit,
        lower,
        upper,
    )





