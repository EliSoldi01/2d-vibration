import numpy as np


def compute_r2_metrics(y_true, y_pred, weights=None):
    """Compute R² metrics with reference to zero and mean, optionally using weights."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if weights is None:
        weights = np.ones_like(y_true)
    else:
        weights = np.asarray(weights)

    # R² with reference to ZERO
    sse = np.sum(
        weights * (y_true - y_pred) ** 2
    )

    sst_zero = np.sum(
        weights * y_true ** 2
    )

    r2_zero = (
        1 - sse / sst_zero
        if sst_zero > 0
        else np.nan
    )

    # R² with reference to MEAN
    y_mean = (
        np.sum(weights * y_true)
        / np.sum(weights)
    )

    sst_mean = np.sum(
        weights * (y_true - y_mean) ** 2
    )

    r2_mean = (
        1 - sse / sst_mean
        if sst_mean > 0
        else np.nan
    )

    return {
        "R2_zero": r2_zero,
        "R2_mean": r2_mean,
    }


def compute_mae(y_true, y_pred, weights=None):
    """
    Compute mean absolute error.

    Parameters
    ----------
    y_true : array-like
        Observed values.
    y_pred : array-like
        Predicted values.
    weights : array-like, optional
        Sample weights.

    Returns
    -------
    float
        Mean absolute error.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors = np.abs(
        y_true - y_pred
    )

    if weights is None:
        return float(np.mean(errors))

    weights = np.asarray(weights, dtype=float)

    return float(
        np.average(
            errors,
            weights=weights
        )
    )


def compute_rmse(y_true, y_pred, weights=None):
    """
    Compute root mean squared error.

    Parameters
    ----------
    y_true : array-like
        Observed values.
    y_pred : array-like
        Predicted values.
    weights : array-like, optional
        Sample weights.

    Returns
    -------
    float
        Root mean squared error.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    squared_errors = (
        y_true - y_pred
    ) ** 2

    if weights is None:
        mse = np.mean(
            squared_errors
        )
    else:
        weights = np.asarray(
            weights,
            dtype=float
        )

        mse = np.average(
            squared_errors,
            weights=weights
        )

    return float(
        np.sqrt(mse)
    )


def compute_aic(n, rss, k):
    """
    Calculate Akaike Information Criterion (AIC).

    Parameters
    ----------
    n : int
        Number of observations.
    rss : float
        Residual sum of squares.
    k : int
        Number of fitted parameters.

    Returns
    -------
    float
        AIC value.
    """

    if n <= 0 or rss <= 0:
        return np.nan

    return float(
        n * np.log(rss / n)
        + 2 * k
    )
