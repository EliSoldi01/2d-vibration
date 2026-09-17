import numpy as np

# ----------------------------
# Funzioni per R² zero e media
# ----------------------------
def compute_r2_metrics(y_true, y_pred, weights=None):
    """Compute R² metrics with reference to zero and mean, optionally using weights."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if weights is None:
        weights = np.ones_like(y_true)
    else:
        weights = np.asarray(weights)

    # R² with reference to ZERO
    sse = np.sum(weights * (y_true - y_pred)**2)
    sst_zero = np.sum(weights * (y_true**2))
    r2_zero = 1 - sse / sst_zero if sst_zero > 0 else np.nan

    # R² with reference to MEAN
    y_mean = np.sum(weights * y_true) / np.sum(weights)
    sst_mean = np.sum(weights * (y_true - y_mean)**2)
    r2_mean = 1 - sse / sst_mean if sst_mean > 0 else np.nan

    return {"R2_zero": r2_zero, "R2_mean": r2_mean}



def compute_r2_unweighted(y, y_pred):
    """Compute R2 metrics without weights."""
    return compute_r2_metrics(y, y_pred, weights=None)