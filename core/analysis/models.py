import numpy as np
from scipy.stats import linregress

# ============================================================
# LINEAR REGRESSION
# ============================================================
def linear_regression(x, y):
    """
    Perform simple linear regression: y ~ x
    Args:
        x : array-like
            Independent variable
        y : array-like
            Dependent variable
    Returns:
        dict
            {
                "slope": float,
                "intercept": float,
                "r_value": float,
                "r_squared": float,
                "p_value": float,
                "stderr": float
            }
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size < 2 or y.size < 2:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r_value": np.nan,
            "r_squared": np.nan,
            "p_value": np.nan,
            "stderr": np.nan
        }

    lr = linregress(x, y)
    return {
        "slope": lr.slope,
        "intercept": lr.intercept,
        "r_value": lr.rvalue,
        "r_squared": lr.rvalue ** 2,
        "p_value": lr.pvalue,
        "stderr": lr.stderr
    }


# ============================================================
# REGRESSION SUMMARY
# ============================================================
def regression_summary(df, x_col, y_col, group_col=None):
    """
    Compute linear regression summary for a DataFrame.
    Can compute per-group if group_col is provided.

    Args:
        df : pd.DataFrame
            Input data containing x_col and y_col
        x_col : str
            Column name for independent variable
        y_col : str
            Column name for dependent variable
        group_col : str, optional
            Column name for grouping (e.g., subject or rep)
    Returns:
        pd.DataFrame
            Summary table with columns:
            [group_col, slope, r_squared, p_value]
    """
    import pandas as pd

    results = []

    if group_col is not None:
        groups = df[group_col].unique()
        for g in groups:
            sub_df = df[df[group_col] == g]
            lr = linear_regression(sub_df[x_col], sub_df[y_col])
            results.append({
                group_col: g,
                "slope": lr["slope"],
                "r_squared": lr["r_squared"],
                "p_value": lr["p_value"]
            })
    else:
        lr = linear_regression(df[x_col], df[y_col])
        results.append({
            "slope": lr["slope"],
            "r_squared": lr["r_squared"],
            "p_value": lr["p_value"]
        })

    return pd.DataFrame(results)
