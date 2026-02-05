import numpy as np

def compute_std_norm(values):
    """
    Compute the standard deviation normalized by the mean of a list of values.
    Args:
        values (list or np.array): List or array of numerical values.
    Returns:
        float: Standard deviation normalized by the mean.
    """
    values = np.array(values)
    return np.std(values) / np.mean(values)
