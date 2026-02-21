import pandas as pd

def add_presentation_order(df, subject_orders, duration_col="duration"):
    """
    Adds subject-specific presentation order.
    Args:
        df : DataFrame
            Must contain 'subject' and duration_col
        subject_orders : dict or DataFrame
            {subject: block_order} or DataFrame with columns ['subject', 'block_order']
        duration_col : str
            Column name in df that indicates the duration (e.g., "duration")
    Returns:
        DataFrame with new column 'presentation_order' (1-based index)
    """
    df = df.copy()

    if isinstance(subject_orders, pd.DataFrame):
        order_map = dict(
            zip(subject_orders["subject"].astype(str),
                subject_orders["block_order"])
        )
    else:
        order_map = subject_orders

    def map_order(row):
        """ Maps duration to presentation order (1-based) for the subject. """
        block_order = str(order_map.get(row["subject"]))
        order = [int(d) for d in block_order]
        return order.index(int(row[duration_col])) + 1
    

    df["presentation_order"] = df.apply(map_order, axis=1)
    return df


def add_presentation_pos(df, subject_orders, duration_col="duration"):
    """
    Adds universal presentation position (1..N),
    consistent across subjects regardless of duration identity.
    """
    df = df.copy()

    if isinstance(subject_orders, pd.DataFrame):
        order_map = dict(
            zip(subject_orders["subject"].astype(str),
                subject_orders["block_order"])
        )
    else:
        order_map = subject_orders

    def map_pos(row):
        block_order = str(order_map.get(row["subject"]))
        order = [int(d) for d in block_order]
        mapping = {dur: i + 1 for i, dur in enumerate(order)}
        return mapping.get(int(row[duration_col]))

    df["presentation_pos"] = df.apply(map_pos, axis=1)
    return df
