import pandas as pd

def add_presentation_order(df, subject_orders, duration_col="duration"):
    """
    Add subject-specific presentation order.

    For each row, the presentation order is the 1-based position of the
    row's duration within the subject's block order (e.g. with block
    order "3-1-2", duration 1 has presentation order 2).

    Parameters
    ----------
    df : pandas.DataFrame
        Trial-level data. Must contain the column "subject" and
        duration_col.

    subject_orders : dict or pandas.DataFrame
        Block order of each subject, either as a dictionary
        {subject: block_order} or as a DataFrame with the columns
        "subject" and "block_order". Each block order is a string of
        durations separated by "-" (e.g. "3-1-2").

    duration_col : str, optional
        Name of the column in df that contains the duration.
        Defaults to "duration".

    Returns
    -------
    pandas.DataFrame
        Copy of df with the new column "presentation_order" (1-based).
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
        """
        Map the duration of a row to its presentation order for the subject.

        Parameters
        ----------
        row : pandas.Series
            Row of df, containing "subject" and the duration column.

        Returns
        -------
        int
            1-based position of the row's duration within the subject's
            block order.
        """
        block_order = str(order_map.get(row["subject"]))
        # Divide la stringa usando il trattino '-' anziché iterare sui singoli caratteri
        order = [int(d) for d in block_order.split("-")]
        return order.index(int(row[duration_col])) + 1


    df["presentation_order"] = df.apply(map_order, axis=1)
    return df
