def pattern_sums(pattern):
    """
    Return the number of active biceps and triceps stimulations.

    Parameters
    ----------
    pattern : str
        Pattern encoded as 'BBB_TTT'.

    Returns
    -------
    tuple[int, int]
        Number of active biceps and triceps stimulations.
    """

    biceps, triceps = pattern.split("_")

    pb_sum = sum(int(value) for value in biceps)
    pt_sum = sum(int(value) for value in triceps)

    return pb_sum, pt_sum


def is_combined_pattern(pattern, pure_patterns):
    """
    Return True if the pattern is not one of the pure stimulation patterns.
    """

    return pattern not in pure_patterns