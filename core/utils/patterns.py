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


def is_complex_pattern(pattern, pure_patterns):
    """
    Return True if the pattern is not one of the pure stimulation patterns.

    Parameters
    ----------
    pattern : str
        Pattern encoded as 'BBB_TTT'.

    pure_patterns : iterable of str
        Pure stimulation patterns (e.g. the keys of cfg.PURE_PATTERNS).

    Returns
    -------
    bool
        True if the pattern is combined, False if it is pure.
    """

    return pattern not in pure_patterns