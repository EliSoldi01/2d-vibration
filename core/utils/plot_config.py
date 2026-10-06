# core/utils/plot_config.py

import matplotlib.pyplot as plt

import config as cfg


def configure_plot_style():
    """
    Configure the global Matplotlib style for the project.

    Sets the font family of all plots to cfg.PLOT_FONT_FAMILY by
    updating plt.rcParams. The change applies to all figures created
    after the call.

    Returns
    -------
    None
    """
    plt.rcParams["font.family"] = cfg.PLOT_FONT_FAMILY