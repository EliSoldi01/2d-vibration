# core/utils/plot_config.py

import matplotlib.pyplot as plt

import config as cfg


def configure_plot_style():
    """Configure the global Matplotlib style for the project."""

    plt.rcParams["font.family"] = cfg.PLOT_FONT_FAMILY