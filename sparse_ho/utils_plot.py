import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def configure_plt():
    params = {'axes.labelsize': 12,
              'font.size': 12,
              'legend.fontsize': 12,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'text.usetex': True,
              'figure.figsize': (8, 6)}
    plt.rcParams.update(params)

    sns.set_palette("colorblind")
    sns.set_context("poster")
    sns.set_style("ticks")


def plot_legend_apart(ax, figname, ncol=None):
    """Do all your plots with fig, ax = plt.subplots(),
    don't call plt.legend() at the end but this instead"""
    if ncol is None:
        ncol = len(ax.lines)
    fig = plt.figure(figsize=(30, 4), constrained_layout=True)
    fig.legend(ax.lines, [l.get_label() for l in ax.lines], ncol=ncol,
               loc="upper center")
    fig.tight_layout()
    fig.savefig(figname)
    os.system("pdfcrop %s %s" % (figname, figname))
    return fig

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(1/3, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return np.floor(n * multiplier) / multiplier
