import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def configure_plt():
    params = {
        'axes.labelsize': 14,
        'font.size': 14,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'text.usetex': True,
    }
    plt.rcParams.update(params)
    sns.set_palette("colorblind")
    sns.set_style("ticks")


def plot_legend_apart(ax, figname, ncol=None, figwidth=10.67, fontsize=18):
    """Do all your plots with fig, ax = plt.subplots(),
    don't call plt.legend() at the end but this instead"""
    if ncol is None:
        ncol = len(ax.lines)
    fig = plt.figure(figsize=(10.67, 3.5), constrained_layout=True)
    fig.legend(ax.lines, [line.get_label() for line in ax.lines], ncol=ncol,
               loc="upper center", fontsize=18)
    # fig = plt.figure(figsize=(30, 4), constrained_layout=True)
    # fig = plt.figure(figsize=(figwidth, 2), constrained_layout=True)
    # fig.legend(ax.lines, [line.get_label() for line in ax.lines], ncol=ncol,
    #           loc="upper center", fontsize=fontsize)
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


def discrete_color(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(1/2, 1, N))
    cmap_name = base.name + str(N)
    cmap = base.from_list(cmap_name, color_list, N)
    return cmap(np.linspace(0, 1, N))


def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return np.floor(n * multiplier) / multiplier


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return np.ceil(n * multiplier) / multiplier


dict_color_2Dplot = {
    'implicit_forward': 'Greens',
    'implicit_forward_approx': 'Greens',
    'grid_search': 'Oranges',
    'random': 'Purples',
    'bayesian': 'Blues'
    }


current_palette = sns.color_palette("colorblind")
dict_color = {}
dict_color["grid_search"] = current_palette[3]
dict_color["random"] = current_palette[5]
dict_color["bayesian"] = current_palette[0]
dict_color["implicit_forward"] = current_palette[2]
dict_color["implicit_forward_approx"] = current_palette[2]
dict_color["forward"] = current_palette[4]
dict_color["implicit"] = current_palette[1]

dict_method = {}
dict_method["forward"] = 'F. Iterdiff.'
dict_method["implicit_forward"] = '1st order'
dict_method["implicit_forward_approx"] = '1st order approx'
dict_method['implicit'] = 'Implicit'
dict_method['grid_search'] = 'Grid-search'
dict_method['bayesian'] = 'Bayesian'
dict_method['random'] = 'Random-search'
dict_method['hyperopt'] = 'Random-search'
dict_method['backward'] = 'B. Iterdiff.'

dict_markers = {}
dict_markers["implicit_forward"] = 'X'
dict_markers["implicit_forward_approx"] = 'x'
dict_markers['implicit'] = 'v'
dict_markers['grid_search'] = '3'
# dict_markers['grid_search'] = 'o'
dict_markers['bayesian'] = 'P'
dict_markers['random'] = '*'

dict_title = {}
dict_title["rcv1_train"] = "rcv1"
dict_title["news20"] = "news20"
dict_title["finance"] = "finance"
dict_title["kdda_train"] = "kdda"
dict_title["climate"] = "climate"
dict_title["leukemia"] = "leukemia"
dict_title["real-sim"] = "real-sim"

dict_n_features = {}
dict_n_features["rcv1_train"] = r"($p=19,959$)"
dict_n_features["real-sim"] = r"($p=20,958$)"
dict_n_features["news20"] = r"($p=632,982$)"
dict_n_features["finance"] = r"($p=1,668,737$)"
dict_n_features["leukemia"] = r"($p=7129$)"
