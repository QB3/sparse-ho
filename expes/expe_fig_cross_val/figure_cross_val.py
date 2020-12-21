import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from sparse_ho.utils_plot import configure_plt

configure_plt()
current_palette = sns.color_palette("colorblind")

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(1/3, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


algorithms = ['grid_search10', 'random', 'bayesian', 'grad_search']

dict_title = {}
dict_title['grid_search10'] = 'Grid-search'
dict_title['random'] = 'Random-search'
dict_title['bayesian'] = 'Bayesian'
dict_title['grad_search'] = '1st order method'

plt.close('all')
fig, axarr = plt.subplots(
    1, len(algorithms), sharex=True, sharey=True,
    figsize=[14, 4.5], constrained_layout=True)

objs_full = np.load("results/objs_grid_search100.npy", allow_pickle=True)
log_alphas_full = np.load(
    "results/log_alphas_grid_search100.npy", allow_pickle=True)

cmap = discrete_cmap(10, 'Reds')
c = np.linspace(1, 10, 10)
# cmap = plt.get_cmap('RdBu', 10)

for i, algorithm in enumerate(algorithms):
    objs = np.load("results/objs_%s.npy" % algorithm, allow_pickle=True)
    log_alphas = np.load(
        "results/log_alphas_%s.npy" % algorithm, allow_pickle=True)

    axarr[i].plot(
        log_alphas_full, objs_full / objs_full[0], color=current_palette[0])
    pcm = axarr[i].scatter(
        log_alphas, objs / objs_full[0], c=c, cmap=cmap, marker='x')
    axarr[i].scatter(
        log_alphas, np.zeros(len(log_alphas)), c=c, cmap=cmap, marker='x',
        clip_on=False)

    axarr[i].set_title(dict_title[algorithm])
    axarr[i].set_xlabel("$\lambda - \lambda_{\max}$")
    axarr[i].set_ylim((0, 1))
    print(objs.min())

axarr[0].set_ylabel(r"$\mathcal{C}(\beta^{(\lambda)})$")

# fig.colorbar(ticks=range(10))
fig.colorbar(pcm, ax=axarr[3], ticks=np.linspace(1, 10, 10))

save_fig = True
# save_fig = False

if save_fig:
    fig_dir = "../../../CD_SUGAR/tex/journal/prebuiltimages/"
    fig_dir_svg = "../../../CD_SUGAR/tex/journal/images/"
    fig.savefig(
        fig_dir + "intro_lassoCV.pdf", bbox_inches="tight")
    fig.savefig(
        fig_dir_svg + "intro_lassoCV.svg", bbox_inches="tight")
plt.show(block=False)
fig.show()
