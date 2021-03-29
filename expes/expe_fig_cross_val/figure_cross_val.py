import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sparse_ho.utils_plot import configure_plt, discrete_cmap


save_fig = True
# save_fig = False

configure_plt()
fontsize = 18
current_palette = sns.color_palette("colorblind")


algorithms = ['grid_search10', 'random', 'bayesian', 'grad_search']

dict_title = {}
dict_title['grid_search10'] = 'Grid-search'
dict_title['random'] = 'Random-search'
dict_title['bayesian'] = 'Bayesian'
dict_title['grad_search'] = '1st order method'

plt.close('all')
fig, axarr = plt.subplots(
    1, len(algorithms), sharex=True, sharey=True,
    figsize=[10.67, 3])

objs_full = np.load("results/objs_grid_search100.npy", allow_pickle=True)
log_alphas_full = np.load(
    "results/log_alphas_grid_search100.npy", allow_pickle=True)

cmap = discrete_cmap(10, 'Reds')
c = np.linspace(1, 10, 10)

for i, algorithm in enumerate(algorithms):
    objs = np.load("results/objs_%s.npy" % algorithm, allow_pickle=True)
    log_alphas = np.load(
        "results/log_alphas_%s.npy" % algorithm, allow_pickle=True)

    axarr[i].plot(
        log_alphas_full, objs_full / objs_full[0], color=current_palette[0],
        zorder=1)
    pcm = axarr[i].scatter(
        log_alphas, objs / objs_full[0], c=c, cmap=cmap, marker='x', zorder=10)
    axarr[i].scatter(
        log_alphas, np.zeros(len(log_alphas)), c=c, cmap=cmap, marker='x',
        # zorder=10)
        clip_on=False, zorder=10)

    axarr[i].set_title(dict_title[algorithm])
    axarr[i].set_xlabel("$\lambda - \lambda_{\max}$", fontsize=fontsize)
    axarr[i].set_ylim((0, 1))
    print(objs.min())

axarr[0].set_ylabel(r"$\mathcal{C}(\beta^{(\lambda)})$", fontsize=fontsize)
cba = fig.colorbar(pcm, ax=axarr[3], ticks=np.linspace(1, 10, 10))
cba.set_label('Iterations', fontsize=fontsize)
fig.tight_layout()

if save_fig:
    fig_dir = "../../../CD_SUGAR/tex/journal/prebuiltimages/"
    fig_dir_svg = "../../../CD_SUGAR/tex/journal/images/"
    fig.savefig(
        fig_dir + "intro_lassoCV.pdf", bbox_inches="tight")
    fig.savefig(
        fig_dir_svg + "intro_lassoCV.svg", bbox_inches="tight")
plt.show(block=False)
fig.show()
