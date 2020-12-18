import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sparse_ho.utils_plot import configure_plt

configure_plt()
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
    figsize=[14, 4.5], constrained_layout=True)

objs_full = np.load("results/objs_grid_search100.npy", allow_pickle=True)
log_alphas_full = np.load(
    "results/log_alphas_grid_search100.npy", allow_pickle=True)

for i, algorithm in enumerate(algorithms):
    objs = np.load("results/objs_%s.npy" % algorithm, allow_pickle=True)
    log_alphas = np.load(
        "results/log_alphas_%s.npy" % algorithm, allow_pickle=True)

    color = [
        plt.cm.Reds((i + len(objs) / 5 + 1) / len(objs))
        for i in np.arange(len(objs))]

    axarr[i].plot(
        log_alphas_full, objs_full / objs_full[0], color=current_palette[0])
    axarr[i].scatter(log_alphas, objs / objs_full[0], color=color, marker='x')
    axarr[i].scatter(
        log_alphas, np.zeros(len(log_alphas)), color=color, marker='x',
        clip_on=False)

    axarr[i].set_title(dict_title[algorithm])
    axarr[i].set_xlabel("$\lambda - \lambda_{\max}$")
    axarr[i].set_ylim((0, 1))
    print(objs.min())

axarr[0].set_ylabel(r"$\mathcal{C}(\beta^{(\lambda)})$")


save_fig = True
# save_fig = False
fig_dir = "../../../CD_SUGAR/tex/journal/prebuiltimages/"
fig_dir_svg = "../../../CD_SUGAR/tex/journal/images/"

if save_fig:
    fig.savefig(
        fig_dir + "intro_lassoCV.pdf", bbox_inches="tight")
    fig.savefig(
        fig_dir_svg + "intro_lassoCV.svg", bbox_inches="tight")
plt.show(block=False)
fig.show()
