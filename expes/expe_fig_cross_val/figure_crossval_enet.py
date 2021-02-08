import numpy as np
import matplotlib.pyplot as plt

from sparse_ho.utils_plot import configure_plt, round_down, discrete_cmap

save_fig = True
# save_fig = False

configure_plt()
fontsize = 18

algorithms = [
    'grid_search10', 'random', 'bayesian', 'grad_search']

dict_title = {}
dict_title['grid_search10'] = 'Grid-search'
dict_title['random'] = 'Random-search'
dict_title['bayesian'] = 'Bayesian'
dict_title['grad_search'] = '1st order method'
dict_title['grad_search_ls'] = '1st order method + LS'

dataset = "rcv1_train"
# dataset = "real-sim"

objs_full = np.load(
    "results/%s_objs_grid_search100_enet.npy" % dataset, allow_pickle=True)
log_alphas_full = np.load(
    "results/%s_log_alphas_grid_search100_enet.npy" % dataset,
    allow_pickle=True)
X = log_alphas_full[:, 0].reshape(30, 30)
log_alphas_full = X[:, 0]
log_alpha_min = log_alphas_full.min()
log_alpha_max = log_alphas_full.max()
X, Y = np.meshgrid(log_alphas_full, log_alphas_full)
Z = objs_full.reshape(30, 30)

fig, axarr = plt.subplots(
    1, len(algorithms), sharex=True, sharey=True,
    figsize=[10.67, 3], constrained_layout=True)

min_grid = Z.min()
for algorithm in algorithms:
    objs = np.load(
        "results/%s_objs_%s_enet.npy" % (dataset, algorithm),
        allow_pickle=True)
    min_grid = min(min_grid, objs.min())
min_grid = min(min_grid, objs_full.min())

levels = np.geomspace(min_grid, objs_full.max(), num=40)
levels = round_down(levels, 2)

plt.figure()
plt.contourf(X, Y, Z)
plt.plot()

for i, algorithm in enumerate(algorithms):
    objs = np.load(
        "results/%s_objs_%s_enet.npy" % (dataset, algorithm),
        allow_pickle=True)
    log_alphas = np.load(
        "results/%s_log_alphas_%s_enet.npy" % (dataset, algorithm),
        allow_pickle=True)
    assert objs.min() >= min_grid
    cmap = discrete_cmap(len(objs), 'Reds')
    c = np.linspace(1, len(objs), len(objs))
    cs = axarr[i].contourf(X, Y, Z.T, levels=levels, cmap='viridis')
    pcm = axarr[i].scatter(
        log_alphas[:, 0], log_alphas[:, 1], c=c, marker='x', cmap=cmap,
        clip_on=False)

    axarr[i].set_title(dict_title[algorithm])
    axarr[i].set_xlabel("$\lambda_1 - \lambda_{\max}$")
    print(objs.min())
    axarr[i].set_xlim([log_alpha_min, log_alpha_max])
    axarr[i].set_ylim([log_alpha_min, log_alpha_max])
    axarr[i].set_aspect('equal', adjustable='box')

cba = fig.colorbar(pcm, ax=axarr[3], ticks=[1, 5, 10, 15, 20, 25], shrink=0.6)
cba.set_label('Iterations', fontsize=fontsize)
cba2 = fig.colorbar(cs, ax=axarr[0], location='left', shrink=0.6)
cba2.set_label(r"$\mathcal{C}(\beta^{(\lambda)})$", fontsize=fontsize)

axarr[0].set_ylabel(r"$\mathcal{C}(\beta^{(\lambda)})$", fontsize=fontsize)
axarr[0].set_ylabel("$\lambda_2 - \lambda_{\max}$", fontsize=fontsize)

if save_fig:
    fig_dir = "../../../CD_SUGAR/tex/journal/prebuiltimages/"
    fig_dir_svg = "../../../CD_SUGAR/tex/journal/images/"
    fig.savefig(
        fig_dir + "%s_intro_enetCV.pdf" % dataset, bbox_inches="tight")
    fig.savefig(
        fig_dir_svg + "%s_intro_enetCV.svg" % dataset, bbox_inches="tight")

fig.show()
