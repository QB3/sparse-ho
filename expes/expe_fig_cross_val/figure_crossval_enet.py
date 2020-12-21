import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
    "results/%s_objs_grid_search10_enet.npy" % dataset, allow_pickle=True)
log_alphas_full = np.load(
    "results/%s_log_alphas_grid_search10_enet.npy" % dataset,
    allow_pickle=True)
X = log_alphas_full[:, 0].reshape(5, 5)
log_alphas_full = X[:, 0]
X, Y = np.meshgrid(log_alphas_full, log_alphas_full)
# Y = log_alphas_full[:, 1].reshape(10, 10)
Z = objs_full.reshape(5, 5)

# objs_full = np.load("results/objs_grid_search100_enet.npy", allow_pickle=True)
# log_alphas_full = np.load(
#     "results/log_alphas_grid_search100_enet.npy", allow_pickle=True)


# cmap = plt.get_cmap('RdBu', 10)

fig, axarr = plt.subplots(
    1, len(algorithms), sharex=True, sharey=True,
    figsize=[14, 4], constrained_layout=True)

min_grid = np.infty
for algorithm in algorithms:
    objs = np.load(
        "results/%s_objs_%s_enet.npy" % (dataset, algorithm),
        allow_pickle=True)
    min_grid = min(min_grid, objs.min())
min_grid = min(min_grid, objs_full.min())

levels = np.geomspace(min_grid, objs_full.max(), num=60)
# levels = np.round(levels, 2)

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
    # cs = ax.contourf(X, Y, Z, levels, cmap=cmap, extend=extend, origin=origin)
    # fig.colorbar(cs, ax=ax, shrink=0.9)
    cs = axarr[i].contourf(X, Y, Z.T, levels=levels, cmap='viridis')
    # axarr[i].plot(
    #     log_alphas_full, objs_full / objs_full[0], color=current_palette[0])
    pcm = axarr[i].scatter(
        log_alphas[:, 0], log_alphas[:, 1], c=c, marker='x', cmap=cmap)
    # axarr[i].scatter(
    #     log_alphas, np.zeros(len(log_alphas)), c=c, cmap=cmap, marker='x',
    #     clip_on=False)

    axarr[i].set_title(dict_title[algorithm])
    axarr[i].set_xlabel("$\lambda_1 - \lambda_{\max}$")
    # axarr[i].set_ylim((0, 1))
    print(objs.min())

fig.colorbar(pcm, ax=axarr[3], ticks=[1, 5, 10, 15, 20, 25])
# fig.colorbar(cs, ax=axarr[3])
#
# fig.subplots_adjust(right=0.7)
# cbar_ax = fig.add_axes([0.73, 0.15, 0.05, 0.7])
# fig.colorbar(
#     cs, cax=cbar_ax,
#     ticks=np.round(np.geomspace(min_grid, objs_full.max(), 5), 2))

# fig.colorbar(cs, cax=cax, orientation="horizontal")

# fig.colorbar(cs, ax=axarr[3], cax=cbaxes, orientation="horizontal", pad=0.2)

axarr[0].set_ylabel(r"$\mathcal{C}(\beta^{(\lambda)})$")
axarr[0].set_ylabel("$\lambda_2 - \lambda_{\max}$")

# fig.colorbar(ticks=range(10))
# fig.colorbar(pcm, ax=axarr[3], ticks=np.linspace(1, 10, 10))

save_fig = True
# save_fig = False

if save_fig:
    fig_dir = "../../../CD_SUGAR/tex/journal/prebuiltimages/"
    fig_dir_svg = "../../../CD_SUGAR/tex/journal/images/"
    fig.savefig(
        fig_dir + "%s_intro_enetCV.pdf" % dataset, bbox_inches="tight")
    fig.savefig(
        fig_dir_svg + "%s_intro_enetCV.svg" % dataset, bbox_inches="tight")

fig.show()
