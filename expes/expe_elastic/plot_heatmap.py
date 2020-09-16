import matplotlib.pyplot as plt
import numpy as np
from sparse_ho.utils_plot import configure_plt

# save_fig = True
save_fig = False
save_fig_grad = True
# save_fig_grad = False
fig_dir = "../../../CD_SUGAR/tex/slides_qbe_long/prebuiltimages/"
fig_dir_svg = "../../../CD_SUGAR/tex/slides_qbe_long/images/"


configure_plt()

objs_grad = np.load("grad_search.npy")
objs_grid = np.load("grid_search.npy")

alpha_1 = np.load("alpha_1.npy")
alpha_2 = np.load("alpha_2.npy")
alphas_grad = np.load("alpha_grad.npy")
alpha_max = np.load("alpha_max.npy")

X, Y = np.meshgrid(alpha_1/alpha_max, alpha_2/alpha_max)
Z = objs_grid

levels = np.geomspace(0.2, 1, num=20)
levels = np.round(levels, 2)

fontsize = 15

plt.set_cmap(plt.cm.viridis)

# fig, ax = plt.subplots(1, 1)
# cp = ax.contourf(X, Y, Z.T, levels)
# ax.scatter(
#     X, Y, s=10, c="orange", marker="o", label="$0$ order (grid search)")
# # cp.ax.tick_params(labelsize=2)
# ax.scatter(
#     alphas_grad[:, 0]/alpha_max,
#     alphas_grad[:, 1]/alpha_max,
#     s=100, color=[plt.cm.Reds((i + len(objs_grad) / 5 + 1) / len(objs_grad)) for i in np.arange(
#         len(objs_grad))], marker="x", label="$1$st order")
# cb = fig.colorbar(cp)
# for t in cb.ax.get_yticklabels():
#     t.set_fontsize(fontsize)
# ax.set_xlim(alpha_1.min()/alpha_max, alpha_1.max()/alpha_max)
# ax.set_ylim(alpha_2.min()/alpha_max, alpha_2.max()/alpha_max)
# plt.xscale('log')
# plt.yscale('log')
# fig.legend(loc=2, ncol=2, fontsize=fontsize, bbox_to_anchor=(0.1653, 1))
# ax.set_xlabel(r'$\lambda_1 / \lambda_\max$', fontsize=fontsize)
# ax.set_ylabel(r'$\lambda_2 / \lambda_\max$', fontsize=fontsize)
# # ax.set_xticklabels(fontsize=fontsize)
# plt.xticks(fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# # plt.tight_layout()

# if save_fig:
#     fig.savefig(
#         fig_dir + "held_out_real_sim_enet.pdf", bbox_inches="tight")
#     fig.savefig(
#         fig_dir_svg + "held_out_real_sim_enet.svg", bbox_inches="tight")

# plt.show(block=False)


color = [
    plt.cm.Reds((i + len(objs_grad) / 5 + 1) / len(objs_grad))
    for i in np.arange(len(objs_grad))]
#################################################################
# code for GRID SEARCH + GRAD SEARCH
for i in np.arange(len(objs_grad)+1):
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(X, Y, Z.T, levels)
    ax.scatter(
        X, Y, s=10, c="orange", marker="o", label="$0$ order (grid search)")
    cb = fig.colorbar(cp)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    # ax.plot(
    #     p_alphas, objs, color=current_palette[0], linewidth=7.0)
    # ax.plot(
    #     p_alphas, objs, 'bo', label='0-order method',
    #     color=current_palette[1], markersize=15)
    ax.scatter(
        alphas_grad[:i, 0]/alpha_max,
        alphas_grad[:i, 1]/alpha_max,
        s=100, color=color[:i], marker="x", label="$1$st order")
    plt.xscale('log')
    plt.yscale('log')
    ax.set_xlim(alpha_1.min()/alpha_max, alpha_1.max()/alpha_max)
    ax.set_ylim(alpha_2.min()/alpha_max, alpha_2.max()/alpha_max)
    plt.xscale('log')
    plt.yscale('log')
    fig.legend(loc=2, ncol=2, fontsize=fontsize, bbox_to_anchor=(0.125, 1))
    ax.set_xlabel(r'$\lambda_1 / \lambda_\max$', fontsize=fontsize)
    ax.set_ylabel(r'$\lambda_2 / \lambda_\max$', fontsize=fontsize)
    # ax.set_xticklabels(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # ax.set_xlim(p_alphas.min(), p_alphas.max())
    # ax.set_ylim(ymin, ymax)
    # plt.xlabel(r"$\lambda / \lambda_{\max}$", fontsize=28)
    # plt.ylabel(
    #     r"$\|y^{\rm{val}} - X^{\rm{val}} \hat \beta^{(\lambda)} \|^2$",
    #     fontsize=28)
    # plt.tick_params(width=5)
    # plt.legend(fontsize=17, loc=2)
    # plt.tight_layout()

    if save_fig_grad:
        fig.savefig(
            fig_dir + "grad_grid_search_real_sim_enet_%i.pdf" % i, bbox_inches="tight")
        fig.savefig(
            fig_dir_svg + "grad_grid_search_real_sim_enet_%i.svg" % i, bbox_inches="tight")
    plt.show(block=False)
