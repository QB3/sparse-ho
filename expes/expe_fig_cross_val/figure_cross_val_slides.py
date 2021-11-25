import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sparse_ho.utils_plot import configure_plt

configure_plt()

# save_fig = False
# save_fig_grid = True
save_fig_grid = False
# save_fig = True
save_fig_grad = True
# save_fig_grad = False
fig_dir = "../../../CD_SUGAR/tex/slides_qbe_long/prebuiltimages/"
fig_dir_svg = "../../../CD_SUGAR/tex/slides_qbe_long/images/"


current_palette = sns.color_palette("colorblind")

p_alphas_grid = np.load("p_alphas.npy")
objs_grid = np.load("objs.npy")

p_alphas_grad = np.load("p_alphas_grad.npy")
objs_grad = np.load("objs_grad.npy")

fontsize = 23
for i in np.arange(len(objs_grad)+1):
    fig, ax = plt.subplots(1, 1)
    ax.plot(
        p_alphas_grid, objs_grid, color=current_palette[0], linewidth=7.0)
    ax.plot(
        p_alphas_grid[:i], objs_grid[:i], 'bo', label='0-order method',
        color=current_palette[1], markersize=15)
    plt.xscale('log')
    ax.set_xlabel(r'$\lambda / \lambda_{\max}$', fontsize=fontsize)
    plt.ylabel(
        r"$\|y^{\rm{val}} - X^{\rm{val}} \hat \beta^{(\lambda)} \|^2$",
        fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_xlim(p_alphas_grid.min(), p_alphas_grid.max())
    plt.tight_layout()

    if save_fig_grid:
        fig.savefig(
            fig_dir + "grid_search_real_sim_lasso_%i.pdf" % i,
            bbox_inches="tight")
        fig.savefig(
            fig_dir_svg + "grid_search_real_sim_lasso_%i.svg" % i,
            bbox_inches="tight")
    else:
        plt.show(block=False)

color = [
    plt.cm.Reds((i + len(objs_grad) / 3 + 1) / len(objs_grad))
    for i in np.arange(len(objs_grad))]

# for i in np.arange(1):
for i in np.arange(len(objs_grad)+1):
    fig, ax = plt.subplots(1, 1)
    ax.plot(
        p_alphas_grid, objs_grid, color=current_palette[0], linewidth=7.0,
        zorder=1)
    # ax.plot(
    #     p_alphas_grid, objs_grid, 'bo', label='0-order method',
    #     color=current_palette[1], markersize=15)
    ax.scatter(
        p_alphas_grid, objs_grid,
        s=300, color=current_palette[1], marker="o", label="$O$ order",
        zorder=2)
    ax.scatter(
        p_alphas_grad[:i], objs_grad[:i],
        s=700, color=color[:i], marker="X", label="$1$st order",
        zorder=3)
    plt.xscale('log')
    ax.set_xlabel(r'$\lambda / \lambda_{\max}$', fontsize=fontsize)
    plt.ylabel(
        r"$\|y^{\rm{val}} - X^{\rm{val}} \hat \beta^{(\lambda)} \|^2$",
        fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_xlim(p_alphas_grid.min(), p_alphas_grid.max())
    ax.set_ylim(0.17, 1)
    plt.tight_layout()

    if save_fig_grad:
        fig.savefig(
            fig_dir + "grid_grad_search_real_sim_lasso_%i.pdf" % i,
            bbox_inches="tight")
        fig.savefig(
            fig_dir_svg + "grid_grad_search_real_sim_lasso_%i.svg" % i,
            bbox_inches="tight")
    # else:
    plt.show(block=False)

# for i in np.arange(len(objs_grad)+1):
#     fig, ax = plt.subplots(1, 1)
#     ax.plot(
#         p_alphas_grid, objs_grid, color=current_palette[0], linewidth=7.0)
#     ax.plot(
#         p_alphas_grid, objs_grid, 'bo', label='0-order',
#         color=current_palette[1], markersize=15)
#     ax.set_xlim(p_alphas_grid.min(), p_alphas_grid.max())
#     ax.plot(
#         p_alphas_grad[:i], objs_grad[:i], 'bo', label='1st-order',
#         color=current_palette[1], markersize=15)
#     plt.xscale('log')
#     # ax.set_ylim(alpha_2.min()/alpha_max, alpha_2.max()/alpha_max)
#     # plt.xscale('log')
#     # plt.yscale('log')
#     # fig.legend(loc=2, ncol=2, fontsize=fontsize, bbox_to_anchor=(0.125, 1))
#     # ax.set_xlabel(r'$\lambda_1 / \lambda_\max$', fontsize=fontsize)
#     # ax.set_ylabel(r'$\lambda_2 / \lambda_\max$', fontsize=fontsize)
#     # # ax.set_xticklabels(fontsize=fontsize)
#     # plt.xticks(fontsize=fontsize)
#     # plt.yticks(fontsize=fontsize)
#     # ax.set_xlim(p_alphas.min(), p_alphas.max())
#     # ax.set_ylim(ymin, ymax)
#     # plt.xlabel(r"$\lambda / \lambda_{\max}$", fontsize=28)
#     # plt.ylabel(
#     #     r"$\|y^{\rm{val}} - X^{\rm{val}} \hat \beta^{(\lambda)} \|^2$",
#     #     fontsize=28)
#     # plt.tick_params(width=5)
#     # plt.legend(fontsize=17, loc=2)
#     # plt.tight_layout()

#     if save_fig_grad:
#         fig.savefig(
#             fig_dir + "grad_grid_search_real_sim_enet_%i.pdf" % i,
#             bbox_inches="tight")
#         fig.savefig(
#             fig_dir_svg + "grad_grid_search_real_sim_enet_%i.svg" % i,
#             bbox_inches="tight")
#     else:
#         plt.show(block=False)


# plt.ylabel(
#     r"$\|y^{\rm{val}} - X^{\rm{val}} \hat \beta^{(\lambda)} \|^2$",
#     fontsize=28)
# plt.tick_params(width=5)
# plt.legend(fontsize=28)
# plt.tight_layout()

# if save_fig:
#     fig.savefig(
#         fig_dir + "cross_val_real_sim.pdf", bbox_inches="tight")
#     fig.savefig(
#         fig_dir_svg + "cross_val_real_sim.svg", bbox_inches="tight")
# fig.show()

# plt.show(block=False)


# plot fig with 0 order and 1rst order methods

# fig = plt.figure()
# plt.semilogx(
#     p_alphas, objs, color=current_palette[0], linewidth=7.0)
# plt.semilogx(
#     p_alphas, objs, 'bo', label='0-order (grid-search)',
#     color=current_palette[1], markersize=15)
# plt.semilogx(
#     p_alphas_grad, objs_grad, 'bX', label='1-st order',
#     color=current_palette[2], markersize=25)
# plt.xlabel(r"$\lambda / \lambda_{\max}$", fontsize=28)
# plt.ylabel(
#     r"$\|y^{\rm{val}} - X^{\rm{val}} \hat \beta^{(\lambda)} \|^2$",
#     fontsize=28)
# plt.tick_params(width=5)
# plt.legend(fontsize=28)
# plt.tight_layout()

# if save_fig:
#     fig.savefig(
#         fig_dir + "cross_val_and_grad_search_real_sim.pdf", bbox_inches="tight")
#     fig.savefig(
#         fig_dir + "cross_val_and_grad_search_real_sim.png", bbox_inches="tight")
#     fig.savefig(
#         fig_dir_svg + "cross_val_and_grad_search_real_sim.svg", bbox_inches="tight")
# fig.show()

# plt.show(block=False)
