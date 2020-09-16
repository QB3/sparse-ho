import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sparse_ho.utils_plot import configure_plt

configure_plt()

# save_fig = False
save_fig_grid = False
# save_fig = True
save_fig_grad = True
# save_fig_grad = False
fig_dir = "../../../CD_SUGAR/tex/slides_qbe_long/prebuiltimages/"
fig_dir_svg = "../../../CD_SUGAR/tex/slides_qbe_long/images/"


current_palette = sns.color_palette("colorblind")

p_alphas = np.load("p_alphas.npy")
objs = np.load("objs.npy")

p_alphas_grad = np.load("p_alphas_grad.npy")
objs_grad = np.load("objs_grad.npy")

# ax = plt.gca()
# ax.tick_params(width=10)

# fig = plt.figure()
# plt.semilogx(
#     p_alphas, objs, color=current_palette[0], linewidth=7.0)
# plt.semilogx(
#     p_alphas, objs, 'bo', label='grid-search',
#     color=current_palette[1], markersize=15)
# plt.xlabel(r"$\lambda / \lambda_{\max}$", fontsize=28)
# # plt.ylabel("validation loss", fontsize=28)
# # plt.ylabel(
# #     r"$\|y^{\rm{val}} - X^{\rm{val}} \beta \|^2$",
# #     fontsize=28)
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


plt.rcParams['lines.markersize'] = 30

color = [plt.cm.Greens((
         i+len(objs_grad)/1.2) / len(objs_grad) / 2) for i in np.arange(
             len(objs_grad))]

ymax = objs.max() + 0.02
ymin = objs.min() - 0.03

#################################################################
# code for GRID SEARCH ONLY
# for i in np.arange(len(objs)+1):
#     fig, ax = plt.subplots(1, 1)
#     ax.plot(
#         p_alphas, objs, color=current_palette[0], linewidth=7.0)
#     ax.plot(
#         p_alphas, objs, 'bo',
#         color="none", markersize=15)
#     ax.plot(
#         p_alphas[:(i)], objs[:(i)], 'bo',
#         color=current_palette[1], markersize=15)
#     plt.xscale('log')
#     plt.yscale('log')
#     ax.set_xlim(p_alphas.min(), p_alphas.max())
#     ax.set_ylim(ymin, ymax)
#     plt.xlabel(r"$\lambda / \lambda_{\max}$", fontsize=28)
#     plt.ylabel(
#         r"$\|y^{\rm{val}} - X^{\rm{val}} \hat \beta^{(\lambda)} \|^2$",
#         fontsize=28)
#     plt.tick_params(width=5)
#     # plt.legend(fontsize=17, loc=2)
#     fig.tight_layout()
#     if save_fig_grid:
#         fig.savefig(
#             fig_dir + "grid_search_real_sim_%i.pdf" % i, bbox_inches="tight")
#         fig.savefig(
#             fig_dir_svg + "grid_search_real_sim_%i.svg" % i,
#             bbox_inches="tight")
#     plt.show(block=False)
########################################################################

#################################################################
# code for GRID SEARCH + GRAD SEARCH
for i in np.arange(len(objs_grad)+1):
    fig, ax = plt.subplots(1, 1)
    ax.plot(
        p_alphas, objs, color=current_palette[0], linewidth=7.0)
    ax.plot(
        p_alphas, objs, 'bo', label='0-order method',
        color=current_palette[1], markersize=15)
    ax.scatter(
        p_alphas_grad[:(i)], objs_grad[:(i)],
        # label='1-st order method',
        # color=current_palette[2], markersize=25, )
        color=color[:i], marker="X", label='1st order method')
    plt.xscale('log')
    plt.yscale('log')
    ax.set_xlim(p_alphas.min(), p_alphas.max())
    ax.set_ylim(ymin, ymax)
    plt.xlabel(r"$\lambda / \lambda_{\max}$", fontsize=28)
    plt.ylabel(
        r"$\|y^{\rm{val}} - X^{\rm{val}} \hat \beta^{(\lambda)} \|^2$",
        fontsize=28)
    plt.tick_params(width=5)
    plt.legend(fontsize=17, loc=2)
    plt.tight_layout()

    if save_fig_grad:
        fig.savefig(
            fig_dir + "grad_grid_search_real_sim_%i.pdf" % i, bbox_inches="tight")
        fig.savefig(
            fig_dir_svg + "grad_grid_search_real_sim_%i.svg" % i, bbox_inches="tight")
    plt.show(block=False)
