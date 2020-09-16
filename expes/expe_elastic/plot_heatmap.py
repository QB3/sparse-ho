import matplotlib.pyplot as plt
import numpy as np
from sparse_ho.utils_plot import configure_plt

# save_fig = True
save_fig = False
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

# plt.set_cmap(plt.cm.viridis)
# fig, ax = plt.subplots(1, 1)
# cp = ax.contourf(np.log10(X), np.log10(Y), Z.T, levels)
# ax.scatter(
#     X, Y, s=10, c="yellow", marker="o")
# ax.scatter(
#     # alphas_grad[:, 0]/alpha_max,
#     # alphas_grad[:, 1]/alpha_max,
#     np.log10(alphas_grad[:, 0]/alpha_max),
#     np.log10(alphas_grad[:, 1]/alpha_max),
#     s=100, color=[plt.cm.Reds((i + 1) / len(objs_grad)) for i in np.arange(
#         len(objs_grad))], marker="x")
# fig.colorbar(cp)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.title(r'Real-sim dataset, elastic net: $\min_\beta \frac{1}{2n} ||y-X\beta||^2 + \lambda_1||\beta||_1 + \frac{\lambda_2}{2}||\beta||_2^2$')
# # fig.legend()
# ax.set_xlabel(r'$\lambda_1 / \lambda_\max$')
# ax.set_ylabel(r'$\lambda_2/ \lambda_\max$')
# plt.show(block=False)


plt.figure()
plt.set_cmap(plt.cm.viridis)
fig, ax = plt.subplots(1, 1)
cp = ax.contourf(X, Y, Z.T, levels)
ax.scatter(
    X, Y, s=10, c="orange", marker="o")
# cp.ax.tick_params(labelsize=2)
ax.scatter(
    alphas_grad[:, 0]/alpha_max,
    alphas_grad[:, 1]/alpha_max,
    # np.log10(alphas_grad[:, 0]/alpha_max),
    # np.log10(alphas_grad[:, 1]/alpha_max),
    s=100, color=[plt.cm.Reds((i + 1) / len(objs_grad)) for i in np.arange(
        len(objs_grad))], marker="x")
cb = fig.colorbar(cp)
for t in cb.ax.get_yticklabels():
    t.set_fontsize(fontsize)
ax.set_xlim(alpha_1.min()/alpha_max, alpha_1.max()/alpha_max)
ax.set_ylim(alpha_2.min()/alpha_max, alpha_2.max()/alpha_max)
# plt.axis('equal')
# plt.xlim(alpha_1.min(), alpha_1.max())
# plt.ylim(alpha_2.min(), alpha_2.max())
plt.xscale('log')
plt.yscale('log')
# plt.title(r'Real-sim dataset, elastic net: $\min_\beta \frac{1}{2n} ||y-X\beta||^2 + \lambda_1||\beta||_1 + \frac{\lambda_2}{2}||\beta||_2^2$')
# fig.legend()
ax.set_xlabel(r'$\lambda_1 / \lambda_\max$', fontsize=fontsize)
ax.set_ylabel(r'$\lambda_2 / \lambda_\max$', fontsize=fontsize)
# ax.set_xticklabels(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.tight_layout()

if save_fig:
    fig.savefig(
        fig_dir + "held_out_real_sim_enet.pdf", bbox_inches="tight")
    fig.savefig(
        fig_dir_svg + "held_out_real_sim_enet.svg", bbox_inches="tight")
fig.show()

plt.tight_layout()
plt.show(block=False)
