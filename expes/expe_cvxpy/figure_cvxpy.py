import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sparse_ho.utils_plot import configure_plt, plot_legend_apart
configure_plt()
fontsize = 18

# save_fig = False
save_fig = True

fig_dir = "results/"
fig_dir_svg = "results/"

current_palette = sns.color_palette("colorblind")
dict_method = {}
dict_method["forward"] = 'PCD Forward Iterdiff'
dict_method["backward"] = 'PCD Backward Iterdiff'
dict_method['cvxpy'] = 'Cvxpylayers'

dict_div_alphas = {}
dict_div_alphas[10] = r'$10$'
dict_div_alphas[100] = r'$10^2$'


dict_title = {}
dict_title["lasso"] = "Lasso"
dict_title["enet"] = "Elastic net"

dict_color = {}
dict_color["cvxpy"] = current_palette[3]
dict_color["backward"] = current_palette[9]
dict_color["forward"] = current_palette[4]

dict_markers = {}
dict_markers["forward"] = 'o'
dict_markers["backward"] = 'o'
dict_markers['cvxpy'] = 'o'

models = ["lasso"]
div_alphas = [10, 100]

fig, axarr = plt.subplots(
     len(models), len(div_alphas), sharex=False, sharey=True,
     figsize=[10.67, 3])
for idx, div_alpha in enumerate(div_alphas):
    for _, model in enumerate(models):
        times_fwd = np.load(
            "results/times_%s_forward_%s.npy" % (model, div_alpha),
            allow_pickle=True)
        times_bwd = np.load(
            "results/times_%s_backward_%s.npy" % (model, div_alpha),
            allow_pickle=True)
        times_cvxpy = np.load(
            "results/times_%s_cvxpy_%s.npy" % (model, div_alpha),
            allow_pickle=True)

        n_features = np.load(
            "results/nfeatures_%s_%s.npy" % (model, div_alpha),
            allow_pickle=True)
        print(n_features)
        axarr[idx].loglog(
            n_features, times_fwd, color=dict_color["forward"],
            marker=dict_markers["forward"], label=dict_method["forward"])
        axarr[idx].loglog(
            n_features, times_bwd, color=dict_color["backward"],
            marker=dict_markers["backward"], label=dict_method["backward"])
        axarr[idx].loglog(
            n_features, times_cvxpy, color=dict_color["cvxpy"],
            marker=dict_markers["cvxpy"], label=dict_method["cvxpy"])

        axarr[idx].set_xlabel(
            '\# features p', fontsize=fontsize)
        axarr[idx].set_yticks([1e-2, 1e0, 1e2])
        axarr.flat[idx].set_title(
            r"$e^\lambda = e^{\lambda_{\max}}/$ %s" % div_alpha)

axarr.flat[0].set_ylabel(
    " Time (s)", fontsize=fontsize)

fig.tight_layout()
fig_dir = "../../../CD_SUGAR/tex/journal/prebuiltimages/"
fig_dir_svg = "../../../CD_SUGAR/tex/journal/images/"
if save_fig:
    fig.savefig(fig_dir + "hypergrad_cvxpy.pdf", bbox_inches="tight")
    fig.savefig(fig_dir_svg + "hypergrad_cvxpy.svg", bbox_inches="tight")

    plot_legend_apart(
        axarr[0],
        fig_dir + "legend_cvxpy.pdf", ncol=3)
fig.show()
