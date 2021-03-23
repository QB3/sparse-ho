import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sparse_ho.utils_plot import configure_plt
configure_plt()
fontsize = 18
current_palette = sns.color_palette("colorblind")

dict_title = {}
dict_title["lasso"] = "Lasso"
dict_title["enet"] = "Elastic net"
dict_title["logreg"] = "Sparse logistic regression"

model_names = ["lasso", "enet", "logreg"]

plt.close('all')
fig, axarr = plt.subplots(
    1, len(model_names), sharex=True, sharey=False,
    figsize=[10.67, 3.5], constrained_layout=True)
# check
# loading results
for i, model_name in enumerate(model_names):
    coefs = np.load("results/coefs_%s.npy" % model_name, allow_pickle=True)
    alphas = np.load(
            "results/alphas_%s.npy" % model_name, allow_pickle=True)

    if model_name == 'logreg':
        coefs = coefs[:, 0, :]
    n_features = coefs.shape[1]

    neg_log_alphas = -np.log(alphas) + np.log(alphas[0])
    for j in range(n_features):
        axarr[i].plot(neg_log_alphas, coefs[:, j], color=current_palette[j])

    axarr[i].set_title(dict_title[model_name], fontsize=fontsize)
    axarr[i].set_xlabel(r"$\lambda_{\max} - \lambda$", fontsize=fontsize)

axarr[0].set_ylabel("Coefficient " + r"$\hat{\beta}^{(\lambda)}$",
                    fontsize=fontsize)
save_fig = True

if save_fig:
    fig_dir = "../../../CD_SUGAR/tex/journal/prebuiltimages/"
    fig_dir_svg = "../../../CD_SUGAR/tex/journal/images/"
    fig.savefig(
        fig_dir + "intro_reg_paths.pdf", bbox_inches="tight")
    fig.savefig(
        fig_dir_svg + "intro_reg_paths.svg", bbox_inches="tight")
plt.show(block=False)
fig.show()
