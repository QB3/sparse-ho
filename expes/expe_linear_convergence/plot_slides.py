# import numpy as np
import pandas
# import seaborn as sns
import matplotlib.pyplot as plt

from sparse_ho.utils_plot import configure_plt


# save_fig = False
save_fig = True
fig_dir = "../../../CD_SUGAR/tex/ICML2020slides/prebuiltimages/"
fig_dir_svg = "../../../CD_SUGAR/tex/ICML2020slides/images/"


configure_plt()

fontsize = 25

# dataset_names = ["leu", "rcv1_train", "news20"]
# dataset_names = ["leu", "rcv1_train", "news20", "finance"]
dataset_names = ["leu", "rcv1_train", "news20", "real-sim"]

dict_title = {}
dict_title["rcv1_train"] = "rcv1"
dict_title["news20"] = "20news"
dict_title["finance"] = "finance"
dict_title["kdda_train"] = "kdda"
dict_title["climate"] = "climate"
dict_title["leu"] = "Leukemia"
dict_title["real-sim"] = "real-sim"

plt.close('all')
fig, axarr = plt.subplots(
    2, 4, sharex=False, sharey=False, figsize=[14, 8],)


for idx, dataset in enumerate(dataset_names):
    df_data = pandas.read_pickle("%s.pkl" % dataset)
    diff_beta = df_data["diff_beta"].to_numpy()[0]
    diff_jac = df_data["diff_jac"].to_numpy()[0]
    supp_id = df_data["supp_id"].to_numpy()[0]
    #
    axarr.flat[idx].semilogy(diff_beta)
    axarr.flat[idx].axvline(x=supp_id, c='red', linestyle="--")

    axarr.flat[idx+4].semilogy(diff_jac)
    axarr.flat[idx+4].axvline(x=supp_id, c='red', linestyle="--")

    axarr.flat[idx+4].set_xlabel(r"$\#$ epochs", size=fontsize)

    axarr.flat[idx].set_title("%s" % (
        dict_title[dataset]), size=fontsize)
    # xarr.flat[idx].set_title("%s %s" % (
    #     dict_title[dataset], dict_n_feature[dataset]), size=fontsize)

axarr.flat[0].set_ylabel(
    r"$||\beta^{(k)} - \hat \beta||$", fontsize=fontsize)
axarr.flat[4].set_ylabel(
    r"$||\mathcal{J}^{(k)} - \hat \mathcal{J}||$", fontsize=fontsize)

fig.tight_layout()

if save_fig:
    fig.savefig(
        fig_dir + "linear_convergence_lasso.pdf", bbox_inches="tight")
    fig.savefig(
        fig_dir_svg + "linear_convergence_lasso.svg", bbox_inches="tight")
fig.show()
