import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

from sparse_ho.utils_plot import configure_plt, plot_legend_apart
from main_hypergradient_svm import dict_max_iter
configure_plt()
fontsize = 18

# save_fig = False
save_fig = True
fig_dir = "../../../CD_SUGAR/tex/journal/prebuiltimages/"
fig_dir_svg = "../../../CD_SUGAR/tex/journal/images/"

current_palette = sns.color_palette("colorblind")
dict_method = {}
dict_method["forward"] = 'Forward-mode PCD'
dict_method["implicit"] = 'Implicit diff.'
dict_method['sota'] = r'Implicit diff. + \texttt{Lightning}'
dict_method['grid_search'] = 'Grid-search'
dict_method['bayesian'] = 'Bayesian'
dict_method['random'] = 'Random-search'
dict_method['hyperopt'] = 'Random-search'

dict_color = {}
dict_color["grid_search"] = current_palette[3]
dict_color["backward"] = current_palette[9]
dict_color["random"] = current_palette[5]
dict_color["bayesian"] = current_palette[0]
dict_color["implicit"] = current_palette[2]
dict_color["forward"] = current_palette[4]
dict_color["sota"] = current_palette[1]

dict_markevery = {}
dict_markevery["forward"] = 2
dict_markevery["implicit"] = 1
dict_markevery["backward"] = 3
dict_markevery["sota"] = 4


dict_div_alphas = {}
dict_div_alphas[10] = "10"
dict_div_alphas[100] = "10^2"

dict_markers = {}
dict_markers["forward"] = 'o'
dict_markers["implicit"] = 'X'
dict_markers['sota'] = 'v'
dict_markers['grid_search'] = 'd'
dict_markers['bayesian'] = 'P'
dict_markers['random'] = '*'
dict_markers["backward"] = ">"

##############################################
y_lims = {}
y_lims["news20", 10] = 1e-10
y_lims["news20", 100] = 1e-10
y_lims["real-sim", 10] = 1e-10
y_lims["real-sim", 5] = 1e-10
y_lims["real-sim", 50] = 1e-10
y_lims["real-sim", 100] = 1e-10

y_lims["rcv1_train", 10] = 1e-10
y_lims["rcv1_train", 5] = 1e-10
y_lims["rcv1_train", 50] = 1e-10
y_lims["rcv1_train", 100] = 1e-10

##############################################

epoch_lims = {}
epoch_lims["news20", 10] = 250
epoch_lims["news20", 5] = 500
epoch_lims["real-sim", 10] = 45
epoch_lims["real-sim", 25] = 195
epoch_lims["rcv1_train", 10] = 145
epoch_lims["rcv1_train", 5] = 990
##############################################
dict_xlim = {}
dict_xlim["rcv1_train"] = 60
dict_xlim["real-sim"] = 37
##############################################

dict_title = {}
dict_title["rcv1_train"] = "rcv1"
dict_title["news20"] = "20news"
dict_title["colon"] = "colon"
dict_title["real-sim"] = "real-sim"

# n_points = 5
# dict_max_iter = {}
# dict_max_iter["real-sim"] = np.linspace(5, 100, n_points, dtype=np.int)
methods = ["implicit", "sota", "forward"]

list_datasets = ["rcv1_train", "real-sim"]


fig, axarr = plt.subplots(
    1, len(list_datasets), sharex=False, sharey=True,
    figsize=[10.67, 3])

for idx1, dataset_name in enumerate(list_datasets):

    str_results = "results_svm/hypergradient_svm_%s_%s_%i.pkl" % (
        dataset_name, 'ground_truth', 5)
    df_data = pandas.read_pickle(str_results)
    true_grad = df_data['grad'].to_numpy()[0]

    for method in methods:
        all_max_iter = dict_max_iter[dataset_name, method]
        grads = np.zeros(len(all_max_iter))
        times = np.zeros(len(all_max_iter))
        for i, max_iter in enumerate(all_max_iter):
            str_results = "results_svm/hypergradient_svm_%s_%s_%i.pkl" % (
                dataset_name, method, max_iter)
            df_data = pandas.read_pickle(str_results)
            grads[i] = df_data['grad'].to_numpy()[0]
            times[i] = df_data['time'].to_numpy()[0]
        # import ipdb; ipdb.set_trace()
        axarr[idx1].semilogy(
            times, np.abs(grads - true_grad), label=dict_method[method],
            color=dict_color[method], marker="o")

        axarr[idx1].set_title(dict_title[dataset_name], fontsize=fontsize)
        axarr[idx1].set_xlabel("Time (s)", fontsize=fontsize)
        axarr[idx1].set_ylim((1e-14, 1e-2))
        axarr[idx1].set_xlim((0, dict_xlim[dataset_name]))

fig.tight_layout()

if save_fig:
    fig.savefig(
        fig_dir + "hypergradient_computation_svm.pdf", bbox_inches="tight")
    fig.savefig(
        fig_dir_svg + "hypergradient_computation_svm.svg", bbox_inches="tight")
    plot_legend_apart(
        axarr[0],
        fig_dir + "legend_hypergradient_computation_svm.pdf", ncol=3)
fig.show()
