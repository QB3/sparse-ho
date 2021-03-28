import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sparse_ho.utils_plot import configure_plt, plot_legend_apart
configure_plt()
fontsize = 18

# save_fig = False
save_fig = True

fig_dir = "results/"
fig_dir_svg = "results/"

current_palette = sns.color_palette("colorblind")
dict_method = {}
dict_method["forward"] = 'PCD Forward Iterdiff.'
dict_method["implicit_forward"] = 'Imp. F. Iterdiff.'
dict_method['celer'] = 'Imp. F. Iterdiff. + Celer'
dict_method['grid_search'] = 'Grid-search'
dict_method['bayesian'] = 'Bayesian'
dict_method['random'] = 'Random-search'
dict_method['hyperopt'] = 'Random-search'
dict_method['backward'] = 'B. Iterdiff.'

dict_color = {}
dict_color["grid_search"] = current_palette[3]
dict_color["backward"] = current_palette[9]
dict_color["random"] = current_palette[5]
dict_color["bayesian"] = current_palette[0]
dict_color["implicit_forward"] = current_palette[2]
dict_color["forward"] = current_palette[4]
dict_color["celer"] = current_palette[1]

dict_markevery = {}
dict_markevery["forward"] = 2
dict_markevery["implicit_forward"] = 1
dict_markevery["backward"] = 3
dict_markevery["celer"] = 4


dict_div_alphas = {}
dict_div_alphas[10] = "10"
dict_div_alphas[100] = "10^2"

dict_markers = {}
dict_markers["forward"] = 'o'
dict_markers["implicit_forward"] = 'X'
dict_markers['celer'] = 'v'
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
time_lims = {}
time_lims["real-sim", 10] = (1e0, 100)
time_lims["real-sim", 5] = (1e0, 100)
time_lims["rcv1_train", 10] = (1e0, 500)
time_lims["rcv1_train", 5] = (1e0, 500)
time_lims["news20", 10] = (1e0, 1000)
time_lims["news20", 25] = (1e0, 10000)
time_lims["colon", 10] = (1e-1, 50)
time_lims["colon", 25] = (1e-1, 100)
##############################################

dict_title = {}
dict_title["rcv1_train"] = "rcv1"
dict_title["news20"] = "20news"
dict_title["colon"] = "colon"
dict_title["real-sim"] = "real-sim"

files = os.listdir('results/')

for i in range(len(files)):
    if i == 0:
        df_data = pandas.read_pickle("results/" + files[i])
    else:
        print(files[i])
        df_temp = pandas.read_pickle("results/" + files[i])
        df_data = pandas.concat([df_data, df_temp])


methods = df_data['method'].unique()
methods = np.delete(methods, np.where(methods == "ground_truth"))
list_datasets = ["rcv1_train", "real-sim", "news20"]
div_alphas = df_data['div_alpha'].unique()
div_alphas = np.sort(div_alphas)


fig, axarr = plt.subplots(
    len(div_alphas), len(list_datasets), sharex=False, sharey=True,
    figsize=[10.67, 5])
fig2, axarr2 = plt.subplots(
    len(div_alphas), len(list_datasets), sharex=False, sharey=True,
    figsize=[10.67, 5])

for idx1, dataset in enumerate(list_datasets):
    df_dataset = df_data[df_data['dataset'] == dataset]
    for idx2, div_alpha in enumerate(div_alphas):
        df_div = df_dataset[df_dataset['div_alpha'] == div_alpha]
        grad = np.float(
            df_div['grad'][df_div['method'] == "ground_truth"].unique())
        lines = []
        # plt.figure()
        for method in methods:
            df_method = df_div[df_div['method'] == method]
            df_method = df_method.sort_values('maxit')
            lines.append(
                axarr2.flat[idx2 * len(list_datasets) + idx1].semilogy(
                    df_method['maxit'], np.abs(df_method['grad'] - grad),
                    label=dict_method[method], color=dict_color[method],
                    marker=""))
            diff_grad = np.abs(df_method['grad'] - grad)
            diff_grad[diff_grad < 1e-9] = 1e-9
            axarr.flat[idx2 * len(list_datasets) + idx1].semilogy(
                df_method['time'], np.abs(df_method['grad'] - grad),
                label=dict_method[method], color=dict_color[method],
                marker="o")

            if dataset == "news20":
                axarr.flat[idx2 * len(list_datasets) + idx1].set_xlim([0, 420])

        axarr2.flat[idx2 * len(list_datasets) + idx1].set_ylim(
                y_lims[dataset, div_alpha])
        axarr.flat[idx2 * len(list_datasets) + idx1].set_ylim(
                y_lims[dataset, div_alpha])
        axarr.flat[idx2 * len(list_datasets)].set_ylabel(
                r"$e^\lambda = e^{\lambda_{\max}}/ %s$" %
                dict_div_alphas[div_alpha], fontsize=fontsize)

        axarr.flat[idx1].set_title(dict_title[dataset], fontsize=fontsize)

        axarr2.flat[idx1].set_title(dict_title[dataset])

for i in np.arange(len(list_datasets)):
    axarr.flat[-(i + 1)].set_xlabel("Time (s)", fontsize=fontsize)
    axarr2.flat[-(i + 1)].set_xlabel(r"$\#$ epochs", fontsize=fontsize)

fig.tight_layout()
fig2.tight_layout()

fig_dir = "../../../CD_SUGAR/tex/journal/prebuiltimages/"
fig_dir_svg = "../../../CD_SUGAR/tex/journal/images/"
if save_fig:
    fig.savefig(fig_dir + "hypergradient_computation.pdf", bbox_inches="tight")
    fig.savefig(fig_dir_svg + "hypergradient_computation.svg",
                bbox_inches="tight")

    plot_legend_apart(
        axarr[0][0],
        fig_dir + "legend_hypergradient_computation.pdf", ncol=3)
fig.show()
fig2.show()
