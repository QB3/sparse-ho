import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sparse_ho.utils_plot import configure_plt, plot_legend_apart
configure_plt()
fontsize = 25

save_fig = False
n_iter_crop = 180

fig_dir = "results/"
fig_dir_svg = "results/"

current_palette = sns.color_palette("colorblind")
dict_method = {}
dict_method["forward"] = 'F. Iterdiff.'
dict_method["implicit_forward"] = 'Imp. F. Iterdiff.'
dict_method['celer'] = 'Imp. F. Iterdiff. + celer'
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

# dict_marker = {}
# dict_marker["forward"] = "o"
# dict_marker["implicit_forward"] = "X"
# dict_marker["backward"] = "s"

dict_markers = {}
dict_markers["forward"] = 'o'
dict_markers["implicit_forward"] = 'X'
dict_markers['celer'] = 'v'
dict_markers['grid_search'] = 'd'
dict_markers['bayesian'] = 'P'
dict_markers['random'] = '*'
dict_markers["backward"] = ">"
##############################################
epoch_lims = {}
epoch_lims["real-sim", 10] = 45
epoch_lims["real-sim", 100] = 195
epoch_lims["rcv1_train", 10] = 145
epoch_lims["rcv1_train", 100] = 990
##############################################
time_lims = {}
time_lims["real-sim", 10] = 50
time_lims["real-sim", 100] = 100
time_lims["rcv1_train", 10] = 50
time_lims["rcv1_train", 100] = 500
time_lims["news20", 10] = 1000
time_lims["news20", 100] = 10000
time_lims["finance", 10] = 1000
time_lims["finance", 100] = 10000
##############################################

dict_title = {}
dict_title["rcv1_train"] = "rcv1"
dict_title["news20"] = "20news"
dict_title["finance"] = "finance"
dict_title["real-sim"] = "real-sim"

# df_data = pandas.read_pickle("results_non_unique.pkl")
files = os.listdir('results/')

for i in range(len(files)):
    if i == 0:
        df_data = pandas.read_pickle("results/" + files[i])
    else:
        df_temp = pandas.read_pickle("results/" +  files[i])
        df_data = pandas.concat([df_data, df_temp])
    

methods = df_data['method'].unique()
methods = np.delete(methods, np.where(methods == "ground_truth"))
list_datasets = df_data['dataset'].unique()
div_alphas = df_data['div_alpha'].unique()


fig, axarr = plt.subplots(
    len(div_alphas), len(list_datasets), sharex=False, sharey=False, figsize=[14, 10],)
fig2, axarr2 = plt.subplots(
    len(div_alphas), len(list_datasets), sharex=False, sharey=False, figsize=[14, 10],)
# for n_features in list_n_features:
for idx1, dataset in enumerate(list_datasets):
    df_dataset = df_data[df_data['dataset'] == dataset]
    for idx2, div_alpha in enumerate(div_alphas):
        df_div = df_dataset[df_dataset['div_alpha'] == div_alpha]
        grad = np.float(df_div['grad'][df_div['method'] == "ground_truth"].unique())
        lines = []
        # plt.figure()
        for method in methods:
            markevery = dict_markevery[method]
            marker = dict_markers[method]
            df_method = df_div[df_div['method'] == method]
            df_method = df_method.sort_values('maxit')
            lines.append(axarr2.flat[idx2 * len(list_datasets) + idx1].semilogy(
                df_method['maxit'], np.abs(df_method['grad'] - grad),
                label=dict_method[method], color=dict_color[method],
                marker=""))
            axarr.flat[idx2 * len(list_datasets) + idx1].loglog(
                df_method['time'], np.abs(df_method['grad'] - grad),
                # df_method['maxit'], np.abs(df_method['criterion grad'] - grad),
                label=dict_method[method], color=dict_color[method],
                marker="")
            # fig.legend()
            # plt.xlim(6, 60)
            # axarr.flat[0].set_xlim(6, 60)
            # axarr.flat[1].set_xlim(6, 60)
            # axarr.flat[0].ylabel("Objective minus optimum")
            # axarr.flat[0].xlabel("Number of iterations")
            # plt.title("p = %i, rho = %0.2f" % (n_features, rho))
            # plt.tight_layout()
            # plt.show(block=False)

        # axarr2.flat[idx2 * len(list_datasets) + idx1].set_xlabel(r"$\#$ epochs", fontsize=18)
        axarr2.flat[idx2 * len(list_datasets) + idx1].set_ylim(1e-12)
        axarr.flat[idx2 * len(list_datasets) + idx1].set_ylim(1e-12)
        axarr.flat[idx2 * len(list_datasets) + idx1].set_xlim((1e0, time_lims[dataset, div_alpha]))
        axarr.flat[idx2 * len(list_datasets)].set_ylabel(
                r"$\lambda_{{\max}} / $" + ("%i" % div_alpha)
                + "\n"
                + "\n"
                +r'$|\mathcal{J}^\top\nabla \mathcal{C}(\beta^{(\lambda)}) - \hat{\mathcal{J}}^\top\nabla \mathcal{C}(\hat{\beta}^{(\lambda)})|$', size=fontsize)
        axarr.flat[idx1].set_title(dict_title[dataset])
        axarr2.flat[idx2 * len(list_datasets)].set_ylabel(
                r"$\lambda_{{\max}} / $" + ("%i" % div_alpha)
                + "\n"
                + "\n"
                +r'$|\mathcal{J}^\top\nabla \mathcal{C}(\beta^{(\lambda)}) - \hat{\mathcal{J}}^\top\nabla \mathcal{C}(\hat{\beta}^{(\lambda)})|$', size=fontsize)
        axarr2.flat[idx1].set_title(dict_title[dataset])
  
        # axarr.flat[1].set_ylabel(r'$|\mathcal{J}^\top\nabla \mathcal{C}(\beta^{(\lambda)}) - \hat{\mathcal{J}}^\top\nabla \mathcal{C}(\hat{\beta}^{(\lambda)})|$', fontsize=18)
for i in np.arange(len(list_datasets)):
    axarr.flat[-(i + 1)].set_xlabel("Times (s)", size=fontsize)
    axarr2.flat[-(i + 1)].set_xlabel(r"$\#$ epochs", size=fontsize)

fig.tight_layout()
fig2.tight_layout()
if save_fig:
    fig.savefig(fig_dir + "intro_influ_niter.pdf", bbox_inches="tight")
    fig.savefig(fig_dir_svg + "intro_influ_niter.svg", bbox_inches="tight")
            #
        # axarr.flat[1].set_title(dataset)
# plot_legend_apart(
#     axarr[0][0],
#     fig_dir + "test.pdf", ncol=3)
fig.show()
fig2.show()


# labels = []
# for method in methods:
#     labels.append(dict_method[method])
# # labels = ['Seq. F. Iterdiff (ours)', 'F. Iterdiff', 'B. Iterdiff', ]
# fig3 = plt.figure(figsize=[9.33, 1])
# fig3.legend(
#     [l[0] for l in lines], labels, ncol=3, loc='upper center', fontsize=18)
# fig3.tight_layout()
# #
# if save_fig:
#     fig3.savefig(
#         fig_dir + "legend_intro_influ_niter.pdf", bbox_inches="tight")
#     fig3.savefig(
#         fig_dir_svg + "legend_intro_influ_niter.svg", bbox_inches="tight")
# fig3.show()
