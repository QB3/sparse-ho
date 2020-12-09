import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

from sparse_ho.utils_plot import configure_plt
configure_plt()

save_fig = False
n_iter_crop = 110

fig_dir = "../../../tex/ICML2020/prebuiltimages/"
fig_dir_svg = "../../../tex/ICML2020/prebuiltimages_svg/"

current_palette = sns.color_palette("colorblind")
dict_method = {}
dict_method["forward"] = 'F. Iterdiff.'
dict_method["implicit_forward"] = 'Imp. F. Iterdiff. (ours)'
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


# df_data = pandas.read_pickle("results_non_unique.pkl")
df_data = pandas.read_pickle("results.pkl")

methods = df_data['method'].unique()
list_datasets = df_data['dataset'].unique()

# for n_features in list_n_features:
for dataset in list_datasets:
    df_dataset = df_data[df_data['dataset'] == dataset]
    fig, axarr = plt.subplots(
        1, 2, sharex=False, sharey=False, figsize=[9.33, 4],)

    grad = df_dataset['criterion true grad'].mean()
    lines = []
    # plt.figure()
    for method in methods:
        markevery = dict_markevery[method]
        marker = dict_markers[method]
        df_method = df_dataset[df_dataset['method'] == method]
        lines.append(axarr.flat[1].semilogy(
                # df_method['maxit'], df_method['time'],
            df_method['maxit'], np.abs(df_method['criterion grad'] - grad),
            label=dict_method[method], color=dict_color[method],
            marker=marker, markevery=markevery))
        axarr.flat[0].loglog(
            df_method['time'], np.abs(df_method['criterion grad'] - grad),
            # df_method['maxit'], np.abs(df_method['criterion grad'] - grad),
            label=dict_method[method], color=dict_color[method],
            marker=marker, markevery=markevery)
        # fig.legend()
        # plt.xlim(6, 60)
        # axarr.flat[0].set_xlim(6, 60)
        # axarr.flat[1].set_xlim(6, 60)
        # axarr.flat[0].ylabel("Objective minus optimum")
        # axarr.flat[0].xlabel("Number of iterations")
        # plt.title("p = %i, rho = %0.2f" % (n_features, rho))
        # plt.tight_layout()
        # plt.show(block=False)

    axarr.flat[0].set_xlabel("Times (s)", fontsize=18)
    axarr.flat[1].set_xlabel(r"$\#$ epochs", fontsize=18)
    axarr.flat[1].set_ylim(1e-14)
    axarr.flat[0].set_ylim(1e-14)
    axarr.flat[1].set_xlim(6, n_iter_crop)
    axarr.flat[0].set_ylabel("Objective minus optimum", fontsize=18)
    axarr.flat[1].set_ylabel("Objective minus optimum", fontsize=18)
    fig.tight_layout()
    if save_fig:
        fig.savefig(fig_dir + "intro_influ_niter.pdf", bbox_inches="tight")
        fig.savefig(fig_dir_svg + "intro_influ_niter.svg", bbox_inches="tight")
        #
    axarr.flat[0].set_title('dataset = %s' % (dataset))
    fig.show()

labels = []
for method in methods:
    labels.append(dict_method[method])
# labels = ['Seq. F. Iterdiff (ours)', 'F. Iterdiff', 'B. Iterdiff', ]
fig3 = plt.figure(figsize=[9.33, 1])
fig3.legend(
    [l[0] for l in lines], labels, ncol=3, loc='upper center', fontsize=18)
fig3.tight_layout()
#
if save_fig:
    fig3.savefig(
        fig_dir + "legend_intro_influ_niter.pdf", bbox_inches="tight")
    fig3.savefig(
        fig_dir_svg + "legend_intro_influ_niter.svg", bbox_inches="tight")
fig3.show()
