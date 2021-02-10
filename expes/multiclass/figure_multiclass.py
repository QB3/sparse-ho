import numpy as np
import pandas
# import seaborn as sns
import matplotlib.pyplot as plt
from sparse_ho.utils_plot import (
    configure_plt, plot_legend_apart, dict_color, dict_method)

# save_fig = False
save_fig = True
fig_dir = "../../../CD_SUGAR/tex/journal/prebuiltimages/"
fig_dir_svg = "../../../CD_SUGAR/tex/journal/images/"

configure_plt()

fontsize = 18


dict_markers = {}
dict_markers["forward"] = 'o'
dict_markers["implicit_forward"] = 'X'
dict_markers["implicit_forward_cdls"] = 'X'
dict_markers["implicit_forward_scipy"] = 'X'
dict_markers["fast_iterdiff"] = 'X'
dict_markers['implicit'] = 'v'
dict_markers['grid_search'] = 'd'
dict_markers['bayesian'] = 'P'
dict_markers['random'] = '*'
dict_markers['lhs'] = 'H'

dict_title = {}
dict_title["mnist"] = "mnist"
dict_title["rcv1_multiclass"] = "rcv1"
dict_title["20news"] = "20news"
dict_title["finance"] = "finance"
dict_title["kdda_train"] = "kdda"
dict_title["climate"] = "climate"
dict_title["leukemia"] = "leukemia"
dict_title["real-sim"] = "real-sim"
dict_title["sensit"] = "sensit"
dict_title["aloi"] = "aloi"
dict_title["usps"] = "usps"
dict_title["sector_scale"] = "sector"

dict_n_classes = {}
dict_n_classes["mnist"] = 10
dict_n_classes["rcv1_multiclass"] = 53
dict_n_classes["20news"] = "20news"
dict_n_classes["finance"] = "finance"
dict_n_classes["kdda_train"] = "kdda"
dict_n_classes["climate"] = "climate"
dict_n_classes["leukemia"] = "leukemia"
dict_n_classes["real-sim"] = "real-sim"
dict_n_classes["sensit"] = "sensit"
dict_n_classes["aloi"] = 1_000
dict_n_classes["usps"] = 10
dict_n_classes["sector_scale"] = 105


dict_markevery = {}
dict_markevery["20news"] = 5
dict_markevery["finance"] = 10
dict_markevery["rcv1"] = 5
dict_markevery["leukemia"] = 1
dict_markevery["real-sim"] = 5
dict_markevery["sensit"] = 1
dict_markevery["aloi"] = 1
dict_markevery["usps"] = 1
dict_markevery["sector_scale"] = 1

dict_n_feature = {}
dict_n_feature["rcv1"] = r"($p=19,959$)"
dict_n_feature["20news"] = r"($p=130,107$)"
dict_n_feature["finance"] = r"($p=1,668,737$)"
dict_n_feature["leukemia"] = r"($p=7,129$)"
dict_n_feature["real-sim"] = r"($p=20,958$)"
dict_n_feature["sensit"] = r"($p=0$)"
dict_n_feature["aloi"] = r"($p=0$)"
dict_n_feature["sector_scale"] = r"($p=0$)"

dict_marker_size = {}
dict_marker_size["forward"] = 4
dict_marker_size["implicit_forward"] = 9
dict_marker_size["implicit_forward_cdls"] = 9
dict_marker_size["implicit_forward_scipy"] = 9
dict_marker_size["fast_iterdiff"] = 4
dict_marker_size['implicit'] = 4
dict_marker_size['grid_search'] = 5
dict_marker_size['bayesian'] = 4
dict_marker_size['random'] = 5
dict_marker_size['lhs'] = 4

dict_xlim = {}
dict_xlim["mnist"] = 500
dict_xlim["usps"] = 1_200
dict_xlim["rcv1_multiclass"] = 1_400
dict_xlim["aloi"] = 2_000

# dataset_names = ["mnist", "rcv1_multiclass", "sector_scale","aloi"]
dataset_names = ["mnist", "usps", "rcv1_multiclass", "aloi"]
# methods = ["implicit_forward_cdls", "random"]
# methods = ["implicit_forward_cdls", "random", "bayesian"]
# methods = ["implicit_forward", "random"]
methods = ["random", "bayesian", "implicit_forward"]
# methods = ["random", "bayesian", "grid_search", "implicit_forward"]

plt.close('all')
fig_acc_val, axarr_acc_val = plt.subplots(
    1, len(dataset_names), sharex=False, sharey=False, figsize=[10.67, 3])

fig_acc_test, axarr_acc_test = plt.subplots(
    1, len(dataset_names), sharex=False, sharey=False, figsize=[10.67, 3])

# figure for cross entropy
fig_ce, axarr_ce = plt.subplots(
    1, len(dataset_names), sharex=False, sharey=False, figsize=[10.67, 3])

all_figs = [fig_acc_val, fig_acc_test, fig_ce]
all_axarr = [axarr_acc_val, axarr_acc_test, axarr_ce]
all_strings = ["acc_val", "acc_test", "crossentropy"]


for idx, dataset_name in enumerate(dataset_names):
    plt.figure()
    for method in methods:
        try:
            df_data = pandas.read_pickle(
                "results/%s_%s.pkl" % (dataset_name, method))

            method = df_data['method'].to_numpy()[0]
            times = df_data['times'].to_numpy()[0]
            objs = df_data['objs'].to_numpy()[0]
            objs = [
                np.min(objs[:k]) for k in np.arange(len(objs)) + 1]
            log_alphas = df_data['log_alphas'].to_numpy()[0]
            acc_vals = df_data['acc_vals'].to_numpy()[0]
            acc_vals = [
                np.max(acc_vals[:k]) for k in np.arange(len(acc_vals)) + 1]
            acc_tests = df_data['acc_tests'].to_numpy()[0]
            acc_tests = [
                np.max(acc_tests[:k]) for k in np.arange(len(acc_tests)) + 1]

            axarr_acc_val.flat[idx].plot(
                times, acc_vals, label=dict_method[method],
                marker=dict_markers[method],
                color=dict_color[method])
            axarr_acc_test.flat[idx].plot(
                times, acc_tests, label=dict_method[method],
                marker=dict_markers[method],
                color=dict_color[method])
            axarr_ce.flat[idx].plot(
                times, objs, label=dict_method[method],
                color=dict_color[method],
                marker=dict_markers[method])
        except Exception:
            print("No dataset found")
    for axarr in all_axarr:
        axarr[idx].set_xlim(0, dict_xlim[dataset_name])

for i, dataset_name in enumerate(dataset_names):
    axarr_ce[i].set_title("%s (K=%i)" % (
        dict_title[dataset_name], dict_n_classes[dataset_name]))
    axarr_acc_test[i].set_xlabel("Time (s)", fontsize=fontsize)


axarr_acc_val.flat[0].set_ylabel("Accuracy validation set", fontsize=fontsize)
axarr_acc_test.flat[0].set_ylabel("Accuracy test set", fontsize=fontsize)
axarr_ce.flat[0].set_ylabel("Multiclass cross entropy", fontsize=fontsize)

for fig in all_figs:
    fig.tight_layout()


if save_fig:
    for string, fig in zip(all_strings, all_figs):
        fig.savefig("%smulticlass_%s.pdf" % (
            fig_dir, string), bbox_inches="tight")
        fig.savefig("%smulticlass_%s.svg" % (
            fig_dir_svg, string), bbox_inches="tight")
    plot_legend_apart(
        axarr_acc_val[0], "%smulticlass_%s_legend.pdf" % (fig_dir, string),
        ncol=3)

for fig in all_figs:
    fig.legend()
    fig.show()


for dataset_name in dataset_names:
    df_data = pandas.read_pickle(
        "results/%s_grid_search.pkl" % dataset_name)
    objs = df_data['objs'].to_numpy()[0]
    print(objs)
    log_alphas = df_data['log_alphas'].to_numpy()[0]
    log_alpha_max = log_alphas[0][0]
    log_alpha_min = log_alphas[-1][0]
    log_alpha_opt = log_alphas[idx][0]

    print("p_alpha %f " % np.exp(log_alpha_opt - log_alpha_max))
    print("range_alpha %f " % np.exp(log_alpha_min - log_alpha_max))
