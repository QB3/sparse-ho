import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sparse_ho.utils_plot import (
    discrete_color, dict_color, dict_color_2Dplot, dict_markers,
    dict_method, dict_title, dict_n_features, configure_plt, plot_legend_apart)

save_fig = True
# save_fig = False
fig_dir = "../../../CD_SUGAR/tex/slides_qbe_long/prebuiltimages/"
fig_dir_svg = "../../../CD_SUGAR/tex/slides_qbe_long/images/"

configure_plt()
fontsize = 28
labelsize = 25


dict_markevery = {}
dict_markevery["news20"] = 1
dict_markevery["finance"] = 10
dict_markevery["rcv1_train"] = 1
dict_markevery["real-sim"] = 1
dict_markevery["leukemia"] = 10

dict_marker_size = {}
dict_marker_size["forward"] = 4
dict_marker_size["implicit_forward"] = 5
dict_marker_size["fast_iterdiff"] = 4
dict_marker_size['implicit'] = 5
dict_marker_size['grid_search'] = 1
dict_marker_size['bayesian'] = 10
dict_marker_size['random'] = 5
dict_marker_size['lhs'] = 4

dict_s = {}
dict_s["implicit_forward"] = 100
dict_s["implicit"] = 100
dict_s["implicit_forward_approx"] = 60
dict_s["implicit_approx"] = 60
dict_s['grid_search'] = 60
dict_s['bayesian'] = 60
dict_s['random'] = 5
dict_s['lhs'] = 4

dict_xmax = {}
dict_xmax["logreg", "rcv1_train"] = 20
dict_xmax["logreg", "real-sim"] = 30
dict_xmax["logreg", "leukemia"] = 5
dict_xmax["logreg", "news20"] = None

dict_xmax["lasso", "rcv1_train"] = 60
dict_xmax["lasso", "real-sim"] = 200
dict_xmax["lasso", "leukemia"] = 5
dict_xmax["lasso", "news20"] = 1200

dict_xticks = {}
dict_xticks["lasso", "rcv1_train"] = (-6, -4, -2, 0)
dict_xticks["lasso", "real-sim"] = (-6, -4, -2, 0)
dict_xticks["lasso", "leukemia"] = (-6, -4, -2, 0)
dict_xticks["lasso", "news20"] = (-8, -6, -4, -2, 0)

dict_xticks["logreg", "rcv1"] = (-8, -6, -4, -2, 0)
dict_xticks["logreg", "real-sim"] = (-8, -6, -4, -2, 0)
dict_xticks["logreg", "leukemia"] = (-8, -6, -4, -2, 0)
dict_xticks["logreg", "news20"] = (-8, -6, -4, -2, 0)

dict_idx = {}
dict_idx['grid_search'] = 0
dict_idx['bayesian'] = 1
dict_idx['implicit_approx'] = 2

markersize = 8

dataset_names = ["rcv1_train", "real-sim", "news20"]
methods = [
    "implicit", "implicit_approx", 'grid_search',
    'random', 'bayesian']

plt.close('all')
fig_grad, axarr_grad = plt.subplots(
    len(dataset_names), 4, sharex=False, sharey=False, figsize=[14, 10])

model_name = "lasso"

for idx, dataset in enumerate(dataset_names):
    for method in methods:
        df_data = pd.read_pickle(
            "results/%s_%s_%s.pkl" % (model_name, dataset, method))
        time = df_data['times'].to_numpy()[0]
        obj = np.array(df_data['objs'].to_numpy()[0])
        alpha = df_data['alphas'].to_numpy()[0]
        alpha_max = df_data['alpha_max'].to_numpy()[0]
        log_alpha_max = np.log(alpha_max)

        E0 = 1

        log_alpha = np.log(alpha)
        if method == 'grid_search':
            for i in range(3):
                axarr_grad[idx, i].plot(
                    np.array(log_alpha) - log_alpha_max, obj / E0,
                    color='grey', zorder=-10)

        log_alpha = np.log(alpha)
        marker = dict_markers[method]
        n_outer = len(obj)
        s = dict_s[method]
        color = discrete_color(n_outer, dict_color_2Dplot[method])

        if method in dict_idx:
            axarr_grad[idx, dict_idx[method]].scatter(
                np.array(log_alpha) - log_alpha_max, obj / E0,
                s=s, color=color, marker=marker, label="todo", clip_on=False)
            axarr_grad[idx, 0].set_ylabel(
                "%s" % (dict_title[dataset]) + \
                "\n" + "CV loss", fontsize=fontsize)
            axarr_grad[0, dict_idx[method]].set_title(
                dict_method[method], size=fontsize)

        marker = dict_markers[method]
        obj = [np.min(obj[:k]) for k in np.arange(len(obj)) + 1]
        obj = np.array(obj)

        axarr_grad[idx, 3].plot(
            time, obj / E0,
            color=dict_color[method], label="%s" % (dict_method[method]),
            marker=marker, markersize=markersize,
            markevery=dict_markevery[dataset])
    #########################################################################
    axarr_grad[idx, 3].set_xlim(0, dict_xmax[model_name, dataset])
    axarr_grad[idx, 3].set_ylim(0.15, 0.4)
    axarr_grad[idx, 0].set_ylim(0.15, 1)

    for i in range(3):
        axarr_grad[idx, i].tick_params(
            axis='both', which='major', labelsize=labelsize)
    axarr_grad[idx, 0].set_yticks([0.2, 0.4, 0.6, 0.8, 1])

    axarr_grad[idx, 3].tick_params(
        axis='both', which='major', labelsize=labelsize)


axarr_grad[2, 3].set_xlabel("Time (s)", fontsize=fontsize)

for j in range(len(dataset_names)):
    axarr_grad[2, j].set_xlabel(
        r"$\lambda - \lambda_{\max}$", fontsize=fontsize)


fig_grad.tight_layout()
if save_fig:
    fig_grad.savefig(
        fig_dir + "%s_val_grad_slides.pdf" % model_name)
    fig_grad.savefig(
        fig_dir_svg + "%s_lasso_val_grad_slides.svg" % model_name,
        bbox_inches="tight")
    plot_legend_apart(
        axarr_grad[0, 3],
        fig_dir + "%s_val_grad_slides_legend.pdf" % model_name, figwidth=12)
fig_grad.show()
