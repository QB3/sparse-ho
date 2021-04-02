import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sparse_ho.utils_plot import (
    discrete_color, dict_color, dict_color_2Dplot, dict_markers,
    dict_method, dict_title, configure_plt, plot_legend_apart)

# save_fig = False
save_fig = True
# fig_dir = "./"
# fig_dir_svg = "./"
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
dict_marker_size["implicit_forward"] = 10
dict_marker_size["implicit_forward_approx"] = 10
dict_marker_size["implicit"] = 10
dict_marker_size["implicit_approx"] = 10
dict_marker_size["fast_iterdiff"] = 4
dict_marker_size['grid_search'] = 5
dict_marker_size['bayesian'] = 10
dict_marker_size['random'] = 5
dict_marker_size['lhs'] = 4

dict_s = {}
dict_s["implicit"] = 50
dict_s["implicit_approx"] = 70
dict_s["implicit_forward"] = 50
dict_s["implicit_forward_approx"] = 70
dict_s['grid_search'] = 40
dict_s['bayesian'] = 70
dict_s['random'] = 5
dict_s['lhs'] = 4

dict_n_feature = {}
dict_n_feature["rcv1_train"] = r"($p=19,959$)"
dict_n_feature["real-sim"] = r"($p=20,958$)"
dict_n_feature["news20"] = r"($p=632,982$)"
dict_n_feature["finance"] = r"($p=1,668,737$)"
dict_n_feature["leukemia"] = r"($p=7129$)"

dict_xmax = {}
dict_xmax["enet", "rcv1_train"] = 250
dict_xmax["enet", "real-sim"] = 400
dict_xmax["enet", "leukemia"] = 5
dict_xmax["enet", "news20"] = 2000

dict_xticks = {}
dict_xticks["enet", "rcv1_train"] = (-6, -4, -2, 0)
dict_xticks["enet", "real-sim"] = (-6, -4, -2, 0)
dict_xticks["enet", "leukemia"] = (-6, -4, -2, 0)
dict_xticks["enet", "news20"] = (-8, -6, -4, -2, 0)

dict_xticks["logreg", "rcv1"] = (-8, -6, -4, -2, 0)
dict_xticks["logreg", "real-sim"] = (-8, -6, -4, -2, 0)
dict_xticks["logreg", "leukemia"] = (-8, -6, -4, -2, 0)
dict_xticks["logreg", "news20"] = (-8, -6, -4, -2, 0)

markersize = 8

dataset_names = ["rcv1_train", "real-sim", "news20"]


plt.close('all')
# fig_val, axarr_val = plt.subplots(
#     1, len(dataset_names), sharex=False, sharey=True, figsize=[10.67, 3.5])

fig_grad, axarr_grad = plt.subplots(
    len(dataset_names), 4, sharex=False, sharey=False, figsize=[14, 10])

dict_idx = {}
dict_idx['grid_search'] = 0
dict_idx['bayesian'] = 1
dict_idx['implicit_forward_approx'] = 2
dict_idx['implicit_approx'] = 2

model_name = "enet"
methods = [
    "implicit", 'grid_search', 'random', 'bayesian', "implicit_approx"]

for idx, dataset in enumerate(dataset_names):
    for method in methods:
        df_data = pd.read_pickle(
            "results/%s_%s_%s.pkl" % (model_name, dataset, method))

        time = df_data['times'].to_numpy()[0]
        obj = np.array(df_data['objs'].to_numpy()[0])
        alphas = np.array(df_data['alphas'].to_numpy()[0])
        alpha_max = df_data['alpha_max'].to_numpy()[0]

        E0 = 1
        if method == 'grid_search':
            min_objs = obj.min()
            alpha1D = np.unique(alphas)
            alpha1D.sort()
            alpha1D = np.log(np.flip(alpha1D) / alpha_max)
            X, Y = np.meshgrid(alpha1D, alpha1D)
            results = obj.reshape(len(alpha1D), -1)
            levels = np.geomspace(5 * 1e-3, 1, num=30) * (
                results.max() - min_objs) / min_objs

            cmap = 'Greys_r'
            for i in range(3):
                axarr_grad[idx, i].contour(
                    X, Y, (results.T - min_objs) / min_objs, levels=levels,
                    cmap=cmap, linewidths=0.5)

        marker = dict_markers[method]
        n_outer = len(obj)
        s = dict_s[method]
        color = discrete_color(n_outer, dict_color_2Dplot[method])

        if method in dict_idx:
            axarr_grad[idx, dict_idx[method]].scatter(
                np.log(alphas[:, 0] / alpha_max),
                np.log(alphas[:, 1] / alpha_max),
                s=s, color=color,
                marker=dict_markers[method], label="todo", clip_on=False)
            axarr_grad[0, dict_idx[method]].set_title(
                dict_method[method], size=fontsize)

        axarr_grad[idx, 0].set_ylabel(
            # "%s \n" % dict_method[method] +
            "%s" % (dict_title[dataset]) + \
            "\n" + r"$\lambda_2 - \lambda_{\max}$",
            fontsize=fontsize)

        axarr_grad[idx, 3].set_ylabel(
            # "%s \n" % dict_method[method] +
            "CV loss",
            fontsize=fontsize)

        marker = dict_markers[method]
        markersize = dict_marker_size[method]
        obj = np.array([np.min(obj[:k]) for k in np.arange(len(obj)) + 1])
        axarr_grad[idx, 3].plot(
            time, obj / E0,
            color=dict_color[method], label="%s" % (dict_method[method]),
            marker=marker, markersize=markersize,
            markevery=dict_markevery[dataset])

    for i in range(3):
        axarr_grad[idx, i].set_xlim((alpha1D.min(), alpha1D.max()))
        axarr_grad[idx, i].set_ylim((alpha1D.min(), alpha1D.max()))
        axarr_grad[idx, i].tick_params(
            axis='both', which='major', labelsize=labelsize)
    axarr_grad[idx, 3].tick_params(
        axis='both', which='major', labelsize=labelsize)

    axarr_grad[idx, 3].set_xlim(0, dict_xmax[model_name, dataset])
    axarr_grad[idx, 3].set_ylim(0.15, 0.6)

axarr_grad[2, 3].set_xlabel("Time (s)", fontsize=fontsize)
# axarr_val.flat[0].set_ylim(0.15, 0.6)

for i in range(len(dataset_names)):
    axarr_grad[2, i].set_xlabel(
        r"$\lambda_1 - \lambda_{\max}$", fontsize=fontsize)
    for j in range(len(dataset_names)):
        axarr_grad[i, j].set_aspect('equal', adjustable='box')
        axarr_grad[i, j].set_xticks([-10, -5, 0])
        axarr_grad[i, j].set_yticks([-10, -5, 0])

fig_grad.tight_layout()


if save_fig:
    fig_grad.savefig(
        fig_dir + "%s_val_grad_slides.pdf" % model_name, bbox_inches="tight")
    fig_grad.savefig(
        fig_dir_svg + "%s_val_grad_slides.svg" % model_name,
        bbox_inches="tight")
    plot_legend_apart(
        axarr_grad[0, 3],
        fig_dir + "%s_val_grad_slides_legend.pdf" % model_name,
        figwidth=12)
fig_grad.show()
