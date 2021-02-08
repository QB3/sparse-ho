import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sparse_ho.utils_plot import (
    discrete_color, dict_color, dict_color_2Dplot, dict_markers,
    dict_method, dict_title, configure_plt)

save_fig = False
# save_fig = True
# fig_dir = "./"
# fig_dir_svg = "./"
fig_dir = "../../../CD_SUGAR/tex/journal/prebuiltimages/"
fig_dir_svg = "../../../CD_SUGAR/tex/journal/images/"

configure_plt()
fontsize = 18

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
dict_marker_size["fast_iterdiff"] = 4
dict_marker_size['implicit'] = 4
dict_marker_size['grid_search'] = 5
dict_marker_size['bayesian'] = 10
dict_marker_size['random'] = 5
dict_marker_size['lhs'] = 4

dict_s = {}
dict_s["implicit_forward"] = 50
dict_s["implicit_forward_approx"] = 70
dict_s['grid_search'] = 40
dict_s['bayesian'] = 70
dict_s['random'] = 5
dict_s['lhs'] = 4

dict_n_feature = {}
dict_n_feature["rcv1_train"] = r"($p=19,959$)"
dict_n_feature["real-sim"] = r"($p=20,958$)"
dict_n_feature["news20"] = r"($p=130,107$)"
dict_n_feature["finance"] = r"($p=1,668,737$)"
dict_n_feature["leukemia"] = r"($p=7129$)"

dict_xmax = {}
dict_xmax["logreg", "rcv1_train"] = 20
dict_xmax["logreg", "real-sim"] = 30
dict_xmax["logreg", "leukemia"] = 5
dict_xmax["logreg", "news20"] = None

dict_xmax["enet", "rcv1_train"] = 200
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
fig_val, axarr_val = plt.subplots(
    1, len(dataset_names), sharex=False, sharey=False, figsize=[10.67, 3.5],)

fig_test, axarr_test = plt.subplots(
    1, len(dataset_names), sharex=False, sharey=False, figsize=[10.67, 3.5],)

fig_grad, axarr_grad = plt.subplots(
    3, len(dataset_names), sharex=False, sharey=False, figsize=[11, 10],
    )


model_name = "enet"

for idx, dataset in enumerate(dataset_names):
    df_data = pd.read_pickle("results/%s_%s.pkl" % (model_name, dataset))
    df_data = df_data[df_data['tolerance_decrease'] == 'constant']

    methods = df_data['method']
    times = df_data['times']
    objs = df_data['objs']
    all_alphas = df_data['alphas']
    alpha_max = df_data['alpha_max'].to_numpy()[0]
    tols = df_data['tolerance_decrease']

    min_objs = np.infty
    for obj in objs:
        min_objs = min(min_objs, obj.min())

    lines = []

    axarr_test.flat[idx].set_xlim(0, dict_xmax[model_name, dataset])
    axarr_test.flat[idx].set_xlabel("Time (s)", fontsize=fontsize)

    E0 = df_data.objs.to_numpy()[0][0]
    for _, (time, obj, alphas, method, _) in enumerate(
            zip(times, objs, all_alphas, methods, tols)):
        if method == 'grid_search':
            alpha1D = np.unique(alphas)
            alpha1D.sort()
            alpha1D = np.log(np.flip(alpha1D) / alpha_max)
            X, Y = np.meshgrid(alpha1D, alpha1D)
            results = obj.reshape(len(alpha1D), -1)
            levels = np.geomspace(5 * 1e-3, 1, num=30) * (
                results.max() - min_objs) / min_objs

            cmap = 'Greys_r'
            for i in range(3):
                axarr_grad[i, idx].contour(
                    X, Y, (results.T - min_objs) / min_objs, levels=levels,
                    cmap=cmap, linewidths=0.5)

    for _, (time, obj, alphas, method, _) in enumerate(
            zip(times, objs, all_alphas, methods, tols)):
        marker = dict_markers[method]
        n_outer = len(obj)
        s = dict_s[method]
        color = discrete_color(n_outer, dict_color_2Dplot[method])
        if method == 'grid_search':
            i = 0
            axarr_grad[i, idx].scatter(
                np.log(alphas[:, 0] / alpha_max),
                np.log(alphas[:, 1] / alpha_max),
                s=s, color=color,
                marker=dict_markers[method], label="todo", clip_on=False)
        elif method == 'bayesian':
            i = 1
            axarr_grad[i, idx].scatter(
                np.log(alphas[:, 0] / alpha_max),
                np.log(alphas[:, 1] / alpha_max),
                s=s, color=color,
                marker=dict_markers[method], label="todo", clip_on=False)
        elif method == 'implicit_forward_approx':
            i = 2
            axarr_grad[i, idx].scatter(
                np.log(alphas[:, 0] / alpha_max),
                np.log(alphas[:, 1] / alpha_max),
                s=s, color=color,
                marker=dict_markers[method], label="todo", clip_on=False)
        else:
            pass
        axarr_grad[i, 0].set_ylabel(
            "%s \n" % dict_method[method] + r"$\lambda_2 - \lambda_{\max}$",
            fontsize=fontsize)

    for i in range(3):
        axarr_grad[i, idx].set_xlim((alpha1D.min(), alpha1D.max()))
        axarr_grad[i, idx].set_ylim((alpha1D.min(), alpha1D.max()))

    for _, (time, obj, method, _) in enumerate(
            zip(times, objs, methods, tols)):
        marker = dict_markers[method]
        markersize = dict_marker_size[method]
        obj = [np.min(obj[:k]) for k in np.arange(len(obj)) + 1]
        lines.append(
            axarr_val.flat[idx].plot(
                time, obj / E0,
                color=dict_color[method], label="%s" % (dict_method[method]),
                marker=marker, markersize=markersize,
                markevery=dict_markevery[dataset]))
    axarr_val.flat[idx].set_xlim(0, dict_xmax[model_name, dataset])
    axarr_val.flat[idx].set_xlabel("Time (s)", fontsize=fontsize)

    axarr_grad.flat[idx].set_title("%s %s" % (
        dict_title[dataset], dict_n_feature[dataset]), size=fontsize)


for i in range(len(dataset_names)):
    axarr_grad[2, i].set_xlabel(
        r"$\lambda_1 - \lambda_{\max}$", fontsize=fontsize)
    for j in range(len(dataset_names)):
        axarr_grad[i, j].set_aspect('equal', adjustable='box')

axarr_val.flat[0].set_ylabel("Cross validation loss", fontsize=fontsize)
axarr_test.flat[0].set_ylabel("Loss on test set", fontsize=fontsize)


fig_val.tight_layout()
fig_test.tight_layout()
fig_grad.tight_layout()


if save_fig:
    fig_val.savefig(
        fig_dir + "%s_val.pdf" % model_name, bbox_inches="tight")
    fig_val.savefig(
        fig_dir_svg + "%s_val.svg" % model_name, bbox_inches="tight")
    fig_test.savefig(
        fig_dir + "%s_test.pdf" % model_name, bbox_inches="tight")
    fig_test.savefig(
        fig_dir_svg + "%s_test.svg" % model_name, bbox_inches="tight")
    fig_grad.savefig(
        fig_dir + "%s_val_grad.pdf" % model_name, bbox_inches="tight")
    fig_grad.savefig(
        fig_dir + "%s_val_grad_grid.pdf" % model_name, bbox_inches="tight")
    fig_grad.savefig(
        fig_dir_svg + "%s_val_grad.svg" % model_name,
        bbox_inches="tight")


fig_val.show()
fig_grad.show()
# fig_grad_bayesian.show()
# fig_grad_sparseho.show()

#################################################################
# plot legend
labels = []
for method in methods:
    labels.append(dict_method[method])

fig_legend = plt.figure(figsize=[9, 2])
fig_legend.legend(
    [l[0] for l in lines], labels,
    ncol=4, loc='upper center', fontsize=fontsize - 4)
fig_legend.tight_layout()
if save_fig:
    fig_legend.savefig(
        fig_dir + "enet_pred_legend.pdf", bbox_inches="tight")
fig_legend.show()
