import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sparse_ho.utils_plot import (
    discrete_color, dict_color, dict_color_2Dplot, dict_markers,
    dict_method, dict_title, configure_plt)

save_fig = True
# save_fig = False
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
dict_marker_size["implicit_forward"] = 5
dict_marker_size["fast_iterdiff"] = 4
dict_marker_size['implicit'] = 4
dict_marker_size['grid_search'] = 1
dict_marker_size['bayesian'] = 10
dict_marker_size['random'] = 5
dict_marker_size['lhs'] = 4

dict_s = {}
dict_s["implicit_forward"] = 100
dict_s["implicit_forward_approx"] = 60
dict_s['grid_search'] = 60
dict_s['bayesian'] = 60
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

markersize = 8

# dataset_names = ["rcv1"]
# dataset_names = ["rcv1", "news20", "finance"]
# dataset_names = ["rcv1", "real-sim"]
dataset_names = ["rcv1_train", "real-sim", "news20"]
# dataset_names = ["leukemia", "rcv1", "real-sim"]
# dataset_names = ["rcv1", "real-sim", "news20"]


plt.close('all')
fig_val, axarr_val = plt.subplots(
    1, len(dataset_names), sharex=False, sharey=True, figsize=[10.67, 3.5],)

# fig_test, axarr_test = plt.subplots(
#     1, len(dataset_names), sharex=False, sharey=False, figsize=[10.67, 3.5],)

fig_grad, axarr_grad = plt.subplots(
    3, len(dataset_names), sharex=False, sharey=True, figsize=[10.67, 10],
    )

model_name = "lasso"

for idx, dataset in enumerate(dataset_names):
    df_data = pd.read_pickle("results/%s_%s.pkl" % (model_name, dataset))
    df_data = df_data[df_data['tolerance_decrease'] == 'constant']

    methods = df_data['method']
    times = df_data['times']
    objs = df_data['objs']
    alphas = df_data['alphas']
    alpha_max = df_data['alpha_max'].to_numpy()[0]
    log_alpha_max = np.log(alpha_max)
    tols = df_data['tolerance_decrease']

    min_objs = np.infty
    for obj in objs:
        min_objs = min(min_objs, obj.min())

    lines = []

    E0 = df_data.objs[2][0]
    for _, (time, obj, alpha, method, _) in enumerate(
            zip(times, objs, alphas, methods, tols)):
        log_alpha = np.log(alpha)
        if method == 'grid_search':
            for i in range(3):
                axarr_grad[i, idx].plot(
                    np.array(log_alpha) - log_alpha_max, obj / E0,
                    color='grey', zorder=-10)

    for _, (time, obj, alpha, method, _) in enumerate(
            zip(times, objs, alphas, methods, tols)):
        log_alpha = np.log(alpha)
        marker = dict_markers[method]
        n_outer = len(obj)
        s = dict_s[method]
        color = discrete_color(n_outer, dict_color_2Dplot[method])
        if method == 'grid_search':
            i = 0
        elif method == 'bayesian':
            i = 1
        elif method == 'implicit_forward_approx':
            i = 2
        else:
            continue
        axarr_grad[i, idx].scatter(
            np.array(log_alpha) - log_alpha_max, obj / E0,
            s=s, color=color,
            marker=marker, label="todo", clip_on=False)
        if method != 'random':
            axarr_grad[i, 0].set_ylabel(
                "%s \n" % dict_method[method] + "CV loss",
                fontsize=fontsize)

    # plot for objective minus optimum on validation set
    for _, (time, obj, method, _) in enumerate(
            zip(times, objs, methods, tols)):
        marker = dict_markers[method]
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


for j in range(len(dataset_names)):
    axarr_grad[2, j].set_xlabel(
        r"$\lambda - \lambda_{\max}$", fontsize=fontsize)
axarr_val.flat[0].set_ylabel(
    "Cross validation \n loss", fontsize=fontsize)
    # "$-----------$"
    # "\n" "CV loss", fontsize=fontsize)
axarr_grad.flat[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
# axarr_test.flat[0].set_ylabel("Loss on test set", fontsize=fontsize)
# for ax in axarr_val:
#     ax.set_aspect('equal', adjustable='box')

fig_val.tight_layout()
# fig_test.tight_layout()
# fig_grad.tight_layout()
if save_fig:
    fig_val.savefig(
        fig_dir + "%s_val.pdf" % model_name)
        # fig_dir + "%s_val.pdf" % model_name, bbox_inches="tight")
    fig_val.savefig(
        fig_dir_svg + "%s_val.svg" % model_name)
        # fig_dir_svg + "%s_val.svg" % model_name, bbox_inches="tight")
    # fig_test.savefig(
    #     fig_dir + "%s_test.pdf" % model_name, bbox_inches="tight")
    # fig_test.savefig(
    #     fig_dir_svg + "%s_test.svg" % model_name, bbox_inches="tight")
    fig_grad.savefig(
        fig_dir + "%s_val_grad.pdf" % model_name)
    fig_grad.savefig(
        fig_dir_svg + "%s_lasso_val_grad.svg" % model_name,
        bbox_inches="tight")


fig_val.show()
# fig_test.show()
fig_grad.show()


#################################################################
# plot legend
labels = []
for method in methods:
    labels.append(dict_method[method])

fig_legend = plt.figure(figsize=[9, 2])
fig_legend.legend(
    [l[0] for l in lines], labels,
    ncol=5, loc='upper center', fontsize=fontsize - 4)
fig_legend.tight_layout()
if save_fig:
    fig_legend.savefig(
        fig_dir + "lasso_pred_legend.pdf", bbox_inches="tight")
fig_legend.show()
