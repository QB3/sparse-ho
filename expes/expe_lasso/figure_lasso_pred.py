import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sparse_ho.utils_plot import configure_plt

# save_fig = False
save_fig = True
fig_dir = "../../../CD_SUGAR/tex/journal/prebuiltimages/"
fig_dir_svg = "../../../CD_SUGAR/tex/journal/images/"


configure_plt()

fontsize = 16

current_palette = sns.color_palette("colorblind")
dict_color = {}
dict_color["grid_search"] = current_palette[3]
dict_color["random"] = current_palette[5]
dict_color["bayesian"] = current_palette[0]
dict_color["implicit_forward"] = current_palette[2]
dict_color["forward"] = current_palette[4]
dict_color["implicit"] = current_palette[1]

dict_method = {}
dict_method["forward"] = 'F. Iterdiff.'
dict_method["implicit_forward"] = 'Imp. F. Iterdiff. (ours)'
dict_method['implicit'] = 'Implicit'
dict_method['grid_search'] = 'Grid-search'
dict_method['bayesian'] = 'Bayesian'
dict_method['random'] = 'Random-search'
dict_method['hyperopt'] = 'Random-search'
dict_method['backward'] = 'B. Iterdiff.'


dict_markers = {}
dict_markers["forward"] = 'o'
dict_markers["implicit_forward"] = 'X'
dict_markers['implicit'] = 'v'
dict_markers['grid_search'] = 'd'
dict_markers['bayesian'] = 'P'
dict_markers['random'] = '*'

dict_title = {}
dict_title["rcv1_train"] = "rcv1"
dict_title["news20"] = "news20"
dict_title["finance"] = "finance"
dict_title["kdda_train"] = "kdda"
dict_title["climate"] = "climate"
dict_title["leukemia"] = "leukemia"
dict_title["real-sim"] = "real-sim"

dict_markevery = {}
dict_markevery["news20"] = 5
dict_markevery["finance"] = 10
dict_markevery["rcv1_train"] = 1
dict_markevery["real-sim"] = 10
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
    1, len(dataset_names), sharex=False, sharey=False, figsize=[14, 4],)

fig_test, axarr_test = plt.subplots(
    1, len(dataset_names), sharex=False, sharey=False, figsize=[14, 4],)

fig_grad, axarr_grad = plt.subplots(
    1, len(dataset_names), sharex=False, sharey=False, figsize=[14, 4],)

model_name = "lasso"
# model_name = "logreg"

for idx, dataset in enumerate(dataset_names):
    df_data = pd.read_pickle("results/%s_%s.pkl" % (model_name, dataset))
    # df_data = pd.read_pickle("%s.pkl" % dataset)

    # df_data = df_data[df_data['tolerance_decrease'] == 'exponential']
    df_data = df_data[df_data['tolerance_decrease'] == 'constant']

    methods = df_data['method']
    times = df_data['times']
    objs = df_data['objs']
    # objs_tests = df_data['objs_test']
    log_alphas = df_data['log_alphas']
    # log_alpha_max = df_data['log_alpha_max'][0]
    log_alpha_max = df_data['log_alpha_max'].to_numpy()[0]
    tols = df_data['tolerance_decrease']
    # norm_y_vals = df_data['norm y_val']
    norm_val = 0
    # for norm_y_valss in norm_y_vals:
    #     norm_val = norm_y_valss

    min_objs = np.infty
    for obj in objs:
        min_objs = min(min_objs, obj.min())

    lines = []

    # plot for performance on test set
    plt.figure()
    for i, (time, obj, method, tol) in enumerate(
            zip(times, objs, methods, tols)):
            # zip(times, objs, objs_tests, methods, tols)):
        # assert not np.allclose(obj, objs_test)
        marker = dict_markers[method]
        # objs_test = [np.min(
        #     objs_test[:k]) for k in np.arange(len(objs_test)) + 1]
        # axarr_test.flat[idx].semilogy(
        # axarr_test.flat[idx].plot(
        #     time, objs_test, color=dict_color[method],
        #     label="%s" % (dict_method[method]),
        #     marker=marker, markersize=markersize,
        #     markevery=dict_markevery[dataset])

    axarr_test.flat[idx].set_xlim(0, dict_xmax[model_name, dataset])
    # axarr_test.flat[idx].set_xticks(dict_xticks[model_name, dataset])

    axarr_grad.flat[idx].set_xlabel(
        r"$\lambda - \lambda_{\max}$", fontsize=fontsize)
    axarr_test.flat[idx].set_xlabel("Time (s)", fontsize=fontsize)
    axarr_test.flat[idx].tick_params(labelsize=fontsize)
    axarr_val.flat[idx].tick_params(labelsize=fontsize)

    E0 = df_data.objs[1][0]
    for i, (time, obj, log_alpha, method, tol) in enumerate(
            zip(times, objs, log_alphas, methods, tols)):
        marker = dict_markers[method]
        # if method.startswith(('grid_search', "implicit_forward", "random")):
        if method.startswith(('grid_search', "implicit_forward", "bayesian")):
            if method.startswith(('grid_search', "random", "bayesian")):
                markevery = dict_markevery[dataset]
                color = dict_color[method]
                s = dict_marker_size[method]
            else:
                markevery = 1
                color = [plt.cm.Greens((
                    i+len(obj)/1.1) / len(
                        obj) / 2) for i in np.arange(len(obj))]
                s = 100
            # axarr_grad.flat[idx].plot(
            axarr_grad.flat[idx].scatter(
                np.array(log_alpha) - log_alpha_max, obj / E0, color=color,
                label="%s" % (dict_method[method]),
                marker=marker, s=s)
            axarr_grad.flat[idx].set_xticks(dict_xticks[model_name, dataset])

    # plot for objective minus optimum on validation set
    for i, (time, obj, method, tol) in enumerate(
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
    axarr_val.flat[idx].set_xlabel("Time (s)")

    axarr_grad.flat[idx].set_title("%s %s" % (
        dict_title[dataset], dict_n_feature[dataset]), size=fontsize)

axarr_grad.flat[0].set_ylabel("Cross validation loss", fontsize=fontsize)
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
        fig_dir_svg + "%s_lasso_val_grad.svg" % model_name,
        bbox_inches="tight")


fig_val.show()
fig_test.show()
fig_grad.show()


#################################################################
# plot legend
labels = []
for method in methods:
    labels.append(dict_method[method])

fig_legend = plt.figure(figsize=[18, 4])
fig_legend.legend(
    [l[0] for l in lines], labels,
    ncol=4, loc='upper center', fontsize=fontsize - 4)
fig_legend.tight_layout()
if save_fig:
    fig_legend.savefig(
        fig_dir + "lasso_pred_legend.pdf", bbox_inches="tight")
fig_legend.show()

# fig5 = plt.figure(figsize=[18, 4])
# fig5.legend([l[0] for l in lines], labels,
#             ncol=4, loc='upper center', fontsize=fontsize - 4)
# fig5.tight_layout()
# fig5.show()
