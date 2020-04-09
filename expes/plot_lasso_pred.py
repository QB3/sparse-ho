import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

from sparse_ho.utils_plot import configure_plt

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
dict_title["rcv1"] = "rcv1"
dict_title["20newsgroups"] = "20news"
dict_title["finance"] = "finance"
dict_title["kdda_train"] = "kdda"
dict_title["climate"] = "climate"
dict_title["leukemia"] = "leukemia"

dict_markevery = {}
dict_markevery["20newsgroups"] = 5
dict_markevery["finance"] = 10
dict_markevery["rcv1"] = 1

dict_n_feature = {}
dict_n_feature["rcv1"] = r"($p=19,959$)"
dict_n_feature["20newsgroups"] = r"($p=130,107$)"
dict_n_feature["finance"] = r"($p=1,668,737$)"

markersize = 8

dataset_names = ["rcv1"]
# dataset_names = ["rcv1", "20newsgroups", "finance"]


plt.close('all')
fig, axarr = plt.subplots(
    1, 3, sharex=False, sharey=False, figsize=[14, 4],)

fig2, axarr2 = plt.subplots(
    1, 3, sharex=False, sharey=False, figsize=[14, 4],)


for idx, dataset in enumerate(dataset_names):
    df_data = pandas.read_pickle("%s.pkl" % dataset)

    df_data = df_data[df_data['tolerance_decrease'] == 'constant']

    methods = df_data['method']
    times = df_data['times']
    objs = df_data['objs']
    objs_tests = df_data['objs_test']
    log_alphas = df_data['log_alphas']
    tols = df_data['tolerance_decrease']
    norm_y_vals = df_data['norm y_val']
    norm_val = 0
    for norm_y_valss in norm_y_vals:
        norm_val = norm_y_valss

    min_objs = np.infty
    for obj in objs:
        min_objs = np.minimum(min_objs, obj.min())
        # obj = [np.min(obj[:k]) for k in np.arange(len(obj)) + 1]

    lines = []

    plt.figure()
    for i, (time, obj, objs_test, method, tol) in enumerate(
            zip(times, objs, objs_tests, methods, tols)):
        marker = dict_markers[method]
        objs_test = [np.min(
            objs_test[:k]) for k in np.arange(len(objs_test)) + 1]
        axarr2.flat[idx].semilogy(
            time, objs_test, color=dict_color[method],
            label="%s" % (dict_method[method]),
            # label="%s, %s" % (dict_method[method], tol),
            marker=marker, markersize=markersize,
            markevery=dict_markevery[dataset])
        # plt.legend()

    if dataset == "rcv1":
        axarr2.flat[idx].set_xlim(0, 1.3)
        axarr2.flat[idx].set_xticks((0, 0.5, 1))
        axarr2.flat[idx].set_yticks((0.1, 1))
    elif dataset == "20newsgroups":
        axarr2.flat[idx].set_xlim(0, 15.5)
        axarr2.flat[idx].set_xticks((0, 5, 10, 15))
        axarr2.flat[idx].set_yticks((10, 100))
    elif dataset == "finance":
        axarr2.flat[idx].set_xlim(0, 390)
        axarr2.flat[idx].set_xticks((0, 100, 200, 300))
        axarr2.flat[idx].set_yticks((0.1, 1, 10))

    axarr2.flat[idx].set_xlabel("Time (s)", fontsize=fontsize)
    axarr2.flat[idx].tick_params(labelsize=fontsize)
    axarr.flat[idx].tick_params(labelsize=fontsize)

    for i, (time, obj, method, tol) in enumerate(
            zip(times, objs, methods, tols)):
        marker = dict_markers[method]
        obj = [np.min(obj[:k]) for k in np.arange(len(obj)) + 1]
        lines.append(
            axarr.flat[idx].semilogy(
                time, (obj-min_objs),
                color=dict_color[method], label="%s" % (dict_method[method]),
                marker=marker, markersize=markersize,
                markevery=dict_markevery[dataset]))
    if dataset == "rcv1":
        axarr.flat[idx].set_xlim(0, 1.3)
        axarr.flat[idx].set_xticks((0, 0.5, 1))
        axarr.flat[idx].set_yticks((0.00001, 0.0001, 0.001, 0.01, 0.1, 1))
    elif dataset == "20newsgroups":
        axarr.flat[idx].set_xlim(0, 15.5)
        axarr.flat[idx].set_xticks((0, 5, 10, 15))
        axarr.flat[idx].set_yticks((0.001, 0.01, 0.1, 1, 10, 100))
    elif dataset == "finance":
        axarr.flat[idx].set_xlim(0, 390)
        axarr.flat[idx].set_xticks((0, 100, 200, 300))
        axarr.flat[idx].set_yticks((0.0001, 0.001, 0.01, 0.1, 1, 10))

    axarr.flat[idx].set_title("%s %s" % (
        dict_title[dataset], dict_n_feature[dataset]), size=fontsize)
    # axarr.flat[idx].title.set_text(dict_title[dataset], size=18)

axarr.flat[0].set_ylabel("Objective minus optimum", fontsize=fontsize)

fig.tight_layout()
fig.show()

axarr2.flat[0].set_ylabel(
    "Loss on test set", fontsize=fontsize)
fig2.tight_layout()
fig2.show()


labels = []
for method in methods:
    labels.append(dict_method[method])

fig3 = plt.figure(figsize=[18, 4])
fig3.legend([l[0] for l in lines], labels,
            ncol=6, loc='upper center', fontsize=fontsize-4)
fig3.tight_layout()
fig3.show()

fig4 = plt.figure(figsize=[18, 4])
fig4.legend([l[0] for l in lines], labels,
            ncol=3, loc='upper center', fontsize=fontsize-4)
fig4.tight_layout()
fig4.show()
