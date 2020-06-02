import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from sparse_ho.utils_plot import configure_plt

configure_plt()
fontsize = 22

current_palette = sns.color_palette("colorblind")
dict_color = {}
dict_color["GridSearch"] = current_palette[3]
dict_color["random"] = current_palette[5]
dict_color["bayesian"] = current_palette[0]
dict_color["implicit_forward"] = current_palette[2]
dict_color["forward"] = current_palette[4]
dict_color["implicit"] = current_palette[1]

dict_label = {}
dict_label["forward"] = 'F. iterdiff.'
dict_label["implicit_forward"] = 'Imp. F. iterdiff. (ours)'
dict_label['implicit'] = 'Implicit diff. (ours)'
dict_label['GridSearch'] = 'Grid-search'
dict_label['bayesian'] = 'Bayesian'
dict_label['random'] = 'Random-search'
dict_label['hyperopt'] = 'Random-search'

dict_markers = {}
dict_markers["forward"] = 'o'
dict_markers["implicit_forward"] = 'X'
dict_markers['implicit'] = 'v'
dict_markers['GridSearch'] = 'd'
dict_markers['bayesian'] = 'P'
dict_markers['random'] = '*'

dict_markers_size = {}
dict_markers_size["forward"] = 10
dict_markers_size["implicit_forward"] = 10
dict_markers_size['implicit'] = 10
dict_markers_size['GridSearch'] = 12
dict_markers_size['bayesian'] = 10
dict_markers_size['random'] = 12

dict_filling = {}
dict_filling["forward"] = 'none'
dict_filling["implicit_forward"] = 'full'
dict_filling['implicit'] = 'full'
dict_filling['GridSearch'] = 'none'
dict_filling['bayesian'] = 'full'
dict_filling['random'] = 'full'

df = pandas.read_pickle("%s.pkl" % "results_lasso_cor")

labels = np.flip([
    "implicit_forward",  "implicit", "forward", "GridSearch", "bayesian",
    "random"])

dict_markevery = [5, (1, 5), (2, 5), (3, 5), 1, (4, 5)]

plt.close('all')
fig, axarr = plt.subplots(1, 2, sharex=False, sharey=False, figsize=[10, 4],)
df_mean = df.groupby(['method', 'p'], as_index=False).mean()
best = df_mean[df_mean['method'].eq("implicit_forward")]
best = np.array(best['Error est'])
lines = []
plt.figure()
for i in range(np.size(labels)):
    df_temp = df_mean[df_mean['method'].eq(labels[i])]
    lines.append(axarr.flat[0].plot(
        df_temp['p'], df_temp['Error est'],
        label=dict_label[labels[i]], c=dict_color[labels[i]],
        markersize=dict_markers_size[labels[i]],
        marker=dict_markers[labels[i]],
        markevery=dict_markevery[i], fillstyle=dict_filling[labels[i]],
        markeredgewidth=2))
    axarr.flat[0].set_ylabel("MSE", fontsize=fontsize)
    axarr.flat[0].set_xlabel("Number of features (p)", fontsize=fontsize)

axarr.flat[0].set_xlim((0, 10000))
axarr.flat[0].set_ylim((0.19, 0.36))
axarr.flat[0].set_xticks((200, 2500, 5000, 7500, 10000))
axarr.flat[0].set_yticks((0.2, 0.25, 0.3, 0.35))
# axarr.flat[0].set_ylim((-0.005, 0.01))

for i in range(np.size(labels)):
    df_temp = df_mean[df_mean['method'].eq(labels[i])]
    axarr.flat[1].semilogy(
        df_temp['p'], df_temp['time'],
        label=dict_label[labels[i]], c=dict_color[labels[i]],
        markersize=dict_markers_size[labels[i]],
        marker=dict_markers[labels[i]],
        markevery=dict_markevery[i], fillstyle=dict_filling[labels[i]],
        markeredgewidth=2)
    axarr.flat[1].set_ylabel("Time (s)", fontsize=fontsize)
    axarr.flat[1].set_xlabel("Number of features (p)", fontsize=fontsize)
axarr.flat[1].set_xlim((0, 10000))
axarr.flat[1].set_xticks((200, 2500, 5000, 7500, 10000))
axarr.flat[1].set_yticks((1., 10, 100))
axarr.flat[0].tick_params(labelsize=fontsize)
axarr.flat[1].tick_params(labelsize=fontsize)
fig.tight_layout()
fig.show()

labels = ([
    "implicit_forward",  "implicit", "forward", "GridSearch", "bayesian",
    "random"])

fig, axarr = plt.subplots(
    1, 2, sharex=False, sharey=False, figsize=[10, 4])
df_mean = df.groupby(['method', 'p'], as_index=False).mean()
best = df_mean[df_mean['method'].eq("implicit_forward")]
best = np.array(best['Error est'])
lines = []
plt.figure()
for i in range(np.size(labels)):
    df_temp = df_mean[df_mean['method'].eq(labels[i])]
    lines.append(
        axarr.flat[0].plot(
            df_temp['p'], df_temp['Error est'],
            label=dict_label[labels[i]], c=dict_color[labels[i]],
            markersize=dict_markers_size[labels[i]],
            marker=dict_markers[labels[i]], markevery=dict_markevery[i],
            fillstyle=dict_filling[labels[i]], markeredgewidth=2))
    axarr.flat[0].set_ylabel("MSE", fontsize=fontsize)
    axarr.flat[0].set_xlabel("Number of features (p)", fontsize=fontsize)

labels = ([
    'Imp. F. Iterdiff. (ours)', 'Implicit', "F. Iterdiff.", "Grid-search",
    "Bayesian", "Random-search"])
fig_legend = plt.figure(figsize=[18, 4])
fig_legend.legend(
    [l[0] for l in lines], labels, ncol=3, loc="upper center",
    fontsize=fontsize)
fig_legend.tight_layout()
fig_legend.show()
