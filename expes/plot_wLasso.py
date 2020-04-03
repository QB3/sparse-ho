from itertools import product
import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

from implicit_forward.utils_plot import configure_plt

configure_plt()
fontsize = 22

current_palette = sns.color_palette("colorblind")
dict_color_lasso = {}
dict_color_lasso["GridSearch"] = current_palette[3]
dict_color_lasso["random"] = current_palette[5]
dict_color_lasso["bayesian"] = current_palette[0]
dict_color_lasso["implicit_forward"] = current_palette[2]
dict_color_lasso["forward"] = current_palette[4]
dict_color_lasso["implicit"] = current_palette[1]

dict_color_alasso = {}
dict_color_alasso["implicit_forward"] = current_palette[9]
dict_color_alasso["forward"] = current_palette[7]



current_palette = sns.color_palette("colorblind")

df = pandas.read_pickle("%s.pkl" % "results_alasso")
df_lasso = df.loc[df['Model'] == "lasso"]
df_alasso = df.loc[df['Model'] == "alasso"]

dict_method_lasso={}
dict_method_alasso={}
dict_method_lasso["implicit_forward"] = "Lasso Imp. F. iterdiff. (ours)"
dict_method_lasso["forward"] = "Lasso F. iterdiff"
dict_method_alasso["implicit_forward"] = "aLasso Imp. F. iterdiff. (ours)"
dict_method_alasso["forward"] = "aLasso F. iterdiff."


dict_markers_lasso = {}
dict_markers_lasso["forward"] = 'o'
dict_markers_lasso["implicit_forward"] = 'X'
dict_markers_alasso = {}
dict_markers_alasso["forward"] = '^'
dict_markers_alasso["implicit_forward"] = 's'


dict_filling_lasso = {}
dict_filling_lasso["forward"] = 'none'
dict_filling_lasso["implicit_forward"] = 'full'


dict_filling_alasso = {}
dict_filling_alasso["forward"] = 'none'
dict_filling_alasso["implicit_forward"] = 'full'

dict_markevery = [(1,2), 2]
markersize = []
plt.close('all')
fig, axarr = plt.subplots(1, 2, sharex=False, sharey=False, figsize=[10,4],)
df_median_lasso = df_lasso.groupby(['Model', 'method', 'p'], as_index=False).mean()
df_median_alasso = df_alasso.groupby(['Model', 'method', 'p'], as_index=False).mean()
lines = []


labels = ["implicit_forward", "forward"]
plt.figure()
for i in range(np.size(labels)):
    df_temp = df_median_lasso[df_median_lasso['method'].eq(labels[i])]
    lines.append(axarr.flat[0].plot(df_temp['p'], df_temp['Error est'], label=dict_method_lasso[labels[i]], c=dict_color_lasso[labels[i]], markersize=10, marker=dict_markers_lasso[labels[i]], markevery=dict_markevery[i],fillstyle=dict_filling_lasso[labels[i]], markeredgewidth=2))


for i in range(np.size(labels)):
    df_temp = df_median_alasso[df_median_alasso['method'].eq(labels[i])]
    lines.append(axarr.flat[0].plot(df_temp['p'], df_temp['Error est'], label=dict_method_alasso[labels[i]], c=dict_color_alasso[labels[i]], markersize=10, marker=dict_markers_alasso[labels[i]], markevery=dict_markevery[i],fillstyle=dict_filling_alasso[labels[i]], markeredgewidth=2))

axarr.flat[0].set_ylabel("MSE", fontsize=fontsize)
axarr.flat[0].set_xlabel("Number of features (p)", fontsize=fontsize)

axarr.flat[0].set_xticks((200., 2500, 5000, 7500, 10000))

for i in range(np.size(labels)):
    df_temp = df_median_lasso[df_median_lasso['method'].eq(labels[i])]
    axarr.flat[1].semilogy(df_temp['p'], df_temp['time'], label=dict_method_lasso[labels[i]], c=dict_color_lasso[labels[i]], markersize=10, marker=dict_markers_lasso[labels[i]], markevery=dict_markevery[i],fillstyle=dict_filling_lasso[labels[i]], markeredgewidth=2)

for i in range(np.size(labels)):
    df_temp = df_median_alasso[df_median_alasso['method'].eq(labels[i])]
    axarr.flat[1].semilogy(df_temp['p'], df_temp['time'], label=dict_method_alasso[labels[i]], c=dict_color_alasso[labels[i]], markersize=10, marker=dict_markers_alasso[labels[i]], markevery=dict_markevery[i],fillstyle=dict_filling_alasso[labels[i]], markeredgewidth=2)
axarr.flat[1].set_ylabel("Time (s)", fontsize=fontsize)
axarr.flat[1].set_xlabel("Number of features (p)", fontsize=fontsize)
axarr.flat[1].set_xticks((200., 2500, 5000, 7500, 10000))
axarr.flat[1].set_yticks((1, 10, 100, 1000))

fig.tight_layout()
fig.show()


labels = ["Lasso Imp. F. Iterdiff. (ours)","Lasso F. Iterdiff.", "wLasso Imp. F. Iterdiff. (ours)", "wLasso F. Iterdiff.",]
fig_legend = plt.figure(figsize=[18, 4])
fig_legend.legend([l[0] for l in lines], labels, ncol= 2, loc="upper center", fontsize=fontsize)
fig_legend.tight_layout()
fig_legend.show()
