import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from expes.utils import configure_plt

save_fig = False
# save_fig = True
fig_dir = "../../../CD_SUGAR/tex/journal/prebuiltimages/"
fig_dir_svg = "../../../CD_SUGAR/tex/journal/images/"

configure_plt()

# init()

fontsize = 16

current_palette = sns.color_palette("colorblind")
dict_color = {}
dict_color["grid_search"] = current_palette[3]
dict_color["random"] = current_palette[5]
dict_color["bayesian"] = current_palette[0]
dict_color["implicit_forward"] = current_palette[2]
dict_color["implicit_forward_cdls"] = current_palette[2]
dict_color["fast_iterdiff"] = current_palette[2]
dict_color["forward"] = current_palette[4]
dict_color["implicit"] = current_palette[1]
dict_color["lhs"] = current_palette[6]

dict_method = {}
dict_method["forward"] = 'F. Iterdiff.'
dict_method["implicit_forward"] = 'Imp. F. Iterdiff. (ours)'
dict_method["implicit_forward_cdls"] = 'Imp. F. Iterdiff. CD LS'
dict_method["fast_iterdiff"] = 'Imp. F. Iterdiff. (ours)'
dict_method['implicit'] = 'Implicit'
dict_method['grid_search'] = 'Grid-search'
dict_method['bayesian'] = 'Bayesian'
dict_method['random'] = 'Random-search'
dict_method['hyperopt'] = 'Random-search'
dict_method['backward'] = 'B. Iterdiff.'
dict_method['lhs'] = 'Lattice Hyp.'


dict_markers = {}
dict_markers["forward"] = 'o'
dict_markers["implicit_forward"] = 'X'
dict_markers["implicit_forward_cdls"] = 'X'
dict_markers["fast_iterdiff"] = 'X'
dict_markers['implicit'] = 'v'
dict_markers['grid_search'] = 'd'
dict_markers['bayesian'] = 'P'
dict_markers['random'] = '*'
dict_markers['lhs'] = 'H'

dict_title = {}
dict_title["rcv1"] = "rcv1"
dict_title["20news"] = "20news"
dict_title["finance"] = "finance"
dict_title["kdda_train"] = "kdda"
dict_title["climate"] = "climate"
dict_title["leukemia"] = "leukemia"
dict_title["real-sim"] = "real-sim"

dict_markevery = {}
dict_markevery["20news"] = 5
dict_markevery["finance"] = 10
dict_markevery["rcv1"] = 5
dict_markevery["leukemia"] = 1
dict_markevery["real-sim"] = 5

dict_n_feature = {}
dict_n_feature["rcv1"] = r"($p=19,959$)"
dict_n_feature["20news"] = r"($p=130,107$)"
dict_n_feature["finance"] = r"($p=1,668,737$)"
dict_n_feature["leukemia"] = r"($p=7,129$)"
dict_n_feature["real-sim"] = r"($p=20,958$)"

markersize = 8

dict_marker_size = {}
dict_marker_size["forward"] = 4
dict_marker_size["implicit_forward"] = 9
dict_marker_size["implicit_forward_cdls"] = 9
dict_marker_size["fast_iterdiff"] = 4
dict_marker_size['implicit'] = 4
dict_marker_size['grid_search'] = 5
dict_marker_size['bayesian'] = 4
dict_marker_size['random'] = 5
dict_marker_size['lhs'] = 4

dataset_names = ["mnist", "rcv1_train"]
# methods = ["implicit_forward_cdls", "random"]
# methods = ["implicit_forward_cdls", "random", "bayesian"]
methods = ["implicit_forward", "random", "bayesian"]

plt.close('all')
fig_acc_val, axarr_acc_val = plt.subplots(
    1, len(dataset_names) + 1, sharex=False, sharey=False, figsize=[14, 4])

fig_acc_test, axarr_acc_test = plt.subplots(
    1, len(dataset_names) + 1, sharex=False, sharey=False, figsize=[14, 4])

# figure for cross entropy
fig_ce, axarr_ce = plt.subplots(
    1, len(dataset_names) + 1, sharex=False, sharey=False, figsize=[14, 4])


for idx, dataset_name in enumerate(dataset_names):
    plt.figure()
    for method in methods:
        df_data = pandas.read_pickle(
            "results/%s_%s.pkl" % (dataset_name, method))

        method = df_data['method'].to_numpy()[0]
        times = df_data['times'].to_numpy()[0]
        objs = df_data['objs'].to_numpy()[0]
        log_alphas = df_data['log_alphas'].to_numpy()[0]
        acc_vals = df_data['acc_vals'].to_numpy()[0]
        acc_tests = df_data['acc_tests'].to_numpy()[0]

        axarr_acc_val.flat[idx].plot(
            times, acc_vals, label=dict_method[method],
            marker=dict_markers[method],
            color=dict_color[method])
        axarr_acc_test.flat[idx].plot(
            times, acc_tests, label=dict_method[method],
            marker=dict_markers[method],
            color=dict_color[method])
        axarr_ce.flat[idx].plot(
            times, objs, label=dict_method[method], color=dict_color[method],
            marker=dict_markers[method])

fig_acc_val.legend()
fig_acc_test.legend()
fig_ce.legend()


axarr_acc_val.flat[0].set_ylabel("Accuracy validation set", fontsize=fontsize)
axarr_acc_test.flat[0].set_ylabel("Accuracy test set", fontsize=fontsize)
axarr_ce.flat[0].set_ylabel("Multiclass cross entropy", fontsize=fontsize)


fig_acc_val.show()
fig_acc_test.show()
fig_ce.show()



# axarr_obj.flat[0].set_ylabel("Loss on test set", fontsize=fontsize)
# axarr_obj.flat[0].set_xlabel("Time (s)", fontsize=fontsize)
# axarr_obj.flat[1].set_xlabel("Time (s)", fontsize=fontsize)
# axarr_obj.flat[2].set_xlabel("Time (s)", fontsize=fontsize)

# axarr_alpha.flat[0].set_ylabel("Loss on validation set", fontsize=fontsize)
# axarr_alpha.flat[0].set_xlabel(
#     r"$\lambda - \lambda_{\max}$", fontsize=fontsize)
# axarr_alpha.flat[1].set_xlabel(
#     r"$\lambda - \lambda_{\max}$", fontsize=fontsize)
# axarr_alpha.flat[2].set_xlabel(
#     r"$\lambda - \lambda_{\max}$", fontsize=fontsize)

# fig_subopt.tight_layout()
# if save_fig:
#     fig_subopt.savefig(
#         fig_dir + "pred_log_reg_validation_set.pdf",
#         bbox_inches="tight")
#     fig_subopt.savefig(
#         fig_dir_svg + "pred_log_reg_validation_set.svg",
#         bbox_inches="tight")
# fig_subopt.show()

# fig_obj.tight_layout()
# if save_fig:
#     fig_obj.savefig(
#         fig_dir + "pred_log_reg_test_set.pdf",
#         bbox_inches="tight")
#     fig_obj.savefig(
#         fig_dir_svg + "pred_log_reg_test_set.svg",
#         bbox_inches="tight")
# fig_obj.show()

# fig_alpha.tight_layout()
# if save_fig:
#     fig_alpha.savefig(
#         fig_dir + "pred_vs_alpha_log_reg_validation_set.pdf",
#         bbox_inches="tight")
#     fig_alpha.savefig(
#         fig_dir_svg + "pred_vs_alpha_log_reg_validation_set.svg",
#         bbox_inches="tight")
# fig_alpha.show()
