import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


COLORS = [
    "#b93f43", # red
    "#548f5c", # green
    "#7f99cc", # light blue
    "#5c3f81", # purple
    "#b4ab30", # yellow
]

# set default line width
mpl.rcParams["lines.linewidth"] = 2

data_S_50 = h5py.File('computed/results-multi/all_0.7_1_S_50.h5', 'r')
data_S_100 = h5py.File('computed/results-multi/all_0.7_1_S_100.h5', 'r')
data_S_200 = h5py.File('computed/results-multi/all_0.7_1_S_200.h5', 'r')
data_M_50 = h5py.File('computed/results-multi/all_0.7_1_M_50.h5', 'r')
data_M_100 = h5py.File('computed/results-multi/all_0.7_1_M_100.h5', 'r')
data_M_200 = h5py.File('computed/results-multi/all_0.7_1_M_200.h5', 'r')

data_logprob = h5py.File('computed/results-base/all_0.7_1.h5', 'r')["avg_logprob_first_score"]


_, axs = plt.subplots(nrows=2, ncols=1, figsize=(4, 6))

TYPE = "score" 
axs[0].plot(
    np.array(data_S_50[f"bayesopt_score"][10:]),
    label=f"BayesOpt+GP\nwith Distilled-S ({np.average(data_S_50[f'bayesopt_score'][10:]):.4f})",
    color=COLORS[0],
)
axs[0].plot(
    np.array(data_S_50[f"proxy_first_score"][10:]),
    label=f"Distilled-S ({np.average(data_S_50[f'proxy_first_score'][10:]):.4f})",
    color=COLORS[0],
    linestyle="--",
)
axs[0].plot(
    np.array(data_logprob[10:]),
    label=f"LogprobAvg ({np.average(data_logprob[10:]):.4f})",
    color=COLORS[1],
    linestyle="--",
)
axs[0].plot(
    np.array(data_M_50[f"bayesopt_score"][10:]),
    label=f"BayesOpt+GP\nwith Distilled-M ({np.average(data_M_50[f'bayesopt_score'][10:]):.4f})",
    color=COLORS[2],
)
axs[0].plot(
    np.array(data_M_50[f"proxy_first_score"][10:]),
    label=f"Distilled-M ({np.average(data_M_50[f'proxy_first_score'][10:]):.4f})",
    color=COLORS[2],
    linestyle="--",
)
# axs[0].plot(
#     list(data[f"avg_logprob_first_{TYPE}"][10:]),
#     label=f"LogprobAvg ({np.average(data[f'avg_logprob_first_{TYPE}'][10:]):.4f})",
#     color=COLORS[1],
# )
# axs[0].plot(
#     list(data[f"random_deduped_{TYPE}"][10:]),
#     label=f"UniqRandom ({np.average(data[f'random_deduped_{TYPE}'][10:]):.4f})",
#     color=COLORS[3],
# )
# axs[0].plot(
#     list(data[f"sum_logprob_first_{TYPE}"][10:]),
#     label=f"LogprobSum ({np.average(data[f'sum_logprob_first_{TYPE}'][10:]):.4f})",
#     color=COLORS[4],
# )

# axs[1].plot(
#     list(data[f"bayesopt_{TYPE}"][10:]),
#     label=f"BayesOpt+GP ({np.average(data[f'bayesopt_{TYPE}'][10:]):.4f})",
#     color=COLORS[0],
# )
# axs[1].plot(
#     list(data[f"avg_logprob_first_{TYPE}"][10:]),
#     label=f"LogprobAvg ({np.average(data[f'avg_logprob_first_{TYPE}'][10:]):.4f})",
#     color=COLORS[1],
# )
# axs[1].plot(
#     list(data[f"hill_climbing_{TYPE}"][10:]),
#     label=f"HillClimbing ({np.average(data[f'hill_climbing_{TYPE}'][10:]):.4f})",
#     color=COLORS[2],
# )
# axs[1].plot(
#     list(data[f"random_deduped_{TYPE}"][10:]),
#     label=f"UniqRandom ({np.average(data[f'random_deduped_{TYPE}'][10:]):.4f})",
#     color=COLORS[3],
# )
# axs[1].plot(
#     list(data[f"sum_logprob_first_{TYPE}"][10:]),
#     label=f"LogprobSum ({np.average(data[f'sum_logprob_first_{TYPE}'][10:]):.4f})",
#     color=COLORS[4],
# )

axs[0].set_ylabel("Selected candidate score")
axs[1].set_ylabel("% Selected candidate is top")
axs[0].set_xlabel("Number of CometKiwi runs")
axs[1].set_xlabel("Number of CometKiwi runs")

axs[0].set_ylim(0.79, None)

for ax in axs.flatten():
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(
        list(np.arange(10, 200, 50)-10)+[200],
        list(np.arange(10, 200, 50))+[200],
    )
    ax.legend(
        edgecolor="white",
        fancybox=False,
        handlelength=1,
        labelspacing=0.3,
        loc="lower right",
        bbox_to_anchor=(1.05, 0),
        framealpha=0,
    )

plt.tight_layout()
plt.savefig("figures/results_multi.pdf")
plt.show()