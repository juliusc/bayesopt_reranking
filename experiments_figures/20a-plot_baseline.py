import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

import matplotlib.ticker as mtick

COLORS = [
    "#b93f43", # red
    "#548f5c", # green
    "#7f99cc", # light blue
    "#5c3f81", # purple
    "#96792b", # yellow
]

# set default line width
mpl.rcParams["lines.linewidth"] = 2

data = h5py.File('computed/results-base/all_0.7_1.h5', 'r')

_, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 2.8))

TYPE = "score" 
axs[0].plot(
    list(data[f"bayesopt_{TYPE}"][10:]),
    label=f"BayesOpt+GP ({np.average(data[f'bayesopt_{TYPE}'][10:]):.4f})",
    color=COLORS[0],
)
axs[0].plot(
    list(data[f"hill_climbing_{TYPE}"][10:]),
    label=f"HillClimbing ({np.average(data[f'hill_climbing_{TYPE}'][10:]):.4f})",
    color=COLORS[2],
)
axs[0].plot(
    list(data[f"avg_logprob_first_{TYPE}"][10:]),
    label=f"LogprobAvg ({np.average(data[f'avg_logprob_first_{TYPE}'][10:]):.4f})",
    color=COLORS[1],
)
axs[0].plot(
    list(data[f"random_deduped_{TYPE}"][10:]),
    label=f"UniqRandom ({np.average(data[f'random_deduped_{TYPE}'][10:]):.4f})",
    color="black",
)
axs[0].plot(
    list(data[f"sum_logprob_first_{TYPE}"][10:]),
    label=f"LogprobSum ({np.average(data[f'sum_logprob_first_{TYPE}'][10:]):.4f})",
    color=COLORS[1],
    linestyle="--",
)

TYPE = "best_retrieved" 
axs[1].plot(
    list(data[f"bayesopt_{TYPE}"][10:]),
    label=f"BayesOpt+GP ({np.average(data[f'bayesopt_{TYPE}'][10:]):.4f})",
    color=COLORS[0],
)
axs[1].plot(
    list(data[f"avg_logprob_first_{TYPE}"][10:]),
    label=f"LogprobAvg ({np.average(data[f'avg_logprob_first_{TYPE}'][10:]):.4f})",
    color=COLORS[1],
)
axs[1].plot(
    list(data[f"hill_climbing_{TYPE}"][10:]),
    label=f"HillClimbing ({np.average(data[f'hill_climbing_{TYPE}'][10:]):.4f})",
    color=COLORS[2],
)
axs[1].plot(
    list(data[f"random_deduped_{TYPE}"][10:]),
    label=f"UniqRandom ({np.average(data[f'random_deduped_{TYPE}'][10:]):.4f})",
    color="black",
)
axs[1].plot(
    list(data[f"sum_logprob_first_{TYPE}"][10:]),
    label=f"LogprobSum ({np.average(data[f'sum_logprob_first_{TYPE}'][10:]):.4f})",
    color=COLORS[1],
    linestyle="--",
)

axs[0].set_ylabel("Selected candidate score")
axs[1].set_ylabel("% Selected candidate is top")
axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
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
        labelspacing=0.3,
        handletextpad=0.1,
        loc="lower right",
        bbox_to_anchor=(1.05, 0),
        framealpha=0,
    )

plt.tight_layout()
plt.savefig("figures/results_baselines.pdf")
plt.show()