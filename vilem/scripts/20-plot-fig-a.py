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

data = h5py.File('computed/results/all_0.7_1.h5', 'r')

_, axs = plt.subplots(nrows=2, ncols=1, figsize=(4, 6))

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
    color=COLORS[3],
)
axs[0].plot(
    list(data[f"sum_logprob_first_{TYPE}"][10:]),
    label=f"LogprobSum ({np.average(data[f'sum_logprob_first_{TYPE}'][10:]):.4f})",
    color=COLORS[4],
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
    color=COLORS[3],
)
axs[1].plot(
    list(data[f"sum_logprob_first_{TYPE}"][10:]),
    label=f"LogprobSum ({np.average(data[f'sum_logprob_first_{TYPE}'][10:]):.4f})",
    color=COLORS[4],
)

axs[0].set_ylabel("Selected candidate score")
axs[1].set_ylabel("% Selected candidate is top")
axs[0].set_xlabel("Number of CometKiwi runs")
axs[1].set_xlabel("Number of CometKiwi runs")

axs[0].set_ylim(0.79, None)

for ax in axs.flatten():
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(
        np.arange(10, 250, 50)-10,
        np.arange(10, 250, 50),
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
plt.savefig("figures/results_baselines.pdf")
plt.show()