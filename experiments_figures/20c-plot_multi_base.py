import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


COLORS = [
    "#b93f43", # red
    "#548f5c", # green
    "#7f99cc", # light blue
    "#5c3f81", # purple
    "#96792b", # yellow
]

# set default line width
mpl.rcParams["lines.linewidth"] = 2

data_S_50 = h5py.File('computed/results-multi/all_0.7_1_S_50.h5', 'r')
data_S_200 = h5py.File('computed/results-multi/all_0.7_1_S_200.h5', 'r')
data_M_50 = h5py.File('computed/results-multi/all_0.7_1_M_50.h5', 'r')
data_M_200 = h5py.File('computed/results-multi/all_0.7_1_M_200.h5', 'r')

data_bayeslogprob = h5py.File('computed/results-multi/all_0.7_1_avg_logprob_200.h5', 'r')["bayesopt_score"]
data_logprob = h5py.File('computed/results-base/all_0.7_1.h5', 'r')["avg_logprob_first_score"]
data_bayesopt = h5py.File('computed/results-base/all_0.7_1.h5', 'r')["bayesopt_score"]
data_random = h5py.File('computed/results-base/all_0.7_1.h5', 'r')["random_deduped_score"]


plt.figure(figsize=(4, 3.1))

TYPE = "score" 
# TODO: modify cost on x-axis
plt.plot(
    np.array(range(len(data_S_200[f"bayesopt_score"][10:]))),
    np.array(data_M_200[f"bayesopt_score"][10:]),
    label=f"BayesOpt+GP with 200 Distilled-M ({np.average(np.array(data_M_200[f'bayesopt_score'][10:])):.4f})",
    color=COLORS[4],
)
plt.plot(
    np.array(range(len(data_S_200[f"bayesopt_score"][10:]))),
    np.array(data_S_200[f"bayesopt_score"][10:]),
    label=f"BayesOpt+GP with 200 Distilled-S ({np.average(np.array(data_S_200[f'bayesopt_score'][10:])):.4f})",
    color=COLORS[3],
)
plt.plot(
    np.array(data_bayesopt[10:]),
    label=f"BayesOpt+GP ({np.average(data_bayesopt[10:]):.4f})",
    color=COLORS[0],
)
plt.plot(
    np.array(data_bayeslogprob[10:]),
    label=f"BayesOpt+GP with LogprobAvg ({np.average(data_bayeslogprob[10:]):.4f})",
    color=COLORS[1],
)
ax = plt.gca()
ax.set_ylabel("Selected candidate score")
ax.set_xlabel("Number of CometKiwi runs")
ax.set_ylim(0.804, 0.8221)
ax.set_xlim(None, 60)

ax.spines[["top", "right"]].set_visible(False)
plt.text(
    0.95, 0.0,
    "(zoomed-in axes)",
    horizontalalignment='right',
    verticalalignment='bottom',
    transform=ax.transAxes,
    fontsize=10,
    fontweight='bold',
)
ax.set_xticks(
    list(np.arange(10, 80, 20)-10),
    list(np.arange(10, 80, 20)),
)
ax.legend(
    edgecolor="white",
    fancybox=False,
    labelspacing=0.1,
    loc='upper center',
    bbox_to_anchor=(0.45, 1.37),
    framealpha=1,
)

plt.subplots_adjust(top=0.56)
plt.tight_layout(rect=(0, 0, 1, 1))
plt.savefig("figures/results_multi.pdf")
plt.show()