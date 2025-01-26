import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import argparse

args = argparse.ArgumentParser()
args.add_argument("proxy", default="S")
args = args.parse_args()

COLORS = [
    "#b93f43", # red
    "#548f5c", # green
    "#7f99cc", # light blue
    "#5c3f81", # purple
    "#96792b", # yellow
]

# set default line width
mpl.rcParams["lines.linewidth"] = 2

data_50 = h5py.File(f'computed/results-multi/all_0.7_1_{args.proxy}_50.h5', 'r')
data_200 = h5py.File(f'computed/results-multi/all_0.7_1_{args.proxy}_200.h5', 'r')

data_bayesopt = h5py.File('computed/results-base/all_0.7_1.h5', 'r')["bayesopt_score"]


plt.figure(figsize=(4, 3.2))

COLOR = COLORS[3] if args.proxy == "S" else COLORS[4]

TYPE = "score" 

# +200*0.76/5.89
plt.plot(
    np.array(range(len(data_200[f"bayesopt_score"][10:]))),
    np.array(data_200[f"bayesopt_score"][10:]),
    label=f"BayesOpt+GP with 200 Distilled-{args.proxy} ({np.average(np.array(data_200[f'bayesopt_score'][10:])):.4f})",
    color=COLOR,
    alpha=1,
)
plt.plot(
    np.array(range(len(data_200[f"proxy_first_score"][10:]))),
    np.array(data_200[f"proxy_first_score"][10:]),
    label=f"ProxyFirst 200 Distilled-{args.proxy} ({np.average(np.array(data_200[f'proxy_first_score'][10:])):.4f})",
    color=COLOR,
    linestyle="--",
)
plt.plot(
    np.array(range(len(data_50[f"proxy_first_score"][10:50]))),
    np.array(data_50[f"proxy_first_score"][10:50]),
    label=f"ProxyFirst 50 Distilled-{args.proxy} ({np.average(np.array(data_50[f'proxy_first_score'][10:])):.4f})",
    color=COLOR,
    linestyle="--",
    alpha=0.5,
)
plt.plot(
    np.array(range(len(data_50[f"bayesopt_score"][10:]))),
    np.array(data_50[f"bayesopt_score"][10:]),
    label=f"BayesOpt+GP with 50 Distilled-{args.proxy} ({np.average(np.array(data_50[f'bayesopt_score'][10:])):.4f})",
    color=COLOR,
    alpha=0.5,
)
plt.plot(
    np.array(data_bayesopt[10:]),
    label=f"BayesOpt+GP ({np.average(data_bayesopt[10:]):.4f})",
    color=COLORS[0],
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
    bbox_to_anchor=(0.45, 1.5),
    framealpha=1,
)
plt.subplots_adjust(top=0.56)
plt.tight_layout(rect=(0, 0, 1, 1))
plt.savefig(f"figures/results_multi_{args.proxy}.pdf")
plt.show()