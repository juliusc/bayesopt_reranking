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
    "#b4ab30", # yellow
]

# set default line width
mpl.rcParams["lines.linewidth"] = 2

data_50 = h5py.File(f'computed/results-multi/all_0.7_1_{args.proxy}_50.h5', 'r')
data_100 = h5py.File(f'computed/results-multi/all_0.7_1_{args.proxy}_100.h5', 'r')
data_200 = h5py.File(f'computed/results-multi/all_0.7_1_{args.proxy}_200.h5', 'r')

data_bayesopt = h5py.File('computed/results-base/all_0.7_1.h5', 'r')["bayesopt_score"]


plt.figure(figsize=(4, 4))

TYPE = "score" 
# TODO: modify cost on x-axis
plt.plot(
    np.array(data_bayesopt[10:]),
    label=f"BayesOpt+GP ({np.average(data_bayesopt[10:]):.4f})",
    color=COLORS[0],
)
plt.plot(
    np.array(range(len(data_200[f"bayesopt_score"][10:])))+200*0.76/5.89,
    np.array(data_200[f"bayesopt_score"][10:])/20,
    label=f"BayesOpt+GP with 200 Distilled-{args.proxy} ({np.average(np.array(data_200[f'bayesopt_score'][10:])/20):.4f})",
    color=COLORS[0],
    alpha=0.75,
)
plt.plot(
    np.array(range(len(data_100[f"bayesopt_score"][10:])))+100*0.76/5.89,
    np.array(data_100[f"bayesopt_score"][10:])/20,
    label=f"BayesOpt+GP with 100 Distilled-{args.proxy} ({np.average(np.array(data_100[f'bayesopt_score'][10:])/20):.4f})",
    color=COLORS[0],
    alpha=0.5,
)
plt.plot(
    np.array(range(len(data_50[f"bayesopt_score"][10:])))+50*0.76/5.89,
    np.array(data_50[f"bayesopt_score"][10:])/20,
    label=f"BayesOpt+GP with 50 Distilled-{args.proxy} ({np.average(np.array(data_50[f'bayesopt_score'][10:])/20):.4f})",
    color=COLORS[0],
    alpha=0.25,
)
plt.plot(
    np.array(range(len(data_100[f"proxy_first_score"][10:])))+50*0.76/5.89,
    np.array(data_100[f"proxy_first_score"][10:])/20,
    label=f"ProxyFirst 100 Distilled-{args.proxy} ({np.average(np.array(data_100[f'proxy_first_score'][10:])/20):.4f})",
    color=COLORS[4],
)

ax = plt.gca()
ax.set_ylabel("Selected candidate score")
ax.set_xlabel("Number of CometKiwi runs")
ax.set_ylim(0.79, None)

ax.spines[["top", "right"]].set_visible(False)
ax.set_xticks(
    list(np.arange(10, 200, 50)-10)+[200],
    list(np.arange(10, 200, 50))+[200],
)
ax.legend(
    edgecolor="white",
    fancybox=False,
    # handlelength=1,
    labelspacing=0.3,
    loc='upper center',
    bbox_to_anchor=(0.45, 1.5),
    framealpha=0,
)
plt.subplots_adjust(top=0.6)
plt.tight_layout(rect=(0, 0, 1, 1))
plt.savefig(f"figures/results_multi_{args.proxy}.pdf")
plt.show()