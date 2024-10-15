import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# set default line width
mpl.rcParams["lines.linewidth"] = 2

COLORS = [
    "#b93f43", # red
    "#548f5c", # green
    "#7f99cc", # light blue
    "#5c3f81", # purple
    "#96792b", # yellow
]

data_1 = h5py.File('computed/results-base/all_0.7_1.h5', 'r')
data_2 = h5py.File('computed/results-base/all_0.7_2.h5', 'r')
data_5 = h5py.File('computed/results-base/all_0.7_5.h5', 'r')
data_10 = h5py.File('computed/results-base/all_0.7_10.h5', 'r')
TYPE = "score" 

plt.figure(figsize=(4, 2.8))
plt.gca().set_clip_on(False)

for data_i, (data, bs) in enumerate([(data_2, 2), (data_5, 5), (data_10, 10)][::-1]):
    # skip the first point at n=10 because that's just initialization
    data_x = [i for i,x in enumerate(data[f'bayesopt_{TYPE}']) if x >= 0.1 and i > 10]
    data_y_bs1 = [x for i,x in enumerate(data_1[f'bayesopt_{TYPE}']) if i in data_x]
    data_y = [x for i,x in enumerate(data[f'bayesopt_{TYPE}']) if i in data_x]
    #  ({np.average(data_y):.4f})
    plt.plot(
        data_x,
        np.array(data_y)-np.array(data_y_bs1),
        label=f"BayesOpt+GP\nbatch size {bs}",
        color=COLORS[0],
        alpha=1-data_i/3,
        linewidth=3,
    )

plt.xlim(-10, 200)
plt.hlines(0, *plt.xlim(), color="black", linewidth=0.7)

plt.gca().spines[["top", "right", "bottom"]].set_visible(False)
plt.ylabel("$\Delta$ in selected candidate score\nagainst batch size 1")
plt.xlabel("Max number of CometKiwi runs")

plt.xticks(
    list(np.arange(10, 200, 50))+[200],
    list(np.arange(10, 200, 50))+[200],
)
plt.legend(
    edgecolor="white",
    fancybox=False,
    framealpha=0,
    handletextpad=0.3,
    labelspacing=0.9,
)

plt.tight_layout()
plt.savefig("figures/batch_size.pdf")
plt.show()