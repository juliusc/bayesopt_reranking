import h5py
import matplotlib.pyplot as plt
import numpy as np

COLORS = [
    "#b93f43", # red
    "#548f5c", # green
    "#7f99cc", # light blue
    "#5c3f81", # purple
    "#b4ab30", # yellow
]

data = h5py.File('computed/results/all_0.7_5.h5', 'r')
print(list(data["bayesopt_score"]))

plt.figure(figsize=(4, 3))

plt.plot(
    list(data["bayesopt_score"][10:]),
    label=f"Bayes opt ({np.average(data['bayesopt_score'][10:]):.4f})",
    color=COLORS[0],
    linewidth=3,
)
plt.plot(
    list(data["avg_logprob_first_score"][10:]),
    label=f"Avg logprob ({np.average(data['avg_logprob_first_score'][10:]):.4f})",
    color=COLORS[1],
    linewidth=3,
)
plt.plot(
    list(data["random_score"][10:]),
    label=f"Random ({np.average(data['random_score'][10:]):.4f})",
    color=COLORS[2],
    linewidth=3,
)

plt.gca().spines[["top", "right"]].set_visible(False)

plt.ylabel("Selected candidate score")
plt.xlabel("Cost")
plt.tight_layout(pad=0.1)
plt.legend(edgecolor="white", fancybox=False)
plt.show()