import h5py
import numpy as np
import matplotlib.pyplot as plt
import copy
import argparse
import scipy.stats
import tqdm

args = argparse.ArgumentParser()
args.add_argument("--model", default="skintle", choices=["skintle", "riem"])
args = args.parse_args()


METRIC_COSTS = {
	"s": 0.7/1000,
	"m": 1.3/1000,
	"l": 2.3/1000,
	"k": 5.7/1000,
}
COST_MAX = METRIC_COSTS["k"]*128
COST_MIN = METRIC_COSTS["s"]*128
MAX_PERF = 0.811477314817054
MIN_PERF = 0.7795202688872814

data_k = h5py.File('data/julius_dev/scores_comet_wmt22-cometkiwi-da.h5', 'r')["scores"]

# simulate lower tier QE
data_l = data_k + np.random.normal(0, 0.075, data_k.shape)
data_m = data_k + np.random.normal(0, 0.105, data_k.shape)
data_s = data_k + np.random.normal(0, 0.170, data_k.shape)

MAX_AUC = (COST_MAX-COST_MIN) * MAX_PERF
MIN_AUC = (COST_MAX-COST_MIN) * MIN_PERF

def auc(zipped):
	# sort by cost
	zipped = sorted(zipped, key=lambda x: x[0])
	area = np.trapz([y for x, y in zipped], [x for x, y in zipped])
	# return proportion to maximum attainable
	return (area-MIN_AUC)/(MAX_AUC-MIN_AUC)


def trim_curve(zipped):
	# sort by cost
	zipped = sorted(zipped, key=lambda x: x[0])

	i_under = [i for i, (x, y) in enumerate(zipped) if x <= COST_MAX][-1]

	if i_under == len(zipped)-1:
		# need to add max cost with same performance
		zipped = zipped + [(COST_MAX, zipped[-1][1])]
	else:
		# extrapolate
		last_cost, last_perf = zipped[i_under]
		next_cost, next_perf = zipped[i_under+1]
		cap_perf = last_perf + (next_perf-last_perf)/(next_cost-last_cost)*(COST_MAX-last_cost)

		# crop to last
		zipped = zipped[:i_under+1]
		zipped = zipped + [(COST_MAX, cap_perf)]

	return zipped



def dynamic_baseline(top_p, METRIC_DATA):
	cands_all = [np.arange(len(row)) for row in data_s]
	cost_all = 0

	for k in METRIC_DATA.keys():
		data = copy.deepcopy(np.array(METRIC_DATA[k]))
		cost = METRIC_COSTS[k]

		cands_new_all = []
		for cands, row in zip(cands_all, data):

			# mask non-candidates so they're never selected
			mask = np.ones(len(row), bool)
			mask[cands] = 0
			row[mask] = -np.inf

			# take top-k
			top_k = int(top_p * len(cands))

			# if we have just one candidate, we don't need to do anything
			if top_k > 1:
				cost_all += cost * len(cands)
				cands_new = np.argsort(row)[-top_k:]
				cands_new_all.append(cands_new)
			else:
				cost_all += 0
				cands_new_all.append(cands)
		cands_all = cands_new_all
	
	return (
		cost_all/len(data_s),
		# take the top
		np.average([row[cand[-1]] for row, cand in zip(data_k, cands_all)]),
	)


# start plotting + computation
plt.figure(figsize=(4, 2.5))
for noise_i, noise in enumerate(tqdm.tqdm(np.linspace(0, 0.9, 5)[::-1])):
	METRIC_DATA = {
		"s": data_s + np.random.normal(0, noise, data_k.shape),
		"m": data_m + np.random.normal(0, noise, data_k.shape),
		"l": data_l + np.random.normal(0, noise, data_k.shape),
		"k": data_k,
	}

	BASELINE_DYNAMIC = [
		dynamic_baseline(top_p, METRIC_DATA)
		for top_p in [0.05, 0.1, 0.2, 0.3, 0.37, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	]

	signature = [
		scipy.stats.kendalltau(data_k, data_local, variant="c").statistic
		for data_key, data_local in METRIC_DATA.items() if data_key != "k"
	]
	signature = "|".join([
		f"{x:.2f}" for x in signature
	])

	BASELINE_DYNAMIC_TRIM = trim_curve(BASELINE_DYNAMIC)
	plt.plot(
		[x[0] for x in BASELINE_DYNAMIC_TRIM],
		[x[1] for x in BASELINE_DYNAMIC_TRIM],
		marker=".",
		color="black",
		alpha=(noise_i+1)/(5+1),
		label=f"{signature}: {auc(BASELINE_DYNAMIC_TRIM):.3f}",
	)


handles, labels = plt.gca().get_legend_handles_labels()
# invert legend
plt.legend(
	handles[::-1], labels[::-1],
	fancybox=False, edgecolor="black"
)
plt.ylabel("Average final $m_0$ score")
plt.xlabel("Average generation cost (s)")

plt.ylim(0.76, 0.815)

plt.tight_layout(pad=0)
plt.savefig(f"figures/results_baselines_simulation.pdf")
plt.show()