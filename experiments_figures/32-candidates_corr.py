import h5py
import argparse
import numpy as np
import scipy.stats

args = argparse.ArgumentParser()
args.add_argument("--model", default="skintle", choices=["skintle", "riem"])
args = args.parse_args()

data_k = h5py.File('data/julius_dev/scores_comet_wmt22-cometkiwi-da.h5', 'r')["scores"]

for size in "SML":
	data = h5py.File(f'data/julius_dev/scores_comet_{args.model}-{size}.h5', 'r')["scores"]
	avg_tau = np.nanmean([
		scipy.stats.kendalltau(x, y, variant='c')[0]
		for x, y in zip(data_k, data)
	])
	print(
		size,
		f"{avg_tau:.3f}",
	)