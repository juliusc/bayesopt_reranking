import h5py
import glob
import scipy.stats
import numpy as np

data_cometkiwi = np.array(
    h5py.File('computed/results-scores/scores_comet_wmt22-cometkiwi-da.h5', 'r')["scores"],
    dtype=np.float64,
).flatten()

for f in glob.glob('computed/results-scores/*.h5'):
    data = h5py.File(f, 'r')
    for k in data.keys():
        data_1 = np.array(data[k], dtype=np.float64).flatten()
        data_2 = data_cometkiwi

        data_2 = data_2[~np.isnan(data_1)]
        data_1 = data_1[~np.isnan(data_1)]
        print(f, k, scipy.stats.kendalltau(
            data_1, data_2,
            variant="c"
        )[0])

