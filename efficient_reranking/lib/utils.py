import h5py
import numpy as np

H5_STRING_DTYPE = h5py.special_dtype(vlen=str)
H5_VLEN_FLOAT_DTYPE = h5py.vlen_dtype(np.dtype('float32'))

DATA_FILENAME_BASE = "data"
CANDIDATES_H5DS_NAME = "candidates"
# Name of COMET model should be appended to this
COMET_SCORES_H5DS_NAME_BASE = "scores_comet_"