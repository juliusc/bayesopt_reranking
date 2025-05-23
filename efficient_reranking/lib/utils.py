import logging
import sys
import h5py
import numpy as np

NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"

LANG_TO_NLLB = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "uk": "ukr_Cyrl",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
    "lt": "lit_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ps": "pbt_Arab",
    "fi": "fin_Latn",
    "zu": "zul_Latn",
    "xh": "xho_Latn",
    "ta": "tam_Taml",
    "pl": "pol_Latn",
    "iu": None,
    "km": "khm_Khmr",
    "gu": "guj_Gujr",
    "is": "isl_Latn",
    "ru": "rus_Cyrl",
    "kk": "kaz_Cyrl",
    "cs": "ces_Latn",
    "ha": "hau_Latn",
}


H5_STRING_DTYPE = h5py.special_dtype(vlen=str)
H5_VLEN_FLOAT_DTYPE = h5py.vlen_dtype(np.dtype('float32'))

CANDIDATES_FILENAME = "candidates"
CANDIDATES_TEXT_H5DS_NAME = "text"
CANDIDATES_TOKEN_LOGPROBS_H5DS_NAME = "token_logprobs"

# Name of COMET model should be appended to this
COMET_SCORES_FILENAME_BASE = "scores_comet_"
COMET_SCORES_H5DS_NAME = "scores"

EMBEDDINGS_FILENAME_BASE = "embeddings_"
EMBEDDINGS_H5DS_NAME = "embeddings"

SIMILARITIES_FILENAME_BASE = "similarities_"
SIMILARITIES_H5DS_NAME = "similarities"


def configure_logger(name, output_filename):
    file_handler = logging.FileHandler(output_filename, mode='w')
    stream_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        force=True,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[file_handler, stream_handler])