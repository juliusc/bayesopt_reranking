import datasets

# kken is in the WMT18 dataset, but the Hugging Face repo is missing the data, so
# we omit it.
WMT18_LANGUAGE_PAIRS = ["csen", "deen", "eten", "fien", "ruen", "tren", "zhen"]
WMT18_LANGUAGE_PAIRS += [lp[2:4] + lp[0:2] for lp in WMT18_LANGUAGE_PAIRS]


def load_dataset(src_lang, tgt_lang, split=None, subset=None):

    def dataset_map_fn(example):
        return {"src": example["translation"][src_lang],
                "tgt": example["translation"][tgt_lang]}

    if src_lang == "en":
        return datasets.load_dataset("wmt18", f'{tgt_lang}-{src_lang}', split=split).map(dataset_map_fn)
    else:
        return datasets.load_dataset("wmt18", f'{src_lang}-{tgt_lang}', split=split).map(dataset_map_fn)
