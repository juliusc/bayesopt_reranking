import tqdm
import json
from transformers import GenerationConfig
from transformers import pipeline
import argparse
import copy

args = argparse.ArgumentParser()
args.add_argument("-i", type=int, default=0)
args = args.parse_args()

BATCH_SIZE = 1
NUM_CANDIDATES = 16

data = [json.loads(x) for x in open("data/jsonl/train.jsonl", "r")][args.i * 100_000:(args.i+1) * 100_000]
print("Will process", len(data), "lines")

# For candidate generation with beam search
CANDIDATE_GENERATION_CONFIG = GenerationConfig(
    max_length=256,
    num_beams=NUM_CANDIDATES,
    num_return_sequences=NUM_CANDIDATES,
    early_stopping=True,
    batch_size=BATCH_SIZE,
)

model = pipeline("translation", model="facebook/nllb-200-distilled-600M", device="cuda:0")

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

data_out = []

for line_i, line in enumerate(tqdm.tqdm(data, miniters=100)):
    lang1, lang2 = line["langs"].split("-")
    if lang1 not in LANG_TO_NLLB or lang2 not in LANG_TO_NLLB:
        continue
    lang1 = LANG_TO_NLLB[lang1]
    lang2 = LANG_TO_NLLB[lang2]
    if lang1 is None or lang2 is None:
        continue

    out = model(line["src"], src_lang=lang1, tgt_lang=lang2, generation_config=CANDIDATE_GENERATION_CONFIG)
    for tgt in [x["translation_text"] for x in out]:
        line_copy = copy.deepcopy(line)
        line_copy["tgt"] = tgt
        line_copy["score"] = None
        data_out.append(line_copy)

    if line_i % 1000 == 0 or line_i == len(data)-1:
        with open(f"data/jsonl/train_nllb_{args.i}.jsonl", "w") as f:
            for line_out in data_out:
                f.write(json.dumps(line_out, ensure_ascii=False) + "\n")

