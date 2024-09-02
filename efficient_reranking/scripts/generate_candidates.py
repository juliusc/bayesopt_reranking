# TODO: Add logging messages
import argparse
import logging
import sys

# Logging format borrowed from Fairseq.
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

from pathlib import Path

import h5py
import torch

from tqdm import tqdm
from transformers import GenerationConfig

from efficient_reranking.lib import datasets, generation, utils

MAX_GENERATION_LENGTH = 256
NUM_CANDIDATES = 256
# For candidate generation with beam search
CANDIDATE_GENERATION_CONFIG = GenerationConfig(
    max_length=MAX_GENERATION_LENGTH,
    num_beams=NUM_CANDIDATES,
    num_return_sequences=NUM_CANDIDATES,
    early_stopping=True
)


def main(args):
    torch.manual_seed(0)
    work_dir = Path(args.work_dir) / f"{args.src_lang}{args.tgt_lang}" / args.split
    work_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_dataset(args.src_lang, args.tgt_lang, args.split)
    if args.subset:
        dataset = dataset.select(range(0, len(dataset), len(dataset) // args.subset))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_h5file = h5py.File(work_dir / (utils.DATA_FILENAME_BASE + ".h5"), 'a')

    logging.info(f"Generating sequences.")

    if utils.CANDIDATES_H5DS_NAME in data_h5file:
        if args.overwrite:
            logging.info(f"Dataset {utils.CANDIDATES_H5DS_NAME} exists but overwriting.")
        else:
            logging.info(f"Dataset {utils.CANDIDATES_H5DS_NAME} exists, aborting. Use --overwrite to overwrite.")
            return
        candidates_h5ds = data_h5file[utils.CANDIDATES_H5DS_NAME]
    else:
        candidates_h5ds = data_h5file.create_dataset(
            utils.CANDIDATES_H5DS_NAME,
            (len(dataset), NUM_CANDIDATES),
            utils.H5_STRING_DTYPE)

    model, tokenizer = generation.load_model_and_tokenizer(args.src_lang, args.tgt_lang)
    model.to(device)

    for i in tqdm(range(len(dataset))):
        inputs = tokenizer(dataset[i]["src"], padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            result = model.generate(**inputs, generation_config=CANDIDATE_GENERATION_CONFIG)
        texts = tokenizer.batch_decode(result, skip_special_tokens=True)
        for j, text in enumerate(texts):
            candidates_h5ds[i, j] = text

    logging.info(f"Finished generating candidates for {len(dataset)} instances.")

    data_h5file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "src_lang",
        help="Source language pair. Supported languages are 'ende' and those supported by"
             " Facebook M2M100 (https://huggingface.co/facebook/m2m100_418M). 'ende',")

    parser.add_argument(
        "tgt_lang",
        help="Source language pair. Supported languages are 'ende' and those supported by"
             " Facebook M2M100 (https://huggingface.co/facebook/m2m100_418M). 'ende',")

    parser.add_argument(
        "split", type=str, help="Data split. Either 'validation' or 'test'.")

    parser.add_argument(
        "work_dir", help="Working directory for all steps. "
                         "Will be created if doesn't exist.")

    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing data.")

    parser.add_argument(
        "--subset", type=int, help="Only process the first n items.")

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for PyTorch.")

    args = parser.parse_args()
    main(args)
