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

from efficient_reranking.lib import datasets, generation, utils, comet

MAX_GENERATION_LENGTH = 256


def main(args):
    torch.manual_seed(0)
    work_dir = Path(args.work_dir) / f"{args.src_lang}{args.tgt_lang}" / args.split
    work_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_dataset(args.src_lang, args.tgt_lang, args.split)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_h5file = h5py.File(work_dir / (utils.DATA_FILENAME_BASE + ".h5"), 'a')

    logging.info(f"Evaluting candidates with COMET model {args.comet_model}.")

    comet_h5ds_name = utils.COMET_SCORES_H5DS_NAME_BASE + args.comet_model
    candidates_h5ds = data_h5file[utils.CANDIDATES_H5DS_NAME]

    if comet_h5ds_name in data_h5file:
        if args.overwrite:
            logging.info(f"Dataset {comet_h5ds_name} exists but overwriting.")
        else:
            logging.info(f"Dataset {comet_h5ds_name} exists, aborting. Use --overwrite to overwrite.")
            return
        scores_h5ds = data_h5file[comet_h5ds_name]
    else:
        scores_h5ds = data_h5file.create_dataset(
            comet_h5ds_name,
            candidates_h5ds.shape,
            float)

    model = comet.load_model(args.comet_model).to(device)

    for i in tqdm(range(candidates_h5ds.shape[0])):
        src = dataset[i]["src"]
        tgts = [candidates_h5ds[i, j].decode() for j in range(candidates_h5ds.shape[1])]
        # inputs = model.encoder.prepare_sample([src] + tgts).to(device)
        data = [
            {"src": src, "mt": tgt}
            for tgt in tgts
        ]
        result = model.predict(samples=data)
        scores_h5ds[i] = result.scores

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
        "comet_model", help="Working directory for all steps. "
                         "Will be created if doesn't exist.")

    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing data.")

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for PyTorch.")

    args = parser.parse_args()
    main(args)
