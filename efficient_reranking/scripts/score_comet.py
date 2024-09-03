import argparse
import logging
import os
import sys

# Logging format borrowed from Fairseq.
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

from pathlib import Path

import comet
import h5py
import torch

from tqdm import tqdm
from transformers import GenerationConfig

from efficient_reranking.lib import datasets, generation, utils

MAX_GENERATION_LENGTH = 256


class MissingArgumentError(ValueError):
    pass

def main(args):
    torch.manual_seed(0)
    work_dir = Path(args.work_dir) / f"{args.src_lang}{args.tgt_lang}" / args.split
    work_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_dataset(args.src_lang, args.tgt_lang, args.split)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_h5file = h5py.File(work_dir / (utils.DATA_FILENAME_BASE + ".h5"), 'a')

    logging.info(f"Evaluating candidates with COMET.")

    if args.comet_repo:
        comet_base_name = args.comet_repo.split("/")[-1]
        model_path = comet.download_model(args.comet_model)
        model = comet.load_from_checkpoint(model_path).eval()
    elif args.comet_path:
        comet_base_name = os.path.splitext(args.comet_path.split("/")[-1])[0]
        model = comet.load_from_checkpoint(args.comet_path)
    else:
        raise MissingArgumentError("Must provide --comet_repo or --comet_path.")

    comet_h5ds_name = utils.COMET_SCORES_H5DS_NAME_BASE + comet_base_name
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
        "--comet_repo", help="Huggingface COMET model name. Must pass --comet_repo or --comet_path")

    parser.add_argument(
        "--comet_path", help="COMET model directory. Must pass --comet_repo or --comet_path")

    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing data.")

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for PyTorch.")

    args = parser.parse_args()
    main(args)
