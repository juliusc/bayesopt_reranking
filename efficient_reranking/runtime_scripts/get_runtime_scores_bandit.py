import argparse
import numpy as np
from collections import defaultdict
from scipy.stats import norm 
import comet
import torch
import torch.nn.functional as F
from tqdm import tqdm
from generate_cands import generate_candidates
import os
import logging
import time
import h5py
import json

# Constants
INITIAL_SIZE = 10
MAX_EVALS = 200
GPUS = 1
NUM_WORKERS = 1
CPU_THREADS = 1

# Environment settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(CPU_THREADS)  # Limit CPU threads
torch.set_num_interop_threads(1)  # Limit inter-operation threads

def main(args):

    # Set up logging to log to a file and the console
    logger = logging.getLogger()  # Use root logger
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler('log_comet_calls.log', mode='a')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    file_path = "efficient_reranking/runtime_scripts/test/candidates.h5"
    # load data
    with h5py.File(file_path, 'r') as h5_file:
        texts = h5_file["text"][:]

    # load comet indices 
    with open("efficient_reranking/runtime_scripts/test/comet_calls.json") as cc:
        comet_calls = json.load(cc)

    # load sources 
    data_path = "efficient_reranking/runtime_scripts/runtime_sample.jsonl"
    data_lines = open(data_path).readlines()
    sources = [json.loads(data_line)["src"] for data_line in data_lines]

    init_cands = []
    assert len(comet_calls["init"]) == len(sources)
    for instance, candidates in comet_calls["init"].items():

        src = sources[int(instance)]
        
        init_cands += [{"src": src, "mt": texts[int(instance), cand].decode()} for cand in candidates]


    later_cands  = defaultdict(list)
    for instance, candidates in comet_calls["calls"].items():
        src = sources[int(instance)]
        for cand_idx, cand in enumerate(candidates):
            later_cands[cand_idx].append({"src": src, "mt": texts[int(instance), cand].decode()} )

    # Global start time
    global_start_time = time.time()


    # Loading Comet models
    comet_loading_start = time.time()
    if args.comet_repo:
        model_path = comet.download_model(args.comet_repo)
        model = comet.load_from_checkpoint(model_path).eval()
    elif args.comet_path:
        model = comet.load_from_checkpoint(args.comet_path)
    else:
        raise ValueError("Must provide --comet_repo or --comet_path.")
    comet_loading_time = time.time() - comet_loading_start

       
    # init cands 
    init_comet_time = time.time()

    scores = model.predict(samples=init_cands, batch_size=200, gpus=GPUS, num_workers=NUM_WORKERS).scores
    init_comet_time_all = time.time() - init_comet_time


    later_time = time.time()
    timings = {}
    for iter_num, comet_inputs in later_cands.items():

        scores = model.predict(samples=comet_inputs, batch_size=200, gpus=GPUS, num_workers=NUM_WORKERS).scores
        if (iter_num +1) % 10 == 0:
            end_time = time.time()
            elapsed_time = end_time - later_time
            timings[iter_num] = elapsed_time
        if iter_num == 101:
            break

    logger.info("=== SUMMARY OF TIMES ===")

    logger.info(f"Total COMET loading time: {comet_loading_time:.2f} seconds")
    logger.info(f"COMET scoring for initalisation {init_comet_time_all:.2f} seconds.")
    for iter_num, elapsed_time in timings.items():
        logger.info(f"COMET scoring for the first {iter_num} iterations took {elapsed_time:.2f} seconds.")

    logger.info(f"Entire process completed in {time.time() - global_start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path", help="Jsonl file with instances.", default="efficient_reranking/runtime_scripts/runtime_sample.jsonl")


    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed.")

    parser.add_argument(
        "--num_candidates", type=int, default=200, help="Number of candidates to generate.")

    parser.add_argument(
        "--epsilon", type=float, default=0.05, help="Threshold for epsilon sampling.")

    parser.add_argument(
        "--max_batch_size", type=int, help="Max batch size used during sampling.", default=200)

    parser.add_argument(
        "--comet_repo", help="Huggingface COMET model name. Must pass --comet_repo or --comet_path", default="Unbabel/wmt22-cometkiwi-da")

    parser.add_argument(
        "--comet_path", help="COMET model directory. Must pass --comet_repo or --comet_path")
    parser.add_argument(
        "--bandwidth", type=float, help="RBF bandwidth parameter.", default=0.2)


    args = parser.parse_args()
    main(args)

# python efficient_reranking/runtime_scripts/get_runtime_scores_bandit.py