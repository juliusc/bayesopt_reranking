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

    file_handler = logging.FileHandler('log_runtime_baseline.log', mode='a')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Global start time
    global_start_time = time.time()

    # Start of candidate generation
    candidate_generation_start = time.time()
    all_sources, all_texts, embeddings, all_counts = generate_candidates(args.data_path, args.num_candidates, args.max_batch_size, args.epsilon)
    candidate_generation_time = time.time() - candidate_generation_start

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

    comet_kiwi_all_start = time.time()
    inputs = []
    for src, all_text, count in zip(all_sources, all_texts, all_counts):
        num_cands = (count > 0).sum()
        text = all_text[:num_cands]

        inputs += [{"src": src, "mt": cand} for cand in text]

    scores = model.predict(samples=inputs, batch_size=200, gpus=GPUS, num_workers=NUM_WORKERS).scores

    kiwi_scoring_time = time.time() - comet_kiwi_all_start

    logger.info("=== SUMMARY OF TIMES ===")
    logger.info(f"Total candidate generation time: {candidate_generation_time:.2f} seconds")
    logger.info(f"Total COMET loading time: {comet_loading_time:.2f} seconds")
    logger.info(f"COMET scoring for all candidates completed in {kiwi_scoring_time:.2f} seconds.")
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