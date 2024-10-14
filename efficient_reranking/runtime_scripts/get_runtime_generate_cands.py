import argparse
import numpy as np
from collections import defaultdict
from scipy.stats import norm 
import comet
import torch
import torch.nn.functional as F
from tqdm import tqdm
from generate_cands import generate_candidates
from generate_cands_no_hidden import generate_candidates as generate_candidates_no_hidden
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

def set_logging():
    # Set up logging to log to a file and the console
    logger = logging.getLogger()  # Use root logger
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler('log_runtime_generate_cands.log', mode='a')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def main(args):
    logger = set_logging()
    # Global start time
    global_start_time = time.time()

    # Start of candidate generation
    candidate_generation_start = time.time()
    all_sources, all_texts, embeddings, all_counts = generate_candidates(args.data_path, args.num_candidates, args.max_batch_size, args.epsilon)
    #all_sources, all_texts, all_counts = generate_candidates_no_hidden(args.data_path, args.num_candidates, args.max_batch_size, args.epsilon)
    candidate_generation_time = time.time() - candidate_generation_start

    logger.info("=== SUMMARY OF TIMES ===")
    logger.info(f"Total candidate generation time: {candidate_generation_time:.2f} seconds")
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

# efficient_reranking/runtime_scripts/get_runtime_baseline.py