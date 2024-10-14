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

def load_data(candidates_path, comet_calls_path, data_path):

    # load data
    with h5py.File(candidates_path, 'r') as h5_file:
        texts = h5_file["text"][:]

    with open(comet_calls_path) as cc:
        comet_calls = json.load(cc)

    data_lines = open(data_path).readlines()
    sources = [json.loads(data_line)["src"] for data_line in data_lines]
    return texts, comet_calls, sources

def prepare_comet_input(comet_calls, sources, texts):
    init_cands = []

    assert len(comet_calls["init"]) == len(sources)
    for instance, candidates in comet_calls["init"].items():
        src = sources[int(instance)]
        init_cands += [{"src": src, "mt": texts[int(instance), cand].decode()} for cand in candidates]

    later_cands  = defaultdict(list)
    for instance, candidates in comet_calls["calls"].items():
        src = sources[int(instance)]
        for cand_idx, cand_batch in enumerate(candidates):
            for c in cand_batch:
                later_cands[cand_idx].append({"src": src, "mt": texts[int(instance), c].decode()} )

    proxy_cands = []
    if "proxy" in comet_calls.keys():
        assert len(comet_calls["proxy"]) == len(sources)
        for instance, candidates in comet_calls["proxy"].items():
            src = sources[int(instance)]
            proxy_cands += [{"src": src, "mt": texts[int(instance), cand].decode()} for cand in candidates]


    return init_cands, later_cands, proxy_cands


def set_logging(need_proxy, comet_path):
    # Set up logging to log to a file and the console
    logger = logging.getLogger()  # Use root logger
    logger.setLevel(logging.INFO)
    breakpoint()
    if need_proxy:
        file_handler = logging.FileHandler(f'log_runtime_comet_calls_proxy.log', mode='a')
    else:
        file_handler = logging.FileHandler(f'log_runtime_comet_calls.log', mode='a')
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



    # get data
    texts, comet_calls, sources = load_data(candidates_path=args.candidates_path,
                                            comet_calls_path=args.comet_calls_path,
                                            data_path=args.data_path)

    # prepare comet input
    init_cands, later_cands, proxy_cands = prepare_comet_input(comet_calls, sources, texts)

    need_proxy = len(proxy_cands) > 0
    if args.comet_path and not need_proxy:
        raise ValueError
    
    if need_proxy and not args.comet_path:
        raise ValueError

    logger = set_logging(need_proxy, args.comet_path)

    # Global start time
    global_start_time = time.time()


    # Loading Comet models
    comet_loading_start = time.time()
    if args.comet_repo:
        model_path = comet.download_model(args.comet_repo)
        model = comet.load_from_checkpoint(model_path).eval()
    if args.comet_path and need_proxy:
        proxy_model = comet.load_from_checkpoint(args.comet_path)
    else:
        raise ValueError("Must provide --comet_repo or --comet_path.")
    comet_loading_time = time.time() - comet_loading_start

       
    # calculate comet for initial candidates
    init_comet_time = time.time()
    scores = model.predict(samples=init_cands, batch_size=args.max_batch_size, gpus=GPUS, num_workers=NUM_WORKERS).scores
    init_comet_time_all = time.time() - init_comet_time

    if need_proxy:
        proxy_comet_time = time.time()
        scores = proxy_model.predict(samples=proxy_cands, batch_size=args.max_batch_size, gpus=GPUS, num_workers=NUM_WORKERS).scores
        proxy_comet_time_all = time.time() - proxy_comet_time

    # calculate comet for everything else
    iteration_time = time.time()
    timings = {}
    for iter_num, comet_inputs in later_cands.items():
        scores = model.predict(samples=comet_inputs, batch_size=args.max_batch_size, gpus=GPUS, num_workers=NUM_WORKERS).scores
        timings[(iter_num+1)*10] = time.time() - iteration_time
        # only are interested in the first 100 (10 with initialisation and 90 here)
        if iter_num == 8:
            break


    logger.info("=== SUMMARY OF TIMES ===")
    logger.info(f"Total COMET loading time: {comet_loading_time:.2f} seconds")
    if need_proxy:
        logger.info(f"Proxy COMET scoring for inital cands {proxy_comet_time_all:.2f} seconds.")
    logger.info(f"COMET scoring for inital cands {init_comet_time_all:.2f} seconds.")
    for iter_num, elapsed_time in timings.items():
        logger.info(f"COMET scoring for {iter_num} more candidates took {elapsed_time:.2f} seconds.")
    logger.info(f"Entire process completed in {time.time() - global_start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path", help="Jsonl file with instances.", default="efficient_reranking/runtime_scripts/runtime_sample.jsonl")

    parser.add_argument(
        "--comet_calls_path", help="Json file with comet calls.") #, default="efficient_reranking/runtime_scripts/test/comet_calls_batch10.json")
    
    parser.add_argument(
        "--candidates_path", help="h5 file with candidates", default="efficient_reranking/runtime_scripts/test/candidates.h5")
    
    parser.add_argument(
        "--max_batch_size", type=int, help="Max batch size used during sampling.", default=200)

    parser.add_argument(
        "--comet_repo", help="Huggingface COMET model name. Must pass --comet_repo or --comet_path", default="Unbabel/wmt22-cometkiwi-da")

    parser.add_argument(
        "--comet_path", help="COMET model directory. Must pass --comet_repo or --comet_path")


    args = parser.parse_args()
    main(args)

# python efficient_reranking/runtime_scripts/get_runtime_comet_calls.py --comet_calls_path  "efficient_reranking/runtime_scripts/test/comet_calls_batch10.json" 


# python efficient_reranking/runtime_scripts/get_runtime_comet_calls.py --comet_calls_path efficient_reranking/runtime_scripts/test/comet_calls/comet_calls_batch10_multi_fid_50_S.json --comet_path models/skintle-M/model/skintle-M-v20.ckpt
# python efficient_reranking/runtime_scripts/get_runtime_comet_calls.py --comet_calls_path efficient_reranking/runtime_scripts/test/comet_calls/comet_calls_batch10_multi_fid_50_M.json --comet_path models/skintle-L/model/skintle-L-v10.ckpt

# python efficient_reranking/runtime_scripts/get_runtime_comet_calls.py --comet_calls_path efficient_reranking/runtime_scripts/test/comet_calls/comet_calls_batch10_multi_fid_200_S.json --comet_path models/skintle-M/model/skintle-M-v20.ckpt
# python efficient_reranking/runtime_scripts/get_runtime_comet_calls.py --comet_calls_path efficient_reranking/runtime_scripts/test/comet_calls/comet_calls_batch10_multi_fid_200_M.json --comet_path models/skintle-L/model/skintle-L-v10.ckp