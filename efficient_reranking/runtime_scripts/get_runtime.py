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
    if args.calculate_kiwi_for_all: 
        file_handler = logging.FileHandler('log_runtime_comet_for_all.log', mode='a')
    else: 
        file_handler = logging.FileHandler('log_runtime_bandit.log', mode='a')
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

    # logger.info(f"Candidate generation completed in {candidate_generation_time:.2f} seconds.")

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

    # logger.info(f"COMET model loaded in {comet_loading_time:.2f} seconds.")

    # as a baseline: Just calculate it for all 
    if args.calculate_kiwi_for_all:
        total_kiwi_scoring_time = 0
        final_candidates = []
        comet_kiwi_all_start = time.time()
        for src, all_text, count in zip(all_sources, all_texts, all_counts):
            num_cands = (count > 0).sum()
            text = all_text[:num_cands]
            scores = np.empty(text.shape[0])
            for cand_idx, cand in enumerate(text):
                comet_input = {"src": src, "mt": cand}
                scores[cand_idx] = model.predict(samples=[comet_input], batch_size=1, gpus=GPUS, progress_bar=False, num_workers=NUM_WORKERS).scores[0]
            final_candidates.append(np.argmax(scores))
        kiwi_scoring_time = time.time() - comet_kiwi_all_start
        total_kiwi_scoring_time += kiwi_scoring_time
        logger.info("=== SUMMARY OF TIMES ===")
        logger.info(f"Total candidate generation time: {candidate_generation_time:.2f} seconds")
        logger.info(f"Total COMET loading time: {comet_loading_time:.2f} seconds")
        logger.info(f"COMET scoring for all candidates completed in {kiwi_scoring_time:.2f} seconds.")
        logger.info(f"Entire process completed in {time.time() - global_start_time:.2f} seconds.")
        return

    # Calculate similarities

    similarity_calc_start = time.time()
    sims_orig = {}
    for idx, emb in enumerate(tqdm(embeddings)):
        emb = F.normalize(torch.tensor(emb))
        sims_orig[idx] = torch.matmul(emb, emb.T).reshape(-1)
    similarity_calc_time = time.time() - similarity_calc_start

    # logger.info(f"Similarity calculations completed in {similarity_calc_time:.2f} seconds.")

    sims = []
    texts = []
    max_cands = all_counts.shape[1]
    for idx, data_idx in enumerate(tqdm(range(all_texts.shape[0]))):
        num_cands = (all_counts[data_idx] > 0).sum()
        texts.append(all_texts[data_idx][:num_cands])
        sims.append(sims_orig[data_idx].reshape(max_cands, max_cands)[:num_cands, :num_cands])
  
    # Bandit 
    np.random.seed(args.seed)
    all_sims = sims
    bandit_total = defaultdict(int)

    # logger.info("Starting bandit process.")

    # Time accumulators
    
    total_bandit_time = 0
    total_comet_initial_time = 0
    total_kernel_calc_time = 0
    total_comet_new_cand_time = 0
    total_posterior_calc_time = 0
    bandit_start = time.time()
    for src, text, sims, counts in tqdm(zip(all_sources, texts, all_sims, all_counts), total=len(all_sources)):

        # For sampling without replacement for the baseline
        candidate_idxs = []
        for i in range(counts.size):
            candidate_idxs.extend([i] * int(counts[i]))
        np.random.shuffle(candidate_idxs)

        all_idxs = np.arange(text.shape[0])
        scores = np.empty(text.shape[0])
        known_idxs = list(np.random.choice(text.shape[0], min(INITIAL_SIZE, all_idxs.shape[0]), replace=False))
        unknown_idxs = [x for x in all_idxs if x not in known_idxs]
        
        # Comet score for the initial known IDs
        comet_initial_scores_start = time.time()
        with torch.no_grad():
            for known_id in known_idxs:
                comet_input = {"src": src, "mt": text[known_id]}
                scores[known_id] = model.predict(samples=[comet_input], batch_size=1, gpus=GPUS, progress_bar=False, num_workers=NUM_WORKERS).scores[0]
        comet_initial_scores_time = time.time() - comet_initial_scores_start
        total_comet_initial_time += comet_initial_scores_time
        # logger.info(f"Initial COMET scores calculated in {comet_initial_scores_time:.2f} seconds.")

        # Kernel calculation
        kernel_calc_start = time.time()
        rbf_cov = np.exp(-(1 - sims.reshape(-1)) / (2 * args.bandwidth ** 2)).reshape(all_idxs.size, all_idxs.size)
        kernel_calc_time = time.time() - kernel_calc_start
        total_kernel_calc_time += kernel_calc_time
        # logger.info(f"RBF kernel calculated in {kernel_calc_time:.2f} seconds.")

        while len(known_idxs) < min(MAX_EVALS, all_idxs.shape[0]):

            bandit_total[len(known_idxs)] += scores[known_idxs].max()

            known_scores = scores[known_idxs]
            known_scores -= known_scores.mean()
            known_scores /= np.std(known_scores)

            unknown_unknown_cov = rbf_cov[unknown_idxs][:, unknown_idxs]
            known_unknown_cov = rbf_cov[known_idxs][:, unknown_idxs]
            known_known_cov = rbf_cov[known_idxs][:, known_idxs]

            posterior_calc_start = time.time()
            try:
                inverse_known_known_plus_prior = np.linalg.inv(known_known_cov)
                term_1 = np.matmul(inverse_known_known_plus_prior, known_unknown_cov)
                term_2 = np.matmul(known_unknown_cov.T, term_1)
                posterior_cov = unknown_unknown_cov - term_2
                mean_term_1 = np.matmul(inverse_known_known_plus_prior, known_scores)
                posterior_mean = np.matmul(known_unknown_cov.T, mean_term_1)
                posterior_var = posterior_cov.diagonal()
                best_score = known_scores.max()
                cdf = (norm.cdf(best_score, loc=posterior_mean, scale=posterior_var ** 0.5))
                best_unknown_idx_idx = (1 - cdf).argmax()
                known_idxs.append(unknown_idxs[best_unknown_idx_idx])
                iteration_time = time.time() - posterior_calc_start
                total_posterior_calc_time += iteration_time
                # logger.info(f"Posterior calculation completed in {time.time() - posterior_calc_start:.2f} seconds.")
                
                # Calculate comet score for the new candidate
                comet_new_cand_start = time.time()
                with torch.no_grad():
                    new_cand = unknown_idxs[best_unknown_idx_idx]                    
                    comet_input = {"src": src, "mt": text[new_cand]}
                    scores[new_cand] = model.predict(samples=[comet_input], batch_size=1, gpus=GPUS, progress_bar=False, num_workers=NUM_WORKERS).scores[0]
                comet_new_cand_time = time.time() - comet_new_cand_start
                total_comet_new_cand_time += comet_new_cand_time
                # logger.info(f"COMET score for new candidate calculated in {comet_new_cand_time:.2f} seconds.")
                del unknown_idxs[best_unknown_idx_idx]
            except Exception as e:
                logger.error("FAIL: Bandit process encountered an error.", exc_info=True)
                break

    bandit_time = time.time() - bandit_start
    total_bandit_time += bandit_time
    # logger.info(f"Bandit process completed in {bandit_time:.2f} seconds.")

    # Log final bandit results
    for k in bandit_total:
        if k % 10 == 0:
            logger.info(f"Bandit total at {k}: {bandit_total[k] / len(texts)}")

    logger.info("=== SUMMARY OF TIMES ===")
    logger.info(f"Total candidate generation time: {candidate_generation_time:.2f} seconds")
    logger.info(f"Total COMET loading time: {comet_loading_time:.2f} seconds")
    logger.info(f"Total similarity calculation time: {similarity_calc_time:.2f} seconds")
    logger.info(f"Total bandit time: {total_bandit_time:.2f} seconds")
    logger.info(f"Total COMET initial scoring time: {total_comet_initial_time:.2f} seconds")
    logger.info(f"Total kernel calculation time: {total_kernel_calc_time:.2f} seconds")
    logger.info(f"Total COMET scoring for new candidates time: {total_comet_new_cand_time:.2f} seconds")
    logger.info(f"Entire process completed in {time.time() - global_start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path", help="Jsonl file with instances.", default="efficient_reranking/runtime_scripts/shortest_sample.jsonl")


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

    parser.add_argument("--calculate_kiwi_for_all", action="store_true")

    args = parser.parse_args()
    main(args)