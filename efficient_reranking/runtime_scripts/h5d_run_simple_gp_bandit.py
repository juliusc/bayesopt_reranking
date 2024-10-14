import argparse
from collections import defaultdict
import json
import logging
import sys
import time
from pathlib import Path
from tqdm import tqdm


import numpy as np
import h5py
from scipy.stats import norm 
import utils

# Set up logging to log to a file and the console
logger = logging.getLogger()  # Use root logger
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('log_runtime_bandit.log', mode='a')
file_handler.setLevel(logging.INFO)
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
# console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# logger.addHandler(console_handler)

def get_data_indices(data_dir, work_dir, split, lang_pair):
    logger.info("Loading data indices.")
    start_time = time.time()
    
    data_indices = []
    data_path = "efficient_reranking/runtime_scripts/runtime_sample.jsonl"
    with (open(data_path) as data_file,
          h5py.File(Path(work_dir) / split / (utils.CANDIDATES_FILENAME + ".h5")) as candidates_h5):
        candidates_h5ds = candidates_h5[utils.CANDIDATES_TEXT_H5DS_NAME]
        for i, data_line in enumerate(data_file):
            data = json.loads(data_line)
            if ((lang_pair == "all" or data["langs"] == lang_pair) and candidates_h5ds[i][0]):
                data_indices.append(i)

    logger.info(f"Data indices loaded in {time.time() - start_time:.2f} seconds.")
    return data_indices


def load_scores_and_similarities(data_path, work_dir, split, model_class_name):
    logger.info("Loading scores and similarities.")
    start_time = time.time()
    
    data_idxs = get_data_indices(args.data_dir, args.work_dir, split, args.lang_pair)
    split_work_dir = Path(args.work_dir) / split

    model_names = ["wmt22-cometkiwi-da"]
    num_metrics = len(model_names)

    with (h5py.File((split_work_dir / utils.CANDIDATES_FILENAME).with_suffix(".h5")) as cand_h5,
          h5py.File((split_work_dir / (utils.SIMILARITIES_FILENAME_BASE + "_cosine")).with_suffix(".h5")) as sim_h5):
        counts_h5ds = cand_h5[utils.CANDIDATES_COUNTS_H5DS_NAME]
        sim_h5ds = sim_h5[utils.SIMILARITIES_H5DS_NAME]

        max_cands = counts_h5ds.shape[1]
        scores = np.zeros((len(data_idxs), max_cands, num_metrics))

        model_metric_idxs = np.arange(len(model_names))
        for metric_idx, model_name in zip(model_metric_idxs, model_names):
            h5_filename = split_work_dir / f"scores_comet_{model_name}.h5"
            with h5py.File(h5_filename) as scores_h5:
                scores_h5ds = scores_h5[utils.COMET_SCORES_H5DS_NAME]
                scores[:, :, metric_idx] = scores_h5ds[data_idxs]

        counts = counts_h5ds[data_idxs]

        instance_scores = []
        sims = []
        for idx, data_idx in enumerate(tqdm(data_idxs)):
            num_cands = (counts_h5ds[data_idx] > 0).sum()
            instance_scores.append(scores[idx, :num_cands])
            sims.append(sim_h5ds[data_idx].reshape(max_cands, max_cands)[:num_cands, :num_cands])

    logger.info(f"Scores and similarities loaded in {time.time() - start_time:.2f} seconds.")
    return instance_scores, sims, counts, None


def main(args):
    np.random.seed(args.seed)

    # Time loading data and similarities
    start_load_time = time.time()
    all_scores, all_sims, all_counts, all_logprobs = load_scores_and_similarities(
        args.data_dir, args.work_dir, args.split, args.model_class_name)
    logger.info(f"Loaded data, scores, and similarities in {time.time() - start_load_time:.2f} seconds.")

    INITIAL_SIZE = 10
    MAX_EVALS = 200

    baseline_max_total = 0
    bandit_total = defaultdict(int)

    comet_calls = {"init": {}, "calls":{}}

    total_loop_time = 0
    total_posterior_calc_time = 0

    iter_times = {i: 0 for i in range(10, MAX_EVALS + 1, 10)}

    logger.info("Starting bandit process loop.")
    iteration_count = 0  # Track total iterations across the entire process

    for idx, (scores, sims, counts) in enumerate(tqdm(zip(all_scores, all_sims, all_counts))):
        loop_start_time = time.time()

        scores = scores[:, -1]
        candidate_idxs = []
        for i in range(counts.size):
            candidate_idxs.extend([i] * int(counts[i]))
        np.random.shuffle(candidate_idxs)

        baseline_max_total += scores.max()

        all_idxs = np.arange(scores.shape[0])
        known_idxs = list(np.random.choice(scores.shape[0], min(INITIAL_SIZE, all_idxs.shape[0]), replace=False))
        unknown_idxs = [x for x in all_idxs if x not in known_idxs]

        comet_calls["init"][idx] = [int(k) for k in known_idxs]
        comet_calls["calls"][idx] = []

        rbf_cov = np.exp(-(1 - sims.reshape(-1)) / (2 * args.bandwidth ** 2)).reshape(all_idxs.size, all_idxs.size)

        iteration_count = 0  # Track how many iterations in the while loop

        while len(known_idxs) < min(MAX_EVALS, all_idxs.shape[0]):
            iteration_count += 1  # Increment global iteration count

            bandit_total[len(known_idxs)] += scores[known_idxs].max()

            known_scores = scores[known_idxs]
            known_scores -= known_scores.mean()
            known_scores /= np.std(known_scores)

            unknown_unknown_cov = rbf_cov[unknown_idxs][:, unknown_idxs]
            known_unknown_cov = rbf_cov[known_idxs][:, unknown_idxs]
            known_known_cov = rbf_cov[known_idxs][:, known_idxs]

            posterior_start_time = time.time()
            try:
                inverse_known_known_plus_prior = np.linalg.inv(known_known_cov)
                term_1 = np.matmul(inverse_known_known_plus_prior, known_unknown_cov)
                term_2 = np.matmul(known_unknown_cov.T, term_1)
                posterior_cov = unknown_unknown_cov - term_2
                mean_term_1 = np.matmul(inverse_known_known_plus_prior, known_scores)
                posterior_mean = np.matmul(known_unknown_cov.T, mean_term_1)
                posterior_var = posterior_cov.diagonal()
                best_score = known_scores.max()
                cdf = norm.cdf(best_score, loc=posterior_mean, scale=posterior_var ** 0.5)
                best_unknown_idx_idx = (1 - cdf).argmax()
                known_idxs.append(unknown_idxs[best_unknown_idx_idx])

                comet_calls["calls"][idx].append(int(unknown_idxs[best_unknown_idx_idx]))
                del unknown_idxs[best_unknown_idx_idx]

                posterior_calc_time = time.time() - posterior_start_time
                total_posterior_calc_time += posterior_calc_time
            except Exception as e:
                logger.error("Error during posterior calculation.", exc_info=True)
                break

            # Log cumulative time every 10 iterations
            if iteration_count % 10 == 0:
                iter_time = time.time() - loop_start_time  # Time spent in this group of 10 iterations
                iter_times[iteration_count] += iter_time  # Add to the corresponding entry in the dictionary


        
        loop_time = time.time() - loop_start_time
        total_loop_time += loop_time

        for total_cands in range(len(known_idxs), MAX_EVALS + 1):
            bandit_total[total_cands] += scores[known_idxs].max()

    logger.info(f"Total time spent in the bandit loop: {total_loop_time:.2f} seconds.")
    logger.info(f"Total time spent on posterior calculations: {total_posterior_calc_time:.2f} seconds.")
    # Log cumulative times for each group of iterations (10, 20, 30, etc.)
    for key, value in iter_times.items():
        logger.info(f"Time taken for {key} iterations: {value:.4f} seconds.")

    with open("comet_calls.json", "w") as file:
        json.dump(comet_calls, file, indent=4)

    logger.info(f"Baseline max total: {baseline_max_total / len(all_scores)}")
    for k in bandit_total:
        if k % 10 == 0:
            logger.info(f"{k}: {bandit_total[k] / len(all_scores)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_dir", help="Data directory generated by the pipeline from vilem/scripts.")

    parser.add_argument(
        "work_dir", help="Working directory for all steps. "
                         "Will be created if doesn't exist.")

    parser.add_argument(
        "model_class_name", help="Name of the model class, e.g. 'quern', 'skintle'.")

    parser.add_argument(
        "lang_pair", help="E.g. 'en-cs'. 'all' for all language pairs.")

    parser.add_argument(
        "split", help="Data split.")

    parser.add_argument(
        "bandwidth", type=float, help="RBF bandwidth parameter.")

    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Confidence threshold.")

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed.")

    args = parser.parse_args()
    main(args)
