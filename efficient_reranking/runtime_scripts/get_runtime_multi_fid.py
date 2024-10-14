import argparse
import json
import logging

import time

from pathlib import Path
from tqdm import tqdm

import numpy as np
import h5py
from scipy.stats import pearsonr, norm

import utils

DEV_CORR = {
    "avg_logprob": 0.30261028985941163,
    "S": 0.6722579350773578,
    "M": 0.7134984155295572
}


# Set up logging to log to a file and the console



def get_data_indices(data_dir, work_dir, split, lang_pair):
    # Only include instances which match the desired language pair and have candidates
    # (some candidates failed due to OOM).
    data_indices = []
    #data_path = Path(data_dir) / "jsonl" / f"{split}.jsonl"
    data_path = "efficient_reranking/runtime_scripts/runtime_sample.jsonl"
    with (open(data_path) as data_file,
          h5py.File(Path(work_dir) / split / (utils.CANDIDATES_FILENAME + ".h5")) as candidates_h5):
        candidates_h5ds = candidates_h5[utils.CANDIDATES_TEXT_H5DS_NAME]
        for i, data_line in enumerate(data_file):
            data = json.loads(data_line)
            if ((lang_pair == "all" or data["langs"] == lang_pair) and
                candidates_h5ds[i][0]):
                data_indices.append(i)

    return data_indices


def load_scores_and_similarities(
    data_path, work_dir, split, model_class_name):
    data_idxs = get_data_indices(
        args.data_dir, args.work_dir, split, args.lang_pair)

    split_work_dir = Path(args.work_dir) / split

    model_names = ([f"{args.model_class_name}-{size}" for size in ("M", "L")] +
                   ["wmt22-cometkiwi-da"])

    num_metrics = len(model_names)
    with (h5py.File((split_work_dir / utils.CANDIDATES_FILENAME).with_suffix(".h5")) as cand_h5,
          h5py.File((split_work_dir / (utils.SIMILARITIES_FILENAME_BASE + "_cosine")).with_suffix(".h5")) as sim_h5):
        counts_h5ds = cand_h5[utils.CANDIDATES_COUNTS_H5DS_NAME]
        sim_h5ds = sim_h5[utils.SIMILARITIES_H5DS_NAME]

        max_cands = counts_h5ds.shape[1]
        scores = np.zeros((len(data_idxs), max_cands, num_metrics))

        # Fetch all scores
        model_metric_idxs = np.arange(len(model_names))
        for metric_idx, model_name in zip(model_metric_idxs, model_names):
            h5_filename = split_work_dir / f"scores_comet_{model_name}.h5"
            with h5py.File(h5_filename) as scores_h5:
                scores_h5ds = scores_h5[utils.COMET_SCORES_H5DS_NAME]
                scores[:, :, metric_idx] = scores_h5ds[data_idxs]

        counts = counts_h5ds[data_idxs]

        # Break out the big score matrix into a list of scores per instance
        # due to varying numbers of candidates per instance
        instance_scores = []
        sims = []
        for idx, data_idx in enumerate(tqdm(data_idxs)):
            num_cands = (counts_h5ds[data_idx] > 0).sum()
            instance_scores.append(scores[idx, :num_cands])
            sims.append(sim_h5ds[data_idx].reshape(max_cands, max_cands)[:num_cands, :num_cands])

        return instance_scores, sims, counts


def normalize(arr):
    return (arr - arr.mean()) / arr.std()


def main(args):
    logger = logging.getLogger()  # Use root logger
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f'log_runtime_bandit_{args.num_proxy_evals}_{args.metric}.log', mode='a')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    np.random.seed(args.seed)

    start_load_time = time.time()

    all_scores, all_sims, all_counts = load_scores_and_similarities(
        args.data_dir, args.work_dir, args.split, args.model_class_name)

    logger.info(f"Loaded data, scores, and similarities in {time.time() - start_load_time:.2f} seconds.")


    INITIAL_SIZE = 10
    MAX_EVALS = 200

    # logging things
    comet_calls = {"init": {}, "calls":{}, "proxy":{}}
    total_loop_time = 0

    iter_times = {i: 0 for i in range(0, 20)}

    logger.info("Starting bandit process loop.")
    iteration_count = 0  # Track total iterations across the entire process

    assert len(all_scores) == len(all_sims) == len(all_counts)
    for instance_idx, (scores, sims, counts) in enumerate(tqdm(zip(all_scores, all_sims, all_counts))):
        
        loop_start_time = time.time()

        if args.metric == "S":
            m_scores_orig = scores[:, 0]
        elif args.metric == "M":
            m_scores_orig = scores[:, 1]
        else:
            raise ValueError(f"Unknown metric '{args.metric}'")

        candidate_idxs = []
        for i in range(counts.size):
            candidate_idxs.extend([i] * int(counts[i]))
        np.random.shuffle(candidate_idxs)
        random_deduped_idxs = list(dict.fromkeys(candidate_idxs))

        m_1_subset_idxs = random_deduped_idxs[:args.num_proxy_evals]

        m_star_scores = scores[:, -1]

        all_idxs = np.arange(m_star_scores.shape[0])

        m_1_sorted_idxs = list(list(zip(*sorted(zip(-m_scores_orig[m_1_subset_idxs], m_1_subset_idxs))))[1])

        known_idxs = m_1_sorted_idxs[:INITIAL_SIZE]
        unknown_idxs = [x for x in all_idxs if x not in known_idxs]

        m_1_used_idxs = list(m_1_sorted_idxs)

        comet_calls["init"][instance_idx] = [int(k) for k in known_idxs]
        comet_calls["proxy"][instance_idx] = [int(m) for m in m_1_used_idxs]
        comet_calls["calls"][instance_idx] = []

        rbf_cov = np.exp(-(1 - sims.reshape(-1)) / (2 * args.bandwidth ** 2)).reshape(all_idxs.size, all_idxs.size)

        known_scores_m_1 = normalize(m_scores_orig[m_1_used_idxs])

        iteration_count = 0  # Track how many iterations in the while loop
        while len(known_idxs) < min(MAX_EVALS, all_idxs.shape[0]):
            iteration_count += 1  # Increment global iteration count
            
            known_scores_m_star = normalize(m_star_scores[known_idxs])

            if args.use_dev_correlation:
                metrics_corr = DEV_CORR[args.metric]
            else:
                metrics_corr = pearsonr(normalize(m_scores_orig[known_idxs]), known_scores_m_star).statistic

            known_scores = np.concatenate([known_scores_m_1, known_scores_m_star])

            unknown_unknown_cov = rbf_cov[unknown_idxs][:, unknown_idxs]

            known_scores = np.concatenate([m_scores_orig[m_1_used_idxs], known_scores_m_star])

            known_unknown_cov_m_1 = rbf_cov[m_1_used_idxs][:, unknown_idxs] * metrics_corr
            known_unknown_cov_m_star = rbf_cov[known_idxs][:, unknown_idxs]
            known_unknown_cov = np.concatenate([known_unknown_cov_m_1, known_unknown_cov_m_star])

            known_known_cov_m_star = rbf_cov[known_idxs][:, known_idxs]
            known_known_cov_m_1_m_star = rbf_cov[known_idxs][:, m_1_used_idxs] * metrics_corr
            known_known_cov_m_1 = rbf_cov[m_1_used_idxs][:, m_1_used_idxs]
            known_known_cov = np.concatenate([np.concatenate([known_known_cov_m_1, known_known_cov_m_1_m_star]), np.concatenate([known_known_cov_m_1_m_star.T, known_known_cov_m_star])], axis=1)

            inverse_known_known_plus_prior = np.linalg.inv(known_known_cov)
            term_1 = np.matmul(inverse_known_known_plus_prior, known_unknown_cov)
            term_2 = np.matmul(known_unknown_cov.T, term_1)
            posterior_cov = unknown_unknown_cov - term_2
            mean_term_1 = np.matmul(inverse_known_known_plus_prior, known_scores)
            posterior_mean = np.matmul(known_unknown_cov.T, mean_term_1)
            posterior_var = posterior_cov.diagonal()

            best_score = known_scores.max()
            z = (best_score - posterior_mean) / (posterior_var ** 0.5)
            ei = (
                posterior_var ** 0.5 *
                (z * norm.cdf(z) + norm.pdf(z))
            )
            best_idxs = np.array(unknown_idxs)[np.argpartition(ei, min(args.batch_size, len(unknown_idxs)-1))[:args.batch_size]]

            known_idxs = known_idxs + list(best_idxs)
            unknown_idxs = [x for x in all_idxs if x not in known_idxs]

            comet_calls["calls"][instance_idx].append([int(x) for x in list(best_idxs)])


            iter_time = time.time() - loop_start_time  # Time spent in this group of 10 iterations
            iter_times[iteration_count] += iter_time  # Add to the corresponding entry in the dictionary

    logger.info(f"Total time spent in the bandit loop: {total_loop_time:.2f} seconds.")

    # Log cumulative times for each group of iterations (10, 20, 30, etc.)
    for key, value in iter_times.items():
        logger.info(f"Time taken for {key*10} candidates: {value:.4f} seconds.")

    with open(f"comet_calls_batch10_multi_fid_{args.num_proxy_evals}_{args.metric}.json", "w") as file:
        json.dump(comet_calls, file, indent=4)



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
        "batch_size", type=int, help="Bayesopt update batch size.")

    parser.add_argument(
        "bandwidth", type=float, help="RBF bandwidth parameter.")

    parser.add_argument(
        "metric", type=str, help="")

    parser.add_argument(
        "num_proxy_evals", type=int, help="")

    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Confidence threshold.")

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed.")

    parser.add_argument(
        "--use_dev_correlation", action="store_true", help="")


    args = parser.parse_args()
    main(args)

# python efficient_reranking/runtime_scripts/get_runtime_multi_fid.py blub efficient_reranking/runtime_scripts skintle all test 10 0.25 S 200
# python efficient_reranking/runtime_scripts/get_runtime_multi_fid.py blub efficient_reranking/runtime_scripts skintle all test 10 0.25 M 200

# python efficient_reranking/runtime_scripts/get_runtime_multi_fid.py blub efficient_reranking/runtime_scripts skintle all test 10 0.25 S 50
# python efficient_reranking/runtime_scripts/get_runtime_multi_fid.py blub efficient_reranking/runtime_scripts skintle all test 10 0.25 M 50