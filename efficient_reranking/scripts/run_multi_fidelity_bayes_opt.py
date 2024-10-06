import argparse
from collections import defaultdict
import json
import logging
import sys

import pandas as pd

from pathlib import Path
from tqdm import tqdm

# Logging format borrowed from Fairseq.
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

import numpy as np
import h5py
from scipy.stats import pearsonr, norm, kendalltau
from pprint import pprint

from efficient_reranking.lib import utils


def conditional_mean_and_covar(known_values, mean, covar):
    num_known_columns = known_values.shape[1]
    # y refers to the conditioned variables, x the condition variables
    x_mean = mean[:num_known_columns]
    y_mean = mean[num_known_columns:]
    xx_covar = covar[:num_known_columns,:num_known_columns]
    yx_covar = covar[num_known_columns:,:num_known_columns]
    xy_covar = yx_covar.T
    yy_covar = covar[num_known_columns:,num_known_columns:]
    y_given_x_mean = np.expand_dims(y_mean, 1) + np.dot(np.dot(yx_covar, np.linalg.inv(xx_covar)), (known_values - x_mean).T)
    # return y_given_x_mean.T
    y_given_x_covar = yy_covar - np.dot(yx_covar, np.dot(np.linalg.inv(xx_covar), xy_covar))
    return y_given_x_mean.T, y_given_x_covar


def get_data_indices(data_dir, work_dir, split, lang_pair):
    # Only include instances which match the desired language pair and have candidates
    # (some candidates failed due to OOM).
    data_indices = []
    data_path = Path(data_dir) / "jsonl" / f"{split}.jsonl"
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

    model_names = ([f"{args.model_class_name}-{size}" for size in ("S", "M", "L")] +
                   ["wmt22-cometkiwi-da"])

    num_metrics = len(model_names)
    with (h5py.File((split_work_dir / utils.CANDIDATES_FILENAME).with_suffix(".h5")) as cand_h5,
          h5py.File((split_work_dir / utils.LOGPROBS_FILENAME_BASE).with_suffix(".h5")) as logprobs_h5,
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

        logprobs = logprobs_h5[utils.SUM_LOGPROBS_H5DS_NAME][data_idxs]
        counts = counts_h5ds[data_idxs]

        # Break out the big score matrix into a list of scores per instance
        # due to varying numbers of candidates per instance
        instance_scores = []
        sims = []
        for idx, data_idx in enumerate(tqdm(data_idxs)):
            num_cands = (counts_h5ds[data_idx] > 0).sum()
            instance_scores.append(scores[idx, :num_cands])
            sims.append(sim_h5ds[data_idx].reshape(max_cands, max_cands)[:num_cands, :num_cands])

        return instance_scores, sims, counts, logprobs


def main(args):
    np.random.seed(args.seed)
    work_dir = Path(args.work_dir)

    all_scores, all_sims, all_counts, all_logprobs = load_scores_and_similarities(
        args.data_dir, args.work_dir, args.split, args.model_class_name)

    INITIAL_SIZE = 10
    MAX_EVALS = 200

    baseline_max_total = 0
    baseline_random_total = defaultdict(int)
    bandit_total = defaultdict(int)

    # corrs = []
    # for scores in all_scores:
    #     corrs.append(pearsonr(scores[:, PROXY_METRIC_IDX], scores[:, -1]).statistic)
    #     # scores = (scores - scores.mean(axis=0)) / scores.std(axis=0)
    #     # corrs.append(np.cov(scores[:, [2, -1]].T)[0, 1])
    # metrics_corr = np.mean(corrs)

    for scores, sims, counts, logprobs in tqdm(zip(all_scores, all_sims, all_counts, all_logprobs)):
        m_scores = scores[:, args.metric_idx]
        m_scores -= m_scores.mean()
        m_scores /= m_scores.std()
        m_star_scores = scores[:, -1]

        candidate_idxs = []
        for i in range(counts.size):
            candidate_idxs.extend([i] * int(counts[i]))
        np.random.shuffle(candidate_idxs)

        baseline_max_total += m_star_scores.max()

        all_idxs = np.arange(m_star_scores.shape[0])
        # known_idxs = list(np.random.choice(m_star_scores.shape[0], min(INITIAL_SIZE, all_idxs.shape[0]), replace=False))
        known_idxs = list(list(zip(*sorted(zip(-m_scores, all_idxs))))[1][:INITIAL_SIZE])
        unknown_idxs = [x for x in all_idxs if x not in known_idxs]

        rbf_cov = np.exp(-(1 - sims.reshape(-1)) / (2 * args.bandwidth ** 2)).reshape(all_idxs.size, all_idxs.size)

        while len(known_idxs) < min(MAX_EVALS, all_idxs.shape[0]):
            baseline_random_total[len(known_idxs)] += m_star_scores[candidate_idxs[:len(known_idxs)]].max()
            bandit_total[len(known_idxs)] += m_star_scores[known_idxs].max()
            # print(len(known_idxs), m_star_scores[known_idxs].max())

            known_scores_m_1 = m_scores
            known_scores_m_1 -= known_scores_m_1.mean()
            known_scores_m_1 /= np.std(known_scores_m_1)

            known_scores_m_star = m_star_scores[known_idxs]
            known_scores_m_star -= known_scores_m_star.mean()
            known_scores_m_star /= np.std(known_scores_m_star)

            metrics_corr = pearsonr(known_scores_m_1[known_idxs], known_scores_m_star).statistic

            known_scores = np.concatenate([known_scores_m_1, known_scores_m_star])

            unknown_unknown_cov = rbf_cov[unknown_idxs][:, unknown_idxs]

            # known_unknown_cov_m_1 = rbf_cov[:, unknown_idxs] * metrics_corr
            # known_unknown_cov_m_star = rbf_cov[known_idxs][:, unknown_idxs]
            # known_unknown_cov = np.concatenate([known_unknown_cov_m_1, known_unknown_cov_m_star])

            # known_known_cov_m_star = rbf_cov[known_idxs][:, known_idxs]
            # known_known_cov_m_1_m_star = rbf_cov[known_idxs] * metrics_corr
            # known_known_cov_m_1 = rbf_cov
            # known_known_cov = np.concatenate([np.concatenate([known_known_cov_m_1, known_known_cov_m_1_m_star]), np.concatenate([known_known_cov_m_1_m_star.T, known_known_cov_m_star])], axis=1)

            known_scores = np.concatenate([known_scores_m_1[unknown_idxs], known_scores_m_star])

            known_unknown_cov_m_1 = rbf_cov[unknown_idxs][:, unknown_idxs] * metrics_corr
            known_unknown_cov_m_star = rbf_cov[known_idxs][:, unknown_idxs]
            known_unknown_cov = np.concatenate([known_unknown_cov_m_1, known_unknown_cov_m_star])

            known_known_cov_m_star = rbf_cov[known_idxs][:, known_idxs]
            known_known_cov_m_1_m_star = rbf_cov[known_idxs][:, unknown_idxs] * metrics_corr
            known_known_cov_m_1 = rbf_cov[unknown_idxs][:, unknown_idxs]
            known_known_cov = np.concatenate([np.concatenate([known_known_cov_m_1, known_known_cov_m_1_m_star]), np.concatenate([known_known_cov_m_1_m_star.T, known_known_cov_m_star])], axis=1)

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
            del unknown_idxs[best_unknown_idx_idx]

        for total_cands in range(len(known_idxs), MAX_EVALS + 1):
            baseline_random_total[total_cands] += m_star_scores[candidate_idxs[:total_cands]].max()
            bandit_total[total_cands] += m_star_scores[known_idxs].max()
            # print(total_cands, m_star_scores[known_idxs].max())

    print(baseline_max_total / len(all_scores))
    for k in sorted(bandit_total):
        if k % 10 == 0:
            print(k, bandit_total[k] / len(all_scores), baseline_random_total[k] / len(all_scores))

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
        "metric_idx", type=int, help="")

    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Confidence threshold.")

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed.")

    args = parser.parse_args()
    main(args)
