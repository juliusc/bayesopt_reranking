import argparse
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


def get_data_indices(data_dir, work_dir, split, model_class_name, lang_pair):
    # Only include instances which match the desired language pair and have candidates
    # (some candidates failed due to OOM).
    data_indices = []
    candidates = []
    data_path = Path(data_dir) / "jsonl" / f"{split}.jsonl"
    with (open(data_path) as data_file,
          h5py.File(Path(work_dir) / split / (utils.CANDIDATES_FILENAME + ".h5")) as candidates_h5):
        candidates_h5ds = candidates_h5[utils.CANDIDATES_TEXT_H5DS_NAME]
        for i, data_line in enumerate(data_file):
            line_data = json.loads(data_line)
            if ((lang_pair == "all" or line_data["langs"] == lang_pair) and
                candidates_h5ds[i][0]):
                data_indices.append(i)
                candidates.append([cand.decode() for cand in candidates_h5ds[i]])

    return data_indices, candidates


def main(args):
    np.random.seed(args.seed)
    work_dir = Path(args.work_dir)

    # (julius) Haven't run the test set yet so just splitting dev set into dev/test.
    logging.info("Fetching scores")

    split = "dev"
    data_indices, candidates = get_data_indices(
        args.data_dir, args.work_dir, split, args.model_class_name, args.lang_pair)

    score_names = []
    scores = []

    # keep_idxs = []
    # with h5py.File((Path(args.work_dir) / split / utils.CANDIDATES_FILENAME).with_suffix(".h5")) as h5_file:
    #     if args.avg_logprob or args.sum_logprob:
    #         token_logprobs = h5_file[utils.CANDIDATES_TOKEN_LOGPROBS_H5DS_NAME][:]
    #     candidates_h5ds = h5_file[utils.CANDIDATES_TEXT_H5DS_NAME]
    #     for instance_idx in range(candidates_h5ds.shape[0]):
    #         if candidates_h5ds[instance_idx][0]:
    #             keep_idxs.append(instance_idx)

    model_names = ([f"{args.model_class_name}-{size}" for size in ("S", "M", "L")] +
                   ["wmt22-cometkiwi-da"])
    for model_name in model_names:
        score_names.append(model_name)
        split_work_dir = Path(work_dir) / split
        h5_filename = split_work_dir / f"scores_comet_{model_name}.h5"
        with h5py.File(h5_filename) as h5_file:
            scores.append(h5_file[utils.COMET_SCORES_H5DS_NAME][data_indices])

    # Shape of (# of scores, # of instances, # of candidates per instance)
    original_scores = np.stack(scores)

    scores = original_scores.copy()
    if args.zero_mean:
        scores -= scores.mean(-1, keepdims=True)

    # Temporary dev/test split from dev set
    idxs = list(range(original_scores.shape[1]))
    np.random.shuffle(idxs)
    split_idx = int(len(idxs) * 0.8)

    train_scores = scores[:, idxs[:split_idx], :]
    test_original_scores = original_scores[:, idxs[split_idx:], :]
    test_scores = scores[:, idxs[split_idx:], :]

    RBF_COV = 2
    RBF_SHAPE = 0.05
    INITIAL_SIZE = 1
    MAX_EVALS = 64

    bandit_total = 0
    baseline_total = 0

    sim_h5 = h5py.File(split_work_dir / "similarity_skintle-S.h5")
    sim_h5ds = sim_h5[utils.SIMILARITIES_H5DS_NAME]
    emb_h5 = h5py.File(split_work_dir / "embed_skintle-S.h5")
    emb_h5ds = emb_h5["embeddings"]

    for i, original_idx in tqdm(zip(range(test_scores.shape[1]), idxs[split_idx:])):
        instance_cands = candidates[original_idx]
        # unique_cand_idxs = []
        # seen_cands = set()
        # for j, cand in enumerate(instance_cands):
        #     if cand not in seen_cands:
        #         seen_cands.add(cand)
        #         unique_cand_idxs.append(j)
        scores = test_original_scores[-1, i, :]
        sims = sim_h5ds[i]
        embs = emb_h5ds[i]

        unique_cand_idxs = []
        seen_emb_sums = set()
        for j, emb in enumerate(embs):
            if emb.sum() not in seen_emb_sums:
                seen_emb_sums.add(emb.sum())
                unique_cand_idxs.append(j)

        # print("UNIQUE", len(unique_cand_idxs))

        scores = scores[unique_cand_idxs]
        sims = sims[unique_cand_idxs][:, unique_cand_idxs]
        embs = emb[unique_cand_idxs]

        sims += np.random.normal(loc=0,scale=0.00001,size=sims.shape)

        # baseline_total += scores.max()
        baseline_total += np.random.choice(scores, size=min(MAX_EVALS, scores.shape[0]), replace=False).max()

        # score_diffs = np.zeros_like(sims)
        # for j in range(test_scores.shape[2]):
        #     for k in range(test_scores.shape[2]):
        #         score_diffs[j, k] = np.abs(scores[j] - scores[k])

        # print(pearsonr(sims.reshape(-1), score_diffs.reshape(-1)))
        # import pdb; pdb.set_trace()
        # continue

        # import sacrebleu
        # for j in range(test_scores.shape[2]):
        #     for k in range(test_scores.shape[2]):
        #         sims[j, k] = sacrebleu.sentence_chrf(instance_cands[j], [instance_cands[k]]).score / 100
        # import pdb; pdb.set_trace()

        all_idxs = np.arange(scores.shape[0])
        known_idxs = list(np.random.choice(scores.shape[0], min(INITIAL_SIZE, all_idxs.shape[0]), replace=False))
        unknown_idxs = [x for x in all_idxs if x not in known_idxs]
        # rbf_cov = np.zeros((unknown_idxs.size, unknown_idxs.size))
        prior_var = np.var(scores[known_idxs])

        rbf_cov = np.zeros((all_idxs.size, all_idxs.size))

        # known_unknown_cov = np.zeros((known_idxs.size, unknown_idxs.size))
        # known_known_cov = np.zeros((known_idxs.size, known_idxs.size))
        for j in all_idxs:
            for k in all_idxs:
                cov = np.exp(-(1 - sims[k, j]) / (2 * RBF_SHAPE ** 2))
                rbf_cov[j, k] = cov

        while len(known_idxs) < min(MAX_EVALS, all_idxs.shape[0]):
            known_scores = scores[known_idxs]
            known_scores -= known_scores.mean()
            known_scores /= np.std(known_scores)
            unknown_unknown_cov = rbf_cov[unknown_idxs][:, unknown_idxs]
            known_unknown_cov = rbf_cov[known_idxs][:, unknown_idxs]
            known_known_cov = rbf_cov[known_idxs][:, known_idxs]
            # prior_cov = np.identity(known_idxs.size) * prior_var
            # prior_cov = np.identity(known_idxs.size)
            prior_cov = 0
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
                del unknown_idxs[best_unknown_idx_idx]
            except Exception as e:
                print("FAIL")
                break
        bandit_total += scores[known_idxs].max()
    sim_h5.close()

    print(baseline_total / test_scores.shape[1])
    print(bandit_total / test_scores.shape[1])

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
        "--alpha", type=float, default=0.2, help="Confidence threshold.")

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed.")

    parser.add_argument(
        "--zero_mean", action="store_true",
        help="Whether to zero-mean each metric ")

    parser.add_argument(
        "--avg_logprob", action="store_true",
        help="Whether to use average token log probability as a metric.")

    parser.add_argument(
        "--sum_logprob", action="store_true",
        help="Whether to use total token log probability as a metric.")

    args = parser.parse_args()
    main(args)
