import argparse

import numpy as np
from collections import defaultdict
from scipy.stats import norm 

from pathlib import Path
import comet

import torch
import torch.nn.functional as F

from tqdm import tqdm

from z_generate_cands import generate_candidates

INITIAL_SIZE = 10
MAX_EVALS = 200
GPUS = 1
NUM_WORKERS = 1

import os
import logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)  # Limit CPU threads
torch.set_num_interop_threads(1)  # Limit inter-operation threads
# Set logging level to suppress info messages
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)


def main(args):

    # get candidates
    all_sources, all_texts, embeddings, all_counts = generate_candidates(args.data_path, args.num_candidates, args.max_batch_size, args.epsilon)

    # load comet models
    if args.comet_repo:
        comet_base_name = args.comet_repo.split("/")[-1]
        model_path = comet.download_model(args.comet_repo)
        model = comet.load_from_checkpoint(model_path).eval()
    elif args.comet_path:
        comet_base_name = args.comet_path.split("/")[-3]
        model = comet.load_from_checkpoint(args.comet_path)
    else:
        raise ValueError("Must provide --comet_repo or --comet_path.")

    if args.calculate_kiwi_for_all:
        final_candidates = []
        for src, all_text, count in tqdm(zip(all_sources, all_texts, all_counts), total=len(all_sources)):
            num_cands = (count > 0).sum()
            text = all_text[:num_cands]
            scores = np.empty(text.shape[0])
            for cand_idx, cand in enumerate(text):
                comet_input = {"src": src, "mt": cand}
                scores[cand_idx] = model.predict(samples=[comet_input], batch_size=1, gpus=GPUS, progress_bar=False, num_workers=NUM_WORKERS).scores[0]
            final_candidates.append(np.argmax(scores))

        return

    # calculate similarities
    sims_orig = {}
    for idx, emb in enumerate(tqdm(embeddings)):
        emb = F.normalize(torch.tensor(emb))
        sims_orig[idx] = torch.matmul(emb, emb.T).reshape(-1)
        
    sims = []
    texts = []
    max_cands = all_counts.shape[1]
    for idx, data_idx in enumerate(tqdm(range(all_texts.shape[0]))):
        num_cands = (all_counts[data_idx] > 0).sum()
        texts.append(all_texts[data_idx][:num_cands])
        sims.append(sims_orig[data_idx].reshape(max_cands, max_cands)[:num_cands, :num_cands])
  
    # bandit 
    np.random.seed(args.seed)

    all_sims = sims

    bandit_total = defaultdict(int)

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
        
        # prior_var = np.var(scores[known_idxs])

        with torch.no_grad():
            for known_id in known_idxs:
                comet_input = {"src": src, "mt": text[known_id]}
                scores[known_id] = model.predict(samples=[comet_input], batch_size=1, gpus=GPUS, progress_bar=False, num_workers=NUM_WORKERS).scores[0]

        rbf_cov = np.exp(-(1 - sims.reshape(-1)) / (2 * args.bandwidth ** 2)).reshape(all_idxs.size, all_idxs.size)

        while len(known_idxs) < min(MAX_EVALS, all_idxs.shape[0]):

            bandit_total[len(known_idxs)] += scores[known_idxs].max()

            known_scores = scores[known_idxs]
            known_scores -= known_scores.mean()
            known_scores /= np.std(known_scores)

            unknown_unknown_cov = rbf_cov[unknown_idxs][:, unknown_idxs]
            known_unknown_cov = rbf_cov[known_idxs][:, unknown_idxs]
            known_known_cov = rbf_cov[known_idxs][:, known_idxs]

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
                
                # calculate comet score for new candidate
                with torch.no_grad():
                    new_cand = unknown_idxs[best_unknown_idx_idx]                    
                    comet_input = {"src": src, "mt": text[new_cand]}
                    scores[new_cand] = model.predict(samples=[comet_input], batch_size=1, gpus=GPUS, progress_bar=False, num_workers=NUM_WORKERS).scores[0]
                del unknown_idxs[best_unknown_idx_idx]
            except Exception as e:
                print("FAIL")
                break




    for k in bandit_total:
        if k % 10 == 0:
            print(k, bandit_total[k] / len(texts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path", help="Data directory generated by the pipeline from vilem/scripts.", default="efficient_reranking/scripts/shortest_sample.jsonl")


    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for PyTorch.")

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
