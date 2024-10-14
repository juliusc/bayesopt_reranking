import argparse
import itertools
import json
import logging
import numpy as np
from collections import defaultdict

from pathlib import Path

import h5py
import torch

from tqdm import tqdm
from transformers import GenerationConfig, M2M100ForConditionalGeneration, AutoTokenizer

import utils

MAX_GENERATION_LENGTH = 256

def process_result(output, tokenizer):
    """Process generation output to extract the data to save: text, token logprobs, and embeddings."""
    texts = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
    decoder_embeddings = []
    for t in range(2, output.sequences.shape[1]):
        scores = output.scores[t-1].log_softmax(dim=-1)
        decoder_embeddings.append(output.decoder_hidden_states[t-1][-1].squeeze())

    decoder_embeddings = torch.stack(decoder_embeddings, dim=1)
    mask = (output.sequences[:, 2:] != tokenizer.pad_token_id).unsqueeze(-1)
    decoder_embeddings = (decoder_embeddings * mask).sum(dim=1) / mask.sum(1)

    return texts, decoder_embeddings

def generate_candidates(data_path, num_candidates, max_batch_size, epsilon_cutoff):


    torch.manual_seed(0)
    data_lines = open(data_path).readlines()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = M2M100ForConditionalGeneration.from_pretrained(utils.NLLB_MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(utils.NLLB_MODEL_NAME)

    print(f"Generating candidates...")
    num_samples = len(data_lines)
    emb_size = model.model.decoder.layers[-1].final_layer_norm.weight.shape[0]
    texts = np.empty((num_samples, num_candidates), dtype=object)  # Array for texts (string objects)
    counts = np.zeros((num_samples, num_candidates), dtype=float)  # Array for counts (float)
    embeddings = np.zeros((num_samples, num_candidates, emb_size), dtype=float)  # Array for embeddings
    sources = []


    for i, data_line in enumerate(tqdm(data_lines)):
        data = json.loads(data_line)
        src_lang, tgt_lang = data["langs"].split("-")
        # The way the tokenizer src_lang is set and tgt_lang is set during generate()
        # is specific to NLLB.
        tokenizer.src_lang = utils.LANG_TO_NLLB[src_lang]
        tgt_lang_token = tokenizer.convert_tokens_to_ids(utils.LANG_TO_NLLB[tgt_lang])
        inputs = tokenizer(data["src"], padding=True, return_tensors="pt").to(model.device)
        sources.append(data["src"])
        all_result_data = []
        
        with torch.no_grad():
            encoder_outputs = model.model.encoder(**inputs)

        max_batch_size = max_batch_size or num_candidates
        num_samples_done = 0
        while num_samples_done < num_candidates:
            try:
                
                batch_size = min(max_batch_size, num_candidates - num_samples_done)
                gen_config = GenerationConfig(
                    max_length=MAX_GENERATION_LENGTH,
                    num_return_sequences=batch_size,
                    epsilon_cutoff=epsilon_cutoff,
                    do_sample=True
                )
                with torch.no_grad():
                    result = model.generate(
                        encoder_outputs=encoder_outputs.copy(),
                        attention_mask=inputs["attention_mask"],
                        generation_config=gen_config,
                        forced_bos_token_id=tgt_lang_token,
                        output_scores=True,
                        output_hidden_states=True,
                        return_dict_in_generate=True)
                    result_data = process_result(result, tokenizer)
                    all_result_data.append(result_data)
                num_samples_done += batch_size
            except torch.OutOfMemoryError:
                if max_batch_size == 1:
                    raise Exception("OOM with batch size 1 :(")
                new_max_batch_size = max_batch_size // 2 + (max_batch_size % 2 > 0)
                print(
                    f"Instance {i} failed with out-of-memory error. Reducing batch "
                    f"size from {max_batch_size} to {new_max_batch_size}.")
                max_batch_size = new_max_batch_size


        text_to_idx = {}
        for text, embedding in zip(*[itertools.chain(*x) for x in zip(*all_result_data)]):
            if text not in text_to_idx:

                text_idx = len(text_to_idx) 
                text_to_idx[text] = text_idx
                texts[i, text_idx] = text
                
                embeddings[i, text_idx] = embedding.cpu().numpy()

            else:
                text_idx = text_to_idx[text]

            counts[i, text_idx] += 1

    return sources, texts, embeddings, counts

