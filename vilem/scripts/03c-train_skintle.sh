#!/usr/bin/bash

. scripts/utils.sh
export MODEL_CODENAME="riem"

sbatch_gpu "riem-s" "comet-train --cfg $(get_config 'MiniLM' 'microsoft/Multilingual-MiniLM-L12-H384' 'riem-s')"
sbatch_gpu "riem-m" "comet-train --cfg $(get_config 'BERT' 'bert-base-multilingual-cased' 'riem-m')"
sbatch_gpu "riem-l" "comet-train --cfg $(get_config 'XLM-RoBERTa' 'xlm-roberta-base' 'riem-l')"