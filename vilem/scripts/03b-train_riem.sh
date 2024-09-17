#!/usr/bin/bash

. scripts/utils.sh
export COMET_CODENAME="riem"

sbatch_gpu "riem-S" "comet-train --cfg $(get_config 'BERT' 'sentence-transformers/all-MiniLM-L12-v2' 'riem-S')" 
sbatch_gpu "riem-M" "comet-train --cfg $(get_config 'MiniLM' 'microsoft/Multilingual-MiniLM-L12-H384' 'riem-M')"
sbatch_gpu "riem-L" "comet-train --cfg $(get_config 'XLM-RoBERTa' 'xlm-roberta-base' 'riem-L')"