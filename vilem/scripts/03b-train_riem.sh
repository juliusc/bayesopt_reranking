#!/usr/bin/bash

. scripts/utils.sh
export COMET_CODENAME="riem"

sbatch_gpu "riem-xlm-roberta-base" "comet-train --cfg $(get_config 'XLM-RoBERTa' 'xlm-roberta-base' 'riem-xlm-roberta-base')"

sbatch_gpu "riem-bert-base-multilingual-cased" "comet-train --cfg $(get_config 'BERT' 'bert-base-multilingual-cased' 'riem-bert-base-multilingual-cased')"

sbatch_gpu "riem-minilm" "comet-train --cfg $(get_config 'MiniLM' 'microsoft/Multilingual-MiniLM-L12-H384' 'riem-minilm')"