#!/usr/bin/bash

. scripts/utils.sh
export COMET_CODENAME="quern"

sbatch_gpu "quern-xlm-roberta-base" "comet-train --cfg $(get_config 'XLM-RoBERTa' 'xlm-roberta-base' 'quern-xlm-roberta-base')"

sbatch_gpu "quern-xlm-roberta-large" "comet-train --cfg $(get_config 'XLM-RoBERTa' 'xlm-roberta-large' 'quern-xlm-roberta-large')"

sbatch_gpu "quern-bert-base-multilingual-cased" "comet-train --cfg $(get_config 'BERT' 'bert-base-multilingual-cased' 'quern-bert-base-multilingual-cased')"

sbatch_gpu "quern-minilm" "comet-train --cfg $(get_config 'MiniLM' 'microsoft/Multilingual-MiniLM-L12-H384' 'quern-minilm')"

sbatch_gpu "quern-rembert" "comet-train --cfg $(get_config 'RemBERT' 'google/rembert' 'quern-rembert')"

# TODO: create quantized versions of the models