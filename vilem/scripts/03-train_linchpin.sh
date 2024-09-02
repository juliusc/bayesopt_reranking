#!/usr/bin/bash

. scripts/utils.sh

sbatch_gpu "linchpin-xlm-roberta-base" "comet-train --cfg $(get_config 'XLM-RoBERTa' 'xlm-roberta-base' 'linchpin-xlm-roberta-base')"

sbatch_gpu "linchpin-xlm-roberta-large" "comet-train --cfg $(get_config 'XLM-RoBERTa' 'xlm-roberta-large' 'linchpin-xlm-roberta-large')"

sbatch_gpu "linchpin-bert-base-multilingual-cased" "comet-train --cfg $(get_config 'BERT' 'bert-base-multilingual-cased' 'linchpin-bert-base-multilingual-cased')"

sbatch_gpu "linchpin-minilm" "comet-train --cfg $(get_config 'MiniLM' 'microsoft/Multilingual-MiniLM-L12-H384' 'linchpin-bert-base-multilingual-cased')"

sbatch_gpu "linchpin-rembert" "comet-train --cfg $(get_config 'RemBERT' 'google/rembert' 'linchpin-rembert')"

# TODO: create quantized versions of the models