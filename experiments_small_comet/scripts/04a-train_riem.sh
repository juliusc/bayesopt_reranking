#!/usr/bin/bash

. scripts/utils.sh

sbatch_gpu "riem-S" "comet-train --cfg $(COMET_CODENAME='riem' get_config 'MiniLM' 'microsoft/Multilingual-MiniLM-L12-H384' 'riem-S')"
sbatch_gpu "riem-M" "comet-train --cfg $(COMET_CODENAME='riem' get_config 'XLM-RoBERTa' 'xlm-roberta-base' 'riem-M')"
sbatch_gpu "riem-L" "comet-train --cfg $(COMET_CODENAME='riem' get_config 'XLM-RoBERTa' 'xlm-roberta-large' 'riem-L')"