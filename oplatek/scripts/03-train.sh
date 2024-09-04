#!/usr/bin/bash

. scripts/utils.sh

printf "Running on $HOSTNAME"

if [[ $HOSTNAME != "sol2" ]] ; then
  # add your cluster head if it differs to sol2 - I recommend machines sol1-8
  printf "Ufal cluster expected. Exiting"; exit 1
fi

# WARNING: These commands are blocking ATM so run them in different terminal 
# Ondra needs to fix --async true setup
# gpu-python comet-train --cfg $(get_config 'XLM-RoBERTa' 'xlm-roberta-base' 'quern-xlm-roberta-base')

gpu-python COMET/comet/cli/train.py --cfg $(get_config 'XLM-RoBERTa' 'xlm-roberta-large' 'quern-xlm-roberta-large')

gpu-python COMET/comet/cli/train.py --cfg $(get_config 'BERT' 'bert-base-multilingual-cased' 'quern-bert-base-multilingual-cased')

gpu-python COMET/comet/cli/train.py --cfg $(get_config 'MiniLM' 'microsoft/Multilingual-MiniLM-L12-H384' 'quern-minilm')

gpu-python COMET/comet/cli/train.py --cfg $(get_config 'RemBERT' 'google/rembert' 'quern-rembert')
