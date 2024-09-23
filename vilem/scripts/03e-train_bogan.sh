#!/usr/bin/bash

. scripts/utils.sh

# prepare data
sbatch_gpu "cometkiwi-score-train" "python3 scripts/04a-score_comet.py -d data/jsonl/train.jsonl -m Unbabel/wmt22-cometkiwi-da -o data/jsonl/train.cometkiwi.jsonl -bs 32"
python3 scripts/02-jsonl_to_csv.py data/jsonl/train.cometkiwi.jsonl data/csv/train_cometkiwi.csv

# launch training
function get_config() {
    mkdir -p models

    TRAIN_DATA=$(realpath data/csv/train_cometkiwi.csv)
    DEV_DATA=$(realpath data/csv/dev.csv)
    CHECKPOINT_DIR=$(realpath "models/")
    ENCODER_MODEL=$1
    PRETRAINED_MODEL=$2
    CHECKPOINT_FILENAME=$3
    
    # should prevent collisions
    mkdir -p tmp
    TMP_CONFIG_DIR=$(mktemp -d -p 'tmp/')
    cp configs/${COMET_CODENAME}/* ${TMP_CONFIG_DIR}

    cat ${TMP_CONFIG_DIR}/model.yaml \
        | sed "s|TRAIN_DATA_PATH|${TRAIN_DATA}|" \
        | sed "s|DEV_DATA_PATH|${DEV_DATA}|" \
        | sed "s|ENCODER_MODEL|${ENCODER_MODEL}|" \
        | sed "s|PRETRAINED_MODEL|${PRETRAINED_MODEL}|" \
        > ${TMP_CONFIG_DIR}/model.yaml
    cat ${TMP_CONFIG_DIR}/model_checkpoint.yaml \
        | sed "s|CHECKPOINT_DIR|${CHECKPOINT_DIR}|" \
        | sed "s|CHECKPOINT_FILENAME|${CHECKPOINT_FILENAME}|" \
        > ${TMP_CONFIG_DIR}/model_checkpoint.yaml
    echo ${TMP_CONFIG_DIR}/model.yaml
}

COMET_CODENAME="riem"     sbatch_gpu "bogan-S" "comet-train --cfg $(get_config 'BERT' 'sentence-transformers/all-MiniLM-L12-v2' 'bogan-S')"
COMET_CODENAME="riem"     sbatch_gpu "bogan-M" "comet-train --cfg $(get_config 'MiniLM' 'microsoft/Multilingual-MiniLM-L12-H384' 'bogan-M')"
COMET_CODENAME="riem-acc" sbatch_gpu "bogan-L" "comet-train --cfg $(get_config 'XLM-RoBERTa' 'xlm-roberta-base' 'bogan-L')"
