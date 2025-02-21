#!/usr/bin/bash

. scripts/utils.sh

function get_config() {
    mkdir -p models

    TRAIN_DATA=$(realpath data/csv/train_cometkiwi_nllb.csv)
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

sbatch_gpu "skintle-S" "comet-train --cfg $(COMET_CODENAME='skintle'     get_config 'MiniLM' 'microsoft/Multilingual-MiniLM-L12-H384' 'skintle-S')"
sbatch_gpu "skintle-M" "comet-train --cfg $(COMET_CODENAME='skintle-acc' get_config 'XLM-RoBERTa' 'xlm-roberta-base' 'skintle-M')"
sbatch_gpu "skintle-L" "comet-train --cfg $(COMET_CODENAME='skintle-acc' get_config 'XLM-RoBERTa' 'xlm-roberta-large' 'skintle-L')"