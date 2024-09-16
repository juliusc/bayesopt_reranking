#!/usr/bin/bash

. scripts/utils.sh

function get_config() {
    mkdir -p models

    TRAIN_DATA=$(realpath data/csv/train.csv)
    DEV_DATA=$(realpath data/csv/dev.csv)
    CHECKPOINT_DIR=$(realpath "models/")
    ENCODER_MODEL=$1
    PRETRAINED_MODEL=$2
    CHECKPOINT_FILENAME=$3
    
    # should prevent collisions
    mkdir -p tmp
    TMP_CONFIG_DIR=$(mktemp -d -p 'tmp/')
    cp configs/forebit/* ${TMP_CONFIG_DIR}

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

# for ARCH in  "BERT" "XLM-RoBERTa" "MiniLM" "XLM-RoBERTa-XL" "RemBERT"; do
#     for MODEL in \
#         "sentence-transformers/all-MiniLM-L6-v2" \
#         "sentence-transformers/all-MiniLM-L12-v2" \
#         "distilbert/distilbert-base-multilingual-cased" \
#         "sentence-transformers/distiluse-base-multilingual-cased-v2" \
#         "sentence-transformers/LaBSE" \
#     ; do
#     done;
# done;

sbatch_gpu "forebit-MiniLML6-XLM" "comet-train --cfg $(get_config 'XLM-RoBERTa' 'sentence-transformers/all-MiniLM-L6-v2' 'forebit-MiniLML6-XLM')"
sbatch_gpu "forebit-MiniLML6-XLMXL" "comet-train --cfg $(get_config 'XLM-RoBERTa-XL' 'sentence-transformers/all-MiniLM-L6-v2' 'forebit-MiniLML6-XLMXL')"

sbatch_gpu "forebit-MiniLML12-XLMXL" "comet-train --cfg $(get_config 'XLM-RoBERTa-XL' 'sentence-transformers/all-MiniLM-L12-v2' 'forebit-MiniLML12-XLMXL')"
sbatch_gpu "forebit-MiniLML12-BERT" "comet-train --cfg $(get_config 'BERT' 'sentence-transformers/all-MiniLM-L12-v2' 'forebit-MiniLML12-BERT')"

sbatch_gpu "forebit-LaBSE-XLMXL" "comet-train --cfg $(get_config 'XLM-RoBERTa-XL' 'sentence-transformers/LaBSE' 'forebit-LaBSE-XLMXL')"
sbatch_gpu "forebit-LaBSE-BERT" "comet-train --cfg $(get_config 'BERT' 'sentence-transformers/LaBSE' 'forebit-LaBSE-BERT')"
sbatch_gpu "forebit-LaBSE-MiniLM" "comet-train --cfg $(get_config 'MiniLM' 'sentence-transformers/LaBSE' 'forebit-LaBSE-MiniLM')"

sbatch_gpu "forebit-distilbert-BERT" "comet-train --cfg $(get_config 'BERT' 'distilbert/distilbert-base-multilingual-cased' 'forebit-distilbert-BERT')"

sbatch_gpu "forebit-distiluse-XLM" "comet-train --cfg $(get_config 'XLM-RoBERTa' 'sentence-transformers/distiluse-base-multilingual-cased-v2' 'forebit-distiluse-XLM')"
