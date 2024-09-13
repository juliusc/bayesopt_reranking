function get_config() {
    mkdir -p models

    TRAIN_DATA=$(realpath data/csv/train.csv)
    DEV_DATA=$(realpath data/csv/dev.csv)
    CHECKPOINT_DIR=$(realpath "models/")
    ENCODER_MODEL=$1
    PRETRAINED_MODEL=$2
    CHECKPOINT_FILENAME=$3

    if [ -v COMET_CODENAME ]; then
        echo "COMET_CODENAME is set to $COMET_CODENAME";
    else
        echo "COMET_CODENAME is unset";
        exit 1;
    fi
    
    # should prevent collisions
    mkdir -p tmp
    TMP_CONFIG_DIR=$(mktemp -d -p 'tmp/')
    cp configs/${COMET_CODENAME}/* ${TMP_CONFIG_DIR}

    cat configs/model.yaml \
        | sed "s|TRAIN_DATA_PATH|${TRAIN_DATA}|" \
        | sed "s|DEV_DATA_PATH|${DEV_DATA}|" \
        | sed "s|ENCODER_MODEL|${ENCODER_MODEL}|" \
        | sed "s|PRETRAINED_MODEL|${PRETRAINED_MODEL}|" \
        > ${TMP_CONFIG_DIR}/model.yaml
    cat configs/model_checkpoint.yaml \
        | sed "s|CHECKPOINT_DIR|${CHECKPOINT_DIR}|" \
        | sed "s|CHECKPOINT_FILENAME|${CHECKPOINT_FILENAME}|" \
        > ${TMP_CONFIG_DIR}/model_checkpoint.yaml
    echo ${TMP_CONFIG_DIR}/model.yaml
}

function sbatch_gpu() {
    JOB_NAME=$1;
    JOB_WRAP=$2;
    mkdir -p logs

    sbatch \
        -J $JOB_NAME --output=logs/%x.out --error=logs/%x.err \
        --gpus=1 --gres=gpumem:22g \
        --ntasks-per-node=1 \
        --cpus-per-task=6 \
        --mem-per-cpu=8G --time=1-0 \
        --wrap="$JOB_WRAP";
}

function sbatch_gpu_short() {
    JOB_NAME=$1;
    JOB_WRAP=$2;
    mkdir -p logs

    sbatch \
        -J $JOB_NAME --output=logs/%x.out --error=logs/%x.err \
        --gpus=1 --gres=gpumem:22g \
        --ntasks-per-node=1 \
        --cpus-per-task=6 \
        --mem-per-cpu=8G --time=0-4 \
        --wrap="$JOB_WRAP";
}

function local_gpu() {
    JOB_NAME=$1;
    JOB_WRAP=$2;
    mkdir -p logs

    $JOB_WRAP 1> logs/${JOB_NAME}.out 2> logs/${JOB_NAME}.err
}