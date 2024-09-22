#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/comet-pruning

# copy trained models

# scp euler:/cluster/work/sachan/vilem/comet-pruning/models/riem-minilm-v1.ckpt models/riem-S/model/epoch8.ckpt
# scp euler:/cluster/work/sachan/vilem/comet-pruning/models/riem-bert-base-multilingual-cased-v1.ckpt models/riem-M/model/epoch8.ckpt
# scp euler:/cluster/work/sachan/vilem/comet-pruning/models/riem-xlm-roberta-base-v1.ckpt models/riem-L/model/epoch8.ckpt

# scp euler:/cluster/work/sachan/vilem/comet-pruning/models/forebit-*-v1.ckpt models/forebit

# scp euler:/cluster/work/sachan/vilem/comet-pruning/models/riem-S.ckpt models/riem-S/model/
# scp euler:/cluster/work/sachan/vilem/comet-pruning/models/skintle-S*.ckpt models/skintle-S/model/