#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/comet-pruning

# copy trained models
# scp euler:/cluster/work/sachan/vilem/comet-pruning/models/quern-*.ckpt tmp/

# scp euler:/cluster/work/sachan/vilem/comet-pruning/models/riem-minilm-v1.ckpt models/riem-S/model/epoch8.ckpt
# scp euler:/cluster/work/sachan/vilem/comet-pruning/models/riem-bert-base-multilingual-cased-v1.ckpt models/riem-M/model/epoch8.ckpt
# scp euler:/cluster/work/sachan/vilem/comet-pruning/models/riem-xlm-roberta-base-v1.ckpt models/riem-L/model/epoch8.ckpt

# scp euler:/cluster/work/sachan/vilem/comet-pruning/models/skintle-s-v{1,5,10,15,20}.ckpt models/skintle/models/
