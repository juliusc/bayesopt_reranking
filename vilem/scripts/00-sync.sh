#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/comet-pruning

# copy trained models
# scp euler:/cluster/work/sachan/vilem/comet-pruning/models/quern-*.ckpt tmp/