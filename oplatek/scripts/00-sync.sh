#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . geri:/cluster/work/sachan/vilem/comet-pruning
