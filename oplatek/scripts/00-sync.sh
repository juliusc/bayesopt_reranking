#!/bin/bash

rsync -azP --filter=":- .gitignore"  --exclude .git/ . geri:/home/oplatek/code/efficient_pruning/oplatek