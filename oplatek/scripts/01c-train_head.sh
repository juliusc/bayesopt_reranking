#!/usr/bin/bash
# if this fails there is probably no csv
# just run
# python scripts/02-jsonl_to_csv.py

# to satisfy comet-train
head -n 100 data/csv/train.csv > data/csv/train_head.csv
