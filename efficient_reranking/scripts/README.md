```
python efficient_reranking/scripts/generate_candidates.py et en validation output
for ckpt in $(ls comet_models/*/model/*.ckpt); do
    python efficient_reranking/scripts/score_comet.py et en validation output --comet_path=$ckpt
done
```