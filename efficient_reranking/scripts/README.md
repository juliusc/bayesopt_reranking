```
# 1. Generate candidates
python efficient_reranking/scripts/generate_candidates.py vilem/scripts/data dev output

# 2A. Run COMET evals for differently sized models
python efficient_reranking/scripts/score_comet.py vilem/scripts/data dev output --comet_repo=Unbabel/wmt22-cometkiwi-da --overwrite

for ckpt in $(ls comet_models/*/model/*.ckpt); do
    python efficient_reranking/scripts/score_comet.py et en validation output --comet_path=$ckpt --overwrite
done
```