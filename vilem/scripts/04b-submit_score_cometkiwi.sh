. scripts/utils.sh

sbatch_gpu "cometkiwi-nllb-0" "python3 scripts/04a-score_comet.py -m Unbabel/wmt22-cometkiwi-da -d data/jsonl/train_nllb_0.jsonl -o computed/train_cometkiwi_nllb_0.jsonl -bs 40"
sbatch_gpu "cometkiwi-nllb-1" "python3 scripts/04a-score_comet.py -m Unbabel/wmt22-cometkiwi-da -d data/jsonl/train_nllb_1.jsonl -o computed/train_cometkiwi_nllb_1.jsonl -bs 40"
sbatch_gpu "cometkiwi-nllb-2" "python3 scripts/04a-score_comet.py -m Unbabel/wmt22-cometkiwi-da -d data/jsonl/train_nllb_2.jsonl -o computed/train_cometkiwi_nllb_2.jsonl -bs 40"
sbatch_gpu "cometkiwi-nllb-3" "python3 scripts/04a-score_comet.py -m Unbabel/wmt22-cometkiwi-da -d data/jsonl/train_nllb_3.jsonl -o computed/train_cometkiwi_nllb_3.jsonl -bs 40"
sbatch_gpu "cometkiwi-nllb-4" "python3 scripts/04a-score_comet.py -m Unbabel/wmt22-cometkiwi-da -d data/jsonl/train_nllb_4.jsonl -o computed/train_cometkiwi_nllb_4.jsonl -bs 40"
sbatch_gpu "cometkiwi-nllb-5" "python3 scripts/04a-score_comet.py -m Unbabel/wmt22-cometkiwi-da -d data/jsonl/train_nllb_5.jsonl -o computed/train_cometkiwi_nllb_5.jsonl -bs 40"
sbatch_gpu "cometkiwi-nllb-6" "python3 scripts/04a-score_comet.py -m Unbabel/wmt22-cometkiwi-da -d data/jsonl/train_nllb_6.jsonl -o computed/train_cometkiwi_nllb_6.jsonl -bs 40"
sbatch_gpu "cometkiwi-nllb-7" "python3 scripts/04a-score_comet.py -m Unbabel/wmt22-cometkiwi-da -d data/jsonl/train_nllb_7.jsonl -o computed/train_cometkiwi_nllb_7.jsonl -bs 40"

# put pieces together
cat computed/train_cometkiwi_nllb_{0,1,2,3,4,5,6,7}.jsonl > data/jsonl/train_cometkiwi_nllb.jsonl