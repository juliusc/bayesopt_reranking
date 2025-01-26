. scripts/utils.sh

sbatch_gpu "nllb-0" "python3 scripts/06a-translate_mt.py -i 0"
sbatch_gpu "nllb-1" "python3 scripts/06a-translate_mt.py -i 1"
sbatch_gpu "nllb-2" "python3 scripts/06a-translate_mt.py -i 2"
sbatch_gpu "nllb-3" "python3 scripts/06a-translate_mt.py -i 3"
sbatch_gpu "nllb-4" "python3 scripts/06a-translate_mt.py -i 4"
sbatch_gpu "nllb-5" "python3 scripts/06a-translate_mt.py -i 5"
sbatch_gpu "nllb-6" "python3 scripts/06a-translate_mt.py -i 6"
sbatch_gpu "nllb-7" "python3 scripts/06a-translate_mt.py -i 7"