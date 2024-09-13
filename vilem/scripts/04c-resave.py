import json
import tqdm

data = [
    json.loads(x)
    for x in tqdm.tqdm(open('data/jsonl/train_cometkiwi_nllb.jsonl', 'r'))
]


for line in data:
    line["score"] = line.pop("model")
with open("data/jsonl/train_cometkiwi_nllb.jsonl", "w") as f:
    for line in tqdm.tqdm(data):
        f.write(json.dumps(line, ensure_ascii=False) + '\n')

# sbatch_cpu "resave" "python3 scripts/04c-resave.py; python3 scripts/02-jsonl_to_csv.py"