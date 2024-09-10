import argparse
from comet import download_model, load_from_checkpoint
import os
import json

args = argparse.ArgumentParser()
args.add_argument('-m', '--model')
args.add_argument('-o', '--out')
args.add_argument('-bs', '--batch-size', type=int, default=16)
args = args.parse_args()

# either local or download
if os.path.isfile(args.model):
    model_path = args.model
else:
    model_path = download_model(args.model)

model = load_from_checkpoint(model_path)

data = [
    json.loads(line)
    for line in open('data/jsonl/test.jsonl')
]

data_comet = [
    {
        "src": x["src"],
        "mt": x["tgt"],
    }
    for x in data
]
print("Data size:", len(data_comet))

output = model.predict(data_comet, batch_size=args.batch_size, gpus=1).scores

with open(args.out, "w") as f:
    for line, score in zip(data, output):
        line["model"] = score
        f.write(json.dumps(line, ensure_ascii=False) + "\n")