import argparse
from comet import download_model, load_from_checkpoint
import os
import json
import time

args = argparse.ArgumentParser()
args.add_argument('-m', '--model')
args.add_argument('-d', '--data', default="/net/tscratch/people/plgzuefle/MT_marathon/efficient_reranking/vilem/data/test.jsonl")
args.add_argument('-o', '--out', default="/dev/null")
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
    for line in open(args.data)
]

data_comet = [
    {
        "src": x["src"],
        "mt": x["tgt"],
    }
    for x in data
]
print("Data size:", len(data_comet))

time_start = time.time()
output = model.predict(data_comet, batch_size=args.batch_size, gpus=1, num_workers=1).scores
time_total = time.time() - time_start
print(f"Time: {time_total:.2f}s")
#print(f"Time/1000 samples: {time_total/len(data)*1000:.2f}s")
print(f"Time/10000 samples: {time_total/len(data)*10000:.2f}s")

with open(args.out, "w") as f:
    for line, score in zip(data, output):
        line["model"] = score
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


