import argparse
import json
import scipy.stats

args = argparse.ArgumentParser()
args.add_argument("-d1", "--data1")
args.add_argument("-d2", "--data2")
args = args.parse_args()

if args.data2 in {"score", "human"}:
    args.data2 = args.data1
    data2_key = "score"
else:
    data2_key = "model"

data1 = [json.loads(line)["model"] for line in open(args.data1)]
data2 = [json.loads(line)[data2_key] for line in open(args.data2)]

print(f"{scipy.stats.kendalltau(data1, data2, variant='c').correlation:.4f}")