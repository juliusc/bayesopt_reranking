import argparse
from comet import download_model, load_from_checkpoint
import os

args = argparse.ArgumentParser()
args.add_argument('-m', '--model')
args = args.parse_args()

# either local or download
if os.path.isfile(args.model):
    model_path = args.model
else:
    model_path = download_model(args.model)

model = load_from_checkpoint(model_path)


for bs in range(10, 2_000, 10):
    # make sure the inputs are long enough
    data_comet = [
        {
            "src": "word " * 10_000,
            "mt": "word "  * 10_000,
        }
        for _ in range(bs)
    ]
    print("Trying batch size of", bs)
    try:
        output = model.predict(data_comet, batch_size=bs, gpus=1).scores
    except Exception as e:
        print(e)
        exit()


"""
python3 scripts/10-find_max_batch_size.py -m Unbabel/wmt22-cometkiwi-da # 60
python3 scripts/10-find_max_batch_size.py -m models/skintle-L/model-L-v10.ckp # 120
python3 scripts/10-find_max_batch_size.py -m models/skintle-M/model-M-v20.ckp # 390
python3 scripts/10-find_max_batch_size.py -m models/skintle-S/model-S-v1.ckp # 390
"""