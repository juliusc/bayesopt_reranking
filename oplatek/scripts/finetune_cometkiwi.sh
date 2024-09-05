#/bin/bash

# Documenting running locally on GPU
# prepand with gpu-python to submit on cluster
./comet/cli/train.py --load_from_checkpoint /home/oplatek/.cache/huggingface/hub/models--Unbabel--wmt22-cometkiwi-da/snapshots/b3a8aea5a5fc22db68a554b92b3d96eb6ea75cc9/checkpoints/model.ckpt --cfg da/configs/cometkiwi_finetuning.yaml
