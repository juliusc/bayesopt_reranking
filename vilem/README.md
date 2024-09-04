Instructions for reproducing Vilém's experiments.

## Basics

Run all scripts from the `vilem/` directory.
The following will set up the training data (up to WMT22 as we reserve WMT23 for testing):

```
bash scripts/01a-get_data.sh
python3 scripts/01b-get_da_data.py
python3 scripts/02-jsonl_to_csv.py
bash scripts/01c-train_head.sh
```

## QE models with base models of varying sizes

```
bash scripts/03-train_quern.sh
```

TODO: describe the `hparams.yaml` shenanigans.