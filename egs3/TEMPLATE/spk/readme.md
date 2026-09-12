# ESPnet3 speaker verification recipe

The template wires the speaker task into the generic ESPnet3 stage runner.
Recipes provide the dataset module and the model configuration; everything
below is inherited from `conf/`.

Training is a closed-set classification over the training speakers, while
validation and evaluation score verification trials, so `best_model_criterion`
tracks `valid/eer` rather than a loss.

## Quick start

```bash
# 1) Prepare Kaldi-style manifests and trial lists
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Train
python run.py --stages train --training_config conf/training.yaml

# 3) Score the trial lists
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 4) Compute EER and minDCF
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

## Dataset contract

`dataset/__init__.py` must expose `Dataset` and `DatasetBuilder`. The dataset
returns two kinds of items, selected by its constructor arguments:

| Split       | Keys                                | Meaning                       |
|---          |---                                  |---                            |
| train       | `speech`, `spk_labels`              | one utterance, speaker string |
| valid, test | `speech`, `speech2`, `spk_labels`   | trial pair, `1` if target     |

`SpkPreprocessor` converts both forms into fixed-length crops and maps the
training speaker strings to integer labels using the `spk2utt` file written by
`create_dataset`.
