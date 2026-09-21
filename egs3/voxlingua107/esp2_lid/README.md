# VoxLingua107 LID recipe

Train an MMS/ECAPA-TDNN model to identify 107 spoken languages.

## Quick start

Run from `egs3/voxlingua107/esp2_lid` with ESPnet installed.
Set `VOXLINGUA107` to the corpus destination or an existing extracted corpus.
Model and training settings are in `conf/training.yaml`.

```bash
export VOXLINGUA107=/path/to/voxlingua107

# 1) Download and prepare the data, collect statistics, and train
python run.py --stages create_dataset collect_stats train \
    --training_config conf/training.yaml

# 2) Predict languages on the development set
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 3) Compute accuracy, precision, recall, and F1
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

## Results

Accuracy (%) of the MMS-1B + ECAPA-TDNN model trained on VoxLingua107 with
[`conf/training.yaml`](conf/training.yaml).

| Dataset | Split | Evaluated languages | Utterances | Accuracy (%) |
| --- | --- | ---: | ---: | ---: |
| VoxLingua107 | dev | 33 | 1,609 | 93.78 |
| FLEURS | test | 83 | 63,885 | 94.49 |
| ML-SUPERB 2.0 | dev | 91 | 17,631 | 88.31 |
| ML-SUPERB 2.0 | dev_dialect | 8 | 7,095 | 75.79 |
| VoxPopuli | test | 16 | 16,991 | 88.42 |

Evaluation includes all utterances whose reference language is supported by
the model; prediction candidates remain all 107 training languages.
VoxLingua107 dev is also used for checkpoint selection.

Model: [shun3232/espnet3_lid_voxlingua107_mms_ecapa](https://huggingface.co/shun3232/espnet3_lid_voxlingua107_mms_ecapa)
