# VoxLingua107 LID recipe

Train an MMS/ECAPA-TDNN model to identify 107 spoken languages.

## Quick start

Run from `egs3/voxlingua107/lid` with ESPnet installed.
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
