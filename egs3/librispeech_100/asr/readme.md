# LibriSpeech 100h ASR recipe

Place the corpus under `download/LibriSpeech`, or set the `LIBRISPEECH`
environment variable to an existing LibriSpeech root, before running
`create_dataset`/`train`.

## Quick start

```bash
# 1) Train the default E-Branchformer model
python run.py --stages train \
    --training_config conf/tuning/training_e_branchformer.yaml

# 2) Decode
python run.py --stages infer \
    --training_config conf/tuning/training_e_branchformer.yaml \
    --inference_config conf/inference.yaml

# 3) Score
python run.py --stages measure \
    --training_config conf/tuning/training_e_branchformer.yaml \
    --metrics_config conf/metrics.yaml
```
