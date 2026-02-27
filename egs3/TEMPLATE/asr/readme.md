# LibriSpeech 100h ESPnet3 recipe

This recipe follows the [lightning-hydra-template](
https://github.com/ashleve/lightning-hydra-template
) layout and uses the latest `espnet3` package utilities.

## Quick start

```bash
# 0) Edit configs to set paths (e.g., conf/training.yaml: dataset_dir, create_dataset.func)

# 1) Convert LibriSpeech to Hugging Face format (run once)
python run.py --stages create_dataset --train_config conf/training.yaml

# 2) Train with the default Branchformer configuration
python run.py --stages train --train_config conf/training.yaml

# 3) Decode
python run.py --stages infer --infer_config conf/inference.yaml

# 4) Score
python run.py --stages measure --measure_config conf/metrics.yaml
```
