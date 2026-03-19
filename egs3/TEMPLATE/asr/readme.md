# ESPnet3 ASR recipe

## Quick start

```bash
# 0) Edit configs to set paths (e.g., conf/training.yaml: dataset_dir, create_dataset.func)

# 1) Convert LibriSpeech to Hugging Face format (run once)
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Train with the default Branchformer configuration
python run.py --stages train --training_config conf/training.yaml

# 3) Decode
python run.py --stages infer --inference_config conf/inference.yaml

# 4) Score
python run.py --stages measure --metrics_config conf/metrics.yaml
```
