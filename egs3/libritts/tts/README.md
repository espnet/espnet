# ESPnet3 LibriTTS VITS recipe

Multi-speaker English TTS on LibriTTS using VITS with x-vector speaker
conditioning.

## Quick start

```bash
# 0) Edit configs to set paths.

# 1) Download LibriTTS and build per-split TSV manifests (run once)
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Extract x-vector speaker embeddings (one .pt file per utterance)
python run.py --stages compute_xvectors --training_config conf/training.yaml

# 3) Filter utterances by duration
python run.py --stages remove_long_short --training_config conf/training.yaml

# 4) Build the phoneme token list
python run.py --stages create_token_list --training_config conf/training.yaml

# 5) Collect feature statistics (resumable: set collect_stats.num_shards>1)
python run.py --stages collect_stats --training_config conf/training.yaml

# 6) Train VITS
python run.py --stages train --training_config conf/training.yaml

# 7) Synthesize from test text
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 8) Compute the metrics
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```
