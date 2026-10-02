# ESPnet3 SVS recipe

See [`egs2/TEMPLATE/svs1`](../../../egs2/TEMPLATE/svs1/README.md) for what
each stage does; the stages below map onto its recipe flow.

## Quick start

```bash
# 0) Edit configs to set paths.
#    Keep `conf/training.yaml:data_dir` as the canonical dataset location.
#    When `--training_config` is also passed to `infer` or `measure`, run.py
#    propagates experiment path fields from training into inference/metrics.
#    Standalone inference or metrics configs must define their own `exp_tag`
#    or `exp_dir`.

# 1) Build the wav files and manifests (run once)
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Filter utterances by duration and build the token list
python run.py --stages remove_long_short create_token_list \
    --training_config conf/training.yaml

# 3) Collect feature statistics
python run.py --stages collect_stats --training_config conf/training.yaml

# 4) Train
python run.py --stages train --training_config conf/training.yaml

# 5) Synthesize the test sets
python run.py --stages infer --inference_config conf/inference.yaml

# 6) Score
python run.py --stages measure --metrics_config conf/metrics.yaml
```
