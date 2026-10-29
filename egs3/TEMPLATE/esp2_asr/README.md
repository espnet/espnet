# ESPnet3 ASR recipe

## Quick start

```bash
# 0) Edit configs to set paths.
#    Keep `conf/training.yaml:dataset_dir` as the canonical dataset location.
#    When `--training_config` is also passed to `infer` or `measure`, each
#    stage inherits experiment path fields from the earlier stages that ran
#    in the same invocation. A standalone `infer` or `measure` instead needs
#    `--exp_dir` (below), which inherits those fields from an earlier run's
#    baked config under that directory.

# 1) Convert LibriSpeech to Hugging Face format (run once)
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Train with the default Branchformer configuration
python run.py --stages train --training_config conf/training.yaml

# 3) Decode
python run.py --stages infer --training_config conf/training.yaml --inference_config conf/inference.yaml

# 4) Score
python run.py --stages measure --training_config conf/training.yaml --metrics_config conf/metrics.yaml

# 4', alternative) Score a previous run via its baked configs, without
#    redeclaring --training_config/--inference_config/--metrics_config.
#    --exp_dir inherits exp_tag/exp_dir/inference_dir and dataset.test back
#    from the baked train.yaml/infer.yaml under that directory:
python run.py --stages measure --exp_dir exp/train_asr_transformer
```
