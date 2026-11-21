# mini_an4 esp2_st recipe

A CI fixture for the ESPnet3 speech translation pipeline, not a result-producing recipe. AN4 has no translation targets, so this fixture creates a synthetic German target by translating its small English vocabulary through a fixed word mapping.

## Data Preparation

The recipe uses the bundled AN4 archive via `downloads.tar.gz`, a symlink to `../esp2_asr/downloads.tar.gz`. `create_dataset` extracts the source, builds manifests, and creates the synthetic target text. This target exists only to exercise the source/target text and metric paths.

## Commands

Run from this directory. The command prepares the data, trains separate source and target tokenizers, then runs the tiny ST model, inference, and BLEU/chrF2/TER measurement:

```bash
source path.sh
python run.py \
  --stages create_dataset train_tokenizer collect_stats \
  --training_config conf/training.yaml
python run.py --stages train infer measure \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

The one-epoch model verifies the ST pipeline and metric wiring only. Its translation scores are not meaningful, and no pretrained model is provided.
