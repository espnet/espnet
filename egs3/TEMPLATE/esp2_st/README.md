# ESPnet3 speech translation template

## Overview

This template uses `espnet3.systems.esp2_st.ESP2STSystem` for an ESPnet2 speech translation task. Corpus recipes provide the dataset implementation and language-pair-specific configuration.

## Data

Each example contains speech and target translation text. A source transcript (`src_text`) can also be supplied when training uses an auxiliary source-text objective.

```yaml
fields:
  speech: audio
  src_text: text
  text: text
```

The inference contract accepts speech and produces translated text. By default, it selects the first audio channel and requires 16 kHz input.

## System Features

The system provides source/target tokenizer preparation where configured, statistics collection, training, inference, measurement, and model/demo packaging.

## Metrics

The system provides BLEU, chrF2, and TER through its BLEU metric. Bind the reference to the dataset's target text and the hypothesis to generated text.

```yaml
metrics:
  - metric:
      _target_: espnet3.systems.esp2_st.metrics.bleu.BLEU
    inputs:
      ref: dataset:text
      hyp: text
```

## Commands

Run from a corpus recipe directory after configuring the language pair, data source, and model. Its `run.py` calls `launch()` to combine template defaults with recipe-specific settings.

```bash
python run.py --stages create_dataset train_tokenizer collect_stats train \
  --training_config conf/training.yaml
python run.py --stages infer \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml
python run.py --stages measure \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

See [mini_an4](../../mini_an4/esp2_st/README.md) for a small runnable example and [MuST-C](../../must_c/esp2_st/README.md) for a corpus recipe.
