# ESPnet3 speech translation template

## Overview

This template uses `espnet3.systems.esp2_st.system.ESP2STSystem` for an ESPnet2 speech translation task. Corpus recipes provide the dataset implementation and language-pair-specific configuration.

## Data

Each item carries speech, the target translation (`text`), and the source transcript (`src_text`). The target side feeds the ST decoder; the source side feeds the auxiliary ASR decoder and CTC. A recipe's dataset class (`dataset/dataset.py`, a `BaseDataset` subclass) declares the fields as a class attribute; both the MuST-C and mini_an4 recipes use:

```python
from espnet3.contract import Field

fields = (Field("speech", "audio"), Field("text", "text"), Field("src_text", "text"))
```

`fields` is declared on the dataset class, not in a configuration file. The inference contract accepts speech and produces translated text. By default it resamples input to 16 kHz (`rate_policy` defaults to `resample`) and keeps the first audio channel (`first_channel`).

## System Features

`train_tokenizer` trains separate target (`tokenizer.tgt`) and source (`tokenizer.src`) SentencePiece models. The system also provides statistics collection, training, inference, measurement, and model/demo packaging.

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

Run from a corpus recipe directory after configuring the language pair, data source, and model. Replace `<training_config>` with the recipe-specific path: `conf/tuning/train_st_conformer.yaml` for MuST-C, `conf/training.yaml` for mini_an4. Its `run.py` calls `launch()` to combine template defaults with recipe-specific settings.

```bash
python run.py --stages create_dataset train_tokenizer collect_stats \
  --training_config <training_config>
python run.py --stages train \
  --training_config <training_config>
python run.py --stages infer \
  --training_config <training_config> \
  --inference_config conf/inference.yaml
python run.py --stages measure \
  --training_config <training_config> \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

See [mini_an4](../../mini_an4/esp2_st/README.md) for a small runnable example and [MuST-C](../../must_c/esp2_st/README.md) for a corpus recipe.
