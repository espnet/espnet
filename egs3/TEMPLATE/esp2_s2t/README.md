# ESPnet3 speech-to-text template

## Overview

This template uses `espnet3.systems.esp2_s2t.system.ESP2S2TSystem` for an
ESPnet2 S2T task with a fixed Whisper vocabulary. The current recipes train
and decode serialized output training (SOT) for multi-talker speech: one
utterance group becomes one sequence, with the speakers separated by a
speaker-change symbol and inline Whisper timestamps. Corpus recipes provide
the dataset implementation and the checkpoint-specific settings.

## Data

Each item carries its id, speech, the serialized target (`text`), and the
two S2T auxiliary texts (`text_prev`, `text_ctc`). A recipe's dataset class
(`dataset/dataset.py`, a `BaseDataset` subclass) declares the fields as a
class attribute; both the AMI and mini_an4 recipes use:

```python
from espnet3.contract import Field

fields = (
    Field("utt_id", "text"),
    Field("speech", "audio"),
    Field("text", "text"),
    Field("text_prev", "text"),
    Field("text_ctc", "text"),
)
```

`text` starts with the language and task symbols, for example
`<|en|><|transcribe|><|0.00|> hello<|1.20|> ???? <|0.80|> hi<|1.50|>`.
The SOT preprocessor drops `utt_id` before training. The inference contract
accepts 16 kHz mono speech and produces `hyp`, the serialized hypothesis with
its timestamps and speaker-change symbols.

## System Features

There is no `train_tokenizer` stage: the vocabulary is Whisper's, and a
recipe's `create_dataset` writes the token list. `collect_stats` keeps a
configured `model.normalize: null`, which the base stage would replace with
the S2T task's default. Inference uses
`espnet3.systems.esp2_s2t.src.sot.s2t_inference.Speech2Text`, which scopes
Whisper's timestamp rules to the current speaker block. SOT decoding runs one
utterance group at a time (`batch_size: null`). The system also provides
training, measurement, and model packaging.

## Metrics

The system provides four utterance-group metrics: cpWER, DER, speaker-counting
accuracy, and WER of the serialized transcript. Each needs the speaker-change
symbol the checkpoint was trained with and has no default for it. Bind the
reference to the dataset's serialized text and the hypothesis to `hyp`:

```yaml
metrics:
  - metric:
      _target_: espnet3.systems.esp2_s2t.metrics.cpwer.UtteranceGroupCpWER
      speaker_change_symbol: "????"
      clean_types: [whisper_en]
    inputs:
      ref: dataset:text
      hyp: hyp
```

## Commands

Run from a corpus recipe directory after configuring the data source and the
checkpoint. Its `run.py` calls `launch()` to combine template defaults with
recipe-specific settings.

```bash
python run.py --stages create_dataset collect_stats \
  --training_config conf/training.yaml
python run.py --stages train \
  --training_config conf/training.yaml
python run.py --stages infer measure \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

See [mini_an4](../../mini_an4/esp2_s2t/README.md) for a small runnable example
and [AMI](../../ami/esp2_s2t/README.md) for a corpus recipe.
