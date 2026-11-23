# ESPnet3 speech-to-text template

## Overview

This template provides the default configuration for recipes that use
`espnet3.systems.esp2_s2t.system.ESP2S2TSystem`. The system trains and decodes
ESPnet2 S2T models with the Whisper vocabulary. The current recipes use it for
multi-talker ASR with serialized output training (SOT). In SOT, the model
transcribes each utterance group as one sequence, with a speaker-change symbol
between speakers and Whisper timestamps in the text. Each corpus recipe
provides its own dataset implementation and checkpoint settings.

## Data

Each item contains an utterance ID, the speech, the serialized target text
(`text`), and the two auxiliary texts of the S2T task (`text_prev` and
`text_ctc`). The dataset class of a recipe (`dataset/dataset.py`, a subclass of
`BaseDataset`) declares these fields as a class attribute. The AMI and mini_an4
recipes both use the following fields.

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
`<|en|><|transcribe|><|0.00|> hello<|1.20|> ???? <|0.80|> hi<|1.50|>`. The SOT
preprocessor removes `utt_id` before training. At inference time, the input is
16 kHz mono speech, and the output `hyp` is the serialized hypothesis with its
timestamps and speaker-change symbols.

## System Features

The system has no `train_tokenizer` stage because it uses the Whisper
vocabulary. The `create_dataset` stage of each recipe writes the token list.
The `collect_stats` stage keeps the setting `model.normalize: null`. The base
stage removes this setting, so the S2T task would add its default normalizer.
Inference uses `espnet3.systems.esp2_s2t.src.sot.s2t_inference.Speech2Text`,
which applies the Whisper timestamp rules within each speaker block. SOT
decoding processes one utterance group at a time, so the inference config sets
`batch_size: null`. All other stages come from `BaseSystem` unchanged.

## Metrics

The system provides four metrics, and each of them is computed per utterance
group: cpWER, DER, speaker-counting accuracy, and WER of the serialized
transcript. Each metric needs the speaker-change symbol that the checkpoint was
trained with, and none of them has a default value. Bind the reference to the
serialized text of the dataset and the hypothesis to `hyp`, as in the following
example.

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

Run these commands from a corpus recipe directory after you set the data source
and the checkpoint. The `run.py` of the recipe calls `launch()`, which combines
the defaults of this template with the settings of the recipe.

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
and [AMI](../../ami/esp2_s2t/README.md) for a full corpus recipe.
