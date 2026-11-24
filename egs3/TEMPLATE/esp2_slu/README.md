# ESPnet3 spoken language understanding template

## Overview

This template uses `espnet3.systems.esp2_slu.system.ESP2SLUSystem` for an ESPnet2 SLU task. Corpus recipes provide the dataset implementation and task-specific configuration.

The model predicts the intent as the first token of the decoded sequence and the transcript after it, so one hypothesis carries both. Intent labels are reserved as SentencePiece symbols, which keeps each one a single token.

A second pass additionally reads a transcript, which a frozen post-decoder encodes and a deliberation encoder fuses with the speech encoder output. That transcript is either the reference or a first pass's own hypothesis -- the two settings `egs2/TEMPLATE/slu1` selects with `--gt true` / `--gt false`.

## Data

Each item carries audio and the target text, which is the intent label followed by the transcript:

```
audio_volume_mute turn off the speakers
^ intent          ^ transcript
```

A recipe's dataset class (`dataset/dataset.py`, a `BaseDataset` subclass) declares the fields as a class attribute; the SLURP and mini_an4 recipes use:

```python
from espnet3.contract import Field

fields = (
    Field("speech", "audio"),
    Field("text", "text"),
    Field("transcript", "text", optional=True),
    Field("intent", "text", optional=True),
    Field("reference_transcript", "text", optional=True),
)
```

Three of those are optional because they are not wanted at every stage, and both recipes gate them behind constructor flags set from `data_src_args`:

- `transcript` is the second pass's **input**, so only a two-pass training or inference config asks for it. It may hold a first-pass hypothesis rather than the truth.
- `intent` and `reference_transcript` are the scoring **references**, which `conf/metrics.yaml` binds with `dataset:`. They are read at measure time from the test set, so only the inference configs ask for them. They are plain strings that no preprocessor tokenizes, and the training collate function cannot pad such a field.

The two are independent: one is what the model reads, the other is what it is scored against.

The default inference contract resamples audio to 16 kHz, averages multichannel input to mono, and emits the decoded sequence (`hyp`) with the intent (`hyp_intent`) and the transcript (`hyp_transcript`) split out of it.

## System Features

`ESP2SLUSystem` extends the ASR system with the two things `SLUTask` needs beyond it:

- **The intent inventory.** `train_tokenizer` passes the recipe's intent labels to SentencePiece as `user_defined_symbols`, so no label is split into pieces.
- **A transcript vocabulary.** `SLUTask` tokenizes the post-decoder's transcript separately, so the system writes a word token list beside the subword one, as stage 5 of `egs2/TEMPLATE/slu1/slu.sh` does.

Both read the recipe's own hooks, named from the training config.

To start a second pass from a first pass's encoder -- the `--pretrained_model` flag of `slu.sh` -- add `espnet3.components.callbacks.pretrained_parameters.PretrainedParametersCallback` to `trainer.callbacks` with `init_param: <path>:encoder:encoder`. It copies values through the state dict and leaves `requires_grad` alone, so it composes with `model.freeze_param`.

## Runners

`SLUInferenceRunner` is the default: it splits the decoded sequence into the intent and the transcript, so the intent label is not charged as a word error. The backend decodes one utterance at a time, so these modes need `batch_size: null`.

A first pass is an ordinary ASR model whose hypothesis is words from its first token, and splitting it would throw the first word away. Decode it with `espnet3.systems.esp2_slu.src.inference_runner.FirstPassTranscriptRunner`, which keeps the hypothesis whole and writes an empty `hyp_intent`.

## Metrics

The system provides intent accuracy; WER and CER come from the ASR system. They read the `*_transcript` fields rather than the raw hypothesis, and the references come from the dataset, so the corpus text is scored even when the model's own transcript input is a first-pass hypothesis.

```yaml
metrics:
  - metric:
      _target_: espnet3.systems.esp2_slu.metrics.intent_accuracy.IntentAccuracy
    inputs:
      ref: dataset:intent
      hyp: hyp_intent
```

## Commands

Run from the corpus recipe directory after configuring the dataset and the SLU model. `run.py` calls `launch()` to combine this template with recipe-specific overrides.

```bash
python run.py --stages create_dataset train_tokenizer collect_stats \
  --training_config conf/training.yaml
python run.py --stages train \
  --training_config conf/training.yaml
python run.py --stages infer \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml
python run.py --stages measure \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

See [mini_an4](../../mini_an4/esp2_slu/README.md) for a small runnable example and [SLURP](../../slurp/esp2_slu/README.md) for a corpus recipe.
