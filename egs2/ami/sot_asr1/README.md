# AMI SOT multi-talker ASR with Whisper

Serialized Output Training (SOT) recipe for multi-talker speech recognition
on the AMI meeting corpus, using OpenAI Whisper as the encoder/decoder
backbone.

Each utterance group contains one or more overlapping speakers; the model
emits a single transcript with all speakers concatenated in FIFO order
(speaker-start-time) and separated by a `<sc>` token. Results are reported
as **utterance-group cpWER** (concatenated minimum-permutation WER) and
**utterance-group DER** (diarization error rate).

## Setup

ESPnet `tools/installers/install_whisper.sh` must have been run; this installs
the [`openai-whisper`](https://github.com/espnet/whisper) dependency that the
recipe relies on at decode time.

## Data preparation

`local/prepare_sot.py` reads source manifests for the AMI utterance-group
splits and writes Kaldi-format data directories. Each utterance group must
expose one or more time-aligned supervisions per speaker. `<sep>` is the
speaker-change symbol that separates consecutive speakers; it must match the
`speaker_change_symbol` set in the training config.

```bash
python local/prepare_sot.py \
    --cutset_paths /path/to/ami_train_manifest \
    --output_dir data/train \
    --use_timestamps true \
    --speaker_change_symbol "<sep>"

python local/prepare_sot.py \
    --cutset_paths /path/to/ami_dev_manifest \
    --output_dir data/dev \
    --use_timestamps true \
    --speaker_change_symbol "<sep>"

python local/prepare_sot.py \
    --cutset_paths /path/to/ami_test_manifest \
    --output_dir data/test \
    --use_timestamps true \
    --speaker_change_symbol "<sep>"
```

The resulting `text` file has one line per utterance group, with consecutive
speakers separated by the speaker-change symbol and per-speaker timestamps
preserved inline.

## Training

Training is driven by `run.sh`, which wraps the standard ESPnet `asr.sh`
pipeline. The default config trains Whisper-small with `preprocessor: multi`
and predicts timestamps:

```bash
# End-to-end (data prep already done)
./run.sh --stage 11 --stop_stage 11   # train
./run.sh --stage 12 --stop_stage 12   # decode (uses stock asr.sh inference)
```

## Inference and evaluation

A checkpoint bundle is `model.pth` + `config.yaml` + `token_list.txt`,
either produced by training above or downloaded from a public release.
A Whisper-small checkpoint is available on the Hugging Face Hub at
[`espnet/multi-talker-whisper-small-ami`](https://huggingface.co/espnet/multi-talker-whisper-small-ami):

```bash
huggingface-cli download espnet/multi-talker-whisper-small-ami \
    --local-dir exp/whisper-sot-small-ami
```

To decode the prepared test set against a checkpoint bundle, pass
`--inference_model <dir>` to `run.sh`:

```bash
./run.sh --inference_model exp/whisper-sot-small-ami \
         --whisper_model small \
         --decode_test_sets test
```

Hypotheses are written to `<dir>/decode_inference/<test_set>/1best_recog/`:
`text` (per-speaker text with `<sc>` separators, for cpWER) and `text_sot`
(same content with inline Whisper timestamps, for DER).

## Results (AMI SDM test)

Decoding uses default settings (`temperature=0.0`, `beam_size=5`, `fp16`).
cpWER is computed with [meeteval](https://github.com/fgnt/meeteval)'s
`cp_word_error_rate_multifile` after normalization via
`whisper.normalizers.EnglishTextNormalizer`. DER is computed with
[pyannote.metrics](https://github.com/pyannote/pyannote-metrics)'
`DiarizationErrorRate(collar=0.25)`.

### cpWER (utterance-group, %)

| Model                | overall | 1-spk | 2-spk | 3-spk | 4-spk |
|----------------------|--------:|------:|------:|------:|------:|
| Whisper-small        |   27.95 | 15.36 | 25.54 | 38.94 | 52.44 |

### DER (utterance-group, collar = 0.25 s, %)

| Model                | overall | 1-spk | 2-spk | 3-spk | 4-spk |
|----------------------|--------:|------:|------:|------:|------:|
| Whisper-small        |    9.84 |  1.47 |  6.99 | 18.65 | 29.43 |
