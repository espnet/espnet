# AMI SOT multi-talker ASR with Whisper

Serialized Output Training (SOT) recipe for multi-talker speech recognition
on the AMI meeting corpus, using OpenAI Whisper as the encoder/decoder
backbone.

Each utterance group contains one or more overlapping speakers; the model
emits a single transcript with all speakers concatenated in FIFO order
(speaker-start-time) and separated by a `<sc>` token. Both **utterance-group
cpWER** (concatenated minimum-permutation WER) and **utterance-group DER**
(diarization error rate) are reported.

## Setup

ESPnet `tools/installers/install_whisper.sh` must have been run; this installs
the [`openai-whisper`](https://github.com/espnet/whisper) and `pyannote.metrics`
dependencies that the recipe relies on at decode/eval time.

## Data preparation

`local/prepare_sot.py` reads source manifests for the AMI utterance-group
splits and writes Kaldi-format data directories. Each utterance group must
expose one or more time-aligned supervisions per speaker.

```bash
python local/prepare_sot.py \
    --cutset_paths /path/to/ami_train_manifest \
    --output_dir data/train \
    --use_timestamps true

python local/prepare_sot.py \
    --cutset_paths /path/to/ami_dev_manifest \
    --output_dir data/dev \
    --use_timestamps true

python local/prepare_sot.py \
    --cutset_paths /path/to/ami_test_manifest \
    --output_dir data/test \
    --use_timestamps true
```

The resulting `text` file has one line per utterance group, with consecutive
speakers separated by `<sc>` and per-speaker timestamps preserved inline.

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

To decode the prepared test set against a checkpoint bundle and compute
both cpWER and DER, pass `--inference_model <dir>` to `run.sh`:

```bash
./run.sh --inference_model exp/whisper-sot-small-ami \
         --whisper_model small \
         --decode_test_sets test
```

## Results (AMI SDM test)

Decoding uses default settings (`temperature=0.0`, `beam_size=5`, `fp16`).

### cpWER (utterance-group, %)

| Model                | overall | 1-spk | 2-spk | 3-spk | 4-spk |
|----------------------|--------:|------:|------:|------:|------:|
| Whisper-small        |   27.63 | 14.61 | 25.36 | 39.25 | 52.71 |

### DER (utterance-group, collar = 0.25 s, %)

| Model                | overall | 1-spk | 2-spk | 3-spk | 4-spk |
|----------------------|--------:|------:|------:|------:|------:|
| Whisper-small        |    9.84 |  1.47 |  6.99 | 18.65 | 29.43 |
