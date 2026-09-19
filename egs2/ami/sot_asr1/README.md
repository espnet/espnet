# AMI SOT multi-talker ASR with Whisper

Serialized Output Training (SOT) recipe for multi-talker speech recognition
on the AMI meeting corpus, using OpenAI Whisper as the encoder/decoder
backbone.

Each utterance group contains one or more overlapping speakers; the model
emits a single transcript with all speakers concatenated in FIFO order
(speaker-start-time), separated by a speaker-change token. Results are reported
as **utterance-group cpWER** (concatenated minimum-permutation WER),
**utterance-group DER** (diarization error rate), and **speaker-counting
accuracy**.

## Setup

Run ESPnet `tools/installers/install_whisper.sh` for the
[`openai-whisper`](https://github.com/espnet/whisper) dependency used at decode
time and for text normalization. Scoring additionally uses SCTK `md-eval.pl`
(`tools/installers/install_sctk.sh`, part of a standard tools build) for DER,
plus `scipy` and `editdistance` (already ESPnet dependencies). No other
packages are required.

## Data preparation

`local/prepare_sot.py` reads source manifests for the AMI utterance-group
splits and writes Kaldi-format data directories. Each utterance group must
expose one or more time-aligned supervisions per speaker. The speaker-change
symbol that separates consecutive speakers must match the
`speaker_change_symbol` in the training config (default `????`, a single
base-vocabulary Whisper BPE token). `local/decode.py` rewrites it to `<sc>` in
its output, and the scorers accept either spelling.

```bash
python local/prepare_sot.py \
    --cutset_paths /path/to/ami_train_manifest \
    --output_dir data/train \
    --use_timestamps true \
    --speaker_change_symbol "????"

python local/prepare_sot.py \
    --cutset_paths /path/to/ami_dev_manifest \
    --output_dir data/dev \
    --use_timestamps true \
    --speaker_change_symbol "????"

python local/prepare_sot.py \
    --cutset_paths /path/to/ami_test_manifest \
    --output_dir data/test \
    --use_timestamps true \
    --speaker_change_symbol "????"
```

The resulting `text` file has one line per utterance group, with consecutive
speakers separated by the speaker-change symbol and per-speaker timestamps
preserved inline.

## Training

Training is driven by `run.sh`, which wraps the standard ESPnet `asr.sh`
pipeline for training only (it passes `--skip_eval`, so the stock decoding and
scoring stages do not run). The default config trains Whisper-small with
`preprocessor: multi` and predicts timestamps:

```bash
# End-to-end (data prep already done)
./run.sh --stage 11 --stop_stage 11   # train
```

Inference is done separately against a trained checkpoint (see below), decoded
with openai-whisper via `local/decode.py`. This SOT model is decoded with
openai-whisper's `transcribe()` pipeline, which provides temperature fallback,
compression-ratio and average-log-prob quality gating (with retry), no-speech
gating, and Whisper's timestamp-pairing rules together with a SOT-aware patch.
The patch scopes the timestamp-pairing rules to the current speaker block, so
the speaker-change token can appear between blocks, and it biases the decoder
toward continuing the current speaker when its timestamp confidence is high.

## Inference and evaluation

A trained checkpoint is `model.pth` + `config.yaml` + `token_list.txt`,
either produced by training above or downloaded from a public release.
A Whisper-small checkpoint is available on the Hugging Face Hub at
[`espnet/multi-talker-whisper-small-ami`](https://huggingface.co/espnet/multi-talker-whisper-small-ami):

```bash
huggingface-cli download espnet/multi-talker-whisper-small-ami \
    --local-dir exp/whisper-sot-small-ami
```

To decode the prepared test set against a trained checkpoint, pass
`--inference_model <dir>` to `run.sh`:

```bash
./run.sh --inference_model exp/whisper-sot-small-ami \
         --whisper_model small \
         --decode_test_sets test
```

Hypotheses are written to `<dir>/decode_inference/<test_set>/1best_recog/`:
`text` (per-speaker text with `<sc>` separators, for cpWER) and `text_sot`
(same content with inline Whisper timestamps, for DER).

Scoring runs automatically after decoding (pass `--no_score` to skip). Per
test set it writes `<dir>/decode_inference/<test_set>/scoring/`:

- `cpwer.json`, `cpwer_by_num_speakers.json`: utterance-group cpWER
- `speaker_count.json`: speaker-counting accuracy and a confusion table
- `der.json`, `der_by_num_speakers.json`: utterance-group DER

To score an existing decode directory on its own:

```bash
local/score_sot.sh <decode_dir> data/<test_set> <out_dir> whisper_en 0.25
```

## Results (AMI SDM test)

Decoding uses default settings (`temperature=0.0`, `beam_size=5`, `fp16`).
All scoring uses dependencies already required by ESPnet, so no extra install
is needed:

- **cpWER** (`local/evaluate_sot.py`): utterance-group concatenated
  minimum-permutation WER. Each group is scored with its own optimal speaker
  permutation (Hungarian assignment via `scipy.optimize.linear_sum_assignment`
  over per-speaker word errors from `editdistance`). Text is normalized with
  `espnet2.text.cleaner.TextCleaner` (`whisper_en`, i.e. openai-whisper's
  `EnglishTextNormalizer`).
- **DER** (`local/score_der.py`): utterance-group diarization error rate from
  the model's own inline `<|t|>` timestamps, scored with SCTK `md-eval.pl`
  (collar 0.25 s), the tool used by ESPnet diarization recipes.
- **Speaker-counting accuracy** (`local/evaluate_sot.py`): fraction of
  utterance groups whose predicted speaker-block count matches the reference.

### cpWER (utterance-group, %)

| Model         | overall | 1-spk | 2-spk | 3-spk | 4-spk |
|---------------|--------:|------:|------:|------:|------:|
| Whisper-small |   27.95 | 14.97 | 26.97 | 41.95 | 55.87 |

### DER (utterance-group, collar = 0.25 s, %)

| Model         | overall | 1-spk | 2-spk | 3-spk | 4-spk |
|---------------|--------:|------:|------:|------:|------:|
| Whisper-small |    8.33 |  0.89 |  5.77 | 17.11 | 27.39 |

### Speaker-counting accuracy (%)

| Model         | overall | 1-spk | 2-spk | 3-spk | 4-spk |
|---------------|--------:|------:|------:|------:|------:|
| Whisper-small |   84.61 | 95.96 | 71.54 | 44.72 | 18.40 |
