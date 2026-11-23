# AMI SOT multi-talker ASR recipe

This recipe trains and evaluates multi-talker ASR on the AMI SDM data with
serialized output training (SOT). It fine-tunes Whisper small with the
`esp2_s2t` system. The model transcribes each utterance group as one sequence
and inserts a speaker-change symbol between speakers, so it does not need a
separate speech separation step. The recipe reports four metrics for each
utterance group: cpWER, DER, speaker-counting accuracy and WER.

## Data Preparation

The `create_dataset` stage downloads the Lhotse CutSet manifests from the
Hugging Face repository set by `builder.cutset_repo` in `dataset/config.yaml`.
It also downloads the AMI SDM audio with `lhotse.recipes.ami.download_audio`.
It then writes one CutSet for each split to `data/train`, `data/dev` and
`data/test`. Each cut stores the recording, channel, start time and duration of
its audio, so the recipe does not copy the audio into separate files. Set
`AMI_SOT_DATA_ROOT` to build the corpus in another directory or to reuse a
corpus that you have already prepared.

Install the packages that the system code (`espnet3/systems/esp2_s2t/`) needs:

```bash
pip install "espnet[s2t]"
```

`create_dataset` also normalizes the transcripts with the CHiME-8 English text
normalizer, with filler removal turned off. The released checkpoint was trained
on text normalized in this way, so the recipe applies the same normalization to
the training text and to the test references. The normalizer comes from
chime-utils, which is not on PyPI. Install it with:

```bash
pip install git+https://github.com/chimechallenge/chime-utils
```

To build the corpus without this normalizer, set `text_norm: none` for each
split under `builder.sot` in `dataset/config.yaml`. The training text and the
test references then differ from the text that the released checkpoint was
trained and evaluated on.

The recipe reads the following environment variables. All of them are optional.

| Variable | Description | Default |
| --- | --- | --- |
| `AMI_SOT_DATA_ROOT` | Directory where the corpus is built and read | The recipe directory |
| `AMI_SOT_CUTSET_DIR` | Directory for the downloaded CutSet manifests | `downloads/cutsets` |
| `AMI_AUDIO_ROOT` | AMI SDM audio, in the `wav_db/<meeting>/audio` layout used by Lhotse | `downloads/ami` |
| `AMI_SOT_SPEAKER_CHANGE_SYMBOL` | Speaker-change symbol used by data preparation, the token list, training, inference and the metrics | `????` |
| `AMI_SOT_TRAIN_CONFIG` | `config.yaml` of the checkpoint to decode | `exp/<tag>/config.yaml` |
| `AMI_SOT_MODEL_FILE` | Model file of the checkpoint to decode | `exp/<tag>/valid.loss.ave_1best.pth` |

The default symbol `????` is a single token in the Whisper vocabulary. If you
use a symbol that is not in the vocabulary, the recipe adds it to the end of
the token list. The training config already sets
`decoder_conf.load_origin_token_embedding: true`, which keeps the pretrained
token embeddings in that case.

## Commands

### Whisper small

Run the following commands from `egs3/ami/esp2_s2t/`.

```bash
source path.sh

# 1) Build the corpus and the token list
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Collect feature statistics
python run.py --stages collect_stats --training_config conf/training.yaml

# 3) Train
python run.py --stages train --training_config conf/training.yaml

# 4) Decode and score
python run.py --stages infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml

# 5) Pack the trained model
python run.py --stages pack_model \
    --training_config conf/training.yaml \
    --publication_config conf/publication.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

The `infer` stage writes the hypotheses to `exp/<tag>/inference/test/hyp.scp`,
with one line for each utterance group. Each hypothesis keeps its Whisper
timestamps and speaker-change symbols. The `measure` stage reads the references
from the test set (`dataset:text`) and writes `metrics.json` to the same
directory. To decode the released checkpoint instead of your own model, set
`AMI_SOT_TRAIN_CONFIG` and `AMI_SOT_MODEL_FILE` to its `config.yaml` and
`model.pth` before step 4.

## Results

### Whisper small

- Training config: `conf/training.yaml`
- Inference config: `conf/inference.yaml`
- Metrics config: `conf/metrics.yaml`

The results are on the AMI SDM test set. Decoding used `conf/inference.yaml`
with beam size 5, CTC weight 0.0, and the Whisper timestamp rules applied
within each speaker block. Scoring used `conf/metrics.yaml` with the
`whisper_en` text cleaner and a DER collar of 0.25 s. The model is the released
checkpoint listed below, decoded and scored with this recipe. This checkpoint
was trained before this recipe was written, and training with
`conf/training.yaml` has not yet been verified to reproduce these numbers.

| Metric | Value (%) |
| --- | ---: |
| cpWER | 27.61 |
| DER | 8.54 |
| Speaker-counting accuracy | 84.09 |
| WER | 27.06 |

Results by the number of speakers in the reference:

| Metric | 1 speaker | 2 speakers | 3 speakers | 4 speakers |
| --- | ---: | ---: | ---: | ---: |
| cpWER | 14.21 | 26.29 | 42.53 | 56.38 |
| DER | 1.15 | 6.71 | 20.17 | 28.92 |
| Speaker-counting accuracy | 96.37 | 70.10 | 40.80 | 12.35 |
| Utterance groups | 3962 | 1234 | 522 | 162 |

The overall results also include 247 utterance groups whose references contain
no words after text cleaning. All metrics score each utterance group
separately. Therefore, they cannot be compared with session-level results, in
which one speaker mapping covers a whole meeting.

- cpWER uses `cp_word_error_rate` from meeteval. It finds the best speaker
  mapping for each utterance group and sums the errors over all groups.
- DER uses `DiarizationErrorRate` from pyannote.metrics on the segments given
  by the timestamps in the text. Each group is scored from time 0 to the end
  of its reference.
- Speaker-counting accuracy is the percentage of utterance groups in which the
  hypothesis has the same number of non-empty speaker blocks as the reference.
- WER compares the reference with the hypothesis without a speaker mapping. The
  reference speakers are ordered by start time, and the hypothesis speakers
  stay in the order that the model produced them. WER is not a standard metric
  for multi-talker ASR, and cpWER is the main metric of this recipe. WER is
  reported next to cpWER because it also counts errors in the speaker order. It
  does not count words assigned to the wrong speaker, so it can be lower than
  cpWER.

#### Pretrained model

- [`espnet/multi-talker-whisper-small-ami`](https://huggingface.co/espnet/multi-talker-whisper-small-ami)
  contains the checkpoint used above and the CutSet manifests that
  `create_dataset` downloads.
