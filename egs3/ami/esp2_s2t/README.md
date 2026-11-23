# AMI SOT multi-talker ASR recipe

Serialized Output Training (SOT) multi-talker ASR on AMI SDM, with Whisper
small and the `esp2_s2t` system. One utterance group is transcribed as a single
sequence, with the speakers separated inline, so overlapping speech is handled
without a separation front end. The recipe reports utterance-group cpWER, DER,
speaker-counting accuracy and WER.

## Data Preparation

`create_dataset` fetches the Lhotse CutSet manifests from the Hugging Face repo
named by `builder.cutset_repo` in `dataset/config.yaml`, downloads the AMI SDM
audio with `lhotse.recipes.ami.download_audio`, and writes one CutSet per
split under `data/{train,dev,test}`. A cut names the offset, duration and
channel its audio comes from, so no per-group wav is written. Set
`AMI_SOT_DATA_ROOT` to build somewhere other than the recipe directory, or to
reuse a prepared corpus.

The system code (`espnet3/systems/esp2_s2t/`) needs:

```bash
pip install "espnet[s2t]"
```

`create_dataset` also normalizes each segment's transcript with the CHiME-8
English text normalizer, with filler removal turned off. The released
checkpoint was trained on text normalized this way, so the recipe applies it
to the training text and to the test references. It comes from chime-utils,
which is not on PyPI:

```bash
pip install git+https://github.com/chimechallenge/chime-utils
```

Set `text_norm: none` in `dataset/config.yaml` to build without it. The text
then differs from the text the released checkpoint was trained on, and the
test references change with it.

The recipe reads these environment variables, all optional:

| Variable | Meaning | Default |
| --- | --- | --- |
| `AMI_SOT_DATA_ROOT` | Corpus root the builder writes and the stages read | the recipe directory |
| `AMI_SOT_CUTSET_DIR` | Where the CutSet manifests are fetched to | `downloads/cutsets` |
| `AMI_AUDIO_ROOT` | AMI SDM audio, in lhotse's `wav_db/<meeting>/audio` layout | `downloads/ami` |
| `AMI_SOT_SPEAKER_CHANGE_SYMBOL` | Speaker-change symbol for the corpus, the token list, training, inference and metrics | `????` |
| `AMI_SOT_TRAIN_CONFIG` | The `config.yaml` of the checkpoint `infer` decodes | `exp/<tag>/config.yaml` |
| `AMI_SOT_MODEL_FILE` | The weights of that checkpoint | `exp/<tag>/valid.loss.ave_1best.pth` |

`????` is a single token in the Whisper vocabulary. A symbol the vocabulary
lacks is appended to the token list, which then needs
`decoder_conf.load_origin_token_embedding: true` in the training config.

## Commands

### Whisper small

Run from `egs3/ami/esp2_s2t/`:

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

`infer` writes `exp/<tag>/inference/test/hyp.scp`, one row per utterance
group, keeping the inline Whisper timestamps and the speaker-change symbol.
`measure` reads the reference from the test set itself (`dataset:text`) and
writes `metrics.json` beside `hyp.scp`. To decode the released checkpoint
instead of a trained one, point `AMI_SOT_TRAIN_CONFIG` and
`AMI_SOT_MODEL_FILE` at its `config.yaml` and `model.pth` before step 4.

## Results

### Whisper small

- Training config: `conf/training.yaml`
- Inference config: `conf/inference.yaml`
- Metrics config: `conf/metrics.yaml`

AMI SDM test, 6127 utterance groups, decoded with `conf/inference.yaml`
(beam 5, `ctc_weight` 0.0, Whisper timestamp rules scoped to the current
speaker) and scored with `conf/metrics.yaml` (`whisper_en` text cleaner, DER
collar 0.25 s). The checkpoint is the released one below, decoded and scored
by this recipe. It was trained before this recipe existed; training with
`conf/training.yaml` has not been verified to reproduce these numbers.

| Metric | Value (%) |
| --- | ---: |
| cpWER | 27.61 |
| DER | 8.54 |
| Speaker-counting accuracy | 84.09 |
| WER | 27.06 |

By number of reference speakers:

| Metric | 1 speaker | 2 speakers | 3 speakers | 4 speakers |
| --- | ---: | ---: | ---: | ---: |
| cpWER | 14.21 | 26.29 | 42.53 | 56.38 |
| DER | 1.15 | 6.71 | 20.17 | 28.92 |
| Speaker-counting accuracy | 96.37 | 70.10 | 40.80 | 12.35 |
| Utterance groups | 3962 | 1234 | 522 | 162 |

The overall numbers also include the 247 groups whose reference cleans to no
words. Every metric scores each utterance group on its own, so none is
comparable with session-level numbers, where one speaker assignment has to
serve a whole meeting.

- cpWER is meeteval's `cp_word_error_rate`, with the speaker permutation
  chosen per group and the errors summed over groups.
- DER is pyannote's `DiarizationErrorRate` over the segments the inline
  timestamps describe, scored from 0 to the end of each group's reference.
- Speaker-counting accuracy is the share of groups whose hypothesis has as
  many non-empty speaker blocks as the reference.
- WER compares the reference speakers in FIFO order (by start time) with the
  hypothesis speakers in the order the model emitted them, without a
  permutation. It does not count words given to the wrong speaker, so it can
  be lower than cpWER.

#### Pretrained model

- [`espnet/multi-talker-whisper-small-ami`](https://huggingface.co/espnet/multi-talker-whisper-small-ami)
  holds the checkpoint above and the CutSet manifests `create_dataset` uses.
