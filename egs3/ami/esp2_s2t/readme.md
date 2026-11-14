# AMI SOT multi-talker ASR recipe

Serialized Output Training (SOT) multi-talker ASR on AMI SDM, with Whisper
small under `espnet2.tasks.s2t.S2TTask`. One utterance group is transcribed as
a single sequence, with the speakers separated inline, so overlapping speech is
handled without a separation front end. Scored with utterance-group
cpWER and DER.

`create_dataset` fetches the Lhotse CutSet manifests from the Hugging Face repo
named by `builder.cutset_repo` in `dataset/config.yaml`, downloads the AMI SDM
audio with `lhotse.recipes.ami.download_audio`, and writes one CutSet per
split under `data/{train,dev,test}`. A cut names the offset, duration and
channel its audio comes from, so no per-group wav is written. Set
`AMI_SOT_DATA_ROOT` to build somewhere other than the recipe directory, or to
reuse a prepared corpus.

The model code lives in the `esp2_s2t` system
(`espnet3/systems/esp2_s2t/`): S2T inference with the Whisper timestamp rules,
the Whisper tokenizer and vocabulary exporter, and the two metrics. Everything
they import comes with:

```bash
pip install "espnet[s2t]"
```

`create_dataset` also normalizes the transcript with the CHiME-8 English text
normalizer, which is not on PyPI and so cannot be an extra:

```bash
cd tools && make chime_utils.done
```

Set `text_norm: none` in `dataset/config.yaml` to build without it.

Every knob the recipe reads from the environment, all optional:

| Variable | Meaning | Default |
| --- | --- | --- |
| `AMI_SOT_DATA_ROOT` | Corpus root the builder writes and the stages read | the recipe directory |
| `AMI_SOT_CUTSET_DIR` | Where the CutSet manifests are fetched to | `downloads/cutsets` |
| `AMI_AUDIO_ROOT` | AMI SDM audio, in lhotse's `wav_db/<meeting>/audio` layout | `downloads/ami` |
| `AMI_SOT_TRAIN_CONFIG` | The `config.yaml` of the checkpoint `infer` decodes | `exp/<tag>/config.yaml` |
| `AMI_SOT_MODEL_FILE` | The weights of that checkpoint | `exp/<tag>/valid.loss.ave_1best.pth` |
| `AMI_SOT_SPEAKER_CHANGE_SYMBOL` | Separator `infer` tells the model about | `????` |

## Quick start

```bash
# 1) Build the corpus
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
```

`infer` writes `exp/<tag>/inference/test/hyp.scp`, one row per utterance
group, keeping the inline Whisper timestamps and the speaker-change symbol.
`measure` reads the reference from the test set itself (`dataset:text`) and
writes `metrics.json` beside `hyp.scp`, with the scores under `ug_cpWER` and
`ug_DER`. cpWER strips the timestamps before comparing words; DER reads
speaker activity from them.

## Pretrained model

[`espnet/multi-talker-whisper-small-ami`](https://huggingface.co/espnet/multi-talker-whisper-small-ami)
holds the checkpoint that produced the results below, and the cutsets
`create_dataset` uses. Fetch `config.yaml` and `model.pth`, point
`AMI_SOT_TRAIN_CONFIG` and `AMI_SOT_MODEL_FILE` at them, then run stage 4.

## Results

AMI SDM test, 6127 utterance groups, beam size 5.

Both metrics score each utterance group on its own, so they are not
comparable with session-level numbers, where one speaker assignment has to
serve a whole meeting.

| Metric | Value |
| --- | --- |
| ug_cpWER | 27.61 % |
| ug_DER (0.25 s collar) | 8.54 % |

cpWER is meeteval's `cp_word_error_rate` over the speaker permutations of one
group; DER is pyannote's `DiarizationErrorRate` over the segments the inline
timestamps describe.

## Packaging

```bash
python run.py --stages pack_model \
    --training_config conf/training.yaml \
    --publication_config conf/publication.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

Run `infer` and `measure` first, so the bundle carries their `metrics.json`.
The bundle holds the checkpoint the train stage wrote, with its token list
written into `config.yaml`, so it decodes on a machine without the corpus.
