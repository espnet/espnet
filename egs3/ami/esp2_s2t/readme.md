# AMI SOT multi-talker ASR recipe

Serialized Output Training (SOT) multi-talker ASR on AMI SDM, with Whisper
small under `espnet2.tasks.s2t.S2TTask`. One utterance group is transcribed as
a single sequence, with the speakers separated inline, so overlapping speech is
handled without a separation front end. Scored with utterance-group
cpWER and DER.

`create_dataset` fetches the Lhotse CutSet manifests from the Hugging Face repo
named by `builder.cutset_repo` in `dataset/config.yaml`, downloads the AMI SDM
audio with [`egs2/ami/asr1/local/ami_download.sh`](../../../egs2/ami/asr1/local/ami_download.sh),
and writes `data/{train,dev,test}`. Set `AMI_SOT_DATA_ROOT` to build somewhere
other than the recipe directory, or to reuse a prepared corpus.

Beyond ESPnet's own requirements the recipe needs `openai-whisper` (the
tokenizer and the pretrained encoder and decoder), `lhotse` (reading the
CutSet manifests in `create_dataset`) and SCTK's `md-eval.pl` for the DER
metric, which `tools/installers/install_sctk.sh` builds.

`create_dataset` normalizes the transcript with the CHiME-8 English text
normalizer, which is not on PyPI:

```bash
pip install git+https://github.com/chimechallenge/chime-utils@main
```

Set `text_norm: none` in `dataset/config.yaml` to build without it.

Every knob the recipe reads from the environment, all optional:

| Variable | Meaning | Default |
| --- | --- | --- |
| `AMI_SOT_DATA_ROOT` | Corpus root the builder writes and the stages read | the recipe directory |
| `AMI_SOT_CUTSET_DIR` | Where the CutSet manifests are fetched to | `downloads/cutsets` |
| `AMI_AUDIO_ROOT` | AMI SDM audio, in `ami_download.sh`'s layout | `downloads/ami` |
| `AMI_SOT_CHECKPOINT_DIR` | Directory holding `config.yaml` and `model.pth` for `infer` | `exp/whisper_sot_s2t` |
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

Run each stage as its own command; `collect_stats` and `train` must not be
combined into one invocation.

`infer` writes `exp/<tag>/inference/test/{hyp,ref,hyp_sot,ref_sot}.scp`, one row
per utterance group. `hyp`/`ref` are the timestamp-free view scored by
`ug_cpWER`;
`hyp_sot`/`ref_sot` keep the inline Whisper timestamps and are scored by
`ug_DER`.
`measure` writes `metrics.json` beside them, with the scores under
`ug_cpWER` and `ug_DER`.

## Pretrained model

[`espnet/multi-talker-whisper-small-ami`](https://huggingface.co/espnet/multi-talker-whisper-small-ami)
holds the checkpoint that produced the results below, and the cutsets
`create_dataset` uses. Fetch `config.yaml` and `model.pth` into
`exp/whisper_sot_s2t`, or point `AMI_SOT_CHECKPOINT_DIR` at wherever you put
them, then run stage 4.

## Results

AMI SDM test, 6127 utterance groups, beam size 5.

Both metrics score each utterance group on its own, so they are not
comparable with session-level numbers, where one speaker assignment has to
serve a whole meeting.

| Metric | Value |
| --- | --- |
| ug_cpWER | 27.61 % |
| ug_DER (0.25 s collar) | 8.54 % |

## Packaging

```bash
python run.py --stages pack_model \
    --training_config conf/training.yaml \
    --publication_config conf/publication.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

Run `infer` and `measure` first, so the bundle carries their `metrics.json`.
