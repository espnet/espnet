# mini_an4 esp2_s2t recipe

This recipe is a small CI test for the ESPnet3 serialized output training (SOT)
pipeline. It runs every stage of the `esp2_s2t` system with Whisper `tiny`,
which is trained for one step. For the full recipe, see
[AMI](../../ami/esp2_s2t/README.md).

## Data Preparation

The recipe uses the AN4 archive `downloads.tar.gz` in this directory, so no
download is needed. Each AN4 utterance has only one speaker, so
`create_dataset` joins two AN4 utterances into one recording and writes a
Lhotse cut for it, with one supervision for each utterance. This matches the
format of an AMI utterance group. The two utterances can come from the same
speaker, so not every group has two speakers. The serialized reference is
stored in the cut as `sot_text`, and the Whisper token list is written next to
the data splits.

| Variable | Description | Default |
| --- | --- | --- |
| `MINI_AN4_SOT_DATA_ROOT` | Directory where the archive is extracted and the data splits and the token list are written. It must contain `downloads.tar.gz`. | The recipe directory |
| `MINI_AN4_SOT_SPEAKER_CHANGE_SYMBOL` | Speaker-change symbol used by data preparation, training, inference and the metrics | `????` |

## Commands

Run the following commands from this directory. They build the data, train the
model for one step, decode the test groups, and compute cpWER, DER,
speaker-counting accuracy and WER.

```bash
source path.sh
python run.py --stages create_dataset collect_stats \
  --training_config conf/training.yaml
python run.py --stages train infer measure \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

`ci/test_integration_espnet3.sh` runs the same stages in one command. Because
the model is trained for only one step, the test only checks that the stages
and the metrics run. The scores are not meaningful, and no pretrained model is
provided.
