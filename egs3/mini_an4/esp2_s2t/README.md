# mini_an4 esp2_s2t recipe

A CI fixture for the ESPnet3 serialized output training (SOT) pipeline, not a
result-producing recipe. It runs every stage the `esp2_s2t` system has, end to
end, with Whisper `tiny` trained for one step. For the full recipe, see
[AMI](../../ami/esp2_s2t/README.md).

## Data Preparation

The recipe uses the bundled AN4 archive, `downloads.tar.gz`; no separate
download is needed. AN4 is single-speaker, so `create_dataset` makes
multi-talker groups: it concatenates utterances from different speakers into
one recording and writes a Lhotse cut over it with one supervision per
utterance, the shape an AMI utterance group has. The serialized reference goes
on the cut as `sot_text`, and the Whisper token list is written beside the
splits.

| Variable | Meaning | Default |
| --- | --- | --- |
| `MINI_AN4_SOT_DATA_ROOT` | Root the built splits and the token list go under | the recipe directory |
| `MINI_AN4_SOT_SPEAKER_CHANGE_SYMBOL` | Speaker-change symbol for the corpus, training, inference and metrics | `????` |

## Commands

Run from this directory. The commands build the corpus, train for one step,
decode the test groups, and score them with cpWER, DER, speaker-counting
accuracy and WER:

```bash
source path.sh
python run.py --stages create_dataset collect_stats \
  --training_config conf/training.yaml
python run.py --stages train infer measure \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

`ci/test_integration_espnet3.sh` runs the same stages in one invocation. The
one-step model only checks the stage and metric wiring: its scores are not
meaningful, and no pretrained model is provided.
