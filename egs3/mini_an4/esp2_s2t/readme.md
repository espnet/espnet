# Mini AN4 SOT recipe

The CI arm of the `esp2_s2t` system. It runs every stage the system has, end
to end, on a corpus small enough to build and train in seconds and with
nothing to download: the AN4 archive is committed beside this file.

AN4 is single-speaker, so there are no multi-talker groups to read.
`create_dataset` makes them: it concatenates utterances from different
speakers into one recording and writes a Lhotse cut over it carrying one
supervision per utterance, which is the shape an AMI utterance group already
has. The serialized reference goes on the cut as `sot_text`.

The model is Whisper `tiny` rather than `small`, and training stops after one
step. Nothing here is meant to learn: the scores `measure` reports are
whatever a one-step model produces.

For the real recipe, see [`egs3/ami/esp2_s2t`](../../ami/esp2_s2t).

## Quick start

```bash
python run.py --stages create_dataset --training_config conf/training.yaml
python run.py --stages collect_stats  --training_config conf/training.yaml
python run.py --stages train          --training_config conf/training.yaml
python run.py --stages infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`ci/test_integration_espnet3.sh` runs the same stages in one invocation.

| Variable | Meaning | Default |
| --- | --- | --- |
| `MINI_AN4_SOT_DATA_ROOT` | Root the built splits and the token list go under | the recipe directory |
