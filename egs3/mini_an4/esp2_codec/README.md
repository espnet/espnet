# mini_an4 codec integration test

A one-batch EnCodec run on the AN4 sample that `ci/test_integration_espnet3.sh`
drives through `create_dataset`, `collect_stats`, `train`, `infer` and
`measure`. The data preparation is the one of `egs3/mini_an4/esp2_asr`; the
dataset here returns only the waveform.

```bash
python run.py --stages create_dataset collect_stats train infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```
