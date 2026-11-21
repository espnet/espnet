# mini_an4 LACE integration test

A one-batch LACE run on the AN4 sample that `ci/test_integration_espnet3.sh`
drives through `create_dataset`, `collect_stats`, `train`, `infer` and
`measure`. The data preparation is the one of `egs3/mini_an4/esp2_asr`; the
dataset here returns only the waveform, under `audio`. The base codec is a
small untrained EnCodec built from `codec` + `codec_conf`, so no pretrained
model is downloaded.

```bash
python run.py --stages create_dataset collect_stats train infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```
