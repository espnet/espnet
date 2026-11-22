# mini_an4 esp2_codec recipe

A tiny end-to-end codec fixture for the ESPnet3 integration test: `ci/test_integration_espnet3.sh` drives a one-batch EnCodec run through `create_dataset`, `collect_stats`, `train`, `infer` and `measure`.
Its model is deliberately small and its scores are not benchmark results.

## Data Preparation

The recipe uses the bundled `downloads.tar.gz` archive of `egs3/mini_an4/esp2_asr`; `create_dataset` extracts it and builds the train, validation and test manifests.
No separate download is needed.
The dataset here returns only the waveform.

## Commands

Run from this directory:

```bash
. ./path.sh
python run.py --stages create_dataset collect_stats train infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

No pretrained model is provided.
