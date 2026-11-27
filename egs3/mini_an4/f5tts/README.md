# mini_an4 f5tts recipe

A tiny end-to-end F5-TTS fixture for the ESPnet3 integration test: `ci/test_integration_espnet3.sh` drives a one-batch run through `create_dataset`, `remove_long_short`, `create_token_list`, `collect_stats`, `train`, `infer` and `measure`.
Its model is deliberately small and its scores are meaningless.

## Data Preparation

The recipe uses the bundled `downloads.tar.gz` archive of `egs3/mini_an4/esp2_asr`; `create_dataset` extracts it and builds the train, validation and test manifests.
No corpus download is needed.
The dataset here resamples the 16 kHz recordings to the model's 24 kHz and, for inference, pairs every test utterance with another one as its voice prompt.

## Commands

Run from this directory:

```bash
. ./path.sh
python run.py \
    --stages create_dataset remove_long_short create_token_list collect_stats \
             train infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`infer` downloads the default Vocos vocoder; nothing else is downloaded.
No pretrained model is provided.
