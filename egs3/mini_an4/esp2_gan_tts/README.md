# mini_an4 esp2_gan_tts recipe

A tiny end-to-end GAN-TTS fixture for the ESPnet3 integration test: `ci/test_integration_espnet3.sh` drives a one-batch VITS run through `create_dataset`, `compute_xvectors`, `remove_long_short`, `create_token_list`, `collect_stats`, `train`, `infer` and `measure`.
The model is the debug-sized VITS of `egs2/mini_an4/tts1`, and its scores are meaningless.

## Data Preparation

The recipe uses the bundled `downloads.tar.gz` archive of `egs3/mini_an4/esp2_asr`; `create_dataset` extracts it and builds the train, validation and test manifests.
No corpus download is needed.
The dataset here adds the speaker embedding `compute_xvectors` wrote for each utterance.
The model trains at AN4's own 16 kHz on character tokens, so no phonemizer is needed.

## Commands

Run from this directory:

```bash
. ./path.sh
python run.py \
    --stages create_dataset compute_xvectors remove_long_short \
             create_token_list collect_stats train infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`compute_xvectors` downloads the template's default speaker model; nothing else is downloaded.
No pretrained model is provided.
