# mini_an4 GAN-TTS integration test

A one-batch VITS run on the AN4 sample that `ci/test_integration_espnet3.sh`
drives through `create_dataset`, `compute_xvectors`, `remove_long_short`,
`create_token_list`, `collect_stats`, `train`, `infer` and `measure`. The data
preparation is the one of `egs3/mini_an4/esp2_asr`; the dataset here adds the
speaker embedding `compute_xvectors` wrote for each utterance. The model is
the debug-sized VITS of `egs2/mini_an4/tts1`, trained at AN4's own 16 kHz on
character tokens, so no phonemizer is needed. `compute_xvectors` downloads the
template's default speaker model; nothing else is downloaded, and the scores
are meaningless.

```bash
python run.py \
    --stages create_dataset compute_xvectors remove_long_short \
             create_token_list collect_stats train infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```
