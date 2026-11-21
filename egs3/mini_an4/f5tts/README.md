# mini_an4 F5-TTS integration test

A one-batch F5-TTS run on the AN4 sample that `ci/test_integration_espnet3.sh`
drives through `create_dataset`, `remove_long_short`, `create_token_list`,
`collect_stats`, `train`, `infer` and `measure`. The data preparation is the
one of `egs3/mini_an4/esp2_asr`; the dataset here resamples the 16 kHz
recordings to the model's 24 kHz and, for inference, pairs every test
utterance with another one as its voice prompt. `infer` downloads the default
Vocos vocoder; nothing else is downloaded, and the scores are meaningless.

```bash
python run.py \
    --stages create_dataset remove_long_short create_token_list collect_stats \
             train infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```
