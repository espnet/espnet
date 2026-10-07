# LibriTTS F5-TTS recipe

`create_dataset` downloads LibriTTS (OpenSLR 60, the training corpus) and
LibriSpeech `test-clean` (OpenSLR 12) with the F5-TTS cross-sentence pair
list, the evaluation set of the F5-TTS paper (1127 same-speaker prompt/target
pairs); `conf/metrics.yaml` scores it with WER, speaker similarity and UTMOS
through VERSA.
`conf/training.yaml` is the F5-TTS Small configuration (158M parameters,
character tokens).

## Quick start

```bash
# 1) Download the corpora and build every manifest (run once), filter by
#    duration, build the token list, collect feature shapes
python run.py --stages create_dataset remove_long_short create_token_list collect_stats \
    --training_config conf/training.yaml

# 2) Train
python run.py --stages train --training_config conf/training.yaml

# 3) Synthesize the LibriSpeech-PC pairs
python run.py --stages infer \
    --training_config conf/training.yaml --inference_config conf/inference.yaml

# 4) Score
python run.py --stages measure \
    --training_config conf/training.yaml --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml

# 5) Pack and upload the model, then the demo
python run.py --stages pack_model upload_model \
    --training_config conf/training.yaml --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml --publication_config conf/publication.yaml
python run.py --stages pack_demo upload_demo \
    --training_config conf/training.yaml --demo_config conf/demo.yaml
```

See `egs3/TEMPLATE/f5tts/README.md` for the stages, `measure`'s dependencies
and loading a packed model.
