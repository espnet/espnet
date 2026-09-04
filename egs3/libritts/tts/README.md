# ESPnet3 LibriTTS VITS recipe

Multi-speaker English TTS on LibriTTS using VITS with x-vector speaker
conditioning.

## Quick start

```bash
# 0) Edit configs to set paths.

# 1) Download LibriTTS and build per-split TSV manifests (run once)
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Extract x-vector speaker embeddings (one .pt file per utterance)
python run.py --stages compute_xvectors --training_config conf/training.yaml

# 3) Filter utterances by duration
python run.py --stages remove_long_short --training_config conf/training.yaml

# 4) Build the phoneme token list
python run.py --stages create_token_list --training_config conf/training.yaml

# 5) Collect feature statistics (resumable: set collect_stats.num_shards>1)
python run.py --stages collect_stats --training_config conf/training.yaml

# 6) Train VITS
python run.py --stages train --training_config conf/training.yaml

# 7) Synthesize from test text
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 8) Compute the metrics
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml

# 9) Bundle the trained model for release (packs the phoneme token list too)
python run.py --stages pack_model \
    --training_config conf/training.yaml \
    --publication_config conf/publication.yaml

# 10) Upload it to the Hugging Face Hub (run `hf auth login` first)
python run.py --stages upload_model \
    --training_config conf/training.yaml \
    --publication_config conf/publication.yaml

# 11) Pack the Gradio demo, then upload it as a Space
python run.py --stages pack_demo   --demo_config conf/demo.yaml
python run.py --stages upload_demo --demo_config conf/demo.yaml
```

## Demo

`pack_demo` builds a Gradio app from `src/app.py`. Because this recipe trains a
multi-speaker VITS, the demo takes **text plus a reference audio clip**: it runs
the same SpeechBrain ECAPA extractor as the `compute_xvectors` stage to turn
that clip into the `spembs` the model needs, then synthesizes the text in that
voice.

`src/app.py` defaults to the same ECAPA model the `compute_xvectors` stage
uses, so `conf/demo.yaml` carries no x-vector settings. If you retrain against
a different embedding model, override `xvector.pretrained_model` in
`conf/demo.yaml` to match, or the embedding the demo builds will not match the
space the model was trained in.
