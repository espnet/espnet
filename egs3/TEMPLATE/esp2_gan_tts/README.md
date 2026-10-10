# ESPnet3 espnet2 GAN-TTS recipe template

Starter files for a recipe that trains an espnet2 GAN-based TTS model (VITS,
JETS, ...) through `espnet3.systems.esp2_gan_tts.system.GANTTSSystem`.
Copy `egs3/libritts/esp2_gan_tts/` when starting a new corpus; the configs
under `conf/` here are the defaults every recipe config is merged over.

## Quick start

```bash
# 0) Edit configs to set paths.
#    Keep `conf/training.yaml:data_dir` as the canonical dataset location.
#    When `--training_config` is also passed to `infer` or `measure`, run.py
#    propagates experiment path fields from training into inference/metrics.
#    Standalone inference or metrics configs must define their own `exp_tag`
#    or `exp_dir`.

# 1) Build manifests (run once)
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Extract speaker embeddings, filter by duration, build the token list
python run.py --stages compute_xvectors  --training_config conf/training.yaml
python run.py --stages remove_long_short --training_config conf/training.yaml
python run.py --stages create_token_list --training_config conf/training.yaml

# 3) Collect feature stats
python run.py --stages collect_stats --training_config conf/training.yaml

# 4) Train
python run.py --stages train --training_config conf/training.yaml

# 5) Synthesize
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 6) Score
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml

# 7) Pack and upload the model
python run.py --stages pack_model \
    --training_config conf/training.yaml \
    --publication_config conf/publication.yaml
python run.py --stages upload_model \
    --training_config conf/training.yaml \
    --publication_config conf/publication.yaml

# 8) Pack and upload the Gradio demo
python run.py --stages pack_demo   --demo_config conf/demo.yaml
python run.py --stages upload_demo --demo_config conf/demo.yaml
```

Stages always execute in the order above, whatever order `--stages` lists
them in, so `--stages all` runs the whole pipeline.

## Demo

The default `src/app.py` reads its input/output components from the packed
`demo.yaml`, so a single-speaker TTS model needs no code: text in, audio out.

A model that needs an input no built-in UI asset can produce - for example the
speaker embedding of a multi-speaker model - should ship its own `src/app.py`
and point `ui.app_script` at it. See `egs3/libritts/esp2_gan_tts/src/app.py`.
