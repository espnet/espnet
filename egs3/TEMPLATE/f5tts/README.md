# ESPnet3 F5-TTS recipe template

Starter files for a recipe that trains F5-TTS (`espnet3.systems.f5tts`) through
`espnet3.systems.f5tts.system.F5TTSSystem`.
The configs under `conf/` are scaffolds: they name the keys each stage reads and set no model, data or optimization values.
`run.py` merges a recipe's config of the same name over them, so a recipe's configs must be complete on their own.
This directory is not a runnable recipe: it has no `dataset/`.
See `egs3/libritts/f5tts` for a complete recipe.

## What a recipe adds

- `dataset/`: a `DatasetBuilder` whose `build` writes one TSV manifest per split to `data/manifest/<split>.tsv`, with `utt_id<TAB>wav_path<TAB>text` on each line (further columns are kept as they are), and a `Dataset` that reads a manifest.
  Training samples provide `speech` and `text`.
  Inference samples provide `text`, `reference_speech` and `reference_text`.
- `conf/training.yaml`: the complete training config (`model._target_: espnet3.systems.f5tts.f5tts.F5TTS`, the `remove_long_short` and `create_token_list` blocks, dataset entries, optimizer, scheduler, dataloader, trainer).
  Keep it complete rather than a list of overrides: inference and a packed model rebuild the model and the tokenizer from this one file.
- `conf/inference.yaml`: `model._target_: espnet3.systems.f5tts.inference.Inference` with its arguments, `input_key: [text, reference_speech, reference_text]`, `output_fn`, `output_artifacts` and the `dataset.test` entries.
- `conf/metrics.yaml`: the metrics to compute.
- `conf/publication.yaml`: `pack_model.include` (the recipe's `src`, `conf` and token list) and `exclude`, plus the README template to render.
- `conf/demo.yaml`: `model.trust_user_code` when the packed inference config names recipe code, `pack.requirements` and the Space README template.
- `src/inference.py`: a copy of this template's `build_output`, extended with the columns the metrics need.
- `src/app.py`: a copy of this template's demo launcher.
- `run.py`: the thin re-export below.

```python
from egs3.TEMPLATE.f5tts.run import (
    DEFAULT_STAGES,
    build_parser,
    main,
    parse_cli_and_stage_args,
)
from espnet3.systems.f5tts.system import F5TTSSystem

if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    main(args=args, system_cls=F5TTSSystem, stages=DEFAULT_STAGES)
```

## Quick start

```bash
# 1) Build manifests (run once)
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Filter by duration, then build the token list from the filtered manifest
python run.py --stages remove_long_short --training_config conf/training.yaml
python run.py --stages create_token_list --training_config conf/training.yaml

# 3) Collect feature shapes for batching
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
    --inference_config conf/inference.yaml \
    --publication_config conf/publication.yaml
python run.py --stages upload_model \
    --training_config conf/training.yaml \
    --publication_config conf/publication.yaml

# 8) Pack and upload the Gradio demo
python run.py --stages pack_demo \
    --training_config conf/training.yaml \
    --demo_config conf/demo.yaml
python run.py --stages upload_demo \
    --training_config conf/training.yaml \
    --demo_config conf/demo.yaml
```

Stages always execute in the order above, whatever order `--stages` lists them in, so `--stages all` runs the whole pipeline.

## Using a packed model

`pack_model` writes a self-contained bundle to `exp/<exp_tag>/model_pack`.
It holds the checkpoint, the training config the model is rebuilt from, the token list, and the recipe's `src/`.
The vocoder is not bundled: Vocos is fetched from the Hugging Face Hub on first use unless `model.vocoder_path` names a local copy.

```python
from espnet3.api.inference import load

model = load("exp/<exp_tag>/model_pack")
output = model(
    "The text to synthesize.",
    "reference.wav",
    "The transcript of the reference recording.",
)
samples, sample_rate = output["wav"].array, output["wav"].rate
```

## Demo

`src/app.py` reads its input and output components from the packed `demo.yaml`: target text, reference speech and reference transcript in, synthesized speech out.
Leaving the reference transcript empty treats the reference as a recording of the target text itself, which is only right when it is one.
