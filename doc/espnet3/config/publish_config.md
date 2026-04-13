---
title: ESPnet3 Publish Configuration
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Publish Configuration

This page explains the `publish.yaml` schema used by pack/upload stages.
It controls how model artifacts are assembled and optionally uploaded.

## Minimum required keys (typical pack/upload)

Required for packing:

- `pack_model.out_dir` (output directory)
- `pack_model.strategy` (or rely on default if set elsewhere)

Required for upload:

- `upload_model.hf_repo`

Common optional:

- `pack_model.infer_dir` (to pull metrics into README)
- `pack_model.include` / `pack_model.exclude`
- `pack_model.files` / `pack_model.yaml_files`

Minimal pack example:

```yaml
pack_model:
  strategy: auto
  out_dir: exp/model_pack
```

Minimal upload example:

```yaml
upload_model:
  hf_repo: yourname/your-model-repo
```

## ✅ Config sections overview

| Section | Description |
| --- | --- |
| `pack_model` | Packaging strategy and file selection. |
| `upload_model` | Upload destination for model artifacts. |

## Core config layout

```yaml
pack_model:
  strategy: auto  # auto|espnet2|espnet3

  # espnet2 packer only
  task: asr
  out_dir: exp/model_pack

  # used to locate scores.json from the measure stage
  # (note: some pipelines may write measures.json instead)
  infer_dir: ${exp_dir}/infer

  # extra files/dirs outside exp/<exp_tag> to include in the archive
  include:
    - ${load_yaml:conf/train.yaml,tokenizer.save_path}
  exclude:
    - "**/*.log"
    - "**/tensorboard/**"
    - "**/wandb/**"

  # used to generate meta.yaml (for espnet2-style from_pretrained)
  files:
    asr_model_file: ${exp_dir}/valid.acc.best.pth
    lm_file: ${exp_dir}/lm.pth
    ngram_file: ${exp_dir}/ngram.arpa
    bpemodel: ${exp_dir}/tokenizer/bpe.model

  # used to generate meta.yaml (for espnet2-style from_pretrained)
  yaml_files:
    asr_train_config: ${exp_dir}/config.yaml
    lm_train_config: ${exp_dir}/lm_config.yaml

upload_model:
  hf_repo: yourname/your-model-repo
```

### Notes

- `upload_model` uses `huggingface-cli`, so login beforehand (`huggingface-cli login`)
  or set an API token via `HF_TOKEN` in the environment.
