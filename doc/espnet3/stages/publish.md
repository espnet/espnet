---
title: ESPnet3 Publish-related Stages
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Publish-related Stages

ESPnet3 provides publish-related stages to package model artifacts and
optionally upload them:

- `pack_model`: build a model package directory (README/meta + artifacts)
- `upload_model`: upload the packed directory (e.g., to Hugging Face Hub)

## Quick usage

### Run

```bash
python run.py \
  --stages pack_model upload_model \
  --train_config conf/train.yaml \
  --publish_config conf/publish.yaml
```

### Configure (in `publish.yaml`)

Keep the core settings in `publish.yaml`. For the full list, see
[Publish configuration](../config/publish_config.md).

| Config section | Description |
| -------------- | ----------- |
| `pack_model` | Packaging strategy, output directory, and files to include. |
| `upload_model` | Hugging Face repo settings for uploading the package. |

### Outputs

Typical output is a package directory (default: `<exp_dir>/model_pack`) containing a `README.md`, `meta.yaml`, and copied artifacts.

## Developer Notes

### `pack_model` details

`pack_model` gathers files from the experiment directory and builds a package
directory (default: `<exp_dir>/model_pack`). It also generates metadata and an
optional README.

Key behaviors:

| Key | Meaning |
| --- | --- |
| `strategy` | `auto` selects `espnet2` when `train_config.task` is set, otherwise `espnet3`. |
| `out_dir` | Output directory for the package. |
| `infer_dir` | Where `scores.json` (or `measures.json`) is searched for README metrics (if present). If unset, it falls back to `infer_config.infer_dir`. |
| `files` / `yaml_files` | Used to generate `meta.yaml` for espnet2-style `from_pretrained`. |
| `include` / `exclude` | Extra file globs to include or skip. |
| `readme_template` | Optional template for README generation. |

Example output tree:

```text
<pack_dir>/
├── README.md
├── meta.yaml
└── exp/
    └── ...
```

### upload_model details

`upload_model` uploads the packed directory to Hugging Face.

Key behaviors:

| Key | Meaning |
| --- | --- |
| `upload_model.hf_repo` | Required (e.g., `yourname/your-model-repo`). |
| `huggingface-cli upload` | Used under the hood; make sure you are logged in or set `HF_TOKEN`. |
| `out_dir` | The directory uploaded as a model repo. |
